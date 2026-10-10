#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Worker observer for managing pipeline frame observers.

This module provides a proxy observer system that manages multiple observers
for pipeline frame events, ensuring that observer processing doesn't block
the main pipeline execution.
"""

import asyncio
from typing import Any

from attr import dataclass

from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed


@dataclass
class Proxy:
    """Proxy data for managing observer tasks and queues.

    This represents is the data received from the main observer that
    is queued for later processing.

    Parameters:
        queue: Queue for frame data awaiting observer processing.
        task: Asyncio task running the observer's frame processing loop.
        observer: The actual observer instance being proxied.
    """

    queue: asyncio.Queue | None
    task: asyncio.Task | None
    observer: BaseObserver
    wants_push: bool = True
    wants_process: bool = True
    ignored: tuple = ()


class _PipelineStartedSignal:
    """Internal sentinel queued to observers when the pipeline has started."""

    pass


class WorkerObserver(BaseObserver):
    """Proxy observer that manages multiple observers without blocking the pipeline.

    This is a pipeline frame observer that is meant to be used as a proxy to
    the user provided observers. That is, this is the observer that should be
    passed to the frame processors. Then, every time a frame is pushed this
    observer will call all the observers registered to the pipeline worker.

    This observer makes sure that passing frames to observers doesn't block the
    pipeline by creating a queue and a worker for each user observer. When a frame
    is received, it will be put in a queue for efficiency and later processed by
    each worker.
    """

    def __init__(
        self,
        *,
        observers: list[BaseObserver] | None = None,
        **kwargs,
    ):
        """Initialize the WorkerObserver.

        Args:
            observers: List of observers to manage. Defaults to empty list.
            **kwargs: Additional arguments passed to the base observer.
        """
        super().__init__(**kwargs)
        self._observers = observers or []
        self._proxies: dict[BaseObserver, Proxy] | None = (
            None  # Becomes a dict after start() is called
        )
        self._wants_process = True

    def add_observer(self, observer: BaseObserver):
        """Add a new observer to the managed list.

        Args:
            observer: The observer to add.
        """
        # Add the observer to the list.
        self._observers.append(observer)

        # If we already started, create a new proxy for the observer.
        # Otherwise, it will be created in start().
        if self._proxies:
            proxy = self._create_proxy(observer)
            self._proxies[observer] = proxy
            self._refresh_wants_process()

    async def remove_observer(self, observer: BaseObserver):
        """Remove an observer and clean up its resources.

        Args:
            observer: The observer to remove.
        """
        # If the observer has a proxy, remove it.
        if self._proxies and observer in self._proxies:
            proxy = self._proxies[observer]
            # Remove the proxy so it doesn't get called anymore.
            del self._proxies[observer]
            self._refresh_wants_process()
            # Cancel the proxy worker right away.
            if proxy.task is not None:
                await self.cancel_task(proxy.task)

        # Remove the observer from the list.
        if observer in self._observers:
            self._observers.remove(observer)

    async def start(self):
        """Start all proxy observer tasks."""
        self._proxies = self._create_proxies(self._observers)
        self._refresh_wants_process()

    async def stop(self):
        """Stop all proxy observer tasks."""
        if not self._proxies:
            return

        for proxy in self._proxies.values():
            if proxy.task is not None:
                await self.cancel_task(proxy.task)

    async def cleanup(self):
        """Cleanup all proxy observers."""
        await super().cleanup()

        if not self._proxies:
            return

        for observer in self._proxies:
            await observer.cleanup()

    async def on_pipeline_started(self):
        """Forward pipeline started signal to all managed observers."""
        await self._send_to_proxy(_PipelineStartedSignal())

    @property
    def wants_process_events(self) -> bool:
        """Whether any managed observer handles ``on_process_frame``.

        When none does, a processor need not even build the event. Read on
        every frame hop, so it is kept up to date rather than computed.
        """
        return self._wants_process

    def _refresh_wants_process(self):
        self._wants_process = self._proxies is None or any(
            p.wants_process for p in self._proxies.values()
        )

    async def on_process_frame(self, data: FrameProcessed):
        """Hand frame-processed data to the observers that handle it.

        Args:
            data: The frame processing event data to distribute to observers.
        """
        await self._send_to_proxy(data, process=True)

    async def on_push_frame(self, data: FramePushed):
        """Hand frame-pushed data to the observers that handle it.

        Args:
            data: The frame push event data to distribute to observers.
        """
        await self._send_to_proxy(data, process=False)

    def _create_proxy(self, observer: BaseObserver) -> Proxy:
        """Create a proxy for a single observer.

        An observer is handed only the events it implements (a handler left as
        ``BaseObserver``'s no-op is never queued) and none for its
        ``ignored_frame_types``; an ``inline_dispatch`` observer is called
        directly, with no queue or task of its own.
        """
        cls = type(observer)
        wants_push = cls.on_push_frame is not BaseObserver.on_push_frame
        wants_process = cls.on_process_frame is not BaseObserver.on_process_frame
        ignored = tuple(getattr(observer, "ignored_frame_types", ()) or ())
        if getattr(observer, "inline_dispatch", False):
            return Proxy(
                queue=None,
                task=None,
                observer=observer,
                wants_push=wants_push,
                wants_process=wants_process,
                ignored=ignored,
            )
        queue = asyncio.Queue()
        task = self.create_task(self._proxy_task_handler(queue, observer))
        proxy = Proxy(
            queue=queue,
            task=task,
            observer=observer,
            wants_push=wants_push,
            wants_process=wants_process,
            ignored=ignored,
        )
        return proxy

    def _create_proxies(self, observers: list[BaseObserver]) -> dict[BaseObserver, Proxy]:
        """Create proxies for all observers."""
        proxies = {}
        for observer in observers:
            proxy = self._create_proxy(observer)
            proxies[observer] = proxy
        return proxies

    async def _send_to_proxy(self, data: Any, process: bool | None = None):
        if not self._proxies:
            return
        frame = getattr(data, "frame", None)
        for proxy in self._proxies.values():
            if process is not None:
                if process and not proxy.wants_process:
                    continue
                if not process and not proxy.wants_push:
                    continue
                if proxy.ignored and isinstance(frame, proxy.ignored):
                    continue
            if proxy.queue is None:
                await self._dispatch(proxy.observer, data)
            else:
                await proxy.queue.put(data)

    @staticmethod
    async def _dispatch(observer: BaseObserver, data: Any):
        if isinstance(data, _PipelineStartedSignal):
            await observer.on_pipeline_started()
        elif isinstance(data, FramePushed):
            await observer.on_push_frame(data)
        elif isinstance(data, FrameProcessed):
            await observer.on_process_frame(data)

    async def _proxy_task_handler(self, queue: asyncio.Queue, observer: BaseObserver):
        """Handle frame processing for a single observer."""
        while True:
            data = await queue.get()

            if isinstance(data, _PipelineStartedSignal):
                await observer.on_pipeline_started()
            elif isinstance(data, FramePushed):
                await observer.on_push_frame(data)
            elif isinstance(data, FrameProcessed):
                await observer.on_process_frame(data)

            queue.task_done()
