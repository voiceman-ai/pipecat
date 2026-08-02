#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame processing pipeline infrastructure for Pipecat.

This module provides the core frame processing system that enables building
audio/video processing pipelines. It includes frame processors, pipeline
management, and frame flow control mechanisms.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
import traceback
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
)

from loguru import logger

from pipecat.clocks.base_clock import BaseClock
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    FrameProcessorPauseFrame,
    FrameProcessorPauseUrgentFrame,
    FrameProcessorResumeFrame,
    FrameProcessorResumeUrgentFrame,
    HeartbeatFrame,
    InterruptionFrame,
    StartFrame,
    SystemFrame,
    TTSAudioRawFrame,
    UninterruptibleFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject

# Frame-push tracing is opt-in: the logger.trace() calls on the per-frame push
# path run twice per frame hop (thousands of times per second per live
# pipeline) and loguru builds their f-string arguments even when the TRACE
# level is filtered out. Set PIPECAT_TRACE_FRAMES=1 (with a TRACE-level sink)
# to enable them.
_TRACE_FRAMES = os.getenv("PIPECAT_TRACE_FRAMES", "").lower() in ("1", "true", "yes")
from pipecat.utils.deprecation import deprecated
from pipecat.utils.frame_queue import FrameQueue

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker


class FrameDirection(Enum):
    """Direction of frame flow in the processing pipeline.

    Parameters:
        DOWNSTREAM: Frames flowing from input to output.
        UPSTREAM: Frames flowing back from output to input.
    """

    DOWNSTREAM = 1
    UPSTREAM = 2


FrameCallback = Callable[["FrameProcessor", Frame, FrameDirection], Awaitable[None]]


@dataclass
class FrameProcessorSetup:
    """Configuration parameters for frame processor initialization.

    Parameters:
        clock: The clock instance for timing operations.
        task_manager: The task manager for handling async operations.
        pipeline_worker: The :class:`PipelineWorker` running this pipeline. Stored
            on each processor as ``self.pipeline_worker`` so processors can
            reach task-scoped state (e.g. ``self.pipeline_worker.app_resources``).
        observer: Optional observer for monitoring frame processing events.
        tool_resources: Deprecated. :class:`PipelineWorker` continues to populate
            this with ``app_resources`` so that custom :class:`FrameProcessor`
            subclasses whose ``setup()`` overrides read ``setup.tool_resources``
            keep working. New code should read
            ``setup.pipeline_worker.app_resources`` instead.

            .. deprecated:: 1.2.0
                Read ``setup.pipeline_worker.app_resources`` instead. Will be
                removed in 2.0.0.
    """

    clock: BaseClock
    task_manager: BaseTaskManager
    pipeline_worker: PipelineWorker
    observer: BaseObserver | None = None
    tool_resources: Any = None

    def __getattribute__(self, name: str) -> Any:
        # Warn when user code reads the deprecated ``tool_resources`` field.
        # Set is unaffected (goes through ``__setattr__``), so PipelineWorker can
        # populate it for backwards compat without tripping the warning.
        if name == "tool_resources":
            value = object.__getattribute__(self, "tool_resources")
            if value is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(
                        "`FrameProcessorSetup.tool_resources` is deprecated since 1.2.0; "
                        "read `setup.pipeline_worker.app_resources` instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
            return value
        return object.__getattribute__(self, name)


class FrameProcessorQueue(asyncio.PriorityQueue):
    """A priority queue for systems frames and other frames.

    This is a specialized queue for frame processors that separates and
    prioritizes system frames over other frames. It ensures that `SystemFrame`
    objects are processed before any other frames by using a priority queue.

    """

    HIGH_PRIORITY = 1
    LOW_PRIORITY = 2

    def __init__(self):
        """Initialize the FrameProcessorQueue."""
        super().__init__()
        self.__high_counter = 0
        self.__low_counter = 0

    async def put(self, item: tuple[Frame, FrameDirection, FrameCallback | None]):
        """Put an item into the priority queue.

        System frames (`SystemFrame`) have higher priority than any other
        frames. If a non-frame item (e.g. a watchdog cancellation sentinel) is
        provided it will have the highest priority.

        Args:
            item (Any): The item to enqueue.

        """
        frame, _, _ = item
        if isinstance(frame, SystemFrame):
            self.__high_counter += 1
            await super().put((self.HIGH_PRIORITY, self.__high_counter, item))
        else:
            self.__low_counter += 1
            await super().put((self.LOW_PRIORITY, self.__low_counter, item))

    async def get(self) -> Any:
        """Retrieve the next item from the queue.

        System frames are prioritized. If both queues are empty, this method
        waits until an item is available.

        Returns:
            Any: The next item from the system or main queue.

        """
        _, _, item = await super().get()
        return item


# CAREFUL — these are grace thresholds, not bounds. `TaskManager.cancel_task`
# cannot abandon a cancel it has started: cancelling an asyncio task and
# awaiting it always waits for the cancellation to complete (asyncio.wait_for
# is no exception — it cancels the inner task and then waits for it before
# raising TimeoutError). Crossing the threshold logs a warning naming the
# blocking frame and switches to periodic re-cancellation (a single
# CancelledError can be absorbed by intermediate layers, leaving a zombie a
# later cancel still reaps), but the wait itself stays unbounded.
#
# The practical consequence, seen in production: a process task that takes 3s to
# cancel serializes 3s of barge-in propagation, because an interruption cancels
# it inline from the input task. The interruption path therefore does NOT use
# these: it bounds its inline wait with PROCESS_TASK_ORPHAN_TIMEOUT_SECS and
# ORPHANS a task that will not die (see `_start_interruption`), so a wedged
# cancel can no longer freeze the pipeline (prod run 1931: an absorbed cancel
# blocked the input task for 53s and the call went silent).
INPUT_TASK_CANCEL_TIMEOUT_SECS = 3
PROCESS_TASK_CANCEL_TIMEOUT_SECS = 3

# How long an interruption waits inline for the process task to cancel before
# orphaning it and continuing on a fresh task. Generous next to a normal
# cancel (sub-millisecond) yet short enough that a barge-in never freezes the
# pipeline behind a wedged cancel. An orphaned task is barred from the frame
# flow — its pushes are dropped and it exits at its next loop turn — and a
# background reaper keeps re-cancelling it until it dies.
PROCESS_TASK_ORPHAN_TIMEOUT_SECS = 1.0


class FrameProcessor(BaseObject):
    """Base class for all frame processors in the pipeline.

    Frame processors are the building blocks of Pipecat pipelines, they can be
    linked to form complex processing pipelines. They receive frames, process
    them, and pass them to the next or previous processor in the chain.  Each
    frame processor guarantees frame ordering and processes frames in its own
    task. System frames are also processed in a separate task which guarantees
    frame priority.

    Event handlers available:

    - on_before_process_frame: Called before a frame is processed
    - on_after_process_frame: Called after a frame is processed
    - on_before_push_frame: Called before a frame is pushed
    - on_after_push_frame: Called after a frame is pushed
    - on_error: Called when an error is raised in the frame processing.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        enable_direct_mode: bool = False,
        metrics: FrameProcessorMetrics | None = None,
        **kwargs,
    ):
        """Initialize the frame processor.

        Args:
            name: Optional name for this processor instance.
            enable_direct_mode: Whether to process frames immediately or use internal queues.
            metrics: Optional metrics collector for this processor.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(name=name, **kwargs)
        self._prev: FrameProcessor | None = None
        self._next: FrameProcessor | None = None

        # Enable direct mode to skip queues and process frames right away.
        self._enable_direct_mode = enable_direct_mode

        # Clock
        self._clock: BaseClock | None = None

        # Observer
        self._observer: BaseObserver | None = None

        # Pipeline Task. Populated by ``setup()``; accessing the
        # ``pipeline_worker`` property before setup raises.
        self._pipeline_worker: PipelineWorker | None = None  # set in setup()

        # Other properties
        self._enable_metrics = False
        self._enable_usage_metrics = False
        self._report_only_initial_ttfb = False

        # Indicates whether we have received the StartFrame.
        self.__started = False

        # Cancellation is done through CancelFrame (a system frame). This could
        # cause other events being triggered (e.g. closing a transport) which
        # could also cause other frames to be pushed from other tasks
        # (e.g. EndFrame). So, when we are cancelling we don't want anything
        # else to be pushed.
        self._cancelling = False

        # Metrics
        self._metrics = metrics or FrameProcessorMetrics()
        self._metrics.set_processor_name(self.name)

        # Processors have an input priority queue which stores any type of
        # frames in order. System frames have higher priority than any other
        # frames, so they will be returned first from the queue.
        #
        # If a system frame is obtained it will be processed immediately any
        # other type of frame (data and control) will be put in a separate queue
        # for later processing. This guarantees that each frame processor will
        # always process system frames before any other frame in the queue.

        # The input task that handles all types of frames. It processes system
        # frames right away and queues non-system frames for later processing.
        self.__should_block_system_frames = False
        self.__input_queue = FrameProcessorQueue()
        self.__input_event: asyncio.Event | None = None
        self.__input_frame_task: asyncio.Task | None = None

        # The process task processes non-system frames.  Non-system frames will
        # be processed as soon as they are received by the processing task
        # (default) or they will block if `pause_processing_frames()` is
        # called. To resume processing frames we need to call
        # `resume_processing_frames()` which will wake up the event.
        self.__should_block_frames = False
        self.__process_queue = FrameQueue(frame_getter=lambda item: item[0])
        self.__process_event: asyncio.Event | None = None
        self.__process_frame_task: asyncio.Task | None = None
        self.__process_current_frame: Frame | None = None

        # Read-only diagnostics (see the `input_queue_depth`,
        # `process_queue_depth` and `seconds_since_last_progress` properties).
        # Monotonic time at which __process_frame last accepted a frame, or 0.0
        # if this processor has never processed one. Deliberately NOT wired
        # into any kill/abort decision: a wedged-before-first-turn leg must be
        # diagnosable from a log dump without changing pipeline behavior.
        self.__last_progress_time: float = 0.0

        # Process tasks an interruption gave up on cancelling (see
        # `_start_interruption`). Membership bars a task from the frame flow:
        # `push_frame` drops its pushes and the handler loop exits instead of
        # consuming queued frames. Entries are removed when the background
        # reaper finally kills the task, so the set stays empty in healthy
        # operation.
        self.__orphaned_process_tasks: set[asyncio.Task] = set()

        # Frame processor events.
        self._register_event_handler("on_before_process_frame", sync=True)
        self._register_event_handler("on_after_process_frame", sync=True)
        self._register_event_handler("on_before_push_frame", sync=True)
        self._register_event_handler("on_after_push_frame", sync=True)
        self._register_event_handler("on_error", sync=True)

    @property
    def id(self) -> int:
        """Get the unique identifier for this processor.

        Returns:
            The unique integer ID of this processor.
        """
        return self._id

    @property
    def name(self) -> str:
        """Get the name of this processor.

        Returns:
            The name of this processor instance.
        """
        return self._name

    @property
    def processors(self) -> list[FrameProcessor]:
        """Return the list of sub-processors contained within this processor.

        Only compound processors (e.g. pipelines and parallel pipelines) have
        sub-processors. Non-compound processors will return an empty list.

        Returns:
            The list of sub-processors if this is a compound processor.
        """
        return []

    @property
    def entry_processors(self) -> list[FrameProcessor]:
        """Return the list of entry processors for this processor.

        Entry processors are the first processors in a compound processor
        (e.g. pipelines, parallel pipelines). Note that pipelines can also be an
        entry processor as pipelines are processors themselves. Non-compound
        processors will simply return an empty list.

        Returns:
            The list of entry processors.
        """
        return []

    @property
    def next(self) -> FrameProcessor | None:
        """Get the next processor.

        Returns:
            The next processor, or None if there's no next processor.
        """
        return self._next

    @property
    def previous(self) -> FrameProcessor | None:
        """Get the previous processor.

        Returns:
            The previous processor, or None if there's no previous processor.
        """
        return self._prev

    @property
    def metrics_enabled(self):
        """Check if metrics collection is enabled.

        Returns:
            True if metrics collection is enabled.
        """
        return self._enable_metrics

    @property
    def usage_metrics_enabled(self):
        """Check if usage metrics collection is enabled.

        Returns:
            True if usage metrics collection is enabled.
        """
        return self._enable_usage_metrics

    @property
    def report_only_initial_ttfb(self):
        """Check if only initial TTFB should be reported.

        Returns:
            True if only initial time-to-first-byte should be reported.
        """
        return self._report_only_initial_ttfb

    @property
    def pipeline_worker(self) -> PipelineWorker:
        """Get the :class:`PipelineWorker` this processor is running in.

        Provides access to worker-scoped state from inside a processor — most
        notably ``self.pipeline_worker.app_resources`` for the application's
        shared bag of resources (DB handles, clients, feature flags, etc.).

        Returns:
            The :class:`PipelineWorker` instance that set up this processor.
        """
        if not self._pipeline_worker:
            raise Exception(f"{self} pipeline worker is still not set.")
        return self._pipeline_worker

    @property
    @deprecated(
        "`FrameProcessor.pipeline_task` is deprecated since 1.3.0 and will be removed in 2.0.0. "
        "Use `pipeline_worker` instead."
    )
    def pipeline_task(self) -> PipelineWorker:
        """Deprecated alias for :attr:`pipeline_worker`.

        .. deprecated:: 1.3.0
            Use :attr:`pipeline_worker` instead. Will be removed in 2.0.0.
        """
        return self.pipeline_worker

    def processors_with_metrics(self):
        """Return processors that can generate metrics.

        Recursively collects all processors that support metrics generation,
        including those from nested processors.

        Returns:
            List of frame processors that can generate metrics.
        """
        return []

    def can_generate_metrics(self) -> bool:
        """Check if this processor can generate metrics.

        Returns:
            True if this processor can generate metrics.
        """
        return False

    def set_core_metrics_data(self, data: MetricsData):
        """Set core metrics data for this processor.

        Args:
            data: The metrics data to set.
        """
        self._metrics.set_core_metrics_data(data)

    async def start_ttfb_metrics(self, *, start_time: float | None = None):
        """Start time-to-first-byte metrics collection.

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_ttfb_metrics(
                start_time=start_time, report_only_initial_ttfb=self._report_only_initial_ttfb
            )

    async def stop_ttfb_metrics(self, *, end_time: float | None = None):
        """Stop time-to-first-byte metrics collection and push results.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_ttfb_metrics(end_time=end_time)
            if frame:
                await self.push_frame(frame)

    async def process_ttfa_metrics(self, frame: TTSAudioRawFrame):
        """Scan a TTS audio frame for the first audible sample and push TTFA.

        Should be called for every audio frame until a measurement is produced;
        the metrics collector tracks leading silence across chunks internally.

        Args:
            frame: The TTS audio frame to inspect.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            metrics_frame = await self._metrics.process_ttfa_metrics(
                audio=frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            if metrics_frame:
                await self.push_frame(metrics_frame)

    async def start_processing_metrics(self, *, start_time: float | None = None):
        """Start processing metrics collection.

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_processing_metrics(start_time=start_time)

    async def stop_processing_metrics(self, *, end_time: float | None = None):
        """Stop processing metrics collection and push results.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_processing_metrics(end_time=end_time)
            if frame:
                await self.push_frame(frame)

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        """Start LLM usage metrics collection.

        Args:
            tokens: Token usage information for the LLM.
        """
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_llm_usage_metrics(tokens)
            if frame:
                await self.push_frame(frame)

    async def start_tts_usage_metrics(self, text: str):
        """Start TTS usage metrics collection.

        Args:
            text: The text being processed by TTS.
        """
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_tts_usage_metrics(text)
            if frame:
                await self.push_frame(frame)

    async def start_text_aggregation_metrics(self):
        """Start text aggregation time metrics collection."""
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_text_aggregation_metrics()

    async def stop_text_aggregation_metrics(self):
        """Stop text aggregation time metrics collection and push results."""
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_text_aggregation_metrics()
            if frame:
                await self.push_frame(frame)

    async def stop_all_metrics(self):
        """Stop all active metrics collection."""
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()
        await self.stop_text_aggregation_metrics()

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup.task_manager)
        self._clock = setup.clock
        self._observer = setup.observer
        self._pipeline_worker = setup.pipeline_worker

        # Create processing tasks.
        self.__create_input_task()

        if self._metrics is not None:
            await self._metrics.setup(self.task_manager)

    async def cleanup(self):
        """Release this processor's resources at teardown.

        This base implementation cancels only the processor's internal
        input/process tasks; tasks created via :meth:`create_task` are released
        by an override.
        """
        await super().cleanup()
        await self.__cancel_input_task()
        await self.__cancel_process_task()
        if self._metrics is not None:
            await self._metrics.cleanup()

    def link(self, processor: FrameProcessor):
        """Link this processor to the next processor in the pipeline.

        Args:
            processor: The processor to link to.
        """
        self._next = processor
        processor._prev = self
        logger.trace(f"Linking {self} -> {self._next}")

    def get_clock(self) -> BaseClock:
        """Get the clock used by this processor.

        Returns:
            The clock instance.

        Raises:
            Exception: If the clock is not initialized.
        """
        if not self._clock:
            raise Exception(f"{self} Clock is still not initialized.")
        return self._clock

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop used by this processor.

        Returns:
            The asyncio event loop.
        """
        return self.task_manager.get_event_loop()

    async def queue_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        callback: FrameCallback | None = None,
    ):
        """Queue a frame for processing.

        Args:
            frame: The frame to queue.
            direction: The direction of frame flow.
            callback: Optional callback to call after processing.
        """
        # If we are cancelling we don't want to process any other frame.
        if self._cancelling:
            return

        if self._enable_direct_mode:
            await self.__process_frame(frame, direction, callback)
        else:
            await self.__input_queue.put((frame, direction, callback))

    async def pause_processing_frames(self):
        """Pause processing of queued frames."""
        logger.trace(f"{self}: pausing frame processing")
        self.__should_block_frames = True
        if self.__process_event:
            self.__process_event.clear()

    async def pause_processing_system_frames(self):
        """Pause processing of queued system frames."""
        logger.trace(f"{self}: pausing system frame processing")
        self.__should_block_system_frames = True
        if self.__input_event:
            self.__input_event.clear()

    async def resume_processing_frames(self):
        """Resume processing of queued frames."""
        logger.trace(f"{self}: resuming frame processing")
        if self.__process_event:
            self.__process_event.set()

    async def resume_processing_system_frames(self):
        """Resume processing of queued system frames."""
        logger.trace(f"{self}: resuming system frame processing")
        if self.__input_event:
            self.__input_event.set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        if self._observer:
            timestamp = self._clock.get_time() if self._clock else 0
            data = FrameProcessed(
                processor=self,
                frame=frame,
                direction=direction,
                timestamp=timestamp,
            )
            await self._observer.on_process_frame(data)

        if isinstance(frame, StartFrame):
            await self.__start(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._start_interruption()
            await self.stop_all_metrics()
        elif isinstance(frame, CancelFrame):
            await self.__cancel(frame)
        elif isinstance(frame, (FrameProcessorPauseFrame, FrameProcessorPauseUrgentFrame)):
            await self.__pause(frame)
        elif isinstance(frame, (FrameProcessorResumeFrame, FrameProcessorResumeUrgentFrame)):
            await self.__resume(frame)

    async def push_error(
        self,
        error_msg: str,
        exception: Exception | None = None,
        fatal: bool = False,
    ):
        """Creates and pushes an ErrorFrame upstream.

        Creates and pushes an ErrorFrame upstream to notify other processors in the
        pipeline about an error condition. The error frame will include context about
        which processor generated the error.

        Args:
            error_msg: Descriptive message explaining the error condition.
            exception: Optional exception object that caused the error, if available.
                This provides additional context for debugging and error handling.
            fatal: Whether this error should be considered fatal to the pipeline.
                Fatal errors typically cause the entire pipeline to stop processing.
                Defaults to False for non-fatal errors.

        Example::

            ```python
            # Non-fatal error
            await self.push_error("Failed to process audio chunk, skipping")

            # Fatal error with exception context
            try:
                result = some_critical_operation()
            except Exception as e:
                await self.push_error("Critical operation failed", exception=e, fatal=True)
            ```
        """
        error_frame = ErrorFrame(error=error_msg, fatal=fatal, exception=exception, processor=self)
        await self.push_error_frame(error=error_frame)

    async def push_error_frame(self, error: ErrorFrame):
        """Push an error frame upstream.

        Args:
            error: The error frame to push.
        """
        if not error.processor:
            error.processor = self
        await self._call_event_handler("on_error", error)

        if error.exception:
            tb = traceback.extract_tb(error.exception.__traceback__)
            last = tb[-1]
            error_message = (
                f"{error.processor} exception ({last.filename}:{last.lineno}): {error.error}"
            )
        else:
            error_message = f"{error.processor} error: {error.error}"

        logger.error(error_message)
        await self.push_frame(error, FrameDirection.UPSTREAM)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame to the next processor in the pipeline.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        # An orphaned process task may still be unwinding an in-flight frame
        # (see __interrupt_process_task); the interruption already flushed
        # everything it produced, so anything more it pushes is stale.
        if self.__orphaned_process_tasks and asyncio.current_task() in self.__orphaned_process_tasks:
            logger.debug(f"{self}: dropping {frame.name} pushed by an orphaned process task")
            return

        if not self._check_started(frame):
            return

        await self._call_event_handler("on_before_push_frame", frame)

        await self.__internal_push_frame(frame, direction)

        await self._call_event_handler("on_after_push_frame", frame)

    async def broadcast_interruption(self):
        """Broadcast an `InterruptionFrame` both upstream and downstream."""
        logger.debug(f"{self}: broadcasting interruption")
        self.__reset_process_task()
        await self.stop_all_metrics()
        await self.broadcast_frame(InterruptionFrame)

    @deprecated(
        "`FrameProcessor.push_interruption_task_frame_and_wait` is deprecated since 0.0.104 "
        "and will be removed in 2.0.0. Use `broadcast_interruption` instead."
    )
    async def push_interruption_task_frame_and_wait(self, *, timeout: float = 5.0):
        """Push an interruption task frame upstream and wait for the interruption.

        .. deprecated:: 0.0.104
            Use :meth:`broadcast_interruption` instead. This method now
            delegates to ``broadcast_interruption()`` and ignores *timeout*.
            Will be removed in 2.0.0.
        """
        await self.broadcast_interruption()

    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Broadcasts a frame of the specified class upstream and downstream.

        This method creates two instances of the given frame class using the
        provided keyword arguments (without deep-copying them) and pushes them
        upstream and downstream.

        Args:
            frame_cls: The class of the frame to be broadcasted.
            **kwargs: Keyword arguments to be passed to the frame's constructor.
        """
        downstream_frame = frame_cls(**kwargs)
        upstream_frame = frame_cls(**kwargs)
        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)

    async def broadcast_frame_instance(self, frame: Frame):
        """Broadcasts a frame instance upstream and downstream.

        This method creates two new frame instances shallow-copying all fields
        from the original frame except `id` and `name`, which get fresh values.

        Args:
            frame: The frame instance to broadcast.

        Note:
            Prefer using `broadcast_frame()` when possible, as it is more
            efficient. This method should only be used when you are not the
            creator of the frame and need to broadcast an existing instance.
        """
        frame_cls = type(frame)
        init_fields = {f.name: getattr(frame, f.name) for f in dataclasses.fields(frame) if f.init}
        extra_fields = {
            f.name: getattr(frame, f.name)
            for f in dataclasses.fields(frame)
            if not f.init and f.name not in ("id", "name")
        }

        downstream_frame = frame_cls(**init_fields)
        for k, v in extra_fields.items():
            setattr(downstream_frame, k, v)

        upstream_frame = frame_cls(**init_fields)
        for k, v in extra_fields.items():
            setattr(upstream_frame, k, v)

        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)

    async def __start(self, frame: StartFrame):
        """Handle the start frame to initialize processor state.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self.__started = True
        self._enable_metrics = frame.enable_metrics
        self._enable_usage_metrics = frame.enable_usage_metrics
        self._report_only_initial_ttfb = frame.report_only_initial_ttfb

        self.__create_process_task()

    async def __cancel(self, frame: CancelFrame):
        """Handle the cancel frame to stop processor operation.

        Args:
            frame: The cancel frame.
        """
        self._cancelling = True
        await self.__cancel_process_task()

    async def __pause(self, frame: FrameProcessorPauseFrame | FrameProcessorPauseUrgentFrame):
        """Handle pause frame to pause processor operation.

        Args:
            frame: The pause frame.
        """
        if frame.processor.name == self.name:
            await self.pause_processing_frames()

    async def __resume(self, frame: FrameProcessorResumeFrame | FrameProcessorResumeUrgentFrame):
        """Handle resume frame to resume processor operation.

        Args:
            frame: The resume frame.
        """
        if frame.processor.name == self.name:
            await self.resume_processing_frames()

    #
    # Handle interruptions
    #

    async def _start_interruption(self):
        """Start handling an interruption by cancelling current tasks."""
        try:
            # HeartbeatFrame is uninterruptible so an interruption cannot drain
            # in-flight heartbeats out of the process queue (that is what made
            # the heartbeat monitor a barge-in detector rather than a health
            # probe). It must NOT, however, protect the process task from being
            # cancelled: a heartbeat can be the current frame while the task is
            # parked in `pause_processing_frames()`, and taking the
            # reset-only branch there leaves the task parked forever. Heartbeats
            # are cheap and re-issued every period, so losing the one in flight
            # at the instant of an interruption costs nothing.
            current = self.__process_current_frame
            current_is_uninterruptible = isinstance(
                current, UninterruptibleFrame
            ) and not isinstance(current, HeartbeatFrame)
            if current_is_uninterruptible:
                # The frame currently being processed is uninterruptible, so we
                # must not cancel it. Just flush non-uninterruptible frames from
                # the queue; any uninterruptible ones will be kept and processed
                # after the current frame finishes.
                self.__reset_process_queue()
            else:
                # Cancel and re-create the process task. Previously this branch
                # was skipped when the queue contained an uninterruptible frame,
                # which caused slow non-uninterruptible frames to block
                # interruptions. Uninterruptible queued frames are safe here
                # because both the fresh-task and orphan paths preserve them.
                # Bounded: this runs inline on the input task, so a cancel that
                # never completes must not freeze the pipeline — past the bound
                # the old task is orphaned and a fresh one takes over.
                await self.__interrupt_process_task()
        except Exception as e:
            await self.push_error(
                error_msg=f"Uncaught exception handling _start_interruption: {e}",
                exception=e,
            )

    async def __internal_push_frame(self, frame: Frame, direction: FrameDirection):
        """Internal method to push frames to adjacent processors.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        try:
            timestamp = self._clock.get_time() if self._clock else 0
            if direction == FrameDirection.DOWNSTREAM and self._next:
                if _TRACE_FRAMES:
                    logger.trace(f"Pushing {frame} downstream from {self} to {self._next}")

                if self._observer:
                    data = FramePushed(
                        source=self,
                        destination=self._next,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await self._observer.on_push_frame(data)
                await self._next.queue_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
                if _TRACE_FRAMES:
                    logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
                if self._observer:
                    data = FramePushed(
                        source=self,
                        destination=self._prev,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await self._observer.on_push_frame(data)
                await self._prev.queue_frame(frame, direction)
        except Exception as e:
            await self.push_error(error_msg=f"Uncaught exception: {e}", exception=e)

    def _check_started(self, frame: Frame):
        """Check if the processor has been started.

        Args:
            frame: The frame being processed.

        Returns:
            True if the processor has been started.
        """
        if not self.__started:
            logger.error(f"{self} Trying to process {frame} but StartFrame not received yet")
        return self.__started

    def __create_input_task(self):
        """Create the frame input processing task."""
        if self._enable_direct_mode:
            return

        if not self.__input_frame_task:
            self.__input_event = asyncio.Event()
            self.__input_frame_task = self.create_task(self.__input_frame_task_handler())

    async def __cancel_input_task(self):
        """Cancel the frame input processing task."""
        if self.__input_frame_task:
            # Apply a timeout as a safeguard: if a library swallows asyncio.CancelledError,
            # the task would otherwise never be cancelled. With a timeout, we can detect this
            # situation and surface it in the logs instead of hanging indefinitely.
            await self.cancel_task(self.__input_frame_task, INPUT_TASK_CANCEL_TIMEOUT_SECS)
            self.__input_frame_task = None

    def __create_process_task(self):
        """Create the non-system frame processing task."""
        if self._enable_direct_mode:
            return

        if not self.__process_frame_task:
            self.__reset_process_task()
            self.__process_frame_task = self.create_task(self.__process_frame_task_handler())

    def __reset_process_task(self):
        """Reset non-system frame processing task."""
        if self._enable_direct_mode:
            return

        self.__should_block_frames = False
        self.__process_event = asyncio.Event()
        self.__reset_process_queue()

    def __reset_process_queue(self):
        """Reset non-system frame processing queue."""
        self.__process_queue.reset()

    def has_queued_frame(self, frame_type: type[Frame] | type[UninterruptibleFrame]) -> bool:
        """Return True if a frame of the given type is waiting in the processing queue.

        Delegates to :meth:`FrameQueue.has_frame` so the check is O(distinct
        enqueued types) with no queue scanning.  ``frame_type`` may be any
        ``Frame`` subclass or ``UninterruptibleFrame`` (a mixin).

        Args:
            frame_type: The frame class (or mixin) to look for.

        Returns:
            True if at least one matching frame is queued.
        """
        return self.__process_queue.has_frame(frame_type)

    #
    # Read-only wedge diagnostics.
    #
    # These properties exist so a stuck pipeline can be diagnosed from a log
    # dump (see `PipelineWorker.dump_processor_diagnostics()`): a processor
    # whose queues grow while `seconds_since_last_progress` ages is wedged; a
    # stale processor with empty queues simply has no traffic. None of these
    # feed kill or abort decisions.
    #

    @property
    def input_queue_depth(self) -> int:
        """Number of frames waiting in this processor's input queue.

        The input queue holds every queued frame (system and non-system) that
        has not yet been dispatched by the input task. Always 0 in direct
        mode, where frames bypass the queues entirely.

        Returns:
            The number of undispatched frames. Read-only diagnostic.
        """
        return self.__input_queue.qsize()

    @property
    def process_queue_depth(self) -> int:
        """Number of non-system frames waiting in the process queue.

        These frames were dispatched by the input task and are waiting for the
        process task to pick them up (e.g. because a frame is still being
        processed, or processing is paused).

        Returns:
            The number of waiting non-system frames. Read-only diagnostic.
        """
        return self.__process_queue.qsize() if self.__process_queue else 0

    @property
    def seconds_since_last_progress(self) -> float | None:
        """Seconds since this processor last accepted a non-system frame.

        The stamp is taken when a frame *enters* processing, so a processor
        stuck inside a single frame ages here just like one that stopped
        receiving frames; combine with the queue depths and
        `processing_frame_name` to tell the two apart.

        System frames and heartbeats do NOT refresh the stamp: system frames
        are handled by the input task (which keeps running while the process
        task is wedged) and heartbeats are liveness probes, so counting either
        as progress would hide exactly the wedge this property exists to
        expose.

        Returns:
            The age in seconds, or None if no non-system frame was ever
            processed.
        """
        if self.__last_progress_time == 0.0:
            return None
        return time.monotonic() - self.__last_progress_time

    @property
    def processing_frame_name(self) -> str | None:
        """Name of the non-system frame currently held by the process task.

        Non-None while a frame is being processed or is parked behind
        `pause_processing_frames()` — i.e. this names the culprit frame when a
        processor is wedged mid-frame. System frames (processed by the input
        task) are not tracked here.

        Returns:
            The in-flight frame's name, or None when the process task is idle.
        """
        frame = self.__process_current_frame
        return frame.name if frame else None

    @property
    def is_frame_processing_paused(self) -> bool:
        """Whether non-system frame processing is currently paused.

        True between `pause_processing_frames()` and
        `resume_processing_frames()`. A processor that stays paused forever
        (e.g. a TTS service whose resume signal never arrived) reads as True
        here with a growing `process_queue_depth`.

        Returns:
            True if the process task is blocked by a pause. Read-only diagnostic.
        """
        return self.__should_block_frames

    async def __cancel_process_task(self):
        """Cancel the non-system frame processing task."""
        if self.__process_frame_task:
            await self.cancel_task(self.__process_frame_task, PROCESS_TASK_CANCEL_TIMEOUT_SECS)
            self.__process_frame_task = None

    async def __interrupt_process_task(self):
        """Replace the process task for an interruption, orphaning a stuck one.

        Runs inline on the input task, so the wait for the old task is bounded
        by ``PROCESS_TASK_ORPHAN_TIMEOUT_SECS``. A task still alive past the
        bound (typically a ``CancelledError`` absorbed inside library
        internals — prod run 1931 froze a pipeline for 53s waiting on one) is
        orphaned instead of awaited: it is barred from the frame flow (pushes
        dropped, handler loop exits), the process queue is replaced so it can
        never consume another frame, and a background reaper keeps
        re-cancelling it until it dies. Queued uninterruptible frames survive
        onto the replacement queue.
        """
        task = self.__process_frame_task
        if task:
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=PROCESS_TASK_ORPHAN_TIMEOUT_SECS)
            if done:
                # Consume the outcome (normally the CancelledError) with the
                # usual bookkeeping; the task is done, so this returns at once.
                await self.cancel_task(task)
            else:
                logger.warning(
                    f"{task.get_name()}: process task did not cancel within "
                    f"{PROCESS_TASK_ORPHAN_TIMEOUT_SECS}s of an interruption — "
                    f"orphaning it and continuing on a fresh task"
                )
                self.__orphaned_process_tasks.add(task)
                # The handler re-reads the queue attribute each loop turn, so
                # replace the queue: if the zombie ever loops again it exits at
                # the orphan check, and if it is parked in get() it starves on
                # the old queue object. Uninterruptible frames move over.
                old_queue = self.__process_queue
                self.__process_queue = FrameQueue(frame_getter=lambda item: item[0])
                old_queue.reset()
                while not old_queue.empty():
                    self.__process_queue.put_nowait(old_queue.get_nowait())
                self.create_task(self.__reap_orphaned_process_task(task))
            self.__process_frame_task = None
        self.__create_process_task()

    async def __reap_orphaned_process_task(self, task: asyncio.Task):
        """Cancel an orphaned process task until it actually dies."""
        try:
            await self.cancel_task(task, PROCESS_TASK_CANCEL_TIMEOUT_SECS)
        finally:
            self.__orphaned_process_tasks.discard(task)

    async def __process_frame(
        self, frame: Frame, direction: FrameDirection, callback: FrameCallback | None
    ):
        # Progress stamp for `seconds_since_last_progress`. Taken at entry so a
        # processor wedged inside this frame keeps aging instead of looking
        # fresh forever. System frames are excluded: __process_frame is also
        # called from the input task, and a live call keeps delivering system
        # frames (e.g. InputAudioRawFrame) even while the process task is
        # wedged inside one non-system frame — stamping them would read
        # "fresh" during exactly the wedge this diagnostic exists to expose.
        # Heartbeats are excluded for the same reason: they are liveness
        # probes, not work.
        if not isinstance(frame, (SystemFrame, HeartbeatFrame)):
            self.__last_progress_time = time.monotonic()
        try:
            await self._call_event_handler("on_before_process_frame", frame)

            # Process the frame.
            await self.process_frame(frame, direction)
            # If this frame has an associated callback, call it now.
            if callback:
                await callback(self, frame, direction)

            await self._call_event_handler("on_after_process_frame", frame)
        except Exception as e:
            await self.push_error(error_msg=f"Error processing frame: {e}", exception=e)

    async def __input_frame_task_handler(self):
        """Handle frames from the input queue.

        It only processes system frames. Other frames are queue for another task
        to execute.

        """
        while True:
            (frame, direction, callback) = await self.__input_queue.get()

            if self.__should_block_system_frames and self.__input_event:
                logger.trace(f"{self}: system frame processing paused")
                await self.__input_event.wait()
                self.__input_event.clear()
                self.__should_block_system_frames = False
                logger.trace(f"{self}: system frame processing resumed")

            if isinstance(frame, SystemFrame):
                await self.__process_frame(frame, direction, callback)
            elif self.__process_queue:
                await self.__process_queue.put((frame, direction, callback))
            else:
                raise RuntimeError(
                    f"{self}: __process_queue is None when processing frame {frame.name}"
                )

            self.__input_queue.task_done()

    async def __process_frame_task_handler(self):
        """Handle non-system frames from the process queue."""
        while True:
            # An orphaned task (an interruption gave up waiting for its cancel
            # — see __interrupt_process_task) must exit here rather than touch
            # the queue: a fresh task owns the frame flow now.
            if (
                self.__orphaned_process_tasks
                and asyncio.current_task() in self.__orphaned_process_tasks
            ):
                logger.debug(f"{self}: orphaned process task exiting")
                return

            self.__process_current_frame = None

            (frame, direction, callback) = await self.__process_queue.get()

            self.__process_current_frame = frame

            # A heartbeat is a liveness probe, not work: it carries no ordering
            # semantics, which is the same argument that keeps it out of the
            # output transport's paced audio queue. `pause_processing_frames()`
            # is held for an entire utterance playout by every TTS service
            # constructed with `pause_frame_processing=True` (they pause on
            # LLMFullResponseEndFrame and resume on BotStoppedSpeakingFrame), so
            # parking the probe here would make its traversal latency measure
            # playout backlog again -- the exact defect the heartbeat routing fix
            # removed, merely relocated from the transport to the TTS processor.
            # The pause flag is deliberately left armed, so the next non-heartbeat
            # frame still blocks and the ordering of real work is unchanged.
            #
            # Residual, deliberately not addressed here: a heartbeat that lands
            # behind a non-heartbeat frame which is itself parked still waits for
            # the pause to lift, and a heartbeat still queues behind a long
            # in-flight frame operation (e.g. a streamed LLM generation). See the
            # traversal-latency caveat on `PipelineWorker`'s `on_heartbeat`.
            if (
                self.__should_block_frames
                and self.__process_event
                and not isinstance(frame, HeartbeatFrame)
            ):
                logger.trace(f"{self}: frame processing paused")
                await self.__process_event.wait()
                self.__process_event.clear()
                self.__should_block_frames = False
                logger.trace(f"{self}: frame processing resumed")

            await self.__process_frame(frame, direction, callback)

            self.__process_queue.task_done()
