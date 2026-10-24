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
from collections import deque
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
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData, STTUsage
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import deprecated, warn_deprecated
from pipecat.utils.errors import ErrorCategory, classify_http_exception
from pipecat.utils.frame_queue import FrameQueue

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.utils.tracing.tracing_context import TracingContext
    from pipecat.workers.runner import WorkerRunner

# Frame-push tracing is opt-in: the logger.trace() calls on the per-frame push
# path run twice per frame hop (thousands of times per second per live
# pipeline) and loguru builds their f-string arguments even when the TRACE
# level is filtered out. Set PIPECAT_TRACE_FRAMES=1 (with a TRACE-level sink)
# to enable them.
_TRACE_FRAMES = os.getenv("PIPECAT_TRACE_FRAMES", "").lower() in ("1", "true", "yes")


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
        audio_in_sample_rate: Input audio sample rate in Hz.
        audio_out_sample_rate: Output audio sample rate in Hz.
        clock: The clock instance for timing operations.
        enable_metrics: Whether to enable performance metrics collection.
        enable_tracing: Whether to enable OpenTelemetry tracing.
        enable_usage_metrics: Whether to enable usage metrics collection.
        pipeline_worker: The :class:`~pipecat.pipeline.worker.PipelineWorker` running this pipeline. Stored
            on each processor as ``self.pipeline_worker`` so processors can
            reach task-scoped state (e.g. ``self.pipeline_worker.app_resources``).
        observer: Optional observer for monitoring frame processing events.
        task_manager: The task manager for handling async operations.
        tool_resources: Deprecated. :class:`~pipecat.pipeline.worker.PipelineWorker` continues to populate
            this with ``app_resources`` so that custom :class:`FrameProcessor`
            subclasses whose ``setup()`` overrides read ``setup.tool_resources``
            keep working. New code should read
            ``setup.pipeline_worker.app_resources`` instead.

            .. deprecated:: 1.2.0
                Read ``setup.pipeline_worker.app_resources`` instead. Will be
                removed in 2.0.0.
        tracing_context: Pipeline-scoped tracing context for span hierarchy.
    """

    clock: BaseClock
    task_manager: BaseTaskManager
    pipeline_worker: PipelineWorker
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    enable_metrics: bool = False
    enable_tracing: bool = False
    enable_usage_metrics: bool = False
    observer: BaseObserver | None = None
    report_only_initial_ttfb: bool = False
    tracing_context: TracingContext | None = None
    # Deprecated fields
    tool_resources: Any = None

    def __getattribute__(self, name: str) -> Any:
        # Warn when user code reads the deprecated ``tool_resources`` field.
        # Set is unaffected (goes through ``__setattr__``), so PipelineWorker can
        # populate it for backwards compat without tripping the warning.
        if name == "tool_resources":
            value = object.__getattribute__(self, "tool_resources")
            if value is not None:
                warn_deprecated(
                    "`FrameProcessorSetup.tool_resources` is deprecated since 1.2.0 and will "
                    "be removed in 2.0.0. Use `setup.pipeline_worker.app_resources` instead.",
                    stacklevel=2,
                )
            return value
        return object.__getattribute__(self, name)


# The starvation bound to enable (see FrameProcessorQueue). 80ms because:
# - it is above the largest single event-loop stall measured on the stalled
#   api pods (max loop lag per 15s: 21-56ms, versus 3ms on the same pod just
#   before the stall), so one slow iteration cannot trip it and it only acts on
#   sustained starvation;
# - it is four 20ms audio frames, well under the VAD stop window (0.2s) and the
#   0.55-0.8s a healthy turn takes to release after its final transcript, so a
#   starved transcript still lands inside the turn it belongs to;
# - a heartbeat (1s period) crossing one starved processor is held at most
#   this long plus one system frame's service time.
INPUT_QUEUE_STARVATION_BOUND_RECOMMENDED_SECS = 0.08


def _starvation_bound_from_env() -> float:
    """Read ``PIPECAT_INPUT_QUEUE_STARVATION_BOUND_MS`` as seconds (0 = off)."""
    raw = os.getenv("PIPECAT_INPUT_QUEUE_STARVATION_BOUND_MS", "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            f"PIPECAT_INPUT_QUEUE_STARVATION_BOUND_MS={raw!r} is not a number; "
            f"input queues keep strict system-frame priority"
        )
        return 0.0
    return max(0.0, value) / 1000.0


class FrameProcessorQueue(asyncio.Queue):
    """A priority queue for the frames arriving at a frame processor.

    Frames are dequeued in three tiers: the `StartFrame` first, then
    `SystemFrame`, then data and control frames. Frames of the same tier keep
    their arrival order. Items are ``(frame, direction, callback)`` tuples;
    anything whose first element is not a `SystemFrame` is non-system.

    Bounded starvation (opt-in). Strict priority starves non-system frames for
    as long as a system frame is always waiting at the next ``get()``. On a
    busy api pod that is what happened: `InputAudioRawFrame` is a system
    frame arriving every 20ms, and `LLMUserAggregator` hops to a thread
    executor for VAD on each one, so once loop and GIL contention pushed its
    per-frame service time to the 20ms arrival interval a system frame was
    always queued and `TranscriptionFrame`/`HeartbeatFrame` got no service for
    20-40s while audio and VAD kept flowing (fleet, c78 at 40 dials/s: 77 of
    1,057 calls stalled and 188 caller utterances were never delivered in 2
    minutes; 0% on pods with <=5 calls/2min, 12.5% on the busiest). The 5s
    turn-stop fuse then committed an empty turn and the caller heard 22-39s of
    dead air.

    With :attr:`starvation_bound_secs` > 0, a non-system item that has waited
    at least that long is no longer overtaken by system frames that arrived
    AFTER it: it takes its arrival-order place again. System frames that
    arrived before it are still served first. So:

    - Nothing changes while the queue keeps up. A fresh item (younger than the
      bound) is ordered exactly as with strict priority, and a healthy input
      queue never holds an item anywhere near the bound.
    - A starved item waits behind newer system frames for at most the bound
      plus the one system frame already in service. Its total wait is at most
      that plus the time to serve the system frames that were already queued
      when it arrived, i.e. what a FIFO would have given it.
    - Order is never inverted beyond arrival order. An aged item is not
      allowed to jump ahead of OLDER system frames: an `InterruptionFrame`
      that arrived first still flushes it, and a `TranscriptionFrame` is never
      served ahead of audio that reached the user aggregator before it.
      Letting it overtake older audio would bound its latency even under a
      growing audio backlog, but it would reorder the aggregator's inputs
      relative to what it was sent, which strict priority never does for
      frames that arrived first; a backlog that large is a throughput deficit
      no queue discipline can fix.

    A bound-served item reaches the process queue while system frames are
    still waiting, so one of them may flush the process queue (a turn start's
    `broadcast_interruption()`, an `InterruptionFrame`) before the process task
    has run. The flush therefore first yields once when such an item is still
    waiting there (see :attr:`last_get_was_bounded`); the input task itself
    never yields for the bound, so a processor that is merely behind still
    moves its whole backlog in one step, as with strict priority.

    Serving a non-system item from the input task only moves it to the
    process queue, so doing so ahead of waiting system frames costs them
    microseconds, not a frame's processing time.

    The bound is read on every ``get()``, from the instance attribute if one
    was set and otherwise from the class, whose default comes from the
    ``PIPECAT_INPUT_QUEUE_STARVATION_BOUND_MS`` environment variable (unset
    or 0 keeps strict priority). See
    :data:`INPUT_QUEUE_STARVATION_BOUND_RECOMMENDED_SECS` for the value to
    use and why.
    """

    START_PRIORITY = 1
    SYSTEM_PRIORITY = 10
    DEFAULT_PRIORITY = 20

    #: Seconds a non-system item may be overtaken by newer system frames
    #: before it regains its arrival-order place. 0 disables the bound
    #: (strict system-frame priority).
    starvation_bound_secs: float = _starvation_bound_from_env()

    def __init__(self):
        """Initialize the FrameProcessorQueue."""
        super().__init__()
        # Non-system items served ahead of a newer waiting system frame
        # because they reached the starvation bound. Read-only diagnostic.
        self._bounded_serves = 0
        self._last_get_bounded = False

    @classmethod
    def set_starvation_bound(cls, seconds: float | None) -> None:
        """Set the process-wide starvation bound for every input queue.

        Takes effect on the next ``get()`` of every existing and future queue
        that has no instance override.

        Args:
            seconds: The bound in seconds; ``None`` or 0 restores strict
                system-frame priority.
        """
        cls.starvation_bound_secs = max(0.0, seconds or 0.0)

    # asyncio.Queue storage hooks. One FIFO per tier replaces a (priority,
    # counter) heap: popping the start FIFO, then the system FIFO, first is
    # the same order the heap gives, and the non-system FIFO carries each
    # item's enqueue time so the starvation bound and the wait diagnostics can
    # read the head's age.

    def _init(self, maxsize):
        self._seq = 0
        self._start_items: deque[Any] = deque()
        self._system_items: deque[tuple[int, Any]] = deque()
        self._non_system_items: deque[tuple[int, float, Any]] = deque()

    def _put(self, item: tuple[Frame, FrameDirection, FrameCallback | None]):
        frame, _, _ = item
        self._seq += 1
        if isinstance(frame, StartFrame):
            self._start_items.append(item)
        elif isinstance(frame, SystemFrame):
            self._system_items.append((self._seq, item))
        else:
            self._non_system_items.append((self._seq, time.monotonic(), item))

    def _get(self) -> Any:
        self._last_get_bounded = False
        # The StartFrame outranks every other frame, aged ones included: a
        # processor drops whatever it is handed before it has started.
        if self._start_items:
            return self._start_items.popleft()
        system_items = self._system_items
        non_system_items = self._non_system_items
        if not non_system_items:
            return system_items.popleft()[1]
        if not system_items:
            return non_system_items.popleft()[2]
        bound = self.starvation_bound_secs
        if bound > 0:
            seq, enqueued_at, item = non_system_items[0]
            # Only an item already overtaken (the waiting system frame is
            # newer) can be starved; an older system frame goes first anyway.
            if seq < system_items[0][0] and time.monotonic() - enqueued_at >= bound:
                non_system_items.popleft()
                self._bounded_serves += 1
                self._last_get_bounded = True
                return item
        return system_items.popleft()[1]

    def qsize(self) -> int:
        """Number of items (system and non-system) in the queue."""
        return len(self._start_items) + len(self._system_items) + len(self._non_system_items)

    def empty(self) -> bool:
        """Return True if the queue holds no items."""
        return not self._start_items and not self._system_items and not self._non_system_items

    @property
    def non_system_qsize(self) -> int:
        """Number of non-system items waiting in the queue."""
        return len(self._non_system_items)

    @property
    def oldest_non_system_wait(self) -> float | None:
        """Seconds the oldest waiting non-system item has been queued.

        Returns:
            The age in seconds, or None when no non-system item is waiting.
        """
        if not self._non_system_items:
            return None
        return time.monotonic() - self._non_system_items[0][1]

    @property
    def bounded_serves(self) -> int:
        """Non-system items served ahead of newer system frames by the bound."""
        return self._bounded_serves

    @property
    def last_get_was_bounded(self) -> bool:
        """Whether the most recent ``get()`` returned an item the bound served.

        That item left system frames waiting behind it, and one of those may
        flush it before its consumer runs (see ``FrameProcessor``'s
        interruption handling, which yields to the process task first).
        """
        return self._last_get_bounded


# How long a processor holds frames waiting for a readiness condition before
# giving up and resuming. See pause_processing_all_frames_until().
PAUSE_UNTIL_READY_TIMEOUT_SECS = 5.0

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
    - on_usable_changed: Called with the new value of `is_usable` when the
      processor stops or starts being able to do its job.

    Example::

        @processor.event_handler("on_usable_changed")
        async def on_usable_changed(processor, is_usable):
            ...
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

        # Processor setup
        self._setup: FrameProcessorSetup | None = None

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
        # always process system frames before any other frame in the queue —
        # unless the opt-in starvation bound is set, in which case a non-system
        # frame that has waited past it is no longer overtaken by newer system
        # frames (see FrameProcessorQueue).

        # The input task that handles all types of frames. It processes system
        # frames right away and queues non-system frames for later processing.
        self.__should_block_system_frames = False
        self.__input_queue = FrameProcessorQueue()
        self.__input_event: asyncio.Event | None = None
        self.__input_frame_task: asyncio.Task | None = None
        # Watches the readiness condition passed to
        # pause_processing_all_frames_until() and lifts the pause it took.
        self.__pause_watcher_task: asyncio.Task | None = None
        # Clear while a pipeline is setting this processor up. The StartFrame
        # is held until it is set, so a processor never starts before its own
        # setup is done, while the ones ahead of it in the pipeline can.
        self.__setup_done = asyncio.Event()
        self.__setup_done.set()
        self.__start_received = False

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
        # The last frame the input queue's starvation bound moved to the
        # process queue (see `__yield_to_bound_served_frame`).
        self.__bound_served_frame: Frame | None = None

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

        # Whether this processor can still do its job. Flipped by the errors it
        # reports, so it is already up to date by the time an error travels.
        self._is_usable = True

        # Frame processor events.
        self._register_event_handler("on_before_process_frame", sync=True)
        self._register_event_handler("on_after_process_frame", sync=True)
        self._register_event_handler("on_before_push_frame", sync=True)
        self._register_event_handler("on_after_push_frame", sync=True)
        self._register_event_handler("on_error", sync=True)
        self._register_event_handler("on_usable_changed")

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
    def is_usable(self) -> bool:
        """Whether this processor can still do its job.

        A processor stays usable through failures it might recover from, and
        becomes unusable once its work can no longer succeed: a provider has
        rejected its API key, model or voice, or it has failed enough times to
        stop trying. Sending it more work would only produce more of the same
        error, so services stop accepting work and stop reconnecting once this
        is False.

        Errors set this as they are reported, so an error handler reading
        ``frame.processor.is_usable`` sees the verdict that came with the error
        it is handling.

        Returns:
            True while the processor can still be given work.
        """
        return self._is_usable

    async def set_usable(self, is_usable: bool):
        """Set whether this processor can be given work.

        Call this to bring back a processor that became unusable, once
        whatever stopped it working has been dealt with — new credentials, or
        a provider that has come back up. Services also do this for themselves
        when their settings change, since new settings may be the fix.

        Args:
            is_usable: Whether the processor can be given work.
        """
        if is_usable == self._is_usable:
            return

        self._is_usable = is_usable
        logger.debug(f"{self}: {'usable' if is_usable else 'no longer usable'}")
        await self._call_event_handler("on_usable_changed", is_usable)

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
    def processor_setup(self) -> FrameProcessorSetup:
        """Get the configuration this processor was set up with.

        Returns:
            The :class:`FrameProcessorSetup` given to :meth:`setup`.
        """
        if not self._setup:
            raise Exception(f"{self} is still not set up.")
        return self._setup

    @property
    def metrics_enabled(self) -> bool:
        """Check if metrics collection is enabled.

        Returns:
            True if metrics collection is enabled.
        """
        return bool(self._setup and self._setup.enable_metrics)

    @property
    def usage_metrics_enabled(self) -> bool:
        """Check if usage metrics collection is enabled.

        Returns:
            True if usage metrics collection is enabled.
        """
        return bool(self._setup and self._setup.enable_usage_metrics)

    @property
    def report_only_initial_ttfb(self) -> bool:
        """Check if only initial TTFB should be reported.

        Returns:
            True if only initial time-to-first-byte should be reported.
        """
        return bool(self._setup and self._setup.report_only_initial_ttfb)

    @property
    def pipeline_worker(self) -> PipelineWorker:
        """Get the :class:`~pipecat.pipeline.worker.PipelineWorker` this processor is running in.

        Provides access to worker-scoped state from inside a processor — most
        notably ``self.pipeline_worker.app_resources`` for the application's
        shared bag of resources (DB handles, clients, feature flags, etc.).

        Returns:
            The :class:`~pipecat.pipeline.worker.PipelineWorker` instance that set up this processor.
        """
        return self.processor_setup.pipeline_worker

    @property
    def worker_runner(self) -> WorkerRunner:
        """Get the :class:`~pipecat.workers.runner.WorkerRunner` hosting this processor's worker.

        Use it to reach another worker on the runner by name, e.g.
        ``self.worker_runner.get_worker("ui-jobs")``.

        Returns:
            The runner this processor's :class:`~pipecat.pipeline.worker.PipelineWorker` is attached to.
        """
        return self.pipeline_worker.worker_runner

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
        return self.processor_setup.pipeline_worker

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
                start_time=start_time, report_only_initial_ttfb=self.report_only_initial_ttfb
            )

    async def cancel_ttfb_metrics(self):
        """Abandon the current time-to-first-byte measurement without reporting it."""
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.cancel_ttfb_metrics()

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

    async def stop_ttfat_metrics(self, *, end_time: float | None = None):
        """Stop time-to-first-answer-token metrics collection and push results.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_ttfat_metrics(end_time=end_time)
            if frame:
                await self.push_frame(frame)

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

    async def start_stt_usage_metrics(self, usage: STTUsage):
        """Start STT usage metrics collection.

        Args:
            usage: Usage information for the STT operation.
        """
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_stt_usage_metrics(usage)
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

        This is where a processor connects and does its other slow start-up
        work, so that the pipeline pays for the slowest processor rather than
        all of them: a pipeline sets its processors up concurrently, so this
        runs alongside every other processor's. A resource shared with another
        processor therefore needs guarding, which
        :func:`~pipecat.utils.shared.acquires` does.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup.task_manager)
        self._setup = setup

        if self._metrics is not None:
            await self._metrics.setup(self.task_manager)

    def _setup_started(self):
        """Hold the StartFrame until :meth:`_setup_finished` is called.

        Called by the pipeline before setting this processor up.
        """
        self.__setup_done.clear()

    def _setup_finished(self):
        """Let this processor start, once its own setup() has returned."""
        self.__setup_done.set()
        self.__maybe_create_input_task()

    async def cleanup(self):
        """Release this processor's resources at teardown.

        This base implementation cancels only the processor's internal
        input/process tasks; tasks created via :meth:`create_task` are released
        by an override. Like :meth:`setup`, this runs concurrently with every
        other processor's, so a resource shared with another processor is
        released with :func:`~pipecat.utils.shared.releases`.
        """
        await super().cleanup()
        await self.__cancel_pause_watcher()
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
        return self.processor_setup.clock

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
            if isinstance(frame, StartFrame):
                await self.__setup_done.wait()
            await self.__process_frame(frame, direction, callback)
            return

        await self.__input_queue.put((frame, direction, callback))

        # Nothing drains the queue until the StartFrame arrives and this
        # processor is set up, so a processor never acts on a frame before it
        # has been started. Frames pushed before that simply wait, and the
        # StartFrame is dequeued ahead of them.
        if isinstance(frame, StartFrame):
            self.__start_received = True
            self.__maybe_create_input_task()

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

    async def pause_processing_all_frames_until(
        self,
        ready: Callable[[], Awaitable[Any]],
        *,
        timeout: float = PAUSE_UNTIL_READY_TIMEOUT_SECS,
    ):
        """Hold frames arriving at this processor until ``ready`` resolves.

        Useful for a processor that cannot act on frames until some condition
        holds, such as one that establishes a connection in the background.
        Frames wait in the processor's queues and are delivered in order once
        the condition resolves, so nothing is lost.

        The frame being processed when this is called is unaffected: the pause
        takes hold from the next frame onwards. A processor pausing while it
        handles its ``StartFrame``, for instance, still passes that frame on
        downstream, so pipeline startup is not delayed.

        Both queues are held, so a processor left paused could not process the
        frames that shut it down. The pause is therefore always lifted: when
        ``ready`` resolves, when ``timeout`` elapses, or at teardown, whichever
        comes first.

        Args:
            ready: Awaited to learn when frames can be acted on, e.g.
                ``some_event.wait``.
            timeout: Seconds to hold frames before giving up and resuming.
        """
        if self._enable_direct_mode:
            logger.warning(f"{self}: cannot hold frames, this processor runs in direct mode")
            return

        await self.__cancel_pause_watcher()
        await self.pause_processing_system_frames()
        await self.pause_processing_frames()
        self.__pause_watcher_task = self.create_task(
            self.__pause_watcher_handler(ready, timeout), name=f"{self}::pause_watcher"
        )

    async def __pause_watcher_handler(self, ready: Callable[[], Awaitable[Any]], timeout: float):
        """Lift the pause taken by pause_processing_all_frames_until()."""
        try:
            await asyncio.wait_for(ready(), timeout=timeout)
        except TimeoutError:
            logger.warning(
                f"{self}: still not ready after {timeout}s, resuming frame processing anyway"
            )
        await self.__resume_processing_all_frames()

    async def __cancel_pause_watcher(self):
        """Stop watching, and lift the pause the watcher was going to lift."""
        if not self.__pause_watcher_task:
            return

        task = self.__pause_watcher_task
        self.__pause_watcher_task = None
        await self.cancel_task(task)
        await self.__resume_processing_all_frames()

    async def __resume_processing_all_frames(self):
        """Resume both queues held by pause_processing_all_frames_until()."""
        await self.resume_processing_system_frames()
        await self.resume_processing_frames()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        observer = self._setup.observer if self._setup else None
        # A worker whose observers all ignore process events says so, and then
        # the event is not even built (see WorkerObserver.wants_process_events).
        if observer and getattr(observer, "wants_process_events", True):
            data = FrameProcessed(
                processor=self,
                frame=frame,
                direction=direction,
                timestamp=self.get_clock().get_time(),
            )
            await observer.on_process_frame(data)

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

    def _classify_error(self, exception: Exception) -> ErrorCategory | None:
        """Classify an exception this processor knows the shape of.

        Override for providers that signal failures through SDK-specific
        exceptions rather than an HTTP status code, or whose credentials can be
        rejected for reasons a reconnection would clear.

        Args:
            exception: The exception to classify.

        Returns:
            The category, or None to fall back to :func:`classify_http_exception`.
        """
        return None

    async def push_error(
        self,
        error_msg: str,
        exception: Exception | None = None,
        fatal: bool = False,
        category: ErrorCategory | None = None,
        force_treat_as_permanent: bool = False,
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

                .. deprecated:: 1.8.0
                    Use ``force_treat_as_permanent=True`` instead, when the
                    error leaves its originating processor unable to do its
                    job: the pipeline worker then applies its
                    :class:`ProcessorUnusablePolicy`, of which ``CANCEL``
                    matches what ``fatal=True`` did. For an error that isn't
                    about that processor's state, push an
                    :class:`EndWorkerFrame` after the error to end the pipeline.
                    Will be removed in 2.0.0.

            category: Why the error occurred, when the caller knows. Leave it
                unset to let the category be worked out from the exception, or
                pass `ErrorCategory.UNKNOWN` to report an error whose cause
                can't be attributed — an unexpected one caught by a broad
                ``except``, say, which may not have come from this processor at
                all.
            force_treat_as_permanent: Whether to treat this error as one that will
                keep recurring, leaving the processor unable to do any more
                work — having failed too many times to keep trying, say. Only
                needed for a failure the category doesn't already convey:
                leaving it False doesn't keep the processor usable, since a
                permanent category costs it its usability on its own. Either way
                the processor stops being given work, and the pipeline worker
                decides what to do about it through its
                :class:`ProcessorUnusablePolicy`: report the error and keep
                running (the default), end the pipeline, or cancel it.

        Example::

            ```python
            # An error this processor can recover from
            await self.push_error("Failed to process audio chunk, skipping")

            # An error that leaves this processor unable to do any more work
            try:
                result = some_critical_operation()
            except Exception as e:
                await self.push_error(
                    "Critical operation failed", exception=e, force_treat_as_permanent=True
                )
            ```
        """
        if fatal:
            warn_deprecated(
                "`push_error(fatal=True)` is deprecated since 1.8.0 and will be removed "
                "in 2.0.0. Use `force_treat_as_permanent=True` or an `EndWorkerFrame` "
                "instead. If the error leaves its originating processor unable to do its "
                "job, `force_treat_as_permanent=True` marks the processor unusable, and the "
                "PipelineWorker acts on it according to its "
                "`processor_unusable_policy` (`ProcessorUnusablePolicy.CANCEL` does what "
                "`fatal=True` did). Otherwise, drop `fatal` and push an "
                "`EndWorkerFrame` after the error to end the pipeline.",
                stacklevel=2,
            )

        error_frame = ErrorFrame(
            error=error_msg,
            exception=exception,
            processor=self,
            category=category,
        )
        # Set after construction so the frame doesn't warn about the deprecated
        # flag on top of the warning already reported above.
        error_frame.fatal = fatal

        # Subclasses may override `push_error_frame` with its original
        # one-argument signature, so only pass the flag when it is set.
        if force_treat_as_permanent:
            await self.push_error_frame(error=error_frame, force_treat_as_permanent=True)
        else:
            await self.push_error_frame(error=error_frame)

    async def push_error_frame(
        self,
        error: ErrorFrame,
        force_treat_as_permanent: bool = False,
    ):
        """Push an error frame upstream.

        Args:
            error: The error frame to push. Its deprecated ``fatal`` flag still
                cancels the pipeline; ``force_treat_as_permanent`` is the replacement.
            force_treat_as_permanent: Whether to treat this error as one that will
                keep recurring, leaving the processor unable to do any more
                work. Leaving it False doesn't keep the processor usable — a
                permanent category costs it its usability either way. See
                :meth:`push_error`.
        """
        if not error.processor:
            error.processor = self
        # Anything still unset by now is going to stay that way, so settle it
        # here and let handlers read a category off every error they receive.
        if error.category is None and error.exception:
            error.category = self._classify_error(error.exception) or classify_http_exception(
                error.exception
            )
        if error.category is None:
            error.category = ErrorCategory.UNKNOWN

        # Before anything sees the error, so that handlers reading
        # `frame.processor.is_usable` get the verdict that came with it.
        if force_treat_as_permanent or error.category.is_permanent:
            await self.set_usable(False)

        await self._call_event_handler("on_error", error)

        # An exception carries a traceback only once it has been raised, so fall
        # back to the plain message rather than losing the error entirely.
        tb = traceback.extract_tb(error.exception.__traceback__) if error.exception else []
        if tb:
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
        if (
            self.__orphaned_process_tasks
            and asyncio.current_task() in self.__orphaned_process_tasks
        ):
            logger.debug(f"{self}: dropping {frame.name} pushed by an orphaned process task")
            return

        await self._call_event_handler("on_before_push_frame", frame)

        await self.__internal_push_frame(frame, direction)

        await self._call_event_handler("on_after_push_frame", frame)

    async def broadcast_interruption(self):
        """Broadcast an `InterruptionFrame` both upstream and downstream."""
        logger.debug(f"{self}: broadcasting interruption")
        await self.__yield_to_bound_served_frame()
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
            # Before `current` is read: the process task may take a waiting
            # bound-served frame now.
            await self.__yield_to_bound_served_frame()
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
            current_is_uninterruptible = (
                current is not None
                and not current.interruptible
                and not isinstance(current, HeartbeatFrame)
            )
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
                # A broad catch: this may not have come from this processor at
                # all, so its cause can't be attributed.
                category=ErrorCategory.UNKNOWN,
            )

    async def __internal_push_frame(self, frame: Frame, direction: FrameDirection):
        """Internal method to push frames to adjacent processors.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        observer = self._setup.observer if self._setup else None
        try:
            timestamp = self.get_clock().get_time() if self._setup else 0
            if direction == FrameDirection.DOWNSTREAM and self._next:
                if _TRACE_FRAMES:
                    logger.trace(f"Pushing {frame} downstream from {self} to {self._next}")

                if observer:
                    data = FramePushed(
                        source=self,
                        destination=self._next,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await observer.on_push_frame(data)
                await self._next.queue_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
                if _TRACE_FRAMES:
                    logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
                if observer:
                    data = FramePushed(
                        source=self,
                        destination=self._prev,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await observer.on_push_frame(data)
                await self._prev.queue_frame(frame, direction)
        except Exception as e:
            # Observers and the downstream processor run inside this
            # block, so the cause of an unexpected failure can't be attributed.
            await self.push_error(
                error_msg=f"Uncaught exception: {e}",
                exception=e,
                category=ErrorCategory.UNKNOWN,
            )

    def __maybe_create_input_task(self):
        """Create the input task once the StartFrame is here and setup is done."""
        if not self.__start_received or not self.__setup_done.is_set():
            return
        if not self._task_manager:
            # Setting up was cut short before FrameProcessor.setup() ran, so
            # there is no task manager to start processing frames with.
            logger.error(
                f"{self}: can't start, setup ended before FrameProcessor.setup() was "
                "called (call super().setup() before anything that can block)"
            )
            return
        self.__create_input_task()

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

    async def __yield_to_bound_served_frame(self):
        """Let the process task take a bound-served frame before a flush drops it.

        The starvation bound moves a frame to the process queue while system
        frames still wait behind it, and one of them may flush the process
        queue before the process task has run: in the load benchmark,
        transcripts that had waited past the bound were dropped by the VAD
        turn start queued right behind them. Strict priority never needed
        this, since it only serves a non-system frame once no system frame
        waits, so the input task suspends and the process task runs first.

        Yielding here, at the flush, rather than after every bound-served frame
        keeps the input task yield-free: a yield per frame capped a processor
        coming back to an aged backlog at one non-system frame per loop
        iteration, below a 60/s token or TTS stream on a 20ms-iteration loop
        (see `TestAgedBacklogDrainsInBulk`). A no-op unless a bound-served
        frame is still queued, and when the flush runs on the process task
        itself, which cannot take that frame while it is the caller.
        """
        frame, self.__bound_served_frame = self.__bound_served_frame, None
        if frame is None or asyncio.current_task() is self.__process_frame_task:
            return
        # A scan of the process queue, only at a flush while the bound is
        # delivering frames.
        queued = self.__process_queue._queue  # pyright: ignore[reportAttributeAccessIssue]
        if any(item[0] is frame for item in queued):
            await asyncio.sleep(0)

    def has_queued_frame(self, frame_type: type[Frame]) -> bool:
        """Return True if a frame of the given type is waiting in the processing queue.

        Delegates to :meth:`FrameQueue.has_frame` so the check is O(distinct
        enqueued types) with no queue scanning.

        Args:
            frame_type: The frame class to look for.

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
    def input_queue_non_system_depth(self) -> int:
        """Number of non-system frames waiting in this processor's input queue.

        Non-system frames (data and control frames, heartbeats included) wait
        here until the input task moves them to the process queue. System
        frames are served first, so a count that stays above zero while
        `input_queue_non_system_wait` grows means this processor is starving
        them. Always 0 in direct mode.

        Returns:
            The number of waiting non-system frames. Read-only diagnostic.
        """
        return self.__input_queue.non_system_qsize

    @property
    def input_queue_non_system_wait(self) -> float | None:
        """Age of the oldest non-system frame waiting in the input queue.

        The starvation signal the heartbeat timeout cannot give on its own.
        A heartbeat that never arrives only says some processor holds it;
        this names which one and says whether it is starved or just behind:

        - a large wait while `input_queue_depth` is barely above
          `input_queue_non_system_depth` means system frames keep overtaking
          the waiting frames (starved by priority);
        - a large wait with many system frames queued as well means the input
          task is behind on everything (a throughput deficit);
        - None means nothing non-system is waiting here.

        Costs one clock read. Safe to call from a heartbeat-timeout handler
        for every processor of the pipeline.

        Returns:
            Seconds the oldest waiting non-system frame has been queued, or
            None if none is waiting. Read-only diagnostic.
        """
        return self.__input_queue.oldest_non_system_wait

    @property
    def input_queue_bounded_serves(self) -> int:
        """Non-system frames the starvation bound served ahead of newer system frames.

        Stays 0 while the bound is disabled or the processor keeps up; a
        count that grows during a call means the bound (see
        :class:`FrameProcessorQueue`) is what delivered those frames.

        Returns:
            The cumulative count for this processor. Read-only diagnostic.
        """
        return self.__input_queue.bounded_serves

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
            # The frame callback runs inside this block, so the cause of an
            # unexpected failure can't be attributed.
            await self.push_error(
                error_msg=f"Error processing frame: {e}",
                exception=e,
                category=ErrorCategory.UNKNOWN,
            )

    async def __input_frame_task_handler(self):
        """Handle frames from the input queue.

        It only processes system frames. Other frames are queue for another task
        to execute.

        """
        while True:
            (frame, direction, callback) = await self.__input_queue.get()

            # This gate runs BEFORE the SystemFrame dispatch below, so it holds
            # every frame class, not just system frames. A HeartbeatFrame is
            # an uninterruptible ControlFrame — not a SystemFrame — so
            # without this exemption the health probe parks here for the whole
            # hold, which is the same defect the process-queue exemption below
            # exists to prevent, merely moved one queue upstream. Both holders
            # of this queue can run for a long time relative to the heartbeat
            # period: `pause_processing_all_frames_until()` holds it for up to
            # PAUSE_UNTIL_READY_TIMEOUT_SECS (e.g. Deepgram STT waiting on its
            # websocket), and ParallelPipeline holds it across lifecycle-frame
            # synchronization. A heartbeat carries no ordering semantics, so
            # letting it through changes nothing those holds are protecting.
            # As with the process queue, the pause flag is deliberately left
            # armed so the next non-heartbeat frame still blocks.
            if (
                self.__should_block_system_frames
                and self.__input_event
                and not isinstance(frame, HeartbeatFrame)
            ):
                logger.trace(f"{self}: system frame processing paused")
                await self.__input_event.wait()
                self.__input_event.clear()
                self.__should_block_system_frames = False
                logger.trace(f"{self}: system frame processing resumed")

            if isinstance(frame, SystemFrame):
                await self.__process_frame(frame, direction, callback)
            elif self.__process_queue:
                await self.__process_queue.put((frame, direction, callback))
                if self.__input_queue.last_get_was_bounded:
                    # Not a yield here: a processor that is only behind (held
                    # past the bound, then facing aged frames interleaved with
                    # 50fps input audio) would move one frame per loop
                    # iteration. See `__yield_to_bound_served_frame`.
                    self.__bound_served_frame = frame
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

            # Bind the queue for THIS iteration. `__interrupt_process_task`
            # replaces `__process_queue` outright when an interruption's cancel
            # exceeds PROCESS_TASK_ORPHAN_TIMEOUT_SECS, so reading the attribute
            # again for `task_done()` below can land on a DIFFERENT queue than
            # the one this frame came from — which raises
            # "task_done() called too many times" and kills the processor's
            # frame task mid-call.
            queue = self.__process_queue

            (frame, direction, callback) = await queue.get()

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

            # ...and mark it done on that same object, never on whatever
            # `__process_queue` points at now.
            queue.task_done()
