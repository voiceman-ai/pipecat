#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the read-only per-processor wedge diagnostics.

Covers the FrameProcessor diagnostic properties (input_queue_depth,
process_queue_depth, seconds_since_last_progress, processing_frame_name,
is_frame_processing_paused) and PipelineWorker.dump_processor_diagnostics().

The motivating failure class: a leg that wedges before its first turn (run-67
wr1872 shape) — frames pile up behind one stuck processor while the rest of
the pipeline looks healthy. These diagnostics must make that leg legible from
a log dump WITHOUT feeding any kill/abort decision, so every assertion here is
about observation only.
"""

import asyncio
from types import SimpleNamespace

import pytest
from loguru import logger

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    HeartbeatFrame,
    StartFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.workers.runner import WorkerRunner


class WedgeProcessor(FrameProcessor):
    """A processor that blocks inside process_frame on TextFrames until released.

    Simulates the wedged-before-first-turn class: the process task is stuck
    inside a single frame while more frames accumulate in the process queue.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.release = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            await self.release.wait()


async def _setup_processor(processor: FrameProcessor) -> TaskManager:
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_event_loop()))
    await processor.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=task_manager,
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )
    return task_manager


@pytest.mark.asyncio
async def test_fresh_processor_reports_no_progress_and_empty_queues():
    """Before any frame flows, the diagnostics must read as 'never ran', not 'wedged'."""
    processor = IdentityFilter()

    assert processor.input_queue_depth == 0
    assert processor.process_queue_depth == 0
    assert processor.seconds_since_last_progress is None
    assert processor.processing_frame_name is None
    assert processor.is_frame_processing_paused is False


@pytest.mark.asyncio
async def test_wedged_processor_shows_growing_queue_and_aging_progress():
    """A processor stuck inside one frame must expose the wedge signature.

    Signature: process_queue_depth grows, seconds_since_last_progress ages,
    and processing_frame_name names the culprit frame. After release,
    everything drains and the in-flight frame clears.
    """
    processor = WedgeProcessor()
    await _setup_processor(processor)
    try:
        await processor.queue_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.05)

        # First TextFrame wedges inside process_frame; three more pile up.
        for i in range(4):
            await processor.queue_frame(TextFrame(text=f"t{i}"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.15)

        assert processor.input_queue_depth == 0, "input task should have dispatched everything"
        assert processor.process_queue_depth == 3, "three frames should be parked behind the wedge"
        in_flight = processor.processing_frame_name
        assert in_flight is not None and in_flight.startswith("TextFrame"), (
            f"the culprit frame must be named, got {in_flight}"
        )
        age = processor.seconds_since_last_progress
        assert age is not None and age >= 0.1, (
            f"progress must age while wedged inside a frame, got {age}"
        )
        assert processor.is_frame_processing_paused is False

        # Release the wedge: the queue must drain and the in-flight frame clear.
        processor.release.set()
        await asyncio.sleep(0.15)
        assert processor.process_queue_depth == 0
        assert processor.processing_frame_name is None
        age_after = processor.seconds_since_last_progress
        assert age_after is not None and age_after < 1.0
    finally:
        await processor.cleanup()


@pytest.mark.asyncio
async def test_paused_processor_reads_as_paused_not_wedged():
    """pause_processing_frames() must be distinguishable from a wedge."""
    processor = WedgeProcessor()
    processor.release.set()  # never wedge inside process_frame
    await _setup_processor(processor)
    try:
        await processor.queue_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.05)

        await processor.pause_processing_frames()
        await processor.queue_frame(TextFrame(text="parked"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.1)

        assert processor.is_frame_processing_paused is True
        # The parked frame was dequeued and is held by the process task.
        in_flight = processor.processing_frame_name
        assert in_flight is not None and in_flight.startswith("TextFrame")

        await processor.resume_processing_frames()
        await asyncio.sleep(0.1)
        assert processor.is_frame_processing_paused is False
        assert processor.processing_frame_name is None
    finally:
        await processor.cleanup()


@pytest.mark.asyncio
async def test_input_queue_depth_counts_undispatched_system_frames():
    """System frames parked behind pause_processing_system_frames() must show up."""
    processor = IdentityFilter()
    await _setup_processor(processor)
    try:
        await processor.queue_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.05)

        await processor.pause_processing_system_frames()
        for _ in range(3):
            await processor.queue_frame(HeartbeatFrame(timestamp=0), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.1)

        # One heartbeat is held by the parked input task; the rest are queued.
        assert processor.input_queue_depth == 2

        await processor.resume_processing_system_frames()
        await asyncio.sleep(0.1)
        assert processor.input_queue_depth == 0
    finally:
        await processor.cleanup()


@pytest.mark.asyncio
async def test_dump_processor_diagnostics_covers_every_processor_and_logs_once():
    """The worker dump must cover source/sink/user processors and log one record."""
    identity = IdentityFilter()
    pipeline = Pipeline([identity])
    worker = PipelineWorker(
        pipeline,
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
    )

    captured_logs: list[str] = []
    sink_id = logger.add(lambda message: captured_logs.append(str(message)), level="WARNING")

    holder: dict = {}

    async def driver():
        await asyncio.sleep(0.1)
        await worker.queue_frame(TextFrame(text="hello"))
        await asyncio.sleep(0.2)
        holder["entries"] = worker.dump_processor_diagnostics(reason="unit-test")
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    try:
        await asyncio.gather(runner.run(), driver())
    finally:
        logger.remove(sink_id)

    entries = holder["entries"]
    names = [e["processor"] for e in entries]
    assert any("Source" in n for n in names), f"worker source missing from dump: {names}"
    assert any("Sink" in n for n in names), f"worker sink missing from dump: {names}"
    assert any(n.startswith("IdentityFilter") for n in names), (
        f"user processor missing from dump: {names}"
    )

    expected_keys = {
        "processor",
        "depth",
        "input_queue_depth",
        "process_queue_depth",
        "seconds_since_last_progress",
        "processing_frame",
        "paused",
    }
    for entry in entries:
        assert set(entry) == expected_keys

    # The pipeline ran a frame through, so at least one processor made progress.
    assert any(e["seconds_since_last_progress"] is not None for e in entries)

    dump_logs = [m for m in captured_logs if "processor diagnostics dump" in m]
    assert len(dump_logs) == 1, "the dump must be a single atomic log record"
    assert "(reason: unit-test)" in dump_logs[0]
    # Every processor appears in the one record.
    for name in names:
        assert name in dump_logs[0]


@pytest.mark.asyncio
async def test_dump_is_safe_before_the_pipeline_starts():
    """Dumping a never-started worker must not raise and must read as 'never'."""
    worker = PipelineWorker(
        Pipeline([IdentityFilter()]),
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
    )
    entries = worker.dump_processor_diagnostics()
    assert entries, "expected at least the source/user/sink processors"
    for entry in entries:
        assert entry["seconds_since_last_progress"] is None
        assert entry["input_queue_depth"] == 0
        assert entry["process_queue_depth"] == 0
        assert entry["paused"] is False
