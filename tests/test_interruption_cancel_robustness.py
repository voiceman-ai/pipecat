#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A barge-in must survive a process task whose cancellation is absorbed.

``Task.cancel()`` raises ``CancelledError`` at exactly one await point, and
library internals (anyio/httpcore cancel scopes) can absorb it, leaving the
task alive with no cancellation left to receive. Production run 1931: the
interruption's inline cancel of the LLM service's process task was absorbed
mid-stream-cleanup, ``_start_interruption`` blocked on the un-dying task for
53s, and no frame traversed the processor for the rest of the call — the
agent went silent while user turns kept transcribing upstream.

Two defenses are covered here:

* ``TaskManager.cancel_task`` re-delivers the cancellation after its grace
  threshold, reaping zombies parked at a later await.
* ``_start_interruption`` bounds its inline wait and ORPHANS a task that
  will not die: a fresh process task takes over immediately, the orphan's
  late pushes are dropped, and a background reaper keeps cancelling it.
"""

import asyncio
import time
import unittest
from dataclasses import dataclass

from pipecat.frames.frames import (
    DataFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.asyncio.task_manager import TaskManager


@dataclass
class WedgeFrame(DataFrame):
    """Data frame whose processing absorbs the first cancellation."""

    pass


class TestCancelTaskRedelivery(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_task_reaps_task_that_absorbs_first_cancel(self):
        """A task that swallows its first CancelledError still gets reaped.

        Without re-delivery past the grace threshold, cancel_task would await
        this task forever (the run-1931 wedge); the outer wait_for is the
        regression bound.
        """
        task_manager = TaskManager(loop=asyncio.get_event_loop())

        async def absorbing():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                # Absorb the first delivery, like a library cancel scope.
                pass
            while True:
                await asyncio.sleep(30)

        task = task_manager.create_task(absorbing(), name="absorbing")
        await asyncio.sleep(0.05)  # let it park in the first sleep

        started = time.monotonic()
        await asyncio.wait_for(task_manager.cancel_task(task, timeout=0.2), timeout=5)
        self.assertTrue(task.done())
        # Threshold (0.2s) + one re-delivery, not the 30s sleep.
        self.assertLess(time.monotonic() - started, 3)

    async def test_cancel_task_fast_path_unchanged(self):
        """A well-behaved task cancels promptly with no re-delivery involved."""
        task_manager = TaskManager(loop=asyncio.get_event_loop())

        async def cooperative():
            await asyncio.sleep(30)

        task = task_manager.create_task(cooperative(), name="cooperative")
        await asyncio.sleep(0.05)

        started = time.monotonic()
        await task_manager.cancel_task(task, timeout=3)
        self.assertTrue(task.done())
        self.assertLess(time.monotonic() - started, 1)


class TestInterruptionOrphansWedgedProcessTask(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_keeps_processing_after_absorbed_cancel(self):
        """Run-1931 shape: the in-flight frame absorbs the interruption's cancel.

        The interruption must still complete (bounded orphan wait), a fresh
        process task must take over, and the frame sent after the interruption
        must flow through. The outer timeout is the regression bound: before
        the orphan path existed, _start_interruption blocked on the zombie and
        this pipeline never finished.
        """

        class AbsorbingProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, WedgeFrame):
                    try:
                        await asyncio.sleep(30)
                    except asyncio.CancelledError:
                        # Absorbed — the task lives on, wedged.
                        pass
                    await asyncio.sleep(30)  # the reaper's re-cancel lands here
                await self.push_frame(frame, direction)

        pipeline = Pipeline([AbsorbingProcessor()])

        frames_to_send = [
            WedgeFrame(),
            SleepFrame(),
            InterruptionFrame(),
            TextFrame(text="after the interruption"),
            SleepFrame(sleep=1.5),  # ride out the orphan bound before EndFrame
        ]
        expected_down_frames = [
            InterruptionFrame,
            TextFrame,
        ]
        await asyncio.wait_for(
            run_test(
                pipeline,
                frames_to_send=frames_to_send,
                expected_down_frames=expected_down_frames,
            ),
            timeout=15,
        )

    async def test_orphaned_task_late_pushes_are_dropped(self):
        """Frames a zombie pushes after being orphaned must not reach the flow.

        The zombie absorbs every cancel until past the orphan bound, then
        pushes a stale frame: the push gate must drop it (an extra TextFrame
        downstream would fail the expected-frames match).
        """

        class LatePushProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, WedgeFrame):
                    deadline = asyncio.get_running_loop().time() + 1.5
                    while asyncio.get_running_loop().time() < deadline:
                        try:
                            await asyncio.sleep(0.05)
                        except asyncio.CancelledError:
                            # Keep absorbing until past the orphan bound.
                            pass
                    # Orphaned by now (bound is 1.0s) — this must be dropped.
                    await self.push_frame(TextFrame(text="stale"), direction)
                    raise asyncio.CancelledError()
                await self.push_frame(frame, direction)

        pipeline = Pipeline([LatePushProcessor()])

        frames_to_send = [
            WedgeFrame(),
            SleepFrame(),
            InterruptionFrame(),
            TextFrame(text="after the interruption"),
            SleepFrame(sleep=2.0),  # let the zombie reach its late push
        ]
        expected_down_frames = [
            InterruptionFrame,
            TextFrame,  # only the probe — "stale" must not appear
        ]
        await asyncio.wait_for(
            run_test(
                pipeline,
                frames_to_send=frames_to_send,
                expected_down_frames=expected_down_frames,
            ),
            timeout=15,
        )


if __name__ == "__main__":
    unittest.main()
