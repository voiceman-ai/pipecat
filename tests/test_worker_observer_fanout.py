#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WorkerObserver hands each observer only the events it handles."""

import asyncio
import unittest

from pipecat.frames.frames import InputAudioRawFrame, TextFrame
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.pipeline.worker_observer import WorkerObserver
from pipecat.utils.asyncio.task_manager import TaskManager


def _pushed(frame):
    return FramePushed(source=None, destination=None, frame=frame, direction=None, timestamp=0)


def _processed(frame):
    return FrameProcessed(processor=None, frame=frame, direction=None, timestamp=0)


class _PushOnly(BaseObserver):
    def __init__(self):
        super().__init__()
        self.pushed = []

    async def on_push_frame(self, data):
        self.pushed.append(data.frame)


class _IgnoresAudio(_PushOnly):
    ignored_frame_types = (InputAudioRawFrame,)


class _Inline(_PushOnly):
    inline_dispatch = True


class _Processes(BaseObserver):
    def __init__(self):
        super().__init__()
        self.processed = []

    async def on_process_frame(self, data):
        self.processed.append(data.frame)


async def _worker(*observers):
    w = WorkerObserver(observers=list(observers))
    await w.setup(TaskManager())
    await w.start()
    return w


def _audio():
    return InputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=8000, num_channels=1)


class TestFanout(unittest.IsolatedAsyncioTestCase):
    async def test_process_events_are_not_queued_to_observers_without_a_handler(self):
        push_only = _PushOnly()
        w = await _worker(push_only)
        self.assertFalse(w.wants_process_events)
        await w.on_process_frame(_processed(TextFrame("x")))
        self.assertEqual(w._proxies[push_only].queue.qsize(), 0)
        await w.stop()

    async def test_an_observer_that_handles_process_events_still_gets_them(self):
        processes = _Processes()
        w = await _worker(processes, _PushOnly())
        self.assertTrue(w.wants_process_events)
        await w.on_process_frame(_processed(TextFrame("x")))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(len(processes.processed), 1)
        await w.stop()

    async def test_ignored_frame_types_are_skipped_and_others_delivered(self):
        ignores, sees_all = _IgnoresAudio(), _PushOnly()
        w = await _worker(ignores, sees_all)
        await w.on_push_frame(_pushed(_audio()))
        await w.on_push_frame(_pushed(TextFrame("hello")))
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual([type(f).__name__ for f in ignores.pushed], ["TextFrame"])
        self.assertEqual(len(sees_all.pushed), 2)
        await w.stop()

    async def test_an_inline_observer_is_called_at_once_without_a_queue(self):
        inline = _Inline()
        w = await _worker(inline)
        self.assertIsNone(w._proxies[inline].queue)
        await w.on_push_frame(_pushed(_audio()))
        self.assertEqual(len(inline.pushed), 1)
        await w.stop()

    async def test_an_observer_added_later_follows_the_same_rules(self):
        w = await _worker(_PushOnly())
        self.assertFalse(w.wants_process_events)
        late = _Processes()
        w.add_observer(late)
        self.assertTrue(w.wants_process_events)
        await w.remove_observer(late)
        self.assertFalse(w.wants_process_events)
        await w.stop()


if __name__ == "__main__":
    unittest.main()
