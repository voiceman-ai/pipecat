#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bounded starvation of non-system frames in a processor's input queue.

Prod shape (api pods, campaign at 40 dials/s): `InputAudioRawFrame` is a system
frame arriving every 20ms, and `LLMUserAggregator` hops to a thread executor
for VAD on each one. Once loop contention pushed that per-frame service time
past the 20ms arrival interval, a system frame was always waiting at the next
``get()``, so the aggregator never served `TranscriptionFrame` or
`HeartbeatFrame`: the caller's words waited 22-37s, the 5s turn-stop fuse
committed an empty turn, and heartbeats timed out while audio and VAD kept
flowing.

These tests pin the queue discipline (strict priority stays byte-for-byte the
same while the bound is off or the queue keeps up; an aged item takes its
arrival-order place and is never inverted past it), then reproduce the prod
shape in a pipeline, with the starving control, for a generic processor and for
the real user aggregator.
"""

import asyncio
import heapq
import os
import random
import subprocess
import sys
import time
import unittest

import pytest

import tests.test_frame_processor as frame_processor_tests
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    HeartbeatFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
    SystemFrame,
    TextFrame,
    TranscriptionFrame,
    UserSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors import frame_processor as frame_processor_module
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import (
    INPUT_QUEUE_STARVATION_BOUND_RECOMMENDED_SECS,
    FrameDirection,
    FrameProcessor,
    FrameProcessorQueue,
    _starvation_bound_from_env,
)
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

BOUND = INPUT_QUEUE_STARVATION_BOUND_RECOMMENDED_SECS

# The prod regime: a 20ms audio frame every 20ms, served in ~21ms.
FRAME_SECS = 0.02
SLOW_SERVICE_SECS = 0.021
SAMPLE_RATE = 8000
FRAME_BYTES = int(SAMPLE_RATE * FRAME_SECS) * 2


@pytest.fixture
def restore_bound():
    saved = FrameProcessorQueue.starvation_bound_secs
    yield
    FrameProcessorQueue.starvation_bound_secs = saved


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(frame_processor_module, "time", clock)
    return clock


def _item(frame: Frame):
    return (frame, FrameDirection.DOWNSTREAM, None)


def _drain(queue: FrameProcessorQueue) -> list[Frame]:
    frames = []
    while not queue.empty():
        frames.append(queue.get_nowait()[0])
        queue.task_done()
    return frames


class _ReferencePriorityQueue:
    """The queue discipline this fork shipped before the bound: a (priority, counter) heap."""

    def __init__(self):
        self._heap = []
        self._high = 0
        self._low = 0

    def put(self, item):
        frame, _, _ = item
        if isinstance(frame, SystemFrame):
            self._high += 1
            heapq.heappush(self._heap, (1, self._high, item))
        else:
            self._low += 1
            heapq.heappush(self._heap, (2, self._low, item))

    def get(self):
        return heapq.heappop(self._heap)[2]

    def empty(self):
        return not self._heap


#
# Queue discipline
#


@pytest.mark.parametrize("bound", [0.0, BOUND])
def test_order_matches_the_priority_heap_while_nothing_ages(bound, fake_clock, restore_bound):
    """Off, or on with every item fresher than the bound: exactly the old heap order."""
    FrameProcessorQueue.set_starvation_bound(bound)
    rng = random.Random(1234)
    queue = FrameProcessorQueue()
    reference = _ReferencePriorityQueue()
    got, expected = [], []
    for _ in range(5000):
        if rng.random() < 0.55:
            frame = InputAudioRawFrame(audio=b"", sample_rate=SAMPLE_RATE, num_channels=1)
            if rng.random() < 0.3:
                frame = TextFrame(text="t") if rng.random() < 0.5 else HeartbeatFrame(timestamp=0)
            queue.put_nowait(_item(frame))
            reference.put(_item(frame))
        elif not reference.empty():
            got.append(queue.get_nowait()[0])
            queue.task_done()
            expected.append(reference.get()[0])
        # Time passes, but no item ever reaches the bound.
        fake_clock.now += 0.00001
    while not reference.empty():
        got.append(queue.get_nowait()[0])
        expected.append(reference.get()[0])
    assert [f.id for f in got] == [f.id for f in expected]
    assert queue.bounded_serves == 0


class _LegacyFrameProcessorQueue(asyncio.PriorityQueue):
    """`FrameProcessorQueue` exactly as c2fef2ac4 shipped it (docstrings trimmed)."""

    HIGH_PRIORITY = 1
    LOW_PRIORITY = 2

    def __init__(self):
        super().__init__()
        self.__high_counter = 0
        self.__low_counter = 0

    async def put(self, item):
        frame, _, _ = item
        if isinstance(frame, SystemFrame):
            self.__high_counter += 1
            await super().put((self.HIGH_PRIORITY, self.__high_counter, item))
        else:
            self.__low_counter += 1
            await super().put((self.LOW_PRIORITY, self.__low_counter, item))

    async def get(self):
        _, _, item = await super().get()
        return item


async def _scripted_queue_trace(queue_cls, frames: list[Frame], seed: int) -> list[tuple]:
    """Drive a queue through the async API the input task uses and record everything visible.

    Concurrent putters, getters parked on an empty queue (some cancelled while
    parked, some cancelled right after a put woke them), task_done/join, and a
    size probe after every step. asyncio scheduling is deterministic, so two
    queues with the same semantics produce the same trace.
    """
    rng = random.Random(seed)
    queue = queue_cls()
    index = {id(frame): i for i, frame in enumerate(frames)}
    trace: list[tuple] = []
    getters: list[asyncio.Task] = []
    joins: list[asyncio.Task] = []
    received = 0
    next_frame = 0

    def settle():
        nonlocal received
        for n, task in enumerate(getters):
            if task is None or not task.done():
                continue
            if task.cancelled():
                trace.append(("cancelled", n))
            else:
                trace.append(("got", n, index[id(task.result()[0])]))
                received += 1
            getters[n] = None
        for n, task in enumerate(joins):
            if task is not None and task.done():
                trace.append(("joined", n))
                joins[n] = None

    async def catch_up():
        # Consume and finish everything so parked join() waiters wake.
        nonlocal received
        while not queue.empty():
            trace.append(("drain", index[id((await queue.get())[0])]))
            received += 1
        while received:
            queue.task_done()
            received -= 1
        await asyncio.sleep(0)
        settle()

    for step in range(3000):
        op = rng.random()
        if step % 500 == 499:
            await catch_up()
        elif op < 0.35 and next_frame < len(frames):
            await queue.put(_item(frames[next_frame]))
            next_frame += 1
        elif op < 0.55:
            getters.append(asyncio.create_task(queue.get()))
        elif op < 0.62:
            pending = [t for t in getters if t is not None and not t.done()]
            if pending:
                rng.choice(pending).cancel()
        elif op < 0.72 and received > 0:
            queue.task_done()
            received -= 1
        elif op < 0.75:
            joins.append(asyncio.create_task(queue.join()))
        else:
            await asyncio.sleep(0)
        settle()
        trace.append(("size", step, queue.qsize(), queue.empty(), queue.full(), queue.maxsize))

    for task in getters:
        if task is not None and not task.done():
            task.cancel()
    await asyncio.sleep(0)
    settle()
    await catch_up()
    for task in joins:
        if task is not None:
            task.cancel()
    await asyncio.gather(*(t for t in joins if t is not None), return_exceptions=True)
    return trace


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [0.0, BOUND])
@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_async_protocol_matches_the_shipped_queue(bound, seed, fake_clock, restore_bound):
    """Bound off, or on while nothing ages: identical to the c2fef2ac4 class, suspension points included."""
    FrameProcessorQueue.set_starvation_bound(bound)
    rng = random.Random(seed)
    frames = [
        rng.choice(
            [
                lambda: InputAudioRawFrame(audio=b"", sample_rate=SAMPLE_RATE, num_channels=1),
                lambda: UserSpeakingFrame(),
                lambda: InterruptionFrame(),
                lambda: TextFrame(text="t"),
                lambda: HeartbeatFrame(timestamp=0),
                lambda: EndFrame(),
            ]
        )()
        for _ in range(1000)
    ]
    legacy = await _scripted_queue_trace(_LegacyFrameProcessorQueue, frames, seed)
    current = await _scripted_queue_trace(FrameProcessorQueue, frames, seed)
    assert current == legacy
    # The schedule really exercises the queue.
    kinds = {entry[0] for entry in legacy}
    assert {"got", "cancelled", "joined", "drain"} <= kinds


def test_fresh_non_system_frame_keeps_strict_priority(fake_clock, restore_bound):
    FrameProcessorQueue.set_starvation_bound(BOUND)
    queue = FrameProcessorQueue()
    text = TextFrame(text="hello")
    audio = [UserSpeakingFrame() for _ in range(3)]
    queue.put_nowait(_item(text))
    for frame in audio:
        queue.put_nowait(_item(frame))
    fake_clock.now += BOUND / 2

    assert _drain(queue) == [*audio, text]
    assert queue.bounded_serves == 0


def test_aged_frame_is_not_overtaken_by_newer_system_frames(fake_clock, restore_bound):
    FrameProcessorQueue.set_starvation_bound(BOUND)
    queue = FrameProcessorQueue()
    older_audio = UserSpeakingFrame()
    text = TextFrame(text="hello")
    newer_audio = [UserSpeakingFrame() for _ in range(3)]
    queue.put_nowait(_item(older_audio))
    queue.put_nowait(_item(text))
    for frame in newer_audio:
        queue.put_nowait(_item(frame))
    fake_clock.now += BOUND

    # The older system frame still goes first; the aged text is next, ahead of
    # the system frames that arrived after it.
    assert _drain(queue) == [older_audio, text, *newer_audio]
    assert queue.bounded_serves == 1


def test_aged_frame_never_jumps_an_older_interruption(fake_clock, restore_bound):
    """Arrival order is the limit: an older InterruptionFrame must still flush it."""
    FrameProcessorQueue.set_starvation_bound(BOUND)
    queue = FrameProcessorQueue()
    interruption = InterruptionFrame()
    text = TextFrame(text="stale")
    queue.put_nowait(_item(interruption))
    queue.put_nowait(_item(text))
    fake_clock.now += 10 * BOUND

    assert _drain(queue) == [interruption, text]
    assert queue.bounded_serves == 0


def test_aged_frames_merge_in_arrival_order_and_keep_fifo_per_class(fake_clock, restore_bound):
    FrameProcessorQueue.set_starvation_bound(BOUND)
    queue = FrameProcessorQueue()
    sequence = []
    for i in range(20):
        frame = UserSpeakingFrame() if i % 3 else TextFrame(text=str(i))
        sequence.append(frame)
        queue.put_nowait(_item(frame))
    fake_clock.now += BOUND

    # Every non-system frame is aged, so the whole queue drains in arrival order.
    assert [f.id for f in _drain(queue)] == [f.id for f in sequence]


def test_one_fresh_frame_behind_an_aged_one_keeps_its_priority(fake_clock, restore_bound):
    FrameProcessorQueue.set_starvation_bound(BOUND)
    queue = FrameProcessorQueue()
    aged = TextFrame(text="aged")
    queue.put_nowait(_item(aged))
    fake_clock.now += BOUND
    audio_1 = UserSpeakingFrame()
    fresh = TextFrame(text="fresh")
    audio_2 = UserSpeakingFrame()
    for frame in (audio_1, fresh, audio_2):
        queue.put_nowait(_item(frame))

    assert _drain(queue) == [aged, audio_1, audio_2, fresh]


def test_instance_override_and_disable(fake_clock, restore_bound):
    FrameProcessorQueue.set_starvation_bound(BOUND)
    queue = FrameProcessorQueue()
    queue.starvation_bound_secs = 0.0
    text = TextFrame(text="hello")
    audio = UserSpeakingFrame()
    queue.put_nowait(_item(text))
    queue.put_nowait(_item(audio))
    fake_clock.now += 100

    assert _drain(queue) == [audio, text]

    FrameProcessorQueue.set_starvation_bound(None)
    assert FrameProcessorQueue.starvation_bound_secs == 0.0


@pytest.mark.asyncio
async def test_queue_protocol_is_preserved():
    """put/get/qsize/empty/task_done/join behave as the input task relies on."""
    queue = FrameProcessorQueue()
    assert queue.empty() and queue.qsize() == 0

    await queue.put(_item(TextFrame(text="a")))
    await queue.put(_item(UserSpeakingFrame()))
    assert queue.qsize() == 2 and not queue.empty()
    assert queue.non_system_qsize == 1

    getter = asyncio.create_task(queue.get())
    first = await getter
    assert isinstance(first[0], UserSpeakingFrame)
    queue.task_done()

    joined = asyncio.create_task(queue.join())
    await asyncio.sleep(0.01)
    assert not joined.done(), "one item is still unfinished"
    assert isinstance((await queue.get())[0], TextFrame)
    queue.task_done()
    await asyncio.wait_for(joined, 1.0)

    # A get() parked on an empty queue wakes on the next put.
    waiter = asyncio.create_task(queue.get())
    await asyncio.sleep(0.01)
    assert not waiter.done()
    await queue.put(_item(TextFrame(text="b")))
    assert (await asyncio.wait_for(waiter, 1.0))[0].text == "b"
    queue.task_done()
    with pytest.raises(ValueError):
        queue.task_done()


def test_non_frame_items_stay_non_system():
    """Nothing in the fork enqueues a sentinel today; a non-frame keeps the old (low) class."""
    queue = FrameProcessorQueue()
    sentinel = object()
    queue.put_nowait((sentinel, None, None))
    queue.put_nowait(_item(UserSpeakingFrame()))
    assert isinstance(queue.get_nowait()[0], UserSpeakingFrame)
    assert queue.get_nowait()[0] is sentinel
    with pytest.raises((TypeError, ValueError)):
        queue.put_nowait(sentinel)
    assert queue.qsize() == 0


def test_non_system_wait_diagnostics(fake_clock):
    queue = FrameProcessorQueue()
    assert queue.oldest_non_system_wait is None
    queue.put_nowait(_item(UserSpeakingFrame()))
    assert queue.oldest_non_system_wait is None
    queue.put_nowait(_item(HeartbeatFrame(timestamp=0)))
    fake_clock.now += 2.5
    queue.put_nowait(_item(TextFrame(text="later")))
    assert queue.non_system_qsize == 2
    assert queue.oldest_non_system_wait == pytest.approx(2.5)


@pytest.mark.parametrize(
    "raw, expected",
    [(None, 0.0), ("", 0.0), ("80", 0.08), ("12.5", 0.0125), ("-3", 0.0), ("fast", 0.0)],
)
def test_bound_from_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("PIPECAT_INPUT_QUEUE_STARVATION_BOUND_MS", raising=False)
    else:
        monkeypatch.setenv("PIPECAT_INPUT_QUEUE_STARVATION_BOUND_MS", raw)
    assert _starvation_bound_from_env() == pytest.approx(expected)


@pytest.mark.parametrize("raw, expected", [(None, "0.0"), ("80", "0.08")])
def test_env_bound_is_what_every_queue_starts_with(raw, expected):
    """The deploy interface: the variable is read at import and is every queue's default.

    Enabling the bound on a pod is only the env var (no api code), so a change
    that stopped reading it at import would silently leave that pod on strict
    priority. Checked in a fresh interpreter, where the import really happens.
    """
    env = {k: v for k, v in os.environ.items() if k != "PIPECAT_INPUT_QUEUE_STARVATION_BOUND_MS"}
    if raw is not None:
        env["PIPECAT_INPUT_QUEUE_STARVATION_BOUND_MS"] = raw
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    code = (
        "from pipecat.processors.frame_processor import FrameProcessorQueue as Q\n"
        "print(Q.starvation_bound_secs, Q().starvation_bound_secs)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == [expected, expected]


#
# The prod shape in a pipeline
#


def _audio(speech: bool = False) -> InputAudioRawFrame:
    marker = b"\x01" if speech else b"\x00"
    return InputAudioRawFrame(audio=marker * FRAME_BYTES, sample_rate=SAMPLE_RATE, num_channels=1)


class ServiceOrder:
    """When each frame reached a processor's input queue, and the order it was served in.

    Lets the load tests assert the bound's guarantee in arrival order instead
    of wall-clock allowances. Under host load the 21ms sleep that models the
    VAD hop runs 40-50ms, the audio backlog ahead of a frame doubles, and a
    latency budget built from the nominal service time fails on the base
    commit and the branch alike, while the order below holds at any speed.
    """

    def __init__(self):
        self.enqueued: dict[int, float] = {}
        self.served: list[tuple[str, int]] = []

    def watch(self, processor: FrameProcessor):
        original = processor.queue_frame

        async def queue_frame(frame, direction=FrameDirection.DOWNSTREAM, callback=None):
            self.enqueued.setdefault(frame.id, time.monotonic())
            await original(frame, direction, callback)

        processor.queue_frame = queue_frame  # type: ignore[method-assign]

    def serve(self, frame: Frame):
        if isinstance(frame, InputAudioRawFrame):
            self.served.append(("audio", frame.id))
        elif isinstance(frame, (TextFrame, HeartbeatFrame)):
            self.served.append((type(frame).__name__, frame.id))

    def newer_audio_served_first(self, frame_id: int) -> int | None:
        """Audio that reached the queue at least the bound after ``frame_id`` yet was served first.

        The bound allows at most the audio frame the input task starts right
        after moving the frame to the process queue (the process task runs at
        that frame's first suspension); strict priority serves all of it first.
        None if ``frame_id`` was never served.
        """
        limit = self.enqueued[frame_id] + BOUND
        count = 0
        for kind, fid in self.served:
            if fid == frame_id:
                return count
            if kind == "audio" and self.enqueued[fid] >= limit:
                count += 1
        return None

    def served_while_audio_flows(self, frame_id: int) -> bool:
        ids = [fid for _, fid in self.served]
        return frame_id in ids and any(
            kind == "audio" for kind, _ in self.served[ids.index(frame_id) + 1 :]
        )


class SlowSystemFrameProcessor(FrameProcessor):
    """Serves each audio frame like LLMUserAggregator's VAD hop on a busy pod."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = ServiceOrder()
        self.order.watch(self)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.order.serve(frame)
        if isinstance(frame, InputAudioRawFrame):
            await asyncio.sleep(SLOW_SERVICE_SECS)
        await self.push_frame(frame, direction)


class ArrivalRecorder(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arrivals: dict[str, float] = {}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            self.arrivals.setdefault(frame.text, time.monotonic())
        await self.push_frame(frame, direction)


async def _stream_audio(worker, secs: float, on_tick=None, speech=lambda t: False):
    """Queue 20ms audio frames at a true 50 fps (absolute schedule, no drift)."""
    start = time.monotonic()
    frames = int(secs / FRAME_SECS)
    for k in range(frames):
        delay = start + k * FRAME_SECS - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        t = time.monotonic() - start
        await worker.queue_frame(_audio(speech(t)))
        if on_tick:
            await on_tick(t)
    return time.monotonic()


async def _run_slow_system_pipeline(stream_secs: float):
    slow = SlowSystemFrameProcessor()
    recorder = ArrivalRecorder()
    worker = PipelineWorker(
        Pipeline([slow, recorder]),
        params=PipelineParams(
            audio_in_sample_rate=SAMPLE_RATE,
            enable_heartbeats=True,
            heartbeats_period_secs=0.5,
            heartbeats_monitor_secs=30.0,
        ),
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
    )
    heartbeats: list[tuple[float, float]] = []

    @worker.event_handler("on_heartbeat")
    async def on_heartbeat(worker, latency_secs):
        heartbeats.append((time.monotonic(), latency_secs))

    result: dict = {"queued": {}, "frame_ids": {}, "order": slow.order}

    async def driver():
        await asyncio.sleep(0.2)
        result["stream_start"] = time.monotonic()

        async def on_tick(t):
            if t >= 1.0 and "text" not in result["queued"]:
                for text in ("text", "transcript"):
                    frame = (
                        TextFrame(text=text)
                        if text == "text"
                        else TranscriptionFrame(text=text, user_id="caller", timestamp="")
                    )
                    await worker.queue_frame(frame)
                    result["queued"][text] = time.monotonic()
                    result["frame_ids"][text] = frame.id

        result["stream_end"] = await _stream_audio(worker, stream_secs, on_tick)
        result["bounded_serves"] = slow.input_queue_bounded_serves
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await asyncio.gather(runner.run(), driver())
    result["arrivals"] = recorder.arrivals
    result["heartbeats"] = heartbeats
    return result


class TestBoundServedFrameReachesTheProcessTask(unittest.IsolatedAsyncioTestCase):
    """A frame the bound served must not be flushed by the system frame queued behind it.

    Found by the load benchmark: a starved TranscriptionFrame was served, moved
    to the process queue, and the input task went straight on to a VAD turn
    start queued behind it, whose `broadcast_interruption()` reset the process
    queue before the process task ever ran. With strict priority a non-system
    frame is only served when no system frame waits, so the process task always
    runs before the next system frame. With the bound, the flush itself first
    lets the process task take a bound-served frame still waiting for it.
    """

    def setUp(self):
        saved = FrameProcessorQueue.starvation_bound_secs
        self.addCleanup(setattr, FrameProcessorQueue, "starvation_bound_secs", saved)

    async def _run(self, flush: str) -> tuple[list[str], int]:
        from tests.test_processor_diagnostics import _setup_processor

        events: list[str] = []

        class TurnStartingProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                if isinstance(frame, (UserSpeakingFrame, InterruptionFrame)):
                    # How many frames the flush is about to hit.
                    events.append(f"flush (queued={self.process_queue_depth})")
                await super().process_frame(frame, direction)
                if isinstance(frame, InputAudioRawFrame):
                    await asyncio.sleep(0.12)
                elif isinstance(frame, UserSpeakingFrame):
                    await self.broadcast_interruption()
                elif isinstance(frame, TextFrame):
                    events.append(frame.text)

        processor = TurnStartingProcessor()
        await _setup_processor(processor)
        processor.push_frame = _discard_push  # type: ignore[method-assign]
        try:
            await processor.queue_frame(StartFrame())
            await asyncio.sleep(0.05)
            await processor.queue_frame(_audio())
            await asyncio.sleep(0.01)
            # Both wait behind the slow audio frame; the text arrived first.
            await processor.queue_frame(TranscriptionFrame(text="hello", user_id="", timestamp=""))
            # A turn start that broadcasts an interruption from the input task,
            # or an InterruptionFrame arriving from a neighbour.
            await processor.queue_frame(
                UserSpeakingFrame() if flush == "broadcast" else InterruptionFrame()
            )
            await asyncio.sleep(0.4)
            bounded = processor.input_queue_bounded_serves
        finally:
            await processor.cleanup()
        return events, bounded

    async def test_strict_priority_delivers_it_after_the_flush(self):
        FrameProcessorQueue.set_starvation_bound(0)
        for flush in ("broadcast", "interruption_frame"):
            events, bounded = await self._run(flush)
            self.assertEqual(events, ["flush (queued=0)", "hello"], flush)
            self.assertEqual(bounded, 0)

    async def test_bound_served_frame_survives_a_broadcast_interruption(self):
        FrameProcessorQueue.set_starvation_bound(BOUND)
        events, bounded = await self._run("broadcast")
        # The text was already in the process queue when the turn start ran,
        # and was still delivered.
        self.assertEqual(bounded, 1)
        self.assertEqual(events, ["flush (queued=1)", "hello"])

    async def test_bound_served_frame_survives_an_interruption_frame(self):
        FrameProcessorQueue.set_starvation_bound(BOUND)
        events, bounded = await self._run("interruption_frame")
        self.assertEqual(bounded, 1)
        self.assertEqual(events, ["flush (queued=1)", "hello"])


class TestAgedBacklogDrainsInBulk(unittest.IsolatedAsyncioTestCase):
    """A processor that is only behind, not starved, must drain an aged backlog in bulk.

    Found in review. Any processor held for longer than the bound (an
    interruption waiting on its process task, a ParallelPipeline lifecycle
    sync, `pause_processing_all_frames_until`) comes back to a backlog of aged
    non-system frames interleaved with the input audio every processor after
    the transport receives at 50fps. Yielding a loop iteration after every
    bound-served frame capped that processor at one non-system frame per loop
    iteration, where strict priority moves the whole backlog in one step. On a
    loaded loop the cap is below the arrival rate: with ~20ms iterations and a
    60/s non-system stream (LLM tokens, TTS audio) a single 300ms hold grew the
    backlog to 238 frames and 4.0s of wait and never recovered, against 0.4s
    to recover with strict priority (scratchpad review/drain_probe.py).
    """

    FRAMES = 50

    def setUp(self):
        saved = FrameProcessorQueue.starvation_bound_secs
        self.addCleanup(setattr, FrameProcessorQueue, "starvation_bound_secs", saved)

    async def _drain(self) -> tuple[int, list[str], int]:
        from tests.test_processor_diagnostics import _setup_processor

        release = asyncio.Event()
        texts: list[str] = []

        class HeldOnce(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, UserSpeakingFrame):
                    await release.wait()
                elif isinstance(frame, TextFrame):
                    texts.append(frame.text)

        processor = HeldOnce()
        await _setup_processor(processor)
        processor.push_frame = _discard_push  # type: ignore[method-assign]
        try:
            await processor.queue_frame(StartFrame())
            await asyncio.sleep(0.05)
            await processor.queue_frame(UserSpeakingFrame())
            await asyncio.sleep(0.01)
            for i in range(self.FRAMES):
                await processor.queue_frame(_audio())
                await processor.queue_frame(TextFrame(text=str(i)))
            await asyncio.sleep(BOUND + 0.02)

            release.set()
            iterations = 0
            while len(texts) < self.FRAMES and iterations < 10 * self.FRAMES:
                await asyncio.sleep(0)
                iterations += 1
            bounded = processor.input_queue_bounded_serves
        finally:
            await processor.cleanup()
        return iterations, texts, bounded

    async def test_strict_priority_drains_the_backlog_in_one_step(self):
        FrameProcessorQueue.set_starvation_bound(0)
        iterations, texts, bounded = await self._drain()
        self.assertEqual(texts, [str(i) for i in range(self.FRAMES)])
        self.assertEqual(bounded, 0)
        self.assertLessEqual(iterations, 5)

    async def test_bound_drains_the_backlog_in_one_step_too(self):
        FrameProcessorQueue.set_starvation_bound(BOUND)
        iterations, texts, bounded = await self._drain()
        self.assertEqual(texts, [str(i) for i in range(self.FRAMES)])
        # Every text but the last was aged and older than the audio behind it...
        self.assertEqual(bounded, self.FRAMES - 1)
        # ...and moving them in arrival order costs no loop iterations.
        self.assertLessEqual(iterations, 5)


async def _discard_push(frame, direction=FrameDirection.DOWNSTREAM):
    return None


class TestSlowSystemFramesDoNotStarveTheRest(unittest.IsolatedAsyncioTestCase):
    STREAM_SECS = 3.0

    def setUp(self):
        saved = FrameProcessorQueue.starvation_bound_secs
        self.addCleanup(setattr, FrameProcessorQueue, "starvation_bound_secs", saved)

    async def test_strict_priority_starves_text_and_heartbeats(self):
        """Control: the bug. With strict priority nothing non-system moves until audio stops."""
        FrameProcessorQueue.set_starvation_bound(0)
        result = await _run_slow_system_pipeline(self.STREAM_SECS)

        end = result["stream_end"]
        for text in ("text", "transcript"):
            self.assertGreater(
                result["arrivals"][text], end, f"{text} should have waited out the audio"
            )
        during_stream = [
            at for at, _ in result["heartbeats"] if result["stream_start"] + 0.6 < at < end
        ]
        self.assertEqual(during_stream, [], "no heartbeat should cross while audio flows")
        self.assertEqual(result["bounded_serves"], 0)
        # In arrival order: every audio frame that came a bound after the text went first.
        self.assertGreater(
            result["order"].newer_audio_served_first(result["frame_ids"]["text"]), 10
        )

    async def test_bound_delivers_text_transcripts_and_heartbeats_while_audio_flows(self):
        FrameProcessorQueue.set_starvation_bound(BOUND)
        result = await _run_slow_system_pipeline(self.STREAM_SECS)
        order: ServiceOrder = result["order"]

        # A frame waits behind the audio that was already queued ahead of it
        # (arrival order) and behind newer audio only until it has aged past
        # the bound: nothing that arrived a bound later goes first, save the
        # one frame started right after the move. The strict-priority control
        # serves every newer audio frame first.
        for text in ("text", "transcript"):
            frame_id = result["frame_ids"][text]
            self.assertLessEqual(order.newer_audio_served_first(frame_id), 1, text)
            self.assertTrue(order.served_while_audio_flows(frame_id), text)

        heartbeats = [fid for kind, fid in order.served if kind == "HeartbeatFrame"]
        during_stream = [fid for fid in heartbeats if order.served_while_audio_flows(fid)]
        self.assertGreaterEqual(len(during_stream), 3, order.served)
        for fid in during_stream:
            self.assertLessEqual(order.newer_audio_served_first(fid), 1)
        self.assertGreater(result["bounded_serves"], 0)


class SlowStubVADAnalyzer(VADAnalyzer):
    """VAD whose analysis costs 21ms of loop time, like the executor hop on a busy pod.

    Speech is marked in the audio payload, so the VAD result is exact while
    the service time is what it would be in prod.
    """

    def __init__(self):
        super().__init__(params=VADParams(start_secs=0.0, stop_secs=0.0))

    def num_frames_required(self) -> int:
        return 160

    def voice_confidence(self, buffer: bytes) -> float:
        return 0.0

    async def analyze_audio(self, buffer: bytes) -> VADState:
        await asyncio.sleep(SLOW_SERVICE_SECS)
        return VADState.SPEAKING if buffer[:1] == b"\x01" else VADState.QUIET


USER_TURN_STOP_FUSE_SECS = 1.5


async def _run_user_aggregator(stream_secs: float):
    """Caller speaks 0.2-0.8s; STT's final lands at 1.0s (after VAD stop), as in run 5155948."""
    aggregator = LLMUserAggregator(
        LLMContext(),
        params=LLMUserAggregatorParams(
            vad_analyzer=SlowStubVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.3)],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_FUSE_SECS,
        ),
    )
    events: dict = {"stopped": [], "timeouts": [], "transcript_seen": None, "order": ServiceOrder()}
    events["order"].watch(aggregator)

    @aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        events["stopped"].append((time.monotonic(), message.content))

    @aggregator.event_handler("on_user_turn_stop_timeout")
    async def on_user_turn_stop_timeout(aggregator):
        events["timeouts"].append(time.monotonic())

    @aggregator.event_handler("on_before_process_frame")
    async def on_before_process_frame(aggregator, frame):
        events["order"].serve(frame)
        if isinstance(frame, TranscriptionFrame) and events["transcript_seen"] is None:
            events["transcript_seen"] = time.monotonic()

    worker = PipelineWorker(
        Pipeline([aggregator]),
        params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE),
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
    )

    async def driver():
        await asyncio.sleep(0.2)

        async def on_tick(t):
            if t >= 1.0 and "transcript_queued" not in events:
                events["transcript_queued"] = time.monotonic()
                transcript = TranscriptionFrame(
                    text="מה שלומך, מאיה?", user_id="caller", timestamp="", finalized=True
                )
                events["transcript_id"] = transcript.id
                await worker.queue_frame(transcript)

        events["stream_end"] = await _stream_audio(
            worker, stream_secs, on_tick, speech=lambda t: 0.2 <= t < 0.8
        )
        await asyncio.sleep(0.5)
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await asyncio.gather(runner.run(), driver())
    return events


class TestUserAggregatorHearsTheCallerUnderLoad(unittest.IsolatedAsyncioTestCase):
    STREAM_SECS = 3.5

    def setUp(self):
        saved = FrameProcessorQueue.starvation_bound_secs
        self.addCleanup(setattr, FrameProcessorQueue, "starvation_bound_secs", saved)

    async def test_strict_priority_commits_an_empty_turn(self):
        """Control: the prod failure. The fuse fires and the turn is released empty."""
        FrameProcessorQueue.set_starvation_bound(0)
        events = await _run_user_aggregator(self.STREAM_SECS)

        self.assertGreater(events["transcript_seen"], events["stream_end"])
        self.assertTrue(events["timeouts"], "the turn-stop fuse should have fired")
        self.assertEqual(events["stopped"][0][1], "", "the first turn is committed empty")

    async def test_bound_delivers_the_transcript_into_its_turn(self):
        FrameProcessorQueue.set_starvation_bound(BOUND)
        events = await _run_user_aggregator(self.STREAM_SECS)

        # Only the audio already queued ahead of it and one frame past the
        # bound go first (see ServiceOrder), and it lands far inside the fuse.
        self.assertLessEqual(events["order"].newer_audio_served_first(events["transcript_id"]), 1)
        self.assertEqual(events["timeouts"], [], "the turn-stop fuse must not fire")
        self.assertEqual(events["stopped"][0][1], "מה שלומך, מאיה?")
        self.assertLess(events["stopped"][0][0], events["stream_end"])


#
# Ordering tests re-run with the bound enabled: interruption, EndFrame/StopFrame
# survival, heartbeat and pause semantics must not change.
#


class _BoundEnabled:
    def setUp(self):
        saved = FrameProcessorQueue.starvation_bound_secs
        self.addCleanup(setattr, FrameProcessorQueue, "starvation_bound_secs", saved)
        FrameProcessorQueue.set_starvation_bound(BOUND)
        super().setUp()


class TestFrameProcessorWithBound(_BoundEnabled, frame_processor_tests.TestFrameProcessor):
    pass


class TestHeartbeatSurvivesInterruptionsWithBound(
    _BoundEnabled, frame_processor_tests.TestHeartbeatSurvivesInterruptions
):
    pass


class TestHeartbeatIgnoresProcessingPauseWithBound(
    _BoundEnabled, frame_processor_tests.TestHeartbeatIgnoresProcessingPause
):
    pass


class _TightBoundEnabled:
    """A 1ms bound ages nearly every queued frame: the worst case for ordering."""

    def setUp(self):
        saved = FrameProcessorQueue.starvation_bound_secs
        self.addCleanup(setattr, FrameProcessorQueue, "starvation_bound_secs", saved)
        FrameProcessorQueue.set_starvation_bound(0.001)
        super().setUp()


class TestFrameProcessorWithTightBound(
    _TightBoundEnabled, frame_processor_tests.TestFrameProcessor
):
    pass
