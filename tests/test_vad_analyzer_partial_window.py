#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""`VADAnalyzer.analyze_audio` skips the thread hop for a buffer that cannot complete a window.

The optimisation is only allowed to change cost, never results: every test
here compares it against the old path (one executor call per buffer) on the
same audio and requires identical state sequences.
"""

import asyncio
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState


class CountingExecutor(ThreadPoolExecutor):
    def __init__(self):
        super().__init__(max_workers=1)
        self.submits = 0

    def submit(self, fn, /, *args, **kwargs):
        self.submits += 1
        return super().submit(fn, *args, **kwargs)


class EnergyVADAnalyzer(VADAnalyzer):
    """Deterministic analyzer: confidence is the window's RMS level."""

    def num_frames_required(self) -> int:
        return 512 if self.sample_rate == 16000 else 256

    def voice_confidence(self, buffer: bytes) -> float:
        samples = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        return float(min(1.0, math.sqrt(float(np.mean(samples * samples))) * 8))


class ExecutorEveryCallAnalyzer(EnergyVADAnalyzer):
    """The old behaviour: overriding `_run_analyzer` opts out of the fast path."""

    def _run_analyzer(self, buffer: bytes) -> VADState:
        return super()._run_analyzer(buffer)


def _make(cls, sample_rate: int, params: VADParams):
    analyzer = cls(params=params)
    analyzer._executor = CountingExecutor()
    analyzer.set_sample_rate(sample_rate)
    return analyzer


def _synthetic_call_audio(sample_rate: int, seconds: float, seed: int) -> bytes:
    """Speech-like bursts of every length the state machine distinguishes.

    Long utterances, blips shorter than start_secs (STARTING -> QUIET), pauses
    shorter than stop_secs (STOPPING -> SPEAKING), long silences, and low-level
    noise throughout.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    total = int(sample_rate * seconds)
    audio = np_rng.normal(0, 60, total)
    t = 0
    while t < total:
        gap = int(sample_rate * rng.choice([0.05, 0.12, 0.3, 0.8, 1.5]))
        burst = int(sample_rate * rng.choice([0.04, 0.1, 0.25, 0.6, 1.2, 2.0]))
        t += gap
        end = min(total, t + burst)
        if t >= total:
            break
        n = np.arange(end - t)
        freq = rng.choice([140, 220, 310])
        amplitude = rng.choice([1500, 6000, 14000])
        envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * n / sample_rate)
        audio[t:end] += amplitude * envelope * np.sin(2 * np.pi * freq * n / sample_rate)
        t = end
    return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()


def _chunks(audio: bytes, sample_rate: int, chunk_ms, seed: int):
    """Split into chunks of ``chunk_ms`` (a number, or "random" for 5-45ms)."""
    rng = random.Random(seed)
    pos = 0
    while pos < len(audio):
        ms = rng.randint(5, 45) if chunk_ms == "random" else chunk_ms
        size = int(sample_rate * ms / 1000) * 2
        yield audio[pos : pos + size]
        pos += size


async def _states(analyzer: VADAnalyzer, chunks) -> list[VADState]:
    return [await analyzer.analyze_audio(chunk) for chunk in chunks]


PARAM_SETS = [
    VADParams(confidence=0.5, start_secs=0.2, stop_secs=0.2, min_volume=0.0),
    VADParams(confidence=0.7, start_secs=0.1, stop_secs=0.8, min_volume=0.6),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", [8000, 16000])
@pytest.mark.parametrize("chunk_ms", [10, 20, 30, 40, "random"])
@pytest.mark.parametrize("params", PARAM_SETS, ids=["defaults", "strict"])
async def test_identical_state_sequence_with_and_without_the_fast_path(
    sample_rate, chunk_ms, params
):
    audio = _synthetic_call_audio(sample_rate, seconds=20.0, seed=sample_rate)
    fast = _make(EnergyVADAnalyzer, sample_rate, params)
    old = _make(ExecutorEveryCallAnalyzer, sample_rate, params)
    chunks = list(_chunks(audio, sample_rate, chunk_ms, seed=7))

    fast_states = await _states(fast, chunks)
    old_states = await _states(old, chunks)

    assert fast_states == old_states
    assert fast._vad_buffer == old._vad_buffer
    if params is PARAM_SETS[0]:
        # The stream really exercises the state machine.
        assert set(old_states) == set(VADState)

    # One hop per buffer before; now only for buffers that complete a window.
    assert old._executor.submits == len(chunks)
    window = fast._vad_frames_num_bytes
    buffered = windows_completed = 0
    for chunk in chunks:
        if (buffered + len(chunk)) // window > buffered // window:
            windows_completed += 1
        buffered += len(chunk)
    assert fast._executor.submits == windows_completed


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", [8000, 16000])
async def test_twenty_ms_frames_skip_three_hops_in_eight(sample_rate):
    analyzer = _make(EnergyVADAnalyzer, sample_rate, PARAM_SETS[0])
    frame = b"\x00" * (int(sample_rate * 0.02) * 2)
    for _ in range(800):
        await analyzer.analyze_audio(frame)
    assert analyzer._executor.submits == 500


@pytest.mark.asyncio
async def test_real_silero_matches_with_and_without_the_fast_path(monkeypatch):
    silero = pytest.importorskip("pipecat.audio.vad.silero")
    # The model resets its recurrent state on a wall clock; pin it so both runs
    # see the same resets however slow the host is.
    monkeypatch.setattr(silero, "_MODEL_RESET_STATES_TIME", 1e9)

    class SileroEveryCall(silero.SileroVADAnalyzer):
        def _run_analyzer(self, buffer: bytes) -> VADState:
            return super()._run_analyzer(buffer)

    sample_rate = 8000
    audio = _synthetic_call_audio(sample_rate, seconds=12.0, seed=3)
    chunks = list(_chunks(audio, sample_rate, 20, seed=0))
    params = VADParams(confidence=0.5, start_secs=0.2, stop_secs=0.2, min_volume=0.0)

    fast = _make(silero.SileroVADAnalyzer, sample_rate, params)
    old = _make(SileroEveryCall, sample_rate, params)
    assert await _states(fast, chunks) == await _states(old, chunks)
    assert fast._executor.submits < old._executor.submits


@pytest.mark.asyncio
async def test_fast_path_waits_for_an_analysis_still_running_on_the_worker():
    """A cancelled await can leave `_run_analyzer` running; appending beside it would race."""
    release = threading.Event()
    started = threading.Event()

    class BlockingAnalyzer(EnergyVADAnalyzer):
        def voice_confidence(self, buffer: bytes) -> float:
            started.set()
            release.wait(5)
            return 0.0

    analyzer = _make(BlockingAnalyzer, 8000, PARAM_SETS[0])
    window = b"\x01\x00" * 256
    partial = b"\x02\x00" * 10

    in_flight = asyncio.create_task(analyzer.analyze_audio(window + partial))
    await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
    in_flight.cancel()
    with pytest.raises(asyncio.CancelledError):
        await in_flight

    # The worker is still inside the first analysis: this partial buffer must be
    # queued behind it, not appended from the loop thread.
    follow_up = asyncio.create_task(analyzer.analyze_audio(partial))
    await asyncio.sleep(0.05)
    assert analyzer._executor.submits == 2
    assert not follow_up.done()
    release.set()
    assert await asyncio.wait_for(follow_up, 5) == VADState.QUIET
    assert analyzer._vad_buffer == partial + partial

    # Once it has finished, the fast path is used again.
    await analyzer.analyze_audio(partial)
    assert analyzer._executor.submits == 2


@pytest.mark.asyncio
async def test_fast_path_waits_behind_a_running_analysis_when_the_queued_one_was_cancelled():
    """A cancelled queued analysis is `done()` but never ran: it proves nothing about the one ahead of it.

    Two cancelled awaits in a row: the first leaves its analysis running on the
    worker, the second cancels its own analysis while it is still queued behind
    that one. The last submitted future is then done (cancelled) while the
    worker is still inside the first `_run_analyzer`, so the fast path must not
    take it as "the previous analysis has finished".
    """
    release = threading.Event()
    started = threading.Event()

    class BlockingAnalyzer(EnergyVADAnalyzer):
        def voice_confidence(self, buffer: bytes) -> float:
            started.set()
            release.wait(5)
            return 0.0

    analyzer = _make(BlockingAnalyzer, 8000, PARAM_SETS[0])
    window = b"\x01\x00" * 256
    partial = b"\x02\x00" * 10

    running = asyncio.create_task(analyzer.analyze_audio(window))
    await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    queued = asyncio.create_task(analyzer.analyze_audio(partial))
    await asyncio.sleep(0.01)
    assert analyzer._executor.submits == 2
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await asyncio.sleep(0)
    assert analyzer._pending_analysis.cancelled(), "the queued analysis never started"

    # The worker is still inside the first analysis: this buffer must queue
    # behind it rather than be appended from the loop thread beside it.
    follow_up = asyncio.create_task(analyzer.analyze_audio(partial))
    await asyncio.sleep(0.05)
    assert analyzer._executor.submits == 3
    assert not follow_up.done()
    release.set()
    assert await asyncio.wait_for(follow_up, 5) == VADState.QUIET
    # The cancelled queued buffer was dropped (as it always was with
    # run_in_executor); the first window was consumed.
    assert analyzer._vad_buffer == partial

    # Once a submitted analysis has really run, the fast path is used again.
    await analyzer.analyze_audio(partial)
    assert analyzer._executor.submits == 3


@pytest.mark.asyncio
async def test_shut_down_analyzer_neither_appends_nor_raises():
    analyzer = _make(EnergyVADAnalyzer, 8000, PARAM_SETS[0])
    await analyzer.analyze_audio(b"\x00\x00" * 100)
    analyzer.shutdown()
    assert await analyzer.analyze_audio(b"\x00\x00" * 100) == VADState.QUIET
    assert len(analyzer._vad_buffer) == 200


@pytest.mark.asyncio
async def test_before_the_sample_rate_is_set_the_executor_path_is_kept():
    """Without window sizing there is no fast path; behaviour is exactly the old one."""
    analyzer = EnergyVADAnalyzer()
    analyzer._executor = CountingExecutor()
    with pytest.raises(AttributeError):
        await analyzer.analyze_audio(b"\x00\x00")
    assert analyzer._executor.submits == 1
