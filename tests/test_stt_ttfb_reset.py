#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""STT TTFB is measured from the VAD stop of the utterance it answers.

Prod run 5155948 logged "STT TTFB 4364ms" and "5996ms" on a Soniox session
whose normal turns measured 222-257ms. Each value was exactly the gap back to
an earlier VAD stop that got no final within the 2s TTFB timeout: a VAD start
cancelled the timeout but left the measurement's start armed, so the next
utterance's final was measured against it.
"""

import time
from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    MetricsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.tests.utils import SleepFrame, run_test

STOP_SECS = 0.2
TTFB_TIMEOUT = 0.3


class FakeSTTService(STTService):
    def __init__(self, **kwargs):
        kwargs.setdefault("settings", STTSettings(model=None, language=None))
        super().__init__(stt_ttfb_timeout=TTFB_TIMEOUT, **kwargs)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield None


def _final(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="caller", timestamp="", finalized=True)


async def _ttfbs(frames_to_send) -> list[float]:
    # Re-stamp VAD stops as the pipeline receives them: run_test builds the
    # list up front, and the measurement starts at `timestamp - stop_secs`.
    class StampingSTT(FakeSTTService):
        async def process_frame(self, frame, direction):
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                frame.timestamp = time.time()
            await super().process_frame(frame, direction)

    down, _ = await run_test(
        StampingSTT(),
        frames_to_send=frames_to_send,
        pipeline_params=PipelineParams(enable_metrics=True, send_initial_empty_metrics=False),
    )
    return [
        data.value
        for frame in down
        if isinstance(frame, MetricsFrame)
        for data in frame.data
        if isinstance(data, TTFBMetricsData)
    ]


@pytest.mark.asyncio
async def test_final_after_a_new_utterance_is_not_measured_against_the_old_stop():
    ttfbs = await _ttfbs(
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=STOP_SECS),
            # No final for this utterance: the TTFB timeout passes without one.
            SleepFrame(sleep=TTFB_TIMEOUT + 0.5),
            VADUserStartedSpeakingFrame(),
            SleepFrame(sleep=0.1),
            # The next utterance's final arrives while the caller is still talking.
            _final("את איתי?"),
            SleepFrame(sleep=0.1),
        ]
    )
    # Before the fix: one TTFB of ~1.0s (0.2 + 0.8 + 0.1 back to the old stop).
    assert ttfbs == []


@pytest.mark.asyncio
async def test_next_utterance_is_measured_from_its_own_stop():
    ttfbs = await _ttfbs(
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=STOP_SECS),
            SleepFrame(sleep=TTFB_TIMEOUT + 0.5),
            VADUserStartedSpeakingFrame(),
            SleepFrame(sleep=0.1),
            VADUserStoppedSpeakingFrame(stop_secs=STOP_SECS),
            SleepFrame(sleep=0.1),
            _final("מאיה?"),
            SleepFrame(sleep=0.1),
        ]
    )
    assert len(ttfbs) == 1
    # stop_secs + the 0.1s wait; nowhere near the ~1.2s back to the first stop.
    assert 0.25 <= ttfbs[0] < 0.6


@pytest.mark.asyncio
async def test_final_for_the_current_utterance_is_still_measured():
    ttfbs = await _ttfbs(
        [
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(stop_secs=STOP_SECS),
            SleepFrame(sleep=0.1),
            _final("מה שלומך, מאיה?"),
            SleepFrame(sleep=0.1),
        ]
    )
    assert len(ttfbs) == 1
    assert 0.25 <= ttfbs[0] < 0.6


@pytest.mark.asyncio
async def test_reset_discards_a_measurement_and_keeps_the_initial_one_owed():
    metrics = FrameProcessorMetrics()
    metrics.set_processor_name("stt")

    await metrics.reset_ttfb_metrics()  # nothing in progress: no-op
    await metrics.start_ttfb_metrics(start_time=time.time() - 5, report_only_initial_ttfb=True)
    await metrics.reset_ttfb_metrics()
    assert await metrics.stop_ttfb_metrics() is None
    assert metrics.ttfb is None

    # The abandoned measurement was never reported, so the initial one is still owed.
    await metrics.start_ttfb_metrics(start_time=time.time() - 0.2, report_only_initial_ttfb=True)
    frame = await metrics.stop_ttfb_metrics()
    assert frame is not None and 0.15 < frame.data[0].value < 1.0

    # ...and only once.
    await metrics.start_ttfb_metrics(start_time=time.time() - 0.2, report_only_initial_ttfb=True)
    assert await metrics.stop_ttfb_metrics() is None
