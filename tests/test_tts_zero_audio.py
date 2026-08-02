#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the TTS zero-audio completion warning.

When a TTS context completes with text sent for synthesis but zero audio
bytes produced (run-67 wr1875 shape: the provider accepted the render and
returned nothing, so the leg played silence), the base TTSService must emit a
distinct TTS_ZERO_AUDIO warning and fire ``on_tts_zero_audio`` so applications
can count these per leg. It must NOT fire for healthy synthesis or for
contexts cut short by an interruption (barge-in is not a provider failure).
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000


class MockNoAudioTTSService(TTSService):
    """HTTP-style service whose provider returns no audio at all.

    run_tts yields nothing, so the audio context holds only the base-class
    TTSStartedFrame; the context's stop-frame timeout then declares it
    finished — the same path a provider-side synthesis failure takes.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            stop_frame_timeout_s=0.2,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if False:
            yield  # async generator that never yields: no audio ever arrives


class MockWebSocketNoAudioTTSService(TTSService):
    """WebSocket-style service whose receive loop delivers only the stop frame.

    Models a provider that closes the stream cleanly without ever sending an
    audio chunk.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async def _deliver():
            await asyncio.sleep(0.01)
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

        self.create_task(_deliver(), name=f"mock_no_audio_deliver_{context_id}")
        if False:
            yield


class MockHealthyTTSService(TTSService):
    """HTTP-style service that synthesizes normally."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


class MockNeverDeliversTTSService(TTSService):
    """WebSocket-style service that never delivers anything (used with barge-in)."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if False:
            yield


def _capture_zero_audio_events(tts: TTSService) -> list[tuple[str, int]]:
    events: list[tuple[str, int]] = []

    @tts.event_handler("on_tts_zero_audio")
    async def on_tts_zero_audio(tts: TTSService, context_id: str, chars_sent: int):
        events.append((context_id, chars_sent))

    return events


@pytest.mark.asyncio
async def test_zero_audio_on_context_timeout_fires_warning_and_event():
    """The wr1875 shape: text rendered and sent, no audio ever returned.

    The context finishes via the stop-frame timeout; the completion must be
    flagged with the char count that was sent for synthesis.
    """
    tts = MockNoAudioTTSService()
    events = _capture_zero_audio_events(tts)

    captured_logs: list[str] = []
    sink_id = logger.add(lambda message: captured_logs.append(str(message)), level="WARNING")
    try:
        frames_received = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="hello world"), SleepFrame(sleep=0.5)],
        )
    finally:
        logger.remove(sink_id)

    down_frames = frames_received[0]
    assert any(isinstance(f, TTSStartedFrame) for f in down_frames)
    assert any(isinstance(f, TTSStoppedFrame) for f in down_frames)
    assert not any(isinstance(f, TTSAudioRawFrame) for f in down_frames)

    assert tts.zero_audio_context_count == 1
    assert len(events) == 1
    context_id, chars_sent = events[0]
    assert context_id
    assert chars_sent == len("hello world")

    zero_logs = [m for m in captured_logs if "TTS_ZERO_AUDIO" in m]
    assert len(zero_logs) == 1, "exactly one distinct zero-audio warning must be logged"
    assert "zero audio bytes" in zero_logs[0]


@pytest.mark.asyncio
async def test_zero_audio_on_clean_provider_close_fires_once():
    """A provider that closes the stream with no audio chunks must be flagged too."""
    tts = MockWebSocketNoAudioTTSService()
    events = _capture_zero_audio_events(tts)

    await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text="silent utterance"), SleepFrame(sleep=0.3)],
    )

    assert tts.zero_audio_context_count == 1
    assert len(events) == 1
    assert events[0][1] == len("silent utterance")


@pytest.mark.asyncio
async def test_healthy_synthesis_is_not_flagged():
    """Normal synthesis must not trip the zero-audio warning."""
    tts = MockHealthyTTSService()
    events = _capture_zero_audio_events(tts)

    frames_received = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text="hello world"), SleepFrame(sleep=0.3)],
    )

    down_frames = frames_received[0]
    assert any(isinstance(f, TTSAudioRawFrame) for f in down_frames)
    assert tts.zero_audio_context_count == 0
    assert events == []


@pytest.mark.asyncio
async def test_interruption_before_audio_is_not_flagged():
    """Barge-in before any audio arrives is not a provider failure.

    The interruption finalizes the synth accounting without a TTSStoppedFrame
    ever flowing for the context, so the zero-audio warning must stay silent.
    """
    tts = MockNeverDeliversTTSService()
    events = _capture_zero_audio_events(tts)

    await run_test(
        tts,
        frames_to_send=[
            TTSSpeakFrame(text="about to be interrupted"),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            SleepFrame(sleep=0.2),
        ],
    )

    assert tts.zero_audio_context_count == 0
    assert events == []
