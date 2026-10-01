#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Text of a response an interruption cancelled is never spoken.

An ``InterruptionFrame`` is a system frame: it overtakes the data frames still
waiting in the TTS input queue, so the cancelled response's last tokens and the
``LLMFullResponseEndFrame`` the cancelled LLM pushes from its ``finally`` reach
the TTS AFTER its interruption reset. Each test below feeds the TTS that
post-overtake order directly — the order is the bug, so it is the fixture.
"""

from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_SAMPLE_RATE = 16000
_FAKE_AUDIO = b"\x00\x01" * 320


class RecordingTTSService(TTSService):
    """HTTP-style TTS that records every text it is asked to synthesize."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self.spoken: list[str] = []

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.spoken.append(text)
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO, sample_rate=_SAMPLE_RATE, num_channels=1, context_id=context_id
        )


def _spoken(tts: RecordingTTSService) -> str:
    return "".join(tts.spoken).strip()


@pytest.mark.asyncio
async def test_fragment_flushed_by_the_cancelled_end_frame_is_not_spoken():
    """Run 11408662: «הבנ» arrived after the reset and the end frame flushed it."""
    tts = RecordingTTSService()
    start = LLMFullResponseStartFrame()
    frames = [
        start,
        InterruptionFrame(),
        TextFrame(text="הבנ"),  # overtaken by the interruption
        LLMFullResponseEndFrame(),  # pushed by the cancelled LLM's finally
        SleepFrame(sleep=0.1),
    ]
    down, _ = await run_test(tts, frames_to_send=frames)
    assert tts.spoken == []
    assert not any(isinstance(f, TTSAudioRawFrame) for f in down)


@pytest.mark.asyncio
async def test_a_start_frame_created_before_the_interruption_is_still_cancelled():
    """The cancelled response's own start can be overtaken too: it must not reopen it."""
    tts = RecordingTTSService()
    stale_start = LLMFullResponseStartFrame()  # created before the interruption
    interruption = InterruptionFrame()
    frames = [
        interruption,
        stale_start,
        TextFrame(text="אני מ"),
        LLMFullResponseEndFrame(),
        SleepFrame(sleep=0.1),
    ]
    await run_test(tts, frames_to_send=frames)
    assert tts.spoken == []


@pytest.mark.asyncio
async def test_a_stale_fragment_is_not_glued_onto_the_next_reply():
    """Run 11406240 spoke «לאאני מבינה.»: the old «לא» waited for the next reply."""
    tts = RecordingTTSService()
    frames = [
        InterruptionFrame(),
        TextFrame(text="לא"),
        SleepFrame(sleep=0.05),
        LLMFullResponseStartFrame(),  # a new response, created after the interruption
        TextFrame(text="אני מבינה."),
        LLMFullResponseEndFrame(),
        SleepFrame(sleep=0.1),
    ]
    await run_test(tts, frames_to_send=frames)
    assert _spoken(tts) == "אני מבינה."


@pytest.mark.asyncio
async def test_a_response_started_after_the_interruption_is_spoken_in_full():
    tts = RecordingTTSService()
    frames = [
        InterruptionFrame(),
        SleepFrame(sleep=0.05),
        LLMFullResponseStartFrame(),
        TextFrame(text="בטח, "),
        TextFrame(text="אני מקשיבה."),
        LLMFullResponseEndFrame(),
        SleepFrame(sleep=0.1),
    ]
    await run_test(tts, frames_to_send=frames)
    assert _spoken(tts) == "בטח, אני מקשיבה."


@pytest.mark.asyncio
async def test_a_speak_frame_after_an_interruption_is_its_own_utterance():
    tts = RecordingTTSService()
    frames = [
        InterruptionFrame(),
        TTSSpeakFrame(text="רגע, אני בודקת."),
        SleepFrame(sleep=0.1),
    ]
    await run_test(tts, frames_to_send=frames)
    assert _spoken(tts) == "רגע, אני בודקת."


@pytest.mark.asyncio
async def test_without_an_interruption_nothing_changes():
    tts = RecordingTTSService()
    frames = [
        LLMFullResponseStartFrame(),
        TextFrame(text="שלום, "),
        TextFrame(text="מה שלומך?"),
        LLMFullResponseEndFrame(),
        SleepFrame(sleep=0.1),
    ]
    await run_test(tts, frames_to_send=frames)
    assert _spoken(tts) == "שלום, מה שלומך?"
