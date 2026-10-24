#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fork: choosing between the bundled Silero models.

``"v6.2"`` is upstream's model and the default. ``"v6"`` is the model the fork
ran before the 1.14 sync, with the reset schedule it ran with: every five
seconds of audio, whatever the audio is.
"""

import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState

SAMPLE_RATE = 16000
FRAME_SECS = 0.02
# Synthesized speech with every pause removed (shared with test_silero_vad.py).
SPEECH_PATH = Path(__file__).parent / "assets" / "continuous_speech_16k.wav"


def _analyzer(**kwargs) -> SileroVADAnalyzer:
    analyzer = SileroVADAnalyzer(sample_rate=SAMPLE_RATE, **kwargs)
    analyzer.set_sample_rate(SAMPLE_RATE)
    return analyzer


def _record_resets(analyzer: SileroVADAnalyzer, position: list[float]) -> list[float]:
    """Record where in the audio the analyzer resets the model.

    The model also calls ``reset_states(batch_size)`` itself to initialize its
    state; those calls carry an argument and aren't the analyzer's resets.
    """
    resets: list[float] = []
    model = analyzer._model
    reset_states = model.reset_states

    def record_reset(*args, **kwargs):
        if not args and not kwargs:
            resets.append(position[0])
        reset_states(*args, **kwargs)

    model.reset_states = record_reset
    return resets


async def _analyze(
    analyzer: SileroVADAnalyzer, audio: np.ndarray, position: list[float]
) -> list[VADState]:
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    frame_samples = int(FRAME_SECS * SAMPLE_RATE)
    states = []
    for start in range(0, len(audio) - frame_samples + 1, frame_samples):
        position[0] = start / SAMPLE_RATE
        states.append(await analyzer.analyze_audio(audio[start : start + frame_samples].tobytes()))
    return states


class TestSileroModelChoice(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        speech, sample_rate = sf.read(SPEECH_PATH, dtype="int16")
        assert sample_rate == SAMPLE_RATE
        cls.speech = speech.astype(np.float64)

    def test_v6_2_is_the_default(self):
        self.assertEqual(_analyzer()._model_version, "v6.2")

    def test_an_unknown_model_is_refused(self):
        with self.assertRaises(ValueError):
            SileroVADAnalyzer(sample_rate=SAMPLE_RATE, model="v5")

    def test_v6_runs_a_different_model(self):
        window = (self.speech[: 512 * 40] / 32768.0).astype(np.float32)
        scores = {}
        for model in ("v6", "v6.2"):
            analyzer = _analyzer(model=model)
            scores[model] = [
                float(np.asarray(analyzer._model(window[i : i + 512], SAMPLE_RATE)).item())
                for i in range(0, len(window), 512)
            ]
        self.assertNotEqual(scores["v6"], scores["v6.2"])
        # Both hear the speech.
        self.assertGreater(max(scores["v6"]), 0.9)
        self.assertGreater(max(scores["v6.2"]), 0.9)

    async def test_v6_resets_every_five_seconds_even_during_speech(self):
        analyzer = _analyzer(model="v6")
        position = [0.0]
        resets = _record_resets(analyzer, position)

        speech = np.tile(self.speech, 6)[: 30 * SAMPLE_RATE]
        states = await _analyze(analyzer, speech, position)

        self.assertIn(VADState.SPEAKING, states)
        self.assertEqual(len(resets), 5, resets)
        spacing = np.diff(resets)
        self.assertTrue(np.allclose(spacing, 5.0, atol=0.05), spacing)

    async def test_v6_2_is_not_reset_during_speech(self):
        analyzer = _analyzer()
        position = [0.0]
        resets = _record_resets(analyzer, position)

        speech = np.tile(self.speech, 6)[: 30 * SAMPLE_RATE]
        await _analyze(analyzer, speech, position)

        self.assertEqual(resets, [])


if __name__ == "__main__":
    unittest.main()
