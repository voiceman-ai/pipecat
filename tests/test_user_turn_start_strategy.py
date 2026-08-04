#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import (
    ExternalUserTurnStartStrategy,
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

# Short enough to keep the suite fast, long enough that a deferral cannot expire
# between two consecutive awaits on a loaded CI box.
DEFER_SECS = 0.05
PAST_DEADLINE = DEFER_SECS + 0.05

# Deliberately longer than any test could take, for the cases that must prove
# something happened for a reason *other* than the deadline elapsing.
NEVER = 3600.0


class FakeBargeInGate:
    """Stand-in for the near-field meter's ``should_defer_barge_in`` callable.

    Returns True to mean "this speech onset is confidently far-field", i.e. not
    the caller. ``raises`` models a meter that blew up mid-call, which the
    strategy must treat as "not far".
    """

    def __init__(self, *, far: bool = False, raises: bool = False):
        self.far = far
        self.raises = raises
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        if self.raises:
            raise RuntimeError("near-field meter exploded")
        return self.far


def audio_tick() -> InputAudioRawFrame:
    """20 ms of silence at 16 kHz.

    A pending deferral is re-examined on ordinary frames rather than from a
    timer the strategy owns, and input audio is what actually supplies that
    clock in production: the user aggregator forwards every frame it sees to the
    turn controller, and the mute set that can swallow frames on the way does
    not include ``InputAudioRawFrame``.
    """
    return InputAudioRawFrame(audio=b"\x00" * 640, sample_rate=16000, num_channels=1)


class TestMinWordsInterruptionStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_bot_speaking_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="Hello", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

        await strategy.process_frame(
            TranscriptionFrame(text="Hello there!", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)

        # Reset and check again
        should_start = None
        await strategy.reset()

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)

    async def test_bot_speaking_singlw_words(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=3)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="One", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

        await strategy.process_frame(TranscriptionFrame(text="Two", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

        await strategy.process_frame(TranscriptionFrame(text="Three", user_id="cat", timestamp=""))
        self.assertFalse(should_start)

    async def test_bot_speaking_interim_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp="")
        )
        self.assertFalse(should_start)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello there!", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)

    async def test_bot_speaking_all_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp="")
        )
        self.assertFalse(should_start)

        await strategy.process_frame(
            TranscriptionFrame(text="Hello there!", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)

    async def test_bot_not_speaking_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(TranscriptionFrame(text="Hello", user_id="cat", timestamp=""))
        self.assertTrue(should_start)

    async def test_bot_not_speaking_interim_transcriptions(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=2)

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)


class TestVADUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_vad_strategy(self):
        strategy = VADUserTurnStartStrategy()

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertTrue(should_start)

    def _build(self, gate=None, defer_max_secs: float = DEFER_SECS):
        """Build a strategy and a list that records every turn start it fires."""
        kwargs = {} if gate is None else {"barge_in_gate": gate}
        strategy = VADUserTurnStartStrategy(defer_max_secs=defer_max_secs, **kwargs)

        starts = []

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            starts.append(params)

        return strategy, starts

    async def test_no_gate_is_unchanged_while_the_bot_speaks(self):
        """With no gate wired, a VAD onset interrupts immediately, as it always has."""
        strategy, starts = self._build()

        await strategy.process_frame(BotStartedSpeakingFrame())
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(len(starts), 1)

    async def test_bot_idle_never_consults_the_gate(self):
        """Deferral is only ever legitimate while the bot has something to lose."""
        gate = FakeBargeInGate(far=True)
        strategy, starts = self._build(gate)

        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(len(starts), 1)
        self.assertEqual(gate.calls, 0)

    async def test_defers_when_the_gate_reports_far_while_the_bot_speaks(self):
        gate = FakeBargeInGate(far=True)
        strategy, starts = self._build(gate)

        await strategy.process_frame(BotStartedSpeakingFrame())
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        # CONTINUE, not STOP: the controller must be free to keep walking the
        # strategy list. See TestDeferredBargeInReachesMinWords for why.
        self.assertEqual(result, ProcessFrameResult.CONTINUE)
        self.assertEqual(starts, [])
        self.assertIsNotNone(strategy._deferred_at)

    async def test_gate_that_raises_fails_open(self):
        gate = FakeBargeInGate(far=True, raises=True)
        strategy, starts = self._build(gate)

        await strategy.process_frame(BotStartedSpeakingFrame())
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(len(starts), 1)

    async def test_deadline_fails_open_when_the_gate_no_longer_reports_far(self):
        """The deadline re-consults; an unsure meter always yields the floor."""
        gate = FakeBargeInGate(far=True)
        strategy, starts = self._build(gate)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(starts, [])

        gate.far = False
        await asyncio.sleep(PAST_DEADLINE)
        result = await strategy.process_frame(audio_tick())

        self.assertEqual(len(starts), 1)
        # The turn is open now, but the frame that happened to tick the clock is
        # not ours to consume.
        self.assertEqual(result, ProcessFrameResult.CONTINUE)

    async def test_deadline_holds_for_the_utterance_and_rearms_on_the_next_one(self):
        """A still-far verdict at the deadline must not degenerate into a metronome.

        Firing unconditionally at the deadline would leave a conversation
        happening next to the caller interrupting the bot every ``defer_max_secs``
        instead of once, which is no fix at all.
        """
        gate = FakeBargeInGate(far=True)
        strategy, starts = self._build(gate)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(PAST_DEADLINE)
        await strategy.process_frame(audio_tick())

        self.assertEqual(starts, [])
        self.assertTrue(strategy._holding_utterance)

        # A fresh far onset during the hold is not even deferred, so no later
        # deadline can release it either.
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(PAST_DEADLINE)
        await strategy.process_frame(audio_tick())
        self.assertEqual(starts, [])

        # The hold is scoped to one bot utterance: it ends with the utterance
        # and does not invent a turn on the way out.
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await strategy.process_frame(BotStoppedSpeakingFrame())
        self.assertEqual(starts, [])
        self.assertFalse(strategy._holding_utterance)

        # ...and the next utterance is protected from scratch, by a deferral
        # rather than by the stale hold.
        await strategy.process_frame(BotStartedSpeakingFrame())
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(result, ProcessFrameResult.CONTINUE)
        self.assertEqual(starts, [])
        self.assertIsNotNone(strategy._deferred_at)

    async def test_near_onset_takes_the_floor_during_a_hold(self):
        """The caller speaking up mid-hold interrupts immediately."""
        gate = FakeBargeInGate(far=True)
        strategy, starts = self._build(gate)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(PAST_DEADLINE)
        await strategy.process_frame(audio_tick())
        self.assertTrue(strategy._holding_utterance)

        gate.far = False
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(len(starts), 1)

    async def test_reset_preserves_bot_speaking(self):
        """``reset()`` clears the pending deferral and nothing else.

        The controller resets *every* start strategy from inside
        ``_trigger_user_turn_start``, so clearing bot-speech state here would
        make the strategy believe the bot went quiet the moment any turn opens
        — the same latent bug that collapses MinWords' own gate to
        ``min_words = 1``.
        """
        gate = FakeBargeInGate(far=True)
        strategy, starts = self._build(gate, defer_max_secs=NEVER)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNotNone(strategy._deferred_at)

        await strategy.reset()

        self.assertTrue(strategy._bot_speaking)
        self.assertIsNone(strategy._deferred_at)

        # And functionally: it still defers, which it could only do if it still
        # believed the bot was speaking.
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(result, ProcessFrameResult.CONTINUE)
        self.assertEqual(starts, [])

    async def test_vad_stop_drops_a_pending_deferral(self):
        """A burst that ends before its deadline must not open a turn afterwards."""
        gate = FakeBargeInGate(far=True)
        strategy, starts = self._build(gate)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Even with the gate now failing open, there is nothing left to release.
        gate.far = False
        await asyncio.sleep(PAST_DEADLINE)
        await strategy.process_frame(audio_tick())

        self.assertEqual(starts, [])

    async def test_bot_stopping_releases_a_pending_deferral(self):
        """Delaying a barge-in is allowed; losing the user turn is not."""
        gate = FakeBargeInGate(far=True)
        strategy, starts = self._build(gate, defer_max_secs=NEVER)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(starts, [])

        result = await strategy.process_frame(BotStoppedSpeakingFrame())

        self.assertEqual(len(starts), 1)
        # CONTINUE even though we triggered: BotStoppedSpeakingFrame is
        # load-bearing for the strategies behind us — MinWords lowers its word
        # threshold on it — and STOP would leave them thinking the bot is still
        # talking.
        self.assertEqual(result, ProcessFrameResult.CONTINUE)


class TestDeferredBargeInReachesMinWords(unittest.IsolatedAsyncioTestCase):
    """A deferred VAD onset must leave the turn *open* for later strategies.

    This is the whole reason the deferral returns CONTINUE without calling
    ``trigger_user_turn_started()``. Triggering with interruptions disabled —
    the shape of an earlier attempt at this feature — still flips the
    controller's ``_user_turn`` to True, after which ``_trigger_user_turn_start``
    refuses every later start for that turn and MinWords can never open it.
    Barge-in becomes impossible rather than delayed.
    """

    async def asyncSetUp(self):
        self.task_manager = TaskManager()

    async def test_min_words_can_still_open_a_deferred_turn(self):
        gate = FakeBargeInGate(far=True)
        vad = VADUserTurnStartStrategy(barge_in_gate=gate, defer_max_secs=NEVER)
        min_words = MinWordsUserTurnStartStrategy(min_words=2, use_interim=True)

        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[vad, min_words],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=NEVER)],
            )
        )
        await controller.setup(self.task_manager)

        starts = []

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            starts.append((strategy, params))

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(starts, [])
        self.assertFalse(controller.has_active_user_turn)

        # min_words=2 is still in force, which additionally proves the VAD
        # strategy returned CONTINUE on BotStartedSpeakingFrame: MinWords only
        # raises its threshold on a frame it actually receives.
        await controller.process_frame(TranscriptionFrame(text="שלום", user_id="", timestamp="now"))
        self.assertEqual(starts, [])
        self.assertFalse(controller.has_active_user_turn)

        await controller.process_frame(
            TranscriptionFrame(text="שלום רגע", user_id="", timestamp="now")
        )

        # The assertion this whole class exists for: `_user_turn` was still
        # False, so MinWords could open the turn and broadcast the interruption.
        self.assertEqual(len(starts), 1)
        strategy, params = starts[0]
        self.assertIs(strategy, min_words)
        self.assertTrue(params.enable_interruptions)
        self.assertTrue(controller.has_active_user_turn)

        await controller.cleanup()


class TestTranscriptionUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_transcription_strategy(self):
        strategy = TranscriptionUserTurnStartStrategy()

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="", timestamp="now"))
        self.assertTrue(should_start)


class TestExternalUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_external_strategy(self):
        strategy = ExternalUserTurnStartStrategy()

        should_start = None

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(UserStartedSpeakingFrame())
        self.assertTrue(should_start)


class TestMinWordsResetPreservesBotState(unittest.IsolatedAsyncioTestCase):
    """``reset()`` must not collapse the word gate mid-utterance.

    The controller resets every start strategy whenever any turn opens, and no
    new ``BotStartedSpeakingFrame`` arrives mid-utterance — so a reset that
    cleared ``_bot_speaking`` silently turned ``min_words=N`` into
    ``min_words=1`` for the rest of the bot's utterance.
    """

    async def test_reset_preserves_bot_speaking(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=3)
        starts = []

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            starts.append(params)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.reset()

        self.assertTrue(strategy._bot_speaking)

        # One word while the bot still speaks must NOT open the turn.
        await strategy.process_frame(
            InterimTranscriptionFrame(text="כן", user_id="u", timestamp="")
        )
        self.assertEqual(starts, [])

        # Three words still do.
        await strategy.process_frame(
            InterimTranscriptionFrame(text="רגע יש לי", user_id="u", timestamp="")
        )
        self.assertEqual(len(starts), 1)

    async def test_reset_after_bot_stopped_keeps_single_word_start(self):
        strategy = MinWordsUserTurnStartStrategy(min_words=3)
        starts = []

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            starts.append(params)

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(BotStoppedSpeakingFrame())
        await strategy.reset()

        await strategy.process_frame(
            TranscriptionFrame(text="כן", user_id="u", timestamp="")
        )
        self.assertEqual(len(starts), 1)


DEBOUNCE_SECS = 0.05
PAST_DEBOUNCE = DEBOUNCE_SECS + 0.05


class TestBargeInDebounce(unittest.IsolatedAsyncioTestCase):
    """``extra_voiced_secs``: a barge-in must sustain before it is honored."""

    def _build(self, extra_voiced_secs: float = DEBOUNCE_SECS, **kwargs):
        strategy = VADUserTurnStartStrategy(
            extra_voiced_secs=extra_voiced_secs, **kwargs
        )
        starts = []

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            starts.append(params)

        return strategy, starts

    async def test_zero_hold_is_unchanged(self):
        strategy, starts = self._build(extra_voiced_secs=0.0)

        await strategy.process_frame(BotStartedSpeakingFrame())
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(len(starts), 1)

    async def test_onset_is_held_not_honored_immediately(self):
        strategy, starts = self._build()

        await strategy.process_frame(BotStartedSpeakingFrame())
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(result, ProcessFrameResult.CONTINUE)
        self.assertEqual(starts, [])
        self.assertIsNotNone(strategy._debounce_at)

    async def test_sustained_onset_is_honored_after_the_hold(self):
        strategy, starts = self._build()

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(PAST_DEBOUNCE)
        result = await strategy.process_frame(audio_tick())

        self.assertEqual(len(starts), 1)
        # The frame that ticked the clock is not ours to consume.
        self.assertEqual(result, ProcessFrameResult.CONTINUE)

    async def test_short_burst_never_interrupts(self):
        """The filter working: cough/backchannel ends before the hold elapses."""
        strategy, starts = self._build()

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(PAST_DEBOUNCE)
        await strategy.process_frame(audio_tick())

        self.assertEqual(starts, [])
        self.assertEqual(strategy.false_starts, 1)

    async def test_bot_stopping_releases_the_hold_as_a_turn(self):
        """Delay a barge-in, never lose the user turn."""
        strategy, starts = self._build()

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(BotStoppedSpeakingFrame())

        self.assertEqual(len(starts), 1)
        self.assertIsNone(strategy._debounce_at)

    async def test_bot_idle_onset_is_instant(self):
        """The hold only applies to barge-ins, never to normal turn-taking."""
        strategy, starts = self._build()

        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(len(starts), 1)

    async def test_reset_clears_a_pending_hold_but_keeps_bot_state(self):
        strategy, starts = self._build()

        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.reset()

        self.assertIsNone(strategy._debounce_at)
        self.assertTrue(strategy._bot_speaking)


class TestConfirmWordsMode(unittest.IsolatedAsyncioTestCase):
    """``confirm_words``: while the bot speaks, only words open the turn."""

    def _build(self):
        strategy = VADUserTurnStartStrategy(confirm_words=True)
        starts = []

        @strategy.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            starts.append(params)

        return strategy, starts

    async def test_vad_never_opens_the_turn_while_bot_speaks(self):
        strategy, starts = self._build()

        await strategy.process_frame(BotStartedSpeakingFrame())
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(result, ProcessFrameResult.CONTINUE)
        self.assertEqual(starts, [])
        # Nothing pending either: words own the decision entirely.
        self.assertIsNone(strategy._debounce_at)
        self.assertIsNone(strategy._deferred_at)

    async def test_vad_still_opens_the_turn_while_bot_is_silent(self):
        strategy, starts = self._build()

        result = await strategy.process_frame(VADUserStartedSpeakingFrame())

        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(len(starts), 1)

    async def test_min_words_behind_it_still_interrupts(self):
        """The controller walks past the VAD strategy to MinWords, which fires."""
        vad, vad_starts = self._build()
        min_words = MinWordsUserTurnStartStrategy(min_words=2)
        mw_starts = []

        @min_words.event_handler("on_user_turn_started")
        async def on_user_turn_started(strategy, params):
            mw_starts.append(params)

        for s in (vad, min_words):
            await s.process_frame(BotStartedSpeakingFrame())

        self.assertEqual(
            await vad.process_frame(VADUserStartedSpeakingFrame()),
            ProcessFrameResult.CONTINUE,
        )
        await min_words.process_frame(
            InterimTranscriptionFrame(text="רגע רגע", user_id="u", timestamp="")
        )
        self.assertEqual(vad_starts, [])
        self.assertEqual(len(mw_starts), 1)


if __name__ == "__main__":
    unittest.main()
