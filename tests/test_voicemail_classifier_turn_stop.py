"""The classifier branch must finalize a turn whose transcript preceded the adopted start.

Regression guard for a production stall (2026-08-17, wf86): the caller answered the
opening question with a single short word the VAD never fired on. The conversation
aggregator opened the turn from the final transcript itself, so the transcript reached
the (upstream) classifier branch before the ``UserStartedSpeakingFrame`` it then adopted.
``ExternalUserTurnStopStrategy`` resets its text on every turn start, the classifier's
turn could never finalize, the classifier never ran, and ``LLMGate`` held the main LLM
until the controller's turn-stop timeout — ~5s of dead air on the first answer.

``ClassifierUserTurnStopStrategy`` keeps transcript state across the start callback.
"""

import asyncio
import unittest

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.extensions.voicemail.voicemail_detector import (
    ClassifierUserTurnStopStrategy,
    VoicemailDetector,
)
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import TaskManager


class _MockLLMService(LLMService):
    """Minimal concrete LLMService (mirrors tests/test_voicemail_classifier_prompt.py)."""

    def __init__(self, **kwargs):
        settings = LLMSettings(
            model="test-model",
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            extra=None,
        )
        super().__init__(settings=settings, **kwargs)

    def create_adapter(self) -> BaseLLMAdapter:
        return BaseLLMAdapter()


def _final(text: str = "כן.") -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="now")


def _interim(text: str = "כן") -> InterimTranscriptionFrame:
    return InterimTranscriptionFrame(text=text, user_id="", timestamp="now")


class _Recorder:
    """Collects every ``on_user_turn_stopped`` emission from a strategy."""

    def __init__(self, strategy):
        self.stops = []

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            self.stops.append(params)

    @property
    def last(self):
        return self.stops[-1] if self.stops else None


async def _adopted_turn(strategy, *, before_start=(), after_start=(), stop=True):
    """Drive one adopted turn: frames before the start, the start, frames after, the stop.

    Mirrors the controller's calls: ``handle_user_turn_started`` runs on the
    stop strategy when the turn opens, then the strategy sees the adopted
    ``UserStartedSpeakingFrame`` itself.
    """
    for frame in before_start:
        await strategy.process_frame(frame)
    await strategy.handle_user_turn_started()
    await strategy.process_frame(UserStartedSpeakingFrame())
    for frame in after_start:
        await strategy.process_frame(frame)
    if stop:
        await strategy.process_frame(UserStoppedSpeakingFrame())


class TestClassifierUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()

    async def _strategy(self, **kwargs) -> ClassifierUserTurnStopStrategy:
        strategy = ClassifierUserTurnStopStrategy(**kwargs)
        await strategy.setup(self.task_manager)
        self.addAsyncCleanup(strategy.cleanup)
        return strategy

    async def test_finalizes_when_final_precedes_adopted_start(self):
        strategy = await self._strategy()
        rec = _Recorder(strategy)
        await _adopted_turn(strategy, before_start=[_final()])
        self.assertIsNotNone(rec.last)
        # Adopted from a real turn frame: emits no second UserStoppedSpeakingFrame.
        self.assertFalse(rec.last.enable_user_speaking_frames)

    async def test_base_strategy_drops_pre_start_final(self):
        """The base class behavior being guarded against: the pre-start final is lost."""
        strategy = ExternalUserTurnStopStrategy()
        await strategy.setup(self.task_manager)
        self.addAsyncCleanup(strategy.cleanup)
        rec = _Recorder(strategy)
        await _adopted_turn(strategy, before_start=[_final()])
        self.assertIsNone(rec.last)

    async def test_final_after_adopted_start_still_finalizes(self):
        """The ordinary (VAD-first) ordering is unchanged."""
        strategy = await self._strategy()
        rec = _Recorder(strategy)
        await _adopted_turn(strategy, after_start=[_final()])
        self.assertIsNotNone(rec.last)
        self.assertFalse(rec.last.enable_user_speaking_frames)

    async def test_pre_start_interim_keeps_waiting_for_its_final(self):
        strategy = await self._strategy()
        rec = _Recorder(strategy)
        await _adopted_turn(strategy, before_start=[_interim()])
        # Interim seen, no final: not finalized on the stop signal.
        self.assertIsNone(rec.last)
        await strategy.process_frame(_final())
        await strategy.process_frame(UserStoppedSpeakingFrame())
        self.assertIsNotNone(rec.last)

    async def test_pre_start_interim_then_pre_start_final(self):
        strategy = await self._strategy()
        rec = _Recorder(strategy)
        await _adopted_turn(strategy, before_start=[_interim(), _final()])
        self.assertIsNotNone(rec.last)

    async def test_finals_on_both_sides_of_the_start_finalize_once(self):
        strategy = await self._strategy()
        rec = _Recorder(strategy)
        await _adopted_turn(strategy, before_start=[_final("כן,")], after_start=[_final("בטח.")])
        self.assertEqual(len(rec.stops), 1)

    async def test_stop_before_final_finalizes_from_the_transcript_timer(self):
        """Adopted stop first, final after: the strategy's own timer finalizes."""
        strategy = await self._strategy(timeout=0.05)
        rec = _Recorder(strategy)
        await _adopted_turn(strategy)
        self.assertIsNone(rec.last)
        await strategy.process_frame(_final())
        await asyncio.sleep(0.15)
        self.assertIsNotNone(rec.last)
        self.assertFalse(rec.last.enable_user_speaking_frames)

    async def test_pre_start_final_never_finalizes_before_the_turn_opens(self):
        """Kept text is inert while no turn is open — the timer path stays quiet."""
        strategy = await self._strategy(timeout=0.05)
        rec = _Recorder(strategy)
        await strategy.handle_user_turn_stopped()
        await strategy.process_frame(_final())
        await asyncio.sleep(0.3)
        self.assertEqual(rec.stops, [])
        # Once the turn opens and the adopted stop lands, that text finalizes it.
        await _adopted_turn(strategy)
        self.assertIsNotNone(rec.last)

    async def test_pre_start_final_does_not_finalize_while_user_still_speaking(self):
        """Adopted start sets speaking state in the same frame: no premature timer stop."""
        strategy = await self._strategy(timeout=0.05)
        rec = _Recorder(strategy)
        await _adopted_turn(strategy, before_start=[_final()], stop=False)
        await asyncio.sleep(0.3)
        self.assertEqual(rec.stops, [])
        await strategy.process_frame(UserStoppedSpeakingFrame())
        self.assertIsNotNone(rec.last)

    async def test_text_clears_at_turn_end_not_at_turn_start(self):
        strategy = await self._strategy()
        await strategy.process_frame(_final())
        await strategy.handle_user_turn_started()
        self.assertEqual(strategy._text, "כן.")
        await strategy.handle_user_turn_stopped()
        self.assertEqual(strategy._text, "")

    async def test_late_final_from_a_finished_turn_does_not_bleed_into_the_next(self):
        """A final that lands after a turn was finalized clears with that turn's end."""
        strategy = await self._strategy()
        rec = _Recorder(strategy)
        await _adopted_turn(strategy, after_start=[_final("שלום.")])
        self.assertEqual(len(rec.stops), 1)
        # Turn end callback (the controller runs it on finalization) resets text.
        await strategy.handle_user_turn_stopped()
        self.assertEqual(strategy._text, "")
        # A brand-new turn with no transcript does not finalize on stale text.
        await _adopted_turn(strategy)
        self.assertEqual(len(rec.stops), 1)

    async def test_proposed_path_is_unchanged(self):
        """A service-proposed turn (Flux-style) still resolves and emits as the base does."""
        strategy = await self._strategy()
        rec = _Recorder(strategy)
        await strategy.handle_user_turn_started()
        await strategy.process_frame(ProposedUserStartedSpeakingFrame())
        await strategy.process_frame(_final("hello"))
        await strategy.process_frame(ProposedUserStoppedSpeakingFrame())
        self.assertIsNotNone(rec.last)
        self.assertTrue(rec.last.enable_user_speaking_frames)

    def test_detector_wires_the_classifier_stop_strategy(self):
        detector = VoicemailDetector(llm=_MockLLMService())
        controller = detector._context_aggregator.user()._user_turn_controller
        stop = controller._user_turn_strategies.stop
        self.assertEqual(len(stop), 1)
        self.assertIsInstance(stop[0], ClassifierUserTurnStopStrategy)
        # Start is still adopted from the conversation aggregator, not detected here.
        start = controller._user_turn_strategies.start
        self.assertEqual([type(s).__name__ for s in start], ["ExternalUserTurnStartStrategy"])
