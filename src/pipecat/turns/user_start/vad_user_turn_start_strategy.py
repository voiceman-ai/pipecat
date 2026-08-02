#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy based on VAD events."""

import time
from collections.abc import Callable

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class VADUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy based on VAD (Voice Activity Detection).

    This strategy assumes the user turn starts as soon as a VAD frame indicates
    that the user has started speaking.

    An optional ``barge_in_gate`` lets an external observer (e.g. a near-field
    level meter) veto the *immediacy* of that decision while the bot is
    speaking, so that a voice coming from across the room does not cut the bot
    off. The veto only ever delays or drops the VAD-driven turn start; it never
    mutes audio and never blocks another start strategy from opening the turn.

    """

    def __init__(
        self,
        *,
        barge_in_gate: Callable[[], bool] | None = None,
        defer_max_secs: float = 1.2,
        **kwargs,
    ):
        """Initialize the VAD user turn start strategy.

        Args:
            barge_in_gate: Optional callable returning True when the *current*
                speech onset is confidently far-field, i.e. not the caller. When
                it returns True while the bot is speaking, the VAD-driven turn
                start is deferred instead of fired. ``None`` (the default)
                disables the whole mechanism and the strategy behaves exactly as
                it always has.
            defer_max_secs: How long a deferral may stand before the gate is
                consulted again. Not a deadline after which the turn starts
                unconditionally — see :meth:`_resolve_deferral`.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._barge_in_gate = barge_in_gate
        self._defer_max_secs = defer_max_secs

        # Bot speech state. Tracked here (rather than read from the controller)
        # because deferral is only ever legitimate while the bot has something
        # to lose by being interrupted.
        self._bot_speaking = False

        # Monotonic timestamp of the pending deferral, or None when no VAD
        # onset is currently being held back.
        self._deferred_at: float | None = None

        # Set once a deferral has survived its re-consultation: the gate is warm
        # and still says "far", so this bot utterance is protected to its end.
        # Cleared when the bot stops, hence re-armed on the next utterance.
        self._holding_utterance = False

    async def reset(self):
        """Reset the strategy to its initial state.

        Only the pending deferral is cleared. ``_bot_speaking`` and the
        per-utterance hold deliberately survive a reset: the controller resets
        *every* start strategy from inside ``_trigger_user_turn_start``, so
        clearing bot-speech state here would make the strategy believe the bot
        went quiet the moment any turn opens — the same latent bug that
        collapses :class:`MinWordsUserTurnStartStrategy`'s own gate to
        ``min_words = 1``. Only Bot{Started,Stopped}SpeakingFrame may move that
        state.
        """
        await super().reset()
        self._deferred_at = None

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process an incoming frame to detect user turn start.

        Args:
            frame: The frame to be analyzed.

        Returns:
            STOP if the user started speaking, CONTINUE otherwise.
        """
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._holding_utterance = False
            return ProcessFrameResult.CONTINUE

        if isinstance(frame, BotStoppedSpeakingFrame):
            return await self._handle_bot_stopped_speaking()

        if isinstance(frame, VADUserStartedSpeakingFrame):
            return await self._handle_vad_user_started_speaking()

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # The speech we were holding back is over, so there is nothing left
            # to barge in with. Dropping the deferral here matters: without it,
            # a short burst of room noise could reach its re-consultation after
            # the span ended, find the gate cold ("not far" by fail-open) and
            # open a user turn out of nowhere, a full second after silence.
            self._deferred_at = None
            return ProcessFrameResult.CONTINUE

        # Any other frame is just a clock tick. The user aggregator forwards
        # every frame it sees to the turn controller, including the 20ms input
        # audio frames, so a pending deferral is re-examined promptly without
        # this strategy having to own a timer task.
        if self._deferred_at is not None:
            return await self._maybe_resolve_deferral()

        return ProcessFrameResult.CONTINUE

    async def _handle_vad_user_started_speaking(self) -> ProcessFrameResult:
        """Start the user turn, unless the onset is vetoed as far-field."""
        if self._bot_speaking and self._gate_says_far():
            # CONTINUE *without* triggering is the entire design. Triggering
            # with interruptions disabled would still flip the controller's
            # `_user_turn` to True, and `_trigger_user_turn_start` then refuses
            # every later start for this turn — barge-in would become impossible
            # rather than delayed. Returning CONTINUE leaves `_user_turn` False,
            # so the controller falls through to the remaining start strategies
            # (MinWords), which can still open the turn on transcribed words and
            # broadcast the interruption.
            if self._deferred_at is None and not self._holding_utterance:
                self._deferred_at = time.monotonic()
                logger.debug(
                    f"{self} deferring barge-in: onset reported far-field while bot is speaking"
                )
            return ProcessFrameResult.CONTINUE

        self._deferred_at = None
        await self.trigger_user_turn_started()
        return ProcessFrameResult.STOP

    async def _handle_bot_stopped_speaking(self) -> ProcessFrameResult:
        """Release any deferral once the utterance it protected has finished."""
        self._bot_speaking = False
        self._holding_utterance = False

        if self._deferred_at is not None:
            # The bot finished on its own, so the deferral did its job. Open the
            # turn now rather than dropping it: this strategy is only allowed to
            # delay a barge-in, never to lose a user turn that today's code
            # would have created.
            self._deferred_at = None
            await self.trigger_user_turn_started()

        # Always CONTINUE, even after triggering. BotStoppedSpeakingFrame is
        # load-bearing for the start strategies behind us — MinWords lowers its
        # word threshold on it — and swallowing it with STOP would leave them
        # believing the bot is still talking.
        return ProcessFrameResult.CONTINUE

    async def _maybe_resolve_deferral(self) -> ProcessFrameResult:
        """Re-consult the gate once a deferral has stood for defer_max_secs."""
        deferred_at = self._deferred_at
        if deferred_at is None or (time.monotonic() - deferred_at) < self._defer_max_secs:
            return ProcessFrameResult.CONTINUE

        self._deferred_at = None

        if self._gate_says_far():
            # Still confidently far-field. Hold for the rest of this utterance.
            # Firing unconditionally at the deadline would defeat the feature:
            # a conversation happening next to the caller lasts far longer than
            # defer_max_secs, so the bot would simply be cut off every 1.2s
            # instead of once. The cost is bounded — one bot utterance — and a
            # fresh onset is still evaluated on its own merits, so the real
            # caller speaking up mid-hold can still take the floor.
            self._holding_utterance = True
            logger.info(
                f"{self} holding barge-in for the remainder of this bot utterance: "
                "onset still reported far-field at re-consultation"
            )
            return ProcessFrameResult.CONTINUE

        # The gate is no longer sure it was somebody else. Fail open and let the
        # turn start, at most defer_max_secs late.
        logger.debug(f"{self} releasing deferred barge-in: onset no longer reported far-field")
        await self.trigger_user_turn_started()

        # CONTINUE rather than STOP: the turn is already open, and this frame is
        # not ours to consume — it is whatever ordinary frame happened to tick
        # the clock.
        return ProcessFrameResult.CONTINUE

    def _gate_says_far(self) -> bool:
        """Ask the barge-in gate whether the current onset is far-field.

        Fails open on every path. No gate wired, or a gate that raises, means
        "not far", which is exactly today's behaviour: interrupt immediately.
        """
        if self._barge_in_gate is None:
            return False

        try:
            return bool(self._barge_in_gate())
        except Exception as e:
            logger.warning(f"{self} barge-in gate raised, failing open: {e}")
            return False
