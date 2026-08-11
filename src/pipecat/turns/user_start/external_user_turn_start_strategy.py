#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn start strategy driven by another component in the pipeline."""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    ProposedUserStartedSpeakingFrame,
    UserStartedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class ExternalUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy driven by another component in the pipeline.

    Rather than detecting the start of a turn itself, this strategy takes its
    cue from another processor. It understands two signals, which differ in how
    much the emitter has already done:

    - :class:`~pipecat.frames.frames.ProposedUserStartedSpeakingFrame` — a
      service with its own turn detection proposing a turn boundary. This
      strategy makes the decision, emitting the
      :class:`~pipecat.frames.frames.UserStartedSpeakingFrame` and broadcasting
      the interruption itself. Subclass it to adjust when, or whether, a
      proposal opens a turn.

    - :class:`~pipecat.frames.frames.UserStartedSpeakingFrame` — the turn was
      already decided and announced elsewhere, typically by a shared
      :class:`~pipecat.turns.user_turn_processor.UserTurnProcessor` fanning turns
      out to several aggregators. This strategy adopts that decision and emits
      nothing, so the turn isn't announced twice.

    A service that emits turn frames directly lands on the adopt path and keeps
    working, but it owns the interruption logic itself. Emitting proposals
    instead hands that job back to the pipeline.
    """

    def __init__(self, *, enable_interruptions: bool = True, **kwargs):
        """Initialize the external user turn start strategy.

        Args:
            enable_interruptions: If True, broadcast an interruption when a
                proposal opens a turn. Ignored on the adopt path, where the
                emitter has already broadcast one.

                The default is True because on the proposal path nothing else
                broadcasts the interruption — the proposing emitter has
                deliberately handed that job to this strategy. Several
                turn-detecting services take that path (Deepgram Flux, Soniox
                and Gladia among them): they emit
                :class:`~pipecat.frames.frames.ProposedUserStartedSpeakingFrame`
                and never interrupt themselves, so ``False`` there silently
                kills barge-in. Deepgram Flux is the one the VoiceMan api wires
                this strategy for by hand.

                It has no effect for the realtime services this fork drives
                (OpenAI Realtime, Grok/xAI Realtime): both broadcast the real
                :class:`~pipecat.frames.frames.UserStartedSpeakingFrame` plus
                their own interruption, which puts them on the adopt path where
                this flag is ignored.

            **kwargs: Additional keyword arguments.
        """
        super().__init__(enable_interruptions=enable_interruptions, **kwargs)

    @property
    def resolves_proposed_turn_start_frames(self) -> bool:
        """Whether this strategy resolves proposals into turn starts."""
        return True

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process an incoming frame to detect user turn start.

        Args:
            frame: The frame to be analyzed.

        Returns:
            STOP if the turn started, CONTINUE otherwise.
        """
        if isinstance(frame, ProposedUserStartedSpeakingFrame):
            logger.debug(f"{self}: resolving a proposed user turn start")
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP

        if isinstance(frame, UserStartedSpeakingFrame):
            # Already announced elsewhere — adopt the decision without repeating it.
            logger.debug(f"{self}: adopting a user turn start decided elsewhere")
            await self.trigger_user_turn_started(
                enable_interruptions=False, enable_user_speaking_frames=False
            )
            return ProcessFrameResult.STOP

        return ProcessFrameResult.CONTINUE
