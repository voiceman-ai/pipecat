#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voicemail detection module for Pipecat.

This module provides voicemail detection capabilities using parallel pipeline
processing to classify incoming calls as either voicemail messages or live
conversations. It's specifically designed for outbound calling scenarios where
a bot needs to determine if a human answered or if the call went to voicemail.

Note:
    The voicemail module is optimized for text LLMs only.
"""

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    StopFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.llm_service import LLMService
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.sync.base_notifier import BaseNotifier
from pipecat.utils.sync.event_notifier import EventNotifier


class ClassifierUserTurnStopStrategy(ExternalUserTurnStopStrategy):
    """Turn stop strategy for the classifier branch's aggregator.

    The classifier branch sits upstream of the conversation aggregator and
    adopts that aggregator's ``UserStartedSpeakingFrame`` /
    ``UserStoppedSpeakingFrame`` broadcasts as its own turn boundaries. When
    the conversation aggregator opens a turn *from a transcript* (a
    transcript-driven start strategy, no VAD onset), that transcript passes
    through this branch before the start it triggers arrives here. Transcript
    state therefore survives the turn-start callback — text seen since the
    previous turn ended belongs to the turn now starting — and is cleared
    only when the turn ends. Signal state resets on start as in the base
    class.

    Scoped to the classifier rather than changed in
    :class:`~pipecat.turns.user_stop.ExternalUserTurnStopStrategy`: an
    aggregator that also opens turns from VAD, ahead of the external start
    signal, would finalize a new turn on stale text during that window. The
    classifier's aggregator opens turns only on start signals, which set the
    speaking state in the same frame, so no such window exists here.
    """

    async def handle_user_turn_started(self):
        """Ready the strategy for the turn now starting, keeping transcript state."""
        text = self._text
        seen_interim_results = self._seen_interim_results
        await super().handle_user_turn_started()
        self._text = text
        self._seen_interim_results = seen_interim_results


class NotifierGate(FrameProcessor):
    """Base gate processor that controls frame flow based on notifier signals.

    This base class provides common gate functionality for processors that need to
    start open and close permanently when a notifier signals. Subclasses define
    which frames are allowed through when the gate is closed.

    The gate starts open to allow initial processing and closes permanently once
    the notifier signals. This ensures controlled frame flow based on external
    decisions or events.
    """

    def __init__(self, notifier: BaseNotifier, worker_name: str = "gate"):
        """Initialize the notifier gate.

        Args:
            notifier: Notifier that signals when the gate should close.
            worker_name: Name for the notification waiting task (for debugging).
        """
        super().__init__()
        self._notifier = notifier
        self._worker_name = worker_name
        self._gate_opened = True
        self._gate_task: asyncio.Task | None = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._gate_task = self.create_task(self._wait_for_notification())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._gate_task:
            await self.cancel_task(self._gate_task)
            self._gate_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and control gate state based on notifier signals.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Gate logic: open gate allows all frames, closed gate filters frames
        if self._gate_opened:
            await self.push_frame(frame, direction)
        elif isinstance(
            frame,
            (SystemFrame, EndFrame, StopFrame),
        ):
            await self.push_frame(frame, direction)

    async def _wait_for_notification(self):
        """Wait for notifier signal and close the gate.

        This method blocks until the notifier signals, then closes the gate
        permanently to change frame filtering behavior.
        """
        await self._notifier.wait()

        if self._gate_opened:
            self._gate_opened = False


class ClassifierGate(NotifierGate):
    """Gate processor that controls frame flow based on classification decisions.

    Inherits from NotifierGate and starts open to allow initial classification
    processing. Closes permanently once a classification decision is made
    (CONVERSATION or VOICEMAIL). This ensures the classifier only runs until a
    definitive decision is reached, preventing unnecessary LLM calls and maintaining
    system efficiency.

    When closed, only allows system frames and user speaking frames to continue.
    Speaking frames are needed for voicemail timing control, but not for conversation.
    """

    def __init__(self, gate_notifier: BaseNotifier, conversation_notifier: BaseNotifier):
        """Initialize the classifier gate.

        Args:
            gate_notifier: Notifier that signals when a classification decision has
                been made and the gate should close.
            conversation_notifier: Notifier that signals when conversation is detected.
        """
        super().__init__(gate_notifier, worker_name="classifier_gate")
        self._conversation_notifier = conversation_notifier
        self._conversation_detected = False
        self._conversation_task: asyncio.Task | None = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._conversation_task = self.create_task(self._wait_for_conversation())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._conversation_task:
            await self.cancel_task(self._conversation_task)
            self._conversation_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and control gate state based on notifier signals.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await FrameProcessor.process_frame(self, frame, direction)

        # Never allow InterruptionFrame into the classifier branch so the
        # classification LLM completes without being cancelled.
        if isinstance(frame, InterruptionFrame):
            return

        # Gate logic: open gate allows all frames, closed gate filters frames
        if self._gate_opened:
            await self.push_frame(frame, direction)
        elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            # Only allow speaking frames if conversation was NOT detected (i.e., voicemail case)
            # This prevents the UserContextAggregator from issuing a warning about no aggregation
            # to push.
            if not self._conversation_detected:
                await self.push_frame(frame, direction)
        elif isinstance(frame, (SystemFrame, EndFrame, StopFrame)):
            # Always allow system frames through
            # This includes the UserStartedSpeakingFrame and UserStoppedSpeakingFrame
            # which are used to detect voicemail timing.
            await self.push_frame(frame, direction)

    async def _wait_for_conversation(self):
        """Wait for conversation detection notification and mark conversation detected."""
        await self._conversation_notifier.wait()
        self._conversation_detected = True


class ConversationGate(NotifierGate):
    """Gate processor that blocks conversation flow when voicemail is detected.

    Inherits from NotifierGate and starts open to allow normal conversation
    processing. Closes permanently when voicemail is detected to prevent the
    main conversation LLM from processing additional input after voicemail
    classification.

    When closed, only allows system frames and user speaking frames to continue.
    """

    def __init__(self, voicemail_notifier: BaseNotifier):
        """Initialize the conversation gate.

        Args:
            voicemail_notifier: Notifier that signals when voicemail has been
                detected and the conversation should be blocked.
        """
        super().__init__(voicemail_notifier, worker_name="conversation_gate")


class ClassifierUpstreamGate(FrameProcessor):
    """Gate that blocks upstream frames to classifier branch when conversation is detected.

    This gate sits at the bottom of the classifier branch and blocks frames flowing
    upstream after conversation is detected. This prevents system frames like
    FunctionCallsStartedFrame from reaching the classifier LLM after classification
    is complete. Downstream frames always pass through. Only essential cleanup frames
    (EndFrame, StopFrame) are allowed upstream when closed.
    """

    def __init__(self, conversation_notifier: BaseNotifier):
        """Initialize the classifier conversation gate.

        Args:
            conversation_notifier: Notifier that signals when conversation has been
                detected and upstream frames should be blocked.
        """
        super().__init__()
        self._conversation_notifier = conversation_notifier
        self._gate_open = True
        self._conversation_task: asyncio.Task | None = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._conversation_task = self.create_task(self._wait_for_conversation())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._conversation_task:
            await self.cancel_task(self._conversation_task)
            self._conversation_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and block upstream frames when conversation is detected.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Never allow InterruptionFrame into the classifier branch so the
        # classification LLM completes without being cancelled.
        if isinstance(frame, InterruptionFrame):
            return

        # Always allow downstream frames through
        if direction == FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
        # Gate upstream frames when conversation is detected
        elif self._gate_open:
            await self.push_frame(frame, direction)

    async def _wait_for_conversation(self):
        """Wait for conversation detection and close the gate."""
        await self._conversation_notifier.wait()
        self._gate_open = False


def classify_verdict(full_response: str) -> str | None:
    """Map a raw classifier LLM response to a voicemail verdict.

    The precedence is load-bearing: "CONVERSATION" is checked before
    "VOICEMAIL", so a response containing both resolves to conversation
    (hanging up on a live person is worse than listening to a recording).
    Offline evals must use this exact function rather than reimplementing
    the match so they measure what production does.

    Args:
        full_response: The complete aggregated response text from the LLM.

    Returns:
        "conversation" or "voicemail", or None when the response contains
        neither marker (e.g. the LLM was interrupted mid-response).
    """
    response = full_response.upper()
    if "CONVERSATION" in response:
        return "conversation"
    if "VOICEMAIL" in response:
        return "voicemail"
    return None


class ClassificationProcessor(FrameProcessor):
    """Processor that handles LLM classification responses and triggers events.

    This processor aggregates LLM text tokens into complete responses and analyzes
    them to determine if the call reached a voicemail system or a live person.
    It uses the LLM response frame delimiters (LLMFullResponseStartFrame and
    LLMFullResponseEndFrame) to ensure complete token aggregation regardless
    of how the LLM tokenizes the response words.

    The processor expects responses containing either "CONVERSATION" (indicating
    a human answered) or "VOICEMAIL" (indicating an automated system). Once a
    decision is made, it triggers the appropriate notifications and event handlers.

    """

    def __init__(
        self,
        *,
        gate_notifier: BaseNotifier,
        conversation_notifier: BaseNotifier,
        voicemail_notifier: BaseNotifier,
    ):
        """Initialize the voicemail processor.

        Args:
            gate_notifier: Notifier to signal the ClassifierGate about classification
                decisions so it can close and stop processing.
            conversation_notifier: Notifier to signal the TTSGate to release
                all gated TTS frames for normal conversation flow.
            voicemail_notifier: Notifier to signal the TTSGate to clear
                gated TTS frames since voicemail was detected.
        """
        super().__init__()
        self._gate_notifier = gate_notifier
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier

        # Register the conversation and voicemail detected events
        self._register_event_handler("on_conversation_detected")
        self._register_event_handler("on_voicemail_detected")

        # Aggregation state for collecting complete LLM responses
        self._processing_response = False
        self._response_buffer = ""
        self._decision_made = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle LLM classification responses.

        This method implements a state machine for aggregating LLM responses:
        1. LLMFullResponseStartFrame: Begin collecting tokens
        2. LLMTextFrame: Accumulate text tokens into buffer
        3. LLMFullResponseEndFrame: Process complete response and make decision

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # Begin aggregating a new LLM response
            self._processing_response = True
            self._response_buffer = ""

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Complete response received - make classification decision
            if self._processing_response and not self._decision_made:
                await self._process_classification(self._response_buffer.strip())
            self._processing_response = False
            self._response_buffer = ""

        elif isinstance(frame, LLMTextFrame) and self._processing_response:
            # Accumulate text tokens from the streaming LLM response
            self._response_buffer += frame.text

        else:
            # Pass all non-LLM frames through
            # Blocking LLM frames prevents interference with the downstream LLM
            await self.push_frame(frame, direction)

    async def _process_classification(self, full_response: str):
        """Process the complete LLM classification response and trigger actions.

        Analyzes the aggregated response text to determine if it contains
        "CONVERSATION" or "VOICEMAIL" and triggers the appropriate notifications
        and callbacks based on the classification result.

        Args:
            full_response: The complete aggregated response text from the LLM.
        """
        if self._decision_made:
            return

        verdict = classify_verdict(full_response)
        logger.debug(f"{self}: Classifying response: '{full_response}'")

        if verdict == "conversation":
            # Human answered - continue normal conversation flow
            self._decision_made = True
            logger.info(f"{self}: CONVERSATION detected")
            await self._gate_notifier.notify()  # Close the classifier gate
            await self._conversation_notifier.notify()  # Release buffered TTS frames
            await self._call_event_handler("on_conversation_detected")

        elif verdict == "voicemail":
            # Voicemail detected - trigger voicemail handling
            self._decision_made = True
            logger.info(f"{self}: VOICEMAIL detected")
            await self._gate_notifier.notify()  # Close the classifier gate
            await self._voicemail_notifier.notify()  # Clear buffered TTS frames

            # Interrupt the current pipeline to stop any ongoing processing
            await self.broadcast_interruption()

            # Immediately trigger the voicemail event handler
            await self._call_event_handler("on_voicemail_detected")

        else:
            # This can happen if the LLM is interrupted before completing the response
            logger.debug(f"{self}: No classification found: '{full_response}'")


class TTSGate(FrameProcessor):
    """Gates TTS frames until voicemail classification decision is made.

    This processor holds TTS output frames in a gate while the voicemail
    classification is in progress. This prevents audio from being played
    to the caller before determining if they're human or a voicemail system.

    The gate operates in two modes based on the classification result:

    - CONVERSATION: Opens the gate to release all held frames for normal dialogue
    - VOICEMAIL: Clears held frames since they're not needed for voicemail

    The gating only applies to TTS-related frames (TTSTextFrame, TTSAudioRawFrame).
    All other frames pass through immediately to maintain proper pipeline flow.
    """

    def __init__(self, conversation_notifier: BaseNotifier, voicemail_notifier: BaseNotifier):
        """Initialize the TTS gate.

        Args:
            conversation_notifier: Notifier that signals when a conversation is
                detected and gated frames should be released for playback.
            voicemail_notifier: Notifier that signals when voicemail is detected
                and gated frames should be cleared (not played).
        """
        super().__init__()
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier
        self._frame_buffer: list[tuple[Frame, FrameDirection]] = []
        self._gating_active = True
        self._conversation_task: asyncio.Task | None = None
        self._voicemail_task: asyncio.Task | None = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)

        self._conversation_task = self.create_task(self._wait_for_conversation())
        self._voicemail_task = self.create_task(self._wait_for_voicemail())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._conversation_task:
            await self.cancel_task(self._conversation_task)
            self._conversation_task = None
        if self._voicemail_task:
            await self.cancel_task(self._voicemail_task)
            self._voicemail_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle gating logic based on frame type.

        TTS frames are gated while classification is active. All other frames
        pass through immediately. The gating state is controlled by the
        classification notifications.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Core gating logic: hold TTS frames, pass everything else through
        if self._gating_active and isinstance(
            frame, (TTSStartedFrame, TTSStoppedFrame, TTSTextFrame, TTSAudioRawFrame)
        ):
            # Gate TTS frames while waiting for classification decision
            self._frame_buffer.append((frame, direction))
        else:
            # Pass through all non-TTS frames immediately
            await self.push_frame(frame, direction)

    async def _wait_for_conversation(self):
        """Wait for conversation detection notification and release gated frames.

        When a conversation is detected, all gated TTS frames are released
        in order to continue normal dialogue flow. This allows the bot to
        respond naturally to the human caller.
        """
        await self._conversation_notifier.wait()

        # Release all gated frames in original order
        self._gating_active = False
        for frame, direction in self._frame_buffer:
            await self.push_frame(frame, direction)
        self._frame_buffer.clear()

    async def _wait_for_voicemail(self):
        """Wait for voicemail detection notification and clear gated frames.

        When voicemail is detected, all gated TTS frames are discarded
        since they were intended for human conversation and are not appropriate
        for voicemail systems. The developer event handlers will handle voicemail-
        specific audio output.
        """
        await self._voicemail_notifier.wait()

        # Clear gated frames without playing them
        self._gating_active = False
        self._frame_buffer.clear()


class LLMGate(FrameProcessor):
    """Gates LLMContextFrame until voicemail classification decision is made.

    This processor buffers the most recent LLMContextFrame while voicemail
    classification is in progress, preventing the main conversation LLM from
    being triggered before determining if the call reached a human or a
    voicemail system. If multiple LLMContextFrame objects arrive while the gate
    is closed, only the latest one is kept (it carries the most up-to-date
    context).

    Based on the classification result:

    - CONVERSATION: Opens the gate and releases the buffered frame to trigger the LLM
    - VOICEMAIL: Discards the buffered frame since the call will be ended

    Only LLMContextFrame is gated. All other frames pass through immediately
    to maintain proper pipeline flow (turn detection, speaking events, etc.).
    """

    def __init__(self, conversation_notifier: BaseNotifier, voicemail_notifier: BaseNotifier):
        """Initialize the LLM gate.

        Args:
            conversation_notifier: Notifier that signals when a conversation is
                detected and gated frames should be released for LLM processing.
            voicemail_notifier: Notifier that signals when voicemail is detected
                and gated frames should be discarded.
        """
        super().__init__()
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier
        self._buffered_context_frame: tuple[Frame, FrameDirection] | None = None
        self._gating_active = True
        self._conversation_task: asyncio.Task | None = None
        self._voicemail_task: asyncio.Task | None = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._conversation_task = self.create_task(self._wait_for_conversation())
        self._voicemail_task = self.create_task(self._wait_for_voicemail())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._conversation_task:
            await self.cancel_task(self._conversation_task)
            self._conversation_task = None
        if self._voicemail_task:
            await self.cancel_task(self._voicemail_task)
            self._voicemail_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, buffering LLMContextFrame while classification is active.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if self._gating_active and isinstance(frame, LLMContextFrame):
            # Keep only the latest LLMContextFrame — it carries the most
            # up-to-date context, so earlier ones are redundant.
            self._buffered_context_frame = (frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _wait_for_conversation(self):
        """Wait for conversation detection and release the gated frame.

        When a conversation is detected, the buffered LLMContextFrame is
        released to trigger the main LLM for normal dialogue flow.
        """
        await self._conversation_notifier.wait()

        self._gating_active = False
        if self._buffered_context_frame:
            frame, direction = self._buffered_context_frame
            self._buffered_context_frame = None
            await self.push_frame(frame, direction)

    async def _wait_for_voicemail(self):
        """Wait for voicemail detection and discard the gated frame.

        When voicemail is detected, the buffered LLMContextFrame is
        discarded since the call will be ended by the voicemail event handler.
        """
        await self._voicemail_notifier.wait()

        self._gating_active = False
        self._buffered_context_frame = None


class VoicemailDetector(ParallelPipeline):
    """Parallel pipeline for detecting voicemail vs. live conversation in outbound calls.

    This detector uses a parallel pipeline architecture to perform real-time
    classification of outbound phone calls without interrupting the conversation
    flow. It determines whether a human answered the phone or if the call went
    to a voicemail system.

    Architecture:

    - Conversation branch: Empty pipeline that allows normal frame flow
    - Classification branch: Contains the LLM classifier and decision logic

    The system uses a gate mechanism to control when classification runs and
    a gating system to prevent TTS output until classification is complete.
    Once a decision is made, the appropriate action is taken:

    - CONVERSATION: Continue normal bot dialogue
    - VOICEMAIL: Trigger developer event handler for custom voicemail handling

    Example::

        classification_llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
        detector = VoicemailDetector(llm=classification_llm)

        @detector.event_handler("on_voicemail_detected")
        async def handle_voicemail(processor):
            await processor.push_frame(TTSSpeakFrame("Please leave a message."))

        pipeline = Pipeline([
            transport.input(),
            stt,
            detector.detector(),          # Classification
            context_aggregator.user(),
            llm,
            tts,
            detector.gate(),              # TTS gating
            transport.output(),
            context_aggregator.assistant(),
        ])

        # For custom prompts, append the required response instruction:
        custom_prompt = "Your custom classification logic here. " + VoicemailDetector.CLASSIFIER_RESPONSE_INSTRUCTION

    Events:
        on_conversation_detected: Triggered when a human conversation is detected. The
            event handler receives one argument: the ClassificationProcessor instance
            which can be used to push frames.
        on_voicemail_detected: Triggered immediately when voicemail is detected. The
            event handler receives one argument: the ClassificationProcessor
            instance which can be used to push frames.

    Constants:
        CLASSIFIER_RESPONSE_INSTRUCTION: The exact text that must be included in custom
            system prompts to ensure proper classification functionality.
    """

    CLASSIFIER_RESPONSE_INSTRUCTION = 'Respond with ONLY "CONVERSATION" if a person answered, or "VOICEMAIL" if it\'s voicemail/recording.'

    DEFAULT_SYSTEM_PROMPT = (
        """You are a voicemail detection classifier for an OUTBOUND calling system. A bot has called a phone number. You receive the transcribed speech of whoever/whatever answered (any language — Hebrew, English, or mixed; transcription may be partial or contain speech-to-text errors). Decide: did a LIVE PERSON answer, or did the call reach a RECORDED SYSTEM (voicemail / answering machine / carrier message / IVR menu)?

THE KEY QUESTION: is this speech ADDRESSED TO THE CALLER AS A RECORDING, or is it a person talking WITH the caller right now?

RECORDED SYSTEM (respond "VOICEMAIL") — complete, scripted announcements:
- Carrier/subscriber messages, third-person scripted grammar: "המנוי שחייגתם אליו אינו זמין", "המנוי עסוק כעת", "אין מענה כרגע", "The subscriber you have dialed is not available", "The person you are calling is unavailable"
- Voicemail-box intros: "שיחתך מועברת לתא הקולי", "הגעתם לתא הקולי של…", "שירות תא קולי משפחתי" (garbled: "שירות תקולים משפחתי"), "Your call has been forwarded to an automated voice messaging system", "You have reached the voice mailbox of…"
- Ad-supported mailboxes that play a promo before the tone: "פרסומת קצרה", "למעבר למפרסם הקישו כוכבית", "לחזרה להאזנה הקישו 1"
- Leave-a-message instructions tied to a tone/beep: "אנא השאירו הודעה לאחר הצליל", "בהישמע הצליל נא להקליט את ההודעה", "השאירו שם ומספר", "please leave a message after the tone/beep", "record your message"
- Recorded personal greetings — a NAME plus can't-answer plus leave-a-message script: "שלום, הגעתם לדני. אני לא זמין כרגע, השאירו הודעה ואחזור אליכם", "Hi, you've reached John, I can't take your call right now, leave a message"
- Disconnected/invalid number: "מספר הטלפון שחייגתם אליו איננו מחובר", "אינו מנוי בשירות", "The number you have dialed is not in service"
- Network/busy announcements: "החיוג אל המנוי המבוקש אינו אפשרי כעת", "כל הקווים תפוסים", "All circuits are busy", "mailbox is full", "תיבת ההודעות מלאה"
- IVR/queue systems: "הקישו 1", "לשירות בעברית הקישו…", "press 1 for…", "our office is currently closed, business hours are…", "כל נציגינו עסוקים, אנא המתינו", "please stay on the line"
- A phone number or bare digit string as the call's opening speech — mailboxes read the subscriber's number back, often transcribed fragmented: "0522568437", "6.  5.", "050-419. 7747", "אפס. 5, 2, 2, 7". A live person never opens a call by reciting digits, so this is VOICEMAIL even though it looks fragmentary. Same for a garbled mailbox intro followed by digits: "גולי של אפס. 5. 43", "התאגודי של 0. 54475" (mangled "התא הקולי של 05…")
- Automated call-screening assistants (any language) that answer for the person: "I'm a call assistant recording this call for the person you're trying to reach", "Please say who you are and why you're calling", "I'll see if this person is available", "The person you're calling is busy now", "Я записываю вызов для человека", "Это ассистент по вызовам" — scripted screeners are machines, not the person
- Carrier missed-call/message services: "הודעה: חיפשתי אותך", "ההודעה נשלחה", "הקלטתכם הגיעה למשך המרבי"
- A cut-off prefix of any of the above (transcription may stop mid-script): "המנוי שחייגתם אליו אינו", "שלום הגעתם לתא הקולי של", "Your call has been forwarded to an automated"

LIVE PERSON (respond "CONVERSATION") — spontaneous, interactive speech:
- Short answers and greetings, any wording: "הלו?", "כן", "כן?", "מי זה?", "דבר", "שומע", "בסדר", "טוב", "מעולה", "סבבה", "מה נשמע?", "Hello?", "Yeah?", "Speaking", "Who is this?"
- ANY reply that engages with what the bot said — answering a question, agreeing, refusing, objecting: "כן, יש לי שתי דקות", "לא, אין לי זמן", "זה זמן לא טוב", "אני לא מעוניין", "כמה זמן זה ייקח?"
- A person asking YOU to call back later or to use another channel — first person, live: "אני עסוק, תחזרו אליי אחר כך", "אני נוהג עכשיו", "תתקשרי בערב", "תשאירי לי הודעה בוואטסאפ", "I'm busy right now, call me back later", "text me instead". A live "call me back" / "message me" is NOT voicemail unless it is part of a recorded greeting structure ("you've reached X… I can't answer…").
- Someone else answering for the target — first-person, present-tense availability talk: "היא לא זמינה כרגע, אני יכול לקחת הודעה?", "דני לא נמצא, מי זה?", "He's not available right now, can I take a message?" — that speaker is a LIVE person offering to help, not a recording.
- A live receptionist/business answer expecting a reply: "משרד עורכי דין כהן, שלום", "מרפאת דר לוי, במה אפשר לעזור?", "Dr. Smith's office, how can I help you?"
- Confused/noisy/hesitant speech, repeats, "הלו? הלו? לא שומעים", "I can't hear you", background chatter, fragments, or garbled words with no voicemail script structure

DECISION RULES:
1. Respond VOICEMAIL only when the text clearly matches a recorded-script pattern above.
2. If the speech is short, ambiguous, fragmentary, or could plausibly be a live person — respond CONVERSATION. Hanging up on a live person is far worse than listening to a recording a few seconds longer.
3. Mentioning words like "הודעה"/"message"/"לא זמין"/"not available" does NOT alone mean voicemail — judge whether it is a recording addressing the caller (third-person script, tone/beep instructions) or a live person talking about themselves or someone else.

"""
        + CLASSIFIER_RESPONSE_INSTRUCTION
    )

    def __init__(
        self,
        *,
        llm: LLMService,
        custom_system_prompt: str | None = None,
        long_speech_timeout: float | None = None,
    ):
        """Initialize the voicemail detector with classification and buffering components.

        Args:
            llm: LLM service used for voicemail vs conversation classification.
                Should be fast and reliable for real-time classification.
            custom_system_prompt: Optional custom system prompt for classification. If None,
                uses the default prompt optimized for outbound calling scenarios.
                Custom prompts should instruct the LLM to respond with exactly
                "CONVERSATION" or "VOICEMAIL" for proper detection functionality.
            long_speech_timeout: Optional timeout in seconds. When the first turn has
                continuous speech exceeding this duration, classification is triggered
                early with whatever transcript is available. Useful for detecting long
                voicemail greetings without waiting for the caller to stop speaking.
                Requires ``eager_eot_threshold`` on Deepgram Flux for interim
                transcripts. Defaults to None (disabled).
        """
        self._classifier_llm = llm
        self._prompt = (
            custom_system_prompt if custom_system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT
        )

        # Validate custom prompts to ensure they work with the detection logic
        if custom_system_prompt is not None:
            self._validate_prompt(custom_system_prompt)

        # Set the system prompt as LLM system_instruction instead of a context message.
        #
        # It must ALSO be recorded as the base the composer rebuilds from. Every
        # LLMContextFrame syncs the tool handlers, and that sync ends in
        # _compose_system_instruction(), which discards the live value and rebuilds
        # from _base_system_instruction:
        #
        #     composed = "\n\n".join(p for p in parts if p)
        #     self._settings.system_instruction = composed or None
        #
        # A classifier whose base was never set therefore has its prompt replaced by
        # None on its very first context frame. It then answers the transcript
        # conversationally, classify_verdict() finds no VOICEMAIL/CONVERSATION marker,
        # no decision is ever made, and the LLMGate holding the main LLM never opens --
        # every caller turn stalls until the api's ResponseGuard retries it 5s later.
        # (Setting _settings.system_instruction alone is not enough: the base is
        # snapshotted only in __init__ or on an LLMUpdateSettingsFrame, and this
        # detector uses neither.)
        self._classifier_llm._settings.system_instruction = self._prompt
        self._classifier_llm._base_system_instruction = self._prompt

        # Create the LLM context and aggregators for conversation management
        self._messages = []
        self._context = LLMContext(self._messages, skip_set_tools_frame=True)

        # Set OTel span name for tracing
        self._context.set_otel_span_name("llm-voicemail-detector")
        # Turn boundaries are adopted from the conversation aggregator downstream
        # (the ``ExternalUserTurnStrategies`` shape), with the classifier's stop
        # variant so a transcript that lands before the adopted start still
        # finalizes the turn — see ClassifierUserTurnStopStrategy.
        self._context_aggregator = LLMContextAggregatorPair(
            self._context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[ExternalUserTurnStartStrategy()],
                    stop=[ClassifierUserTurnStopStrategy()],
                )
            ),
        )

        # Create notification system for coordinating between components
        self._gate_notifier = EventNotifier()  # Signals classification completion
        self._conversation_notifier = EventNotifier()  # Signals conversation detected
        self._voicemail_notifier = EventNotifier()  # Signals voicemail detected

        # Create the processor components
        self._classifier_gate = ClassifierGate(self._gate_notifier, self._conversation_notifier)
        self._classifier_upstream_gate = ClassifierUpstreamGate(self._conversation_notifier)
        self._conversation_gate = ConversationGate(self._voicemail_notifier)
        self._classification_processor = ClassificationProcessor(
            gate_notifier=self._gate_notifier,
            conversation_notifier=self._conversation_notifier,
            voicemail_notifier=self._voicemail_notifier,
        )
        self._voicemail_gate = TTSGate(self._conversation_notifier, self._voicemail_notifier)
        self._llm_gate = LLMGate(self._conversation_notifier, self._voicemail_notifier)

        # Create first-turn speech monitor if long speech timeout is configured
        self._first_turn_speech_monitor = None
        if long_speech_timeout is not None:
            self._first_turn_speech_monitor = FirstTurnSpeechMonitor(
                timeout_secs=long_speech_timeout,
                context=self._context,
                gate_notifier=self._gate_notifier,
            )

        # Build classification branch
        classification_pipeline: list[FrameProcessor] = [self._classifier_gate]
        if self._first_turn_speech_monitor:
            classification_pipeline.append(self._first_turn_speech_monitor)
        classification_pipeline.extend(
            [
                self._context_aggregator.user(),
                self._classifier_llm,
                self._classification_processor,
                self._context_aggregator.assistant(),
                self._classifier_upstream_gate,
            ]
        )

        # Initialize the parallel pipeline with conversation and classifier branches
        super().__init__(
            # Conversation pipeline: gate that blocks after voicemail detection
            [self._conversation_gate],
            # Classification pipeline
            classification_pipeline,
        )

        # Register the voicemail detected event after super().__init__()
        self._register_event_handler("on_voicemail_detected")

    def _validate_prompt(self, prompt: str) -> None:
        """Validate custom prompt contains required response format instructions.

        Custom prompts must instruct the LLM to respond with exactly "CONVERSATION"
        or "VOICEMAIL" for the detection logic to work properly. This method
        checks for the presence of these keywords and warns if they're missing.

        Args:
            prompt: The custom system prompt to validate.
        """
        has_conversation = "CONVERSATION" in prompt
        has_voicemail = "VOICEMAIL" in prompt

        if not has_conversation or not has_voicemail:
            logger.warning(
                "Custom system prompt should instruct the LLM to respond with exactly "
                '"CONVERSATION" or "VOICEMAIL" for proper detection functionality. '
                f"Consider appending VoicemailDetector.CLASSIFIER_RESPONSE_INSTRUCTION to your prompt: "
                f'"{self.CLASSIFIER_RESPONSE_INSTRUCTION}"'
            )

    def detector(self) -> "VoicemailDetector":
        """Get the detector pipeline for placement after STT in the main pipeline.

        This should be placed after the STT service and before the context
        aggregator in your main pipeline to enable voicemail classification.

        Returns:
            The VoicemailDetector instance itself (which is a ParallelPipeline).
        """
        return self

    def gate(self) -> TTSGate:
        """Get the gate processor for placement after TTS in the main pipeline.

        This should be placed after the TTS service and before the transport
        output to enable TTS frame gating during classification.

        Returns:
            The TTSGate processor instance.
        """
        return self._voicemail_gate

    def llm_gate(self) -> LLMGate:
        """Get the LLM gate for placement before the main LLM in the pipeline.

        This should be placed after the user context aggregator and before the
        main LLM service to prevent the LLM from being triggered before
        voicemail classification completes. Only ``LLMContextFrame`` is gated;
        all other frames pass through immediately.

        Returns:
            The LLMGate processor instance.
        """
        return self._llm_gate

    def add_event_handler(self, event_name: str, handler):
        """Add an event handler for voicemail detection events.

        Args:
            event_name: The name of the event to handle.
            handler: The function to call when the event occurs.
        """
        if event_name in ("on_conversation_detected", "on_voicemail_detected"):
            self._classification_processor.add_event_handler(event_name, handler)
        else:
            super().add_event_handler(event_name, handler)


class FirstTurnSpeechMonitor(FrameProcessor):
    """Monitors the first conversation turn for extended continuous speech.

    Used internally by :class:`VoicemailDetector`. When continuous speech in
    the first turn exceeds the configured timeout, it adds the accumulated
    transcript to the classifier ``LLMContext`` and pushes an
    ``LLMContextFrame`` downstream. Because this processor lives inside the
    classification branch of the parallel pipeline, the frame is consumed by
    the classifier LLM and never leaks into the main pipeline.

    For Deepgram Flux, ensure ``eager_eot_threshold`` is set so that
    ``InterimTranscriptionFrame`` s are pushed during speech pauses.
    """

    def __init__(self, *, timeout_secs: float, context: LLMContext, gate_notifier: BaseNotifier):
        """Initialize the first turn speech monitor.

        Args:
            timeout_secs: Duration in seconds of continuous first-turn speech
                before classification is triggered.
            context: The classifier LLM context to inject the transcript into.
            gate_notifier: Notifier to close the ClassifierGate after triggering
                early classification, preventing subsequent LLM calls.
        """
        super().__init__()
        self._timeout_secs = timeout_secs
        self._context = context
        self._gate_notifier = gate_notifier
        self._is_first_turn = True
        self._final_transcripts: list[str] = []
        self._latest_interim = ""
        self._timer = None
        self._triggered = False

    async def cleanup(self):
        """Clean up timer resources."""
        await super().cleanup()
        self._cancel_timer()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, tracking first-turn speech and interim transcripts."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Start timer now since Voicemail Detector relies on
            # External Turn Stop strategies, and the main pipeline
            # might be muted till first Bot Stopped Speaking
            self._schedule_timer()
        elif not self._triggered and self._is_first_turn:
            if isinstance(frame, UserStoppedSpeakingFrame):
                self._cancel_timer()
                self._is_first_turn = False

            elif isinstance(frame, TranscriptionFrame):
                self._final_transcripts.append(frame.text)
                self._latest_interim = ""

            elif isinstance(frame, InterimTranscriptionFrame):
                self._latest_interim = frame.text

        await self.push_frame(frame, direction)

    def _get_best_transcript(self) -> str:
        """Return accumulated finals + latest interim."""
        parts = list(self._final_transcripts)
        if self._latest_interim:
            parts.append(self._latest_interim)
        return " ".join(parts).strip()

    def _schedule_timer(self):
        self._cancel_timer()
        loop = asyncio.get_event_loop()
        self._timer = loop.call_later(
            self._timeout_secs,
            lambda: asyncio.create_task(self._on_timeout()),
        )

    def _cancel_timer(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    async def _on_timeout(self):
        if self._is_first_turn and not self._triggered:
            self._triggered = True
            self._timer = None
            transcript = self._get_best_transcript()
            if not transcript:
                logger.warning(
                    f"{self}: First turn exceeded {self._timeout_secs}s "
                    "but no transcript available yet"
                )
                return
            logger.info(
                f"{self}: First turn exceeded {self._timeout_secs}s, "
                f"triggering classification with: {transcript!r}"
            )
            # Close the classifier gate to prevent subsequent UserStoppedSpeaking
            # from triggering another classification LLM call
            await self._gate_notifier.notify()

            # Inject transcript into classifier context and trigger LLM
            self._context.add_message({"role": "user", "content": transcript})
            await self.push_frame(LLMContextFrame(self._context))
