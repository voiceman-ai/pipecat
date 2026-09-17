#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice Activity Detection (VAD) analyzer base classes and utilities.

This module provides the abstract base class for VAD analyzers and associated
data structures for voice activity detection in audio streams. Includes state
management, parameter configuration, and audio analysis framework.
"""

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import exp_smoothing
from pipecat.audio.volume import AudioVolumeTracker

VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6


class VADState(Enum):
    """Voice Activity Detection states.

    Parameters:
        QUIET: No voice activity detected.
        STARTING: Voice activity beginning, transitioning from quiet.
        SPEAKING: Active voice detected and confirmed.
        STOPPING: Voice activity ending, transitioning to quiet.
    """

    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class VADParams(BaseModel):
    """Configuration parameters for Voice Activity Detection.

    Parameters:
        confidence: Minimum confidence threshold for voice detection.
        start_secs: Duration to wait before confirming voice start.
        stop_secs: Duration to wait before confirming voice stop.
        min_volume: Minimum audio volume threshold for voice detection.
    """

    confidence: float = VAD_CONFIDENCE
    start_secs: float = VAD_START_SECS
    stop_secs: float = VAD_STOP_SECS
    min_volume: float = VAD_MIN_VOLUME


class VADAnalyzer(ABC):
    """Abstract base class for Voice Activity Detection analyzers.

    Provides the framework for implementing VAD analysis with configurable
    parameters, state management, and audio processing capabilities.
    Subclasses must implement the core voice confidence calculation.
    """

    def __init__(self, *, sample_rate: int | None = None, params: VADParams | None = None):
        """Initialize the VAD analyzer.

        Args:
            sample_rate: Audio sample rate in Hz. If None, will be set later.
            params: VAD parameters for detection configuration.
        """
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._params = params or VADParams()
        self._num_channels = 1

        self._vad_buffer = b""
        self._volume_tracker = AudioVolumeTracker()

        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0.0

        # Thread executor that will run the model. We only need one thread per
        # analyzer because one analyzer just handles one audio stream.
        #
        # MEASURED NEGATIVE RESULT — per-frame CPU in this path is NOT what
        # limits per-pod concurrency. Numbers from a developer Mac (2026-07-30,
        # Python 3.13); a GKE e2-standard-4 node will be perhaps 2-4x slower,
        # but the ratios are what matter and they are two orders of magnitude
        # below the budget:
        #
        #   - Silero voice_confidence: 109us per 32ms frame @16kHz, 76us @8kHz.
        #     Runs on this worker thread and ONNX releases the GIL, so it is off
        #     the event loop entirely.
        #   - The executor round trip below — the only loop-side cost — is 35us,
        #     paid once per input audio frame per leg. At 50fps x 6 legs that is
        #     ~1% of loop time.
        #   - soxr stream resampling, for comparison: 3.6us per 20ms frame at
        #     16k->8k VHQ, 5.1us at 24k->8k. Dropping quality below VHQ saves
        #     ~1us and is not worth the audio cost.
        #
        # Do not tune these paths looking for concurrency headroom. Measure
        # instead: with heartbeats no longer routed through the paced media
        # queue, no longer drained by an interruption and exempt from
        # `pause_processing_frames()`, heartbeat traversal latency
        # (PipelineWorker's `on_heartbeat`) is usable as a per-pipeline load
        # signal. Read it for what it is: scheduling delay PLUS the longest
        # in-flight per-processor frame operation — a streamed LLM generation
        # still sits in front of it — so compare like-for-like across pods
        # rather than treating an absolute value as pure event-loop delay.
        #
        # The thread itself, however, is per-analyzer and is only reclaimed when
        # the executor is garbage collected — arbitrarily late for an object
        # held in the pipeline's reference cycles. `shutdown()` releases it
        # deterministically at teardown.
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)
        # The last analysis handed to the executor (see analyze_audio's
        # partial-window fast path, which must not run beside it).
        self._pending_analysis: Future | None = None
        self._partial_window_fast_path = type(self)._run_analyzer is VADAnalyzer._run_analyzer

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate.

        Returns:
            Current audio sample rate in Hz.
        """
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """Get the number of audio channels.

        Returns:
            Number of audio channels (always 1 for mono).
        """
        return self._num_channels

    @property
    def params(self) -> VADParams:
        """Get the current VAD parameters.

        Returns:
            Current VAD configuration parameters.
        """
        return self._params

    @abstractmethod
    def num_frames_required(self) -> int:
        """Get the number of audio frames required for analysis.

        Returns:
            Number of frames needed for VAD processing.
        """
        pass

    @abstractmethod
    def voice_confidence(self, buffer: bytes) -> float:
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Voice confidence score between 0.0 and 1.0.
        """
        pass

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self._sample_rate = self._init_sample_rate or sample_rate
        self.set_params(self._params)

    def set_params(self, params: VADParams):
        """Set VAD parameters and recalculate internal values.

        Args:
            params: VAD parameters for detection configuration.
        """
        logger.debug(f"Setting VAD params to: {params}")
        self._params = params
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        vad_frames_per_sec = self._vad_frames / self.sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
        # VAD state resets, but volume state doesn't: the rolling window and its
        # smoothing follow the audio stream, which is continuous across
        # parameter changes.
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET

    def _get_smoothed_volume(self, audio: bytes) -> float:
        """Calculate smoothed audio volume using exponential smoothing."""
        self._volume_tracker.update(audio, self.sample_rate)
        return exp_smoothing(self._volume_tracker.volume, self._prev_volume, self._smoothing_factor)

    async def analyze_audio(self, buffer: bytes) -> VADState:
        """Analyze audio buffer and return current VAD state.

        Processes incoming audio data, maintains internal state, and determines
        voice activity status based on confidence and volume thresholds.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Current VAD state after processing the buffer.
        """
        executor = self._executor
        if executor is None:
            # Shut down: the analyzer is torn down but audio is still arriving.
            # Report the last known state rather than raising into the pipeline.
            return self._vad_state

        # Partial-window fast path. A buffer that cannot complete one model
        # window only gets appended: `_run_analyzer` returns the unchanged
        # state before touching the model, the counters or the thresholds.
        # Doing that append here instead of on the worker thread gives the
        # identical state sequence and skips the thread hop, which is not free
        # on a busy loop: the awaiting task only resumes one to two loop
        # iterations after the worker finishes, and `LLMUserAggregator` pays
        # that on every 20ms audio frame from its input task. On busy api pods
        # (loop lag max 21-56ms per 15s) that per-frame cost is what let audio
        # starve the aggregator's transcripts and heartbeats (see
        # FrameProcessorQueue). With 20ms frames, 3 of every 8 frames cannot
        # complete a window (160 samples against 256 at 8kHz, 320 against 512
        # at 16kHz), so this removes 37.5% of the hops.
        #
        # Safe without a lock because every caller awaits `analyze_audio`
        # sequentially per analyzer (VADController runs it inline on its
        # processor's input task), and it is only taken when the previous
        # analysis has finished: a cancelled await can leave its
        # `_run_analyzer` still running on the worker, and appending beside it
        # would race on `_vad_buffer`, so that case keeps the executor, which
        # queues behind it in order. A subclass that overrides `_run_analyzer`
        # always keeps the executor.
        num_required_bytes = getattr(self, "_vad_frames_num_bytes", None)
        pending = self._pending_analysis
        if (
            self._partial_window_fast_path
            and num_required_bytes is not None
            and (pending is None or pending.done())
            and len(self._vad_buffer) + len(buffer) < num_required_bytes
        ):
            self._vad_buffer += buffer
            return self._vad_state

        future = executor.submit(self._run_analyzer, buffer)
        self._pending_analysis = future
        return await asyncio.wrap_future(future)

    def shutdown(self) -> None:
        """Release the analyzer's worker thread. Idempotent.

        Safe to call while an analysis is in flight: pending work is cancelled,
        the running one is not waited for, and later ``analyze_audio`` calls
        return the last known state instead of raising.
        """
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _run_analyzer(self, buffer: bytes) -> VADState:
        """Analyze audio buffer and return current VAD state."""
        self._vad_buffer += buffer

        num_required_bytes = self._vad_frames_num_bytes
        if len(self._vad_buffer) < num_required_bytes:
            return self._vad_state

        while len(self._vad_buffer) >= num_required_bytes:
            audio_frames = self._vad_buffer[:num_required_bytes]
            self._vad_buffer = self._vad_buffer[num_required_bytes:]

            confidence = self.voice_confidence(audio_frames)

            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume

            if speaking:
                match self._vad_state:
                    case VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._vad_starting_count = 1
                    case VADState.STARTING:
                        self._vad_starting_count += 1
                    case VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
            else:
                match self._vad_state:
                    case VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._vad_starting_count = 0
                    case VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._vad_stopping_count = 1
                    case VADState.STOPPING:
                        self._vad_stopping_count += 1

        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0

        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0

        return self._vad_state

    async def cleanup(self):
        """Clean up resources.

        This method should be called when the object is no longer needed.
        It waits for all currently executing event handler tasks to finish
        before returning.
        """
        self.shutdown()
