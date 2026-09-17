- Added an opt-in bound on how long `FrameProcessorQueue` lets system frames
  starve non-system frames. Strict priority starved every
  `TranscriptionFrame` and `HeartbeatFrame` at a `LLMUserAggregator` whose
  per-audio-frame service time (the VAD executor hop, slowed by a busy loop)
  reached the 20ms audio interval: on campaign-loaded api pods the caller's
  words waited 22-37s, the 5s turn-stop fuse committed an empty turn, and 77
  of 1,057 calls stalled in two minutes. With
  `PIPECAT_INPUT_QUEUE_STARVATION_BOUND_MS` (or
  `FrameProcessorQueue.set_starvation_bound()`) set, a non-system frame that
  has waited that long is no longer overtaken by system frames that arrived
  after it; system frames that arrived before it still go first, so nothing is
  reordered past arrival order and an `InterruptionFrame` still flushes the
  frames queued before it. After moving such a frame to the process queue the
  input task yields once, so a turn start queued behind it cannot flush it
  before the process task runs. Unset or 0 (the default) keeps strict
  priority.
  `INPUT_QUEUE_STARVATION_BOUND_RECOMMENDED_SECS` (80ms) is the value to use.
  The queue's storage moved from a heap to two FIFOs; with the bound off the
  order is identical.

- `FrameProcessor` gained `input_queue_non_system_depth`,
  `input_queue_non_system_wait` (age of the oldest waiting non-system frame)
  and `input_queue_bounded_serves`; `PipelineWorker.dump_processor_diagnostics()`
  reports the first two (and logs `waiting=<count>/<age>s`), and the new
  `PipelineWorker.starved_processors(min_wait_secs=...)` returns the processors
  holding a non-system frame past a threshold, oldest first, without logging —
  meant for an `on_heartbeat_timeout` handler to name the starved processor.
