- `FrameProcessor` now exposes read-only wedge diagnostics —
  `input_queue_depth`, `process_queue_depth`, `seconds_since_last_progress`,
  `processing_frame_name` and `is_frame_processing_paused` — and
  `PipelineWorker` gained `dump_processor_diagnostics()`, which logs a single
  atomic WARNING record (token: `processor diagnostics dump`) with one line
  per processor and returns the snapshot as data. A leg that wedges before
  its first turn (frames piling up behind one stuck processor) is now
  diagnosable from a log dump: the wedged processor shows a growing queue,
  an aging progress stamp and the name of the culprit frame, while a paused
  processor is explicitly labeled as paused rather than stuck. None of this
  feeds kill or abort decisions — it is observation only.
