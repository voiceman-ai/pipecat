- `TTSService` now flags contexts that complete with zero audio. When text
  was sent for synthesis but no non-empty `TTSAudioRawFrame` was ever
  accounted against the context by the time its `TTSStoppedFrame` flows, the
  base class emits a distinct `TTS_ZERO_AUDIO` warning, increments
  `zero_audio_context_count` and fires the new `on_tts_zero_audio` event
  (context ID + characters sent), so applications can count provider-side
  silent synthesis failures per leg instead of discovering them post-hoc from
  billing data. Interrupted contexts are deliberately excluded — barge-in
  finalizes the accounting before a stop frame ever reaches the check — and
  no `ErrorFrame` is pushed, because the pipeline itself is healthy.
