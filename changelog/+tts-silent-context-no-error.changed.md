- A TTS context that completes with no audio is logged and fires
  `on_tts_zero_audio`; it no longer pushes an `ErrorFrame` of its own, and
  `max_consecutive_zero_audio_contexts` defaults to 0 (the service is never
  written off for silence unless an application sets a limit). An application
  that ends the call on an `ErrorFrame` would otherwise hang up on a
  provider's brief silence.
