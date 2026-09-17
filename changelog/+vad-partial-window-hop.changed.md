- `VADAnalyzer.analyze_audio` no longer hops to its worker thread for a buffer
  that cannot complete a model window; it appends the audio on the loop thread
  and returns the unchanged state, exactly as `_run_analyzer` would have. With
  20ms frames that is 3 of every 8 frames at 8kHz and 16kHz. The state
  sequence is identical; the saving is the one to two loop iterations the
  awaiting input task used to wait per frame. The executor is still used while
  an earlier analysis is running on the worker, and always for subclasses that
  override `_run_analyzer`.
