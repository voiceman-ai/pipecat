- `SileroVADAnalyzer` takes a `model` argument: `"v6.2"` (the default, the
  model upstream ships) or `"v6"`, the earlier bundled model
  (`silero_vad_v6.onnx`), which runs with the reset schedule it was tuned with
  (every five seconds of audio, speech or not). v6.2 scores quiet and far-field
  voices higher, so an application whose VAD thresholds were tuned on v6 can
  stay on it until it re-tunes.
