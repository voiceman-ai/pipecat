- Fixed STT TTFB being measured against a stale VAD stop. When no final
  transcript arrived within the TTFB timeout, `VADUserStartedSpeakingFrame`
  cancelled the timeout but left the measurement armed, so the next
  utterance's final reported the time back to the old stop (4.4s and 6.0s on a
  Soniox session whose real TTFB was 222-257ms). A VAD start now abandons the
  measurement through the new `FrameProcessor.reset_ttfb_metrics()` /
  `FrameProcessorMetrics.reset_ttfb_metrics()`.
