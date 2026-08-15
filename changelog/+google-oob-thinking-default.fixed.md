- `GoogleLLMService.run_inference_with_usage` (and therefore `run_inference`)
  now applies the same low-latency thinking default as the streaming path —
  `thinking_budget=0` for Gemini 2.5 Flash, `thinking_level="minimal"` for
  Gemini 3 Flash models. Out-of-band inferences previously ran at the model's own
  thinking default, so most of their billed completion tokens were thought
  rather than answer. An explicit `thinking` setting still wins. A model that
  answers the default with a 400 mentioning thinking is retried once without it
  and the default is not applied to that model again for the life of the service.
