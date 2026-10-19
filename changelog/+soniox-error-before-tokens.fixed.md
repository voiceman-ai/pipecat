- Fixed `SonioxSTTService` dropping server error messages that carry no
  `tokens` key. Such messages raised a `KeyError` that was logged as a warning,
  so the error was never pushed upstream.
