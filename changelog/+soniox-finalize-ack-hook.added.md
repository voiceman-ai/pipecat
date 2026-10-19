- Added `SonioxSTTService._on_finalize_ack()`, called whenever the server
  answers a finalize request with a `<fin>` token, including acknowledgements
  that carry no transcript. Subclasses override it to observe finalize
  round-trips.
