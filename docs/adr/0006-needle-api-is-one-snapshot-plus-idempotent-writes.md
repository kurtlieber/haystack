# needle's API is one snapshot read plus idempotent writes

needle Reviews offline (ADR 0004), so it reads the whole Review state in one call (`GET /v1/state`: both Review queues as full track cards, the Baseline, the next batch time) and caches it, rather than paging through queues or fetching a track at a time. It writes through an outbox: every verdict and play carries an id needle made, so a replay after a dropped connection is a no-op, and a verdict also carries the time the listener gave it, so a late replay can never override a newer verdict (ADR 0005). The server answers a write as soon as its own state has changed and makes the Spotify writes afterwards, in the background, so a Review never waits on Spotify, and a Spotify outage or a missing scope shows up as `spotify.pending_writes` in the next snapshot rather than as a failed request.

The paths are versioned (`/v1`); new fields are additive, and anything that would break an installed needle becomes `/v2`. A later move of the API (to Cloudflare Workers, say) keeps this contract.

**Considered Options**: paged queue endpoints (smaller responses, but Review offline needs everything anyway, and the full snapshot is about 400 KB); making Spotify writes synchronous (a verdict would fail whenever Spotify did, and the outbox would retry Spotify problems it can't fix).

**Consequences**: the snapshot grows with Liked Songs (about 1 KB a track). A verdict's Spotify writes can lag the response by seconds, or by an hour when they fall back to the sync.
