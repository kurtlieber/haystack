# Verdicts are an append-only log keyed by Spotify track id

needle can give a Verdict on any track (a discovery, an Unrated like, later anything the listener plays), and its outbox replays verdicts given offline, possibly out of order and more than once. So the server stores every verdict as a row in one `verdicts` table keyed by the Spotify track id, with a client-made id (a replay is a no-op) and the time the listener gave it; the current Verdict is the latest by that time, and nothing is ever updated in place. The old `feedback` table (yes/no on candidates only, one row per candidate) was migrated into it once: yes became 3★, a mild or normal no a Pass, a strong no a Dislike.

The server's own state (the candidate leaves the pool, the offer closes) follows a verdict at once; the Spotify writes it implies (the heart, Discoveries, To Rip) are applied afterwards for the current verdict only, and retried until they succeed.

**Considered Options**: a `current_verdict` column on candidates (only discoveries could be judged, and history and offline ordering would be lost); updating one row per track (a late replay of an older verdict would overwrite a newer one).

**Consequences**: a verdict needs a Spotify id, so a candidate that was never mapped to Spotify can't be judged from the CLI. The one mild "no" became a normal-strength Pass (0.5 → 1.0). Verdicts given on another device's clock order by that clock.
