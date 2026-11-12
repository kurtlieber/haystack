# A client that writes songs reads the listener's taste as a Taste brief

spindle is a client that writes song ideas from what haystack knows about the listener and has a music service
make them. Until now a script read haystack's database directly and printed a brief: how often each style, tempo
band and instrumental level gets 5★, the recent 5★ and disliked songs that were new to the listener, and the
most-played 5★ songs. Ideas written from that brief gave 7 songs of 12 rated 4★ or 5★. spindle runs on a
desktop and haystack on the server, so spindle can't read the database; the API served only the Baseline.

So from 0.15.0:

1. **`GET /v1/taste` serves the Taste brief** (feature `taste`, docs/api.md): the same content as the script, as
   JSON. It is a new term, because "Taste profile" already names what haystack tells its own language model.
2. **Counts, not rates.** Each table row is a tally (`n`, 5★, 4-5★, Pass or Dislike), so a client can show a
   share, weigh a small group down or drop it. Every band is present even when empty.
3. **The script's rules, unchanged.** One Song per ISRC with its newest Verdict, "new to the listener" means the
   Verdict came from the Discoveries or Calibrate queue, and a style counts at an activation of 0.10 or more. The
   endpoint's numbers match the script's on the same database copy (checked on 2026-11-12 on a copy of the live
   database, with the default window and smallest group and two others: every line of the two briefs is equal).
4. **Read-only, cheap and always on.** It reads stored Verdicts, styles, Readings and plays; no model, no network,
   no writes. `days` and `min_n` in the query set the window of the recent lists and the smallest style group, as
   the script's flags did.
5. **Its own module.** `taste.py` imports the Baseline and the Readings only. The language model's code pulls in
   the Review and the re-ranker, and an import cycle between the API and the Review once broke the API.

## Considered

- **Serving the Taste profile.** It is shaped for one model request, built around the songs that request asks
  about, and carries no tallies a client could weigh.
- **Serving raw Verdicts and Readings for the client to count.** That moves haystack's rules about Songs and
  queues into each client, and sends far more data.
- **Reading the database over SSH from the desktop.** A client depends on the API, never on the database
  (docs/api.md).

## Consequences

- A second client reads `/v1`, so the contract covers its fields too (`tests/fixtures/v1-contract.json`).
- The brief is computed on each request, over every Verdict: under 0.1 s for about 1,500 Songs.
- The tallies reflect only what haystack has stored: a Song with no styles is counted in the totals and in no
  style table.
