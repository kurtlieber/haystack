# Songs a client has made are kept in an optional store, apart from Verdicts

spindle writes song ideas from the Taste brief (ADR 0028) and has a music service make them. Each idea gives
several takes; the listener rates them, says in a note what worked, and files the ones they want as their own.
Until now a skill kept that record in a JSON file on the desktop: the ideas, the prompts sent, and a log of
verdicts. The next ideas are written from those notes, so they belong where spindle can read them from any host,
with the same rules as needle's verdicts. spindle reaches haystack only through `/v1`.

A take is not a Song. It has no Spotify track, no ISRC and no Deezer preview, and the listener judges it against
the idea it came from, not against everything they might hear. A 5★ take says the idea worked, not that haystack
should look for songs like it.

So from 0.15.0:

1. **An optional store, `[generated] enabled`, off by default.** Off, haystack creates none of its tables, the
   `/v1/generated/` routes answer `404` like any unknown path, and `generated` is not in `server.features`: a
   haystack without spindle is unchanged. `generated` is the first feature listed only when switched on, so
   `api.FEATURES` stays the features every server has and `api.OPTIONAL_FEATURES` holds the switched ones.
2. **Three tables of its own.** `generated_ideas` holds each Idea: the idea as the client's own JSON, the reason
   and the exact request sent to the service. `generated_takes` holds each Take, keyed by service and the
   service's id, with its Idea, title, the service's model and when it was kept. `generated_verdicts` holds the
   verdicts on Takes. The idea stays the client's object, so who wrote the lyrics and the prompt goes in it, in
   whatever form the client uses; the model that made a take is a column, because every service has one.
3. **Verdicts on takes follow ADR 0005.** Append-only, a client-made id so a resend is a no-op, and a take's
   current verdict is its latest by `given_at`. A verdict is 1-5 stars or a Dislike, with an optional note; there
   is no Pass, because the listener asked for this song. Older notes stay readable, with `current` saying which
   is the take's verdict now.
4. **Outbox routes like needle's.** `POST /v1/generated/ideas`, `verdicts` and `kept` take a list, store each
   good item and reject each bad one with its reason, in one transaction. A verdict or a keep may name a take
   haystack has not seen, such as one made before the client recorded its ideas: the take is stored without an
   Idea, and an Idea sent later claims it. The first keep stays.
5. **Nothing else reads them.** Harvest, scoring, the re-ranker, the Baseline, the Review and the language model
   never read these tables, and only `generated.py` names them. A test watches every table read while a batch is
   planned, the Baseline is computed and the Review state is built, and fails on any of these; another checks
   that `/v1/state` and the Baseline are the same with and without generated rows.

## Considered

- **Storing takes as Verdicts on tracks.** A take has no Spotify track to hang a Verdict on, and a Verdict
  teaches the re-ranker and moves the Baseline. A song the listener commissioned would skew both.
- **A separate service or database for spindle.** One more process, token, backup and host to keep running, for
  three small tables. haystack already has the API, the token, the backups and an ADR for append-only verdicts.
- **Always on.** Every haystack would carry tables and routes that only spindle uses, and a client could not tell
  a server that has the store from one that merely answers.
- **Kept marks as verdicts.** Keeping is the listener filing a file, not a judgment; a take can be kept at 3★.

## Consequences

- A second client writes to haystack, so the contract fixture covers the store's answers
  (`tests/fixtures/v1-contract.json`); its entries are checked against a server with the store on.
- Switching the store off again hides the routes and the feature but leaves the tables and their rows in the
  database; switching it back on serves them again.
- A note-only remark with no stars has no place: a verdict is a Rating or a Dislike. A client that wants to say
  something about a take it has not rated holds the note until it does.
- haystack never fetches a take's audio or checks a take id with the service; it stores what the client says.
