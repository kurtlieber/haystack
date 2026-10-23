# HTTP API (v1)

The contract between the server and its clients (needle today). The server owns it: a client depends on this
document, never on the server's code or database. Why it looks this way: ADR 0004 (a private HTTP API), ADR 0005
(Verdicts), ADR 0006 (one snapshot plus idempotent writes), ADR 0022 (possible covers and the Swap).

## Transport

- Plain HTTP/1.1 on the port in `api.port` (default 8517), JSON in and out (`Content-Type: application/json`).
  Responses carry `Cache-Control: no-store`.
- Every request, `/v1/health` included, needs `Authorization: Bearer <token>`. The token is at least 32 characters:
  the server's `api-token` secret (`haystack token new`; `haystack token show` prints it for a client).
- Requests are accepted only from loopback and the networks in `api.allow_networks`. The server checks this
  itself; a host firewall should say the same.
- Errors are `{"error": "<reason>"}` with `400` (bad body), `401` (missing or wrong token), `403` (not from an
  allowed network; checked before the token) or `404` (no such endpoint or object).
- Times are UTC ISO 8601 ending in `Z`. Track and playlist ids are 22-character Spotify base-62 ids.

## Versioning

Everything is under `/v1`. New fields and new endpoints are additive: a client ignores fields it doesn't know.
Removing or changing the meaning of a field is a breaking change and goes to `/v2`, served alongside `/v1` until
clients have moved.

## Server and features

`GET /v1/health` and `GET /v1/state` both carry a `server` object:

```json
{"version": "0.6.0", "api": "1",
 "features": ["why.band", "rerank", "why.reasons", "queue.calibrate", "plays.ms_played", "card.label",
              "card.song", "card.original", "swaps"]}
```

- `version`: the haystack release. Shown to people; a client never branches on it.
- `api`: the API version, the same as `/v1` (and as health's older `version` field).
- `features`: the names of optional additions this server has. A client shows a new element only when its
  feature's name is listed, and treats every field a feature adds as optional. A server without a `server`
  object predates features: a client behaves as it did before any of them.

Each feature is listed here with the fields it brings, and in "Changes" below:

- `why.band`: a Discoveries offer's `why` carries `band` (Track card).
- `rerank`: the top of each batch is ordered by a model trained on the listener's Verdicts (ADR 0011); a
  Discoveries offer's `why` carries `model` and `confidence` (Track card).
- `why.reasons`: a Discoveries offer's `why` carries `reasons` (Track card).
- `queue.calibrate`: `GET /v1/state` carries `queues.calibrate`, the pool tracks haystack is least sure about
  (ADR 0015). A Verdict there is an ordinary Verdict, sent with `"queue": "calibrate"`. A Calibrate card's `why`
  carries `calibrate` and no `band` (Track card).
- `plays.ms_played`: the server stores a play's `ms_played` and `completed` and learns from them (ADR 0016): a
  short, unfinished play of an offered or Calibrate track with no Verdict counts a little against it.
- `card.label`: a track card carries `label`, the record label Deezer gives for the track's album (ADR 0017).
- `card.song`: a track card carries `song`, which tells the releases of one recording apart from other songs (ADR
  0021). A Verdict on one release counts for all of them: a playlist card's `verdict`, `playlists` and `liked`,
  `playlists.verdicts` in `GET /v1/state` and `queues.likes` go by the Song (Track card).
- `card.original`: a track card carries `original` and `original_check`, which say whether the track may stand in
  for another recording, a cover or a live take, and which one (ADR 0022, Track card).
- `swaps`: `POST /v1/swaps` replaces a cover with its Original in the listener's own playlists and Liked Songs and
  takes back the cover's Verdicts (ADR 0022).

## Endpoints

| Method | Path | Result |
|---|---|---|
| GET | `/v1/health` | `{"ok": true, "version": "1", "time": "…Z", "server": <server>}` |
| GET | `/v1/state` | the snapshot a Review needs, in one call |
| POST | `/v1/verdicts` | records Verdicts; idempotent by id |
| POST | `/v1/plays` | records plays; idempotent by track and start time |
| POST | `/v1/swaps` | records Swaps; idempotent by id |
| GET | `/v1/tracks/{id}` | one track card; `202` while the track is being measured |
| GET | `/v1/playlists/{id}` | one playlist with a track card per song |
| PUT | `/v1/playlists/{id}/tracks` | the songs of a playlist, as the client read them |

### `GET /v1/state`

```json
{"generated_at": "…Z",
 "baseline": {"tracks": 412, "traits": {"tempo": {"mean": 118.2, "lo": 98.0, "hi": 128.0, "n": 380}, "…": null}},
 "queues": {
   "discoveries": {"total": 25, "done": 3, "next_batch_at": "…Z", "items": [<track card>, …]},
   "calibrate": {"total": 30, "done": 4, "items": [<track card>, …]},
   "likes": {"total": 140, "items": [<track card>, …]}},
 "playlists": {"synced_at": "…Z", "items": [<playlist summary>, …], "verdicts": {"<track id>": <verdict>, …}},
 "spotify": {"pending_writes": 0, "error": null},
 "server": {"version": "0.6.0", "api": "1", "features": ["why.band", "rerank", "why.reasons", "queue.calibrate",
            "plays.ms_played", "card.label", "card.song", "card.original", "swaps"]}}
```

- **`baseline`**: per Trait, over the tracks the listener likes (weighted by stars; an Unrated like counts as 3,
  or up to 5 when the listener plays it often):
  the weighted mean and the 25th–75th percentile (`lo`, `hi`), or `null` for a Trait with no Readings. Traits:
  `tempo`, `valence`, `energy`, `danceability`, `acousticness`, `instrumentalness`, `loudness`.
- **`queues.discoveries`**: the latest batch's offers plus any older offer still open, unreviewed first (in batch
  order), then the reviewed ones with their Verdict. `done` counts those with a Verdict. `next_batch_at` may be
  `null`.
- **`queues.calibrate`** (feature `queue.calibrate`): the Calibrate picks, the pool tracks the server is least
  sure about (ADR 0015). Every pick with no Verdict yet, unreviewed first in pick order, then the picks given a
  Verdict this ISO week, with their Verdict. `done` counts those with a Verdict. The server keeps up to 30 open
  and adds at most 30 a week (`calibrate.outstanding`, `calibrate.per_week`). A pick is never also a Discoveries
  offer. A pick the listener hearts in Spotify with no Verdict leaves this queue for `queues.likes`.
- **`queues.likes`**: Liked Songs with no Verdict, on the track or on another release of its Song (ADR 0021),
  newest like first. A cover whose Swap is still being made on Spotify is left out.
- **`playlists`**: every playlist in the listener's library (minus the server's own Discoveries and To Rip), with
  its songs' ids but no cards, plus the current Verdict of every one of those songs that has one, so a client can
  count progress for any pick offline. A song's Verdict there is its Song's: one given to another release of it
  counts (ADR 0021).
- **`spotify`**: how many tracks and Swaps still have Spotify writes waiting, and the latest error, if any. Writes
  are made in the background after a Verdict or a Swap is accepted, so a Spotify problem shows up here, not as a
  failed request.
- **`server`**: as in `GET /v1/health` (Server and features).

Reading the snapshot also starts measuring, in the background, the next 20 tracks of each queue that have no
Readings, checks the next 20 unreviewed tracks of each queue for a possible cover, and refreshes the playlists at
most every 10 minutes.

### Track card

```json
{"id": "4MbtaySL1453JwADvGryCL", "song": "…", "title": "…", "artists": ["…"], "album": "…", "album_type": "single",
 "year": 2018, "label": "…", "duration_ms": 179000, "cover": "https://…", "cover_small": "https://…",
 "verdict": {"kind": "rating", "stars": 4, "given_at": "…Z"},
 "why": {"sounds_like": {"title": "…", "artist": "…"}, "similarity": 0.96, "source": "deezer_artist_radio",
         "source_label": "Deezer artist radio", "exploration": false,
         "band": {"kind": "top", "percentile": 98.4}, "model": "rerank", "confidence": 0.81,
         "reasons": ["close to your 4-5★ tracks", "in styles you like"]},
 "styles": [["Dubstep", 0.25], ["Bassline", 0.11], ["Downtempo", 0.10]],
 "readings": {"tempo": 102.0, "valence": 0.14, "energy": 0.83, "danceability": 0.61, "acousticness": 0.02,
              "instrumentalness": 0.77, "loudness": -7.1},
 "measure": "ok",
 "original": {"title": "Zombie", "artist": "The Cranberries", "id": "7EZC6E7UjZe63f1jRmkWxt", "reason": "marker"},
 "original_check": "done"}
```

- Any of `title`, `album`, `album_type`, `year`, `duration_ms`, `cover`, `cover_small` may be `null`; `artists`
  may be empty.
- `label` (feature `card.label`): the record label Deezer gives for the track's album, as Deezer writes it.
  `null` when the server doesn't know it: the track has no Deezer match, Deezer gives no label, or the album
  hasn't been fetched yet (`haystack metadata backfill`, and each daily harvest).
- `song` (feature `card.song`): the track's Song, the ISRC of its recording. Every release of a recording (the
  album track, a remaster, a deluxe edition, a best-of) has its own track id but the same `song`, so two cards
  with the same `song` are one song to the listener. `null` when the server knows no ISRC for the track: it is a
  Song of its own. A client compares it and never parses it.
- `verdict` is the current Verdict, or `null`. On a Discoveries or Calibrate card it is the track's own, since an
  offer or a pick is closed by a Verdict on that track. On any other card it is the Song's: the latest current
  Verdict of any track with the same `song` (ADR 0021).
- `why` is set only for a Discoveries offer or a Calibrate pick: the liked track it sounded most like, the raw
  similarity, and where the candidate came from. `null` otherwise.
- `why.band` (feature `why.band`): which part of the ranking the offer was picked from. `kind` is `top` (the
  best-scored picks) or `explore` (a random pick from the middle of the ranking, the same as `exploration`), and
  `percentile` is the offer's score percentile, 0–100, in the pool its batch was picked from. `percentile` is
  `null` for offers made before haystack 0.2.0.
- `why.model` (feature `rerank`): `rerank` when the re-ranker ordered this offer (a top pick in a batch where it
  passed its check against similarity alone, ADR 0011), else `similarity`. `why.confidence`: the re-ranker's
  chance, 0–1, that the listener keeps the track; `null` when `model` is `similarity` (always for an exploration
  pick). A client may show it as a percentage.
- `why.reasons` (feature `why.reasons`): up to two short phrases, strongest first, for what made the re-ranker
  rate the offer up, such as `close to your 4-5★ tracks` or `brighter than you usually like`. Empty when `model`
  is `similarity`. Phrases are display text and may change between releases; a client never parses them.
- `why.calibrate` (feature `queue.calibrate`), on a Calibrate card only, which has no `band`:
  `{"band": "90-97", "percentile": 93.1, "chance": 0.52, "reason": "the re-ranker gives it even odds"}`. `band`
  names the part of the ranking it was picked from (stage-1 percentiles, `calibrate.bands`), `percentile` its
  score percentile then, `chance` the re-ranker's chance that the listener keeps it (`null` when no model was
  fitted), and `reason` a short phrase for why the server is unsure, display text a client never parses. `model`,
  `confidence` and `reasons` show the re-ranker's view whenever a model was fitted, used for a batch or not.
- `styles`: the top three Discogs styles with their scores; empty when unmeasured.
- `readings`: only the Traits that were measured; `null` when none. `tempo` is BPM, `loudness` integrated LUFS,
  the rest 0–1.
- `measure`: `ok` (has Readings), `pending` (being measured now), `no_preview` (nothing to measure) or `unknown`
  (not measured yet).
- `original` (feature `card.original`): the recording this track may stand in for, its Original, or `null` when
  the server found no sign of one. It is a hint for the listener, never a Verdict (ADR 0022). `title` is the
  Original's title; `artist` its artist, `null` when the server couldn't tell; `id` the Original on Spotify, `null`
  when a search didn't find it (no Swap is possible then); `reason` the sign that was found: `marker` (the title
  says cover, karaoke, tribute, "originally performed by" and the like, or the artist is a karaoke or tribute act),
  `live` (a live, unplugged or BBC recording: the Original is the same artist's studio one), `isrc` (Deezer files
  the track's ISRC under another artist's track of the same title) or `rank` (another artist's track of the same
  title is at least three times as popular on Deezer). A client shows `reason` in its own words.
- `original_check` (feature `card.original`): `done` (checked: `original` is the answer), `pending` (being checked
  now; fetch the card again later) or `unknown` (not checked yet, or the last check failed and is tried again after
  a day). `original` is `null` unless it is `done`.

Cards from `GET /v1/playlists/{id}` add `playlists` (the ids of the listener's playlists the song is on, in
library order), `liked` (in Liked Songs) and `in_library` (certainly in the local music library already). The first
two go by the Song: a playlist or a like of another release of it counts. `in_library` is the track's own: a file
tagged with its Spotify id or its ISRC, or matched to its like. A playlist with two releases of a song has a card for
each; a client that lists each song once merges them by `song`.

### Verdict

`{"kind": "rating" | "pass" | "dislike", "stars": 1..5 | null, "given_at": "…Z"}`. A track's current Verdict is
its Verdict with the latest `given_at` (on a tie, the one the server received last). A Song's current Verdict is
the latest of the current Verdicts of its tracks, on a tie again the one received last (ADR 0021).

### `POST /v1/verdicts`

```json
{"verdicts": [{"id": "0b6c…", "track": "4MbtaySL1453JwADvGryCL", "kind": "rating", "stars": 4,
               "given_at": "2026-10-12T09:14:03.120Z", "queue": "discoveries"}]}
```

→ `{"accepted": ["0b6c…"], "rejected": [{"id": "…", "error": "…"}]}`

- At most 500 per request; the body at most 1 MiB.
- `id`: made by the client, 8–64 characters. Sending the same id again is a no-op, so a client resends its whole
  outbox after a dropped connection.
- `given_at`: when the listener gave it, by the client's clock. It orders the history, so a late resend never
  overrides a newer Verdict.
- `stars`: 1–5 for a `rating`, absent or `null` otherwise.
- `queue` (optional): `discoveries`, `likes`, `playlists` or `calibrate`.
- Each item is checked on its own: one bad item is rejected with a reason and the rest are still accepted.

### `POST /v1/plays`

```json
{"plays": [{"id": "…", "track": "4MbtaySL1453JwADvGryCL", "started_at": "…Z", "ms_played": 61000,
            "completed": false}]}
```

→ `{"accepted": [ids], "rejected": [{"id": "…", "error": "…"}]}`. At most 2000 per request. `id`, `track` and
`started_at` are required; the other fields are optional. A play counts an offer as played; a second play of the
same track with the same `started_at` is a no-op.

- `ms_played` (feature `plays.ms_played`): how long it played, a non-negative integer in milliseconds.
  `completed`: `true` when it played to the end. Either may be absent or `null`, which counts as absent; any other
  value rejects that play with a reason. A play that stopped before 30 s without finishing, of an offered or
  Calibrate track with no Verdict, is a soft no for the re-ranker (ADR 0016).

### `POST /v1/swaps`

Feature `swaps`. The listener's decision that a track is a cover (or a live take) and its Original should take its
place (ADR 0022).

```json
{"swaps": [{"id": "5d1e…", "track": "<the cover's id>", "original": "<the Original's id>",
            "given_at": "2026-10-23T09:14:03.120Z"}]}
```

→ `{"accepted": ["5d1e…"], "rejected": [{"id": "…", "error": "…"}]}`

- At most 500 per request; the body at most 1 MiB. `id` (8–64 characters, made by the client) and `given_at` work
  as for Verdicts: sending the same id again is a no-op. `track` and `original` are different track ids;
  `original` is normally the card's `original.id`, but any track is accepted.
- At once: the cover's Verdicts given up to `given_at` are taken back, so it is unreviewed again, and a Verdict on
  it given before the Swap that arrives later is accepted and dropped. The server keeps what it took back.
- In the background, as for Verdicts (`spotify` in the snapshot): in each playlist the listener owns that holds the
  cover, the Original goes where the cover first was, unless its Song is there already, and every copy of the
  cover comes out. If the cover is in Liked Songs, its heart moves to the Original (unless the Original's Song has
  a Verdict below 3★) and the cover leaves To Rip if the server put it there. Followed playlists, Discoveries and
  the cover's plays are left alone.

### `GET /v1/tracks/{id}`

One track card, for any Spotify track. A track the server has never measured is queued for measuring and answered
`202` with `"measure": "pending"`; ask again later. A track with no preview stays `no_preview`. A Discoveries offer
or a Calibrate pick comes with its `why`. A track not yet checked for a possible cover is checked ahead of the
background checks, with `"original_check": "pending"` (the status stays `200` when only that is pending).

### `GET /v1/playlists/{id}`

A playlist summary plus `items`, a track card per song in playlist order. `404` for a playlist not in the
listener's library. Every song not yet checked for a possible cover is queued for a check.

Playlist summary:

```json
{"id": "…", "name": "…", "cover": "https://…", "mine": true, "owner": "…", "total": 87, "snapshot_id": "…",
 "tracks": ["<track id>", …], "current": true, "reader": "server", "artists": ["…", "…", "…"]}
```

- `total` is Spotify's song count; `tracks` the stored songs (each once), or `null` when none are stored yet.
- `current`: the stored songs are the playlist's current version (`snapshot_id`).
- `reader`: who reads its songs. `server` (the Web API), or `engine` when Spotify refuses the server and the
  client should read them itself and send them with `PUT /v1/playlists/{id}/tracks` (ADR 0008).
- `artists`: the three most frequent first artists.

### `PUT /v1/playlists/{id}/tracks`

```json
{"snapshot_id": "…", "tracks": [{"id": "…", "title": "…", "artists": ["…"], "album": "…", "album_type": "album",
  "release_date": "2018-03-02", "duration_ms": 179000, "cover": "https://…", "cover_small": "https://…",
  "isrc": "…"}]}
```

→ `{"stored": 87, "snapshot_id": "…"}`

- At most 10,000 songs; the body at most 8 MiB. All or nothing: one bad song rejects the request with `400`
  naming it, and nothing is stored.
- `id`, `title` and `artists` are required. `album_type` is `album`, `single` or `compilation`; `release_date` is
  `YYYY`, `YYYY-MM` or `YYYY-MM-DD`; covers are https URLs.
- `404` for a playlist not in the listener's library. This never writes the playlist to Spotify; the only edit
  haystack makes to the listener's own playlists is a Swap's.

## Changes

One line per addition to `/v1`: the date, the haystack release that has it, and its feature name, if any.
`tests/fixtures/v1-contract.json` lists every documented key and type; a change may add to it, never remove.

- 2026-10-19, haystack 0.2.0: the `server` object on `GET /v1/health` and `GET /v1/state`, with its `features`.
- 2026-10-19, haystack 0.2.0, `why.band`: `band` (`kind`, `percentile`) in a track card's `why`.
- 2026-10-19, haystack 0.3.0, `rerank`: `model` and `confidence` in a track card's `why`.
- 2026-10-19, haystack 0.3.0, `why.reasons`: `reasons` in a track card's `why`.
- 2026-10-20, haystack 0.5.0, `queue.calibrate`: `queues.calibrate` in `GET /v1/state`, `calibrate` as a
  Verdict's `queue`, and `calibrate` in a Calibrate card's `why`.
- 2026-10-20, haystack 0.5.0, `plays.ms_played`: the server stores a play's `ms_played` and `completed`.
- 2026-10-20, haystack 0.6.0, `card.label`: `label` in a track card.
- 2026-10-22, haystack 0.8.0, `card.song`: `song` in a track card; a playlist card's `verdict`, `playlists` and
  `liked`, `playlists.verdicts` and `queues.likes` go by the Song.
- 2026-10-23, haystack 0.9.0, `card.original`: `original` and `original_check` in a track card.
- 2026-10-23, haystack 0.9.0, `swaps`: `POST /v1/swaps`; `spotify.pending_writes` counts Swaps too.
