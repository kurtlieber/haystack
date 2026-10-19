# HTTP API (v1)

The contract between the server and its clients (needle today). The server owns it: a client depends on this
document, never on the server's code or database. Why it looks this way: ADR 0004 (a private HTTP API), ADR 0005
(Verdicts), ADR 0006 (one snapshot plus idempotent writes).

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
{"version": "0.4.0", "api": "1", "features": ["why.band", "rerank", "why.reasons"]}
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

## Endpoints

| Method | Path | Result |
|---|---|---|
| GET | `/v1/health` | `{"ok": true, "version": "1", "time": "…Z", "server": <server>}` |
| GET | `/v1/state` | the snapshot a Review needs, in one call |
| POST | `/v1/verdicts` | records Verdicts; idempotent by id |
| POST | `/v1/plays` | records plays; idempotent by track and start time |
| GET | `/v1/tracks/{id}` | one track card; `202` while the track is being measured |
| GET | `/v1/playlists/{id}` | one playlist with a track card per song |
| PUT | `/v1/playlists/{id}/tracks` | the songs of a playlist, as the client read them |

### `GET /v1/state`

```json
{"generated_at": "…Z",
 "baseline": {"tracks": 412, "traits": {"tempo": {"mean": 118.2, "lo": 98.0, "hi": 128.0, "n": 380}, "…": null}},
 "queues": {
   "discoveries": {"total": 25, "done": 3, "next_batch_at": "…Z", "items": [<track card>, …]},
   "likes": {"total": 140, "items": [<track card>, …]}},
 "playlists": {"synced_at": "…Z", "items": [<playlist summary>, …], "verdicts": {"<track id>": <verdict>, …}},
 "spotify": {"pending_writes": 0, "error": null},
 "server": {"version": "0.4.0", "api": "1", "features": ["why.band", "rerank", "why.reasons"]}}
```

- **`baseline`**: per Trait, over the tracks the listener likes (weighted by stars; an Unrated like counts as 3):
  the weighted mean and the 25th–75th percentile (`lo`, `hi`), or `null` for a Trait with no Readings. Traits:
  `tempo`, `valence`, `energy`, `danceability`, `acousticness`, `instrumentalness`, `loudness`.
- **`queues.discoveries`**: the latest batch's offers plus any older offer still open, unreviewed first (in batch
  order), then the reviewed ones with their Verdict. `done` counts those with a Verdict. `next_batch_at` may be
  `null`.
- **`queues.likes`**: Liked Songs with no Verdict, newest like first.
- **`playlists`**: every playlist in the listener's library (minus the server's own Discoveries and To Rip), with
  its songs' ids but no cards, plus the current Verdict of every one of those songs that has one, so a client can
  count progress for any pick offline.
- **`spotify`**: how many tracks still have Spotify writes waiting, and the latest error, if any. Writes are made
  in the background after a Verdict is accepted, so a Spotify problem shows up here, not as a failed request.
- **`server`**: as in `GET /v1/health` (Server and features).

Reading the snapshot also starts measuring, in the background, the next 20 tracks of each queue that have no
Readings, and refreshes the playlists at most every 10 minutes.

### Track card

```json
{"id": "4MbtaySL1453JwADvGryCL", "title": "…", "artists": ["…"], "album": "…", "album_type": "single",
 "year": 2018, "duration_ms": 179000, "cover": "https://…", "cover_small": "https://…",
 "verdict": {"kind": "rating", "stars": 4, "given_at": "…Z"},
 "why": {"sounds_like": {"title": "…", "artist": "…"}, "similarity": 0.96, "source": "deezer_artist_radio",
         "source_label": "Deezer artist radio", "exploration": false,
         "band": {"kind": "top", "percentile": 98.4}, "model": "rerank", "confidence": 0.81,
         "reasons": ["close to your 4-5★ tracks", "in styles you like"]},
 "styles": [["Dubstep", 0.25], ["Bassline", 0.11], ["Downtempo", 0.10]],
 "readings": {"tempo": 102.0, "valence": 0.14, "energy": 0.83, "danceability": 0.61, "acousticness": 0.02,
              "instrumentalness": 0.77, "loudness": -7.1},
 "measure": "ok"}
```

- Any of `title`, `album`, `album_type`, `year`, `duration_ms`, `cover`, `cover_small` may be `null`; `artists`
  may be empty.
- `verdict` is the track's current Verdict, or `null`.
- `why` is set only for a Discoveries offer: the liked track it sounded most like, the raw similarity, and where
  the candidate came from. `null` otherwise.
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
- `styles`: the top three Discogs styles with their scores; empty when unmeasured.
- `readings`: only the Traits that were measured; `null` when none. `tempo` is BPM, `loudness` integrated LUFS,
  the rest 0–1.
- `measure`: `ok` (has Readings), `pending` (being measured now), `no_preview` (nothing to measure) or `unknown`
  (not measured yet).

Cards from `GET /v1/playlists/{id}` add `playlists` (the ids of the listener's playlists the song is on, in
library order), `liked` (in Liked Songs) and `in_library` (certainly in the local music library already).

### Verdict

`{"kind": "rating" | "pass" | "dislike", "stars": 1..5 | null, "given_at": "…Z"}`. A track's current Verdict is
its Verdict with the latest `given_at` (on a tie, the one the server received last).

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
- `queue` (optional): `discoveries`, `likes` or `playlists`.
- Each item is checked on its own: one bad item is rejected with a reason and the rest are still accepted.

### `POST /v1/plays`

```json
{"plays": [{"id": "…", "track": "4MbtaySL1453JwADvGryCL", "started_at": "…Z", "ms_played": 61000,
            "completed": false}]}
```

→ `{"accepted": [ids], "rejected": [{"id": "…", "error": "…"}]}`. At most 2000 per request. `id`, `track` and
`started_at` are required; the other fields are optional. A play counts an offer as played; a second play of the
same track with the same `started_at` is a no-op.

### `GET /v1/tracks/{id}`

One track card, for any Spotify track. A track the server has never measured is queued for measuring and answered
`202` with `"measure": "pending"`; ask again later. A track with no preview stays `no_preview`.

### `GET /v1/playlists/{id}`

A playlist summary plus `items`, a track card per song in playlist order. `404` for a playlist not in the
listener's library.

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
- `404` for a playlist not in the listener's library. Playlists themselves are never written to Spotify.

## Changes

One line per addition to `/v1`: the date, the haystack release that has it, and its feature name, if any.
`tests/fixtures/v1-contract.json` lists every documented key and type; a change may add to it, never remove.

- 2026-10-19, haystack 0.1.0: the `server` object on `GET /v1/health` and `GET /v1/state`; `features` is empty.
- 2026-10-19, haystack 0.2.0, `why.band`: `band` (`kind`, `percentile`) in a track card's `why`.
- 2026-10-19, haystack 0.3.0, `rerank`: `model` and `confidence` in a track card's `why`.
- 2026-10-19, haystack 0.3.0, `why.reasons`: `reasons` in a track card's `why`.
