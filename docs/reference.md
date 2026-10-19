# haystack reference

The technical side of haystack: how it works, every command, the scoring details, the two ways to run it,
secrets, sign-in and alerts. To install and configure it, start with the [README](../README.md). For the same
ideas without the jargon, see [How haystack works](how-it-works.md). needle's API is in [api.md](api.md); the
decisions behind the design are in [adr/](adr/).

## Where things live

| | Container | Without a container |
|---|---|---|
| Config: `config.toml`, `secrets/` | `/config` | `~/.config/haystack/` |
| Data: `discovery.db`, `embeddings/`, `models/` | `/data` | `~/.local/share/haystack/` |
| State: `status.json`, `first-batch.done`, throttle files, auth log | `/data/state` | `~/.local/state/haystack/` |
| Logs | `docker compose logs` | the journal (`journalctl --user -u haystack-*`) |

`HAYSTACK_CONFIG_DIR`, `HAYSTACK_DATA` and `HAYSTACK_STATE` move the three directories (the image sets them);
`HAYSTACK_CONFIG` names a config file elsewhere. Defaults for every setting are in
`src/haystack_music/config.py`.

## How it works

```
library.root (read-only) --scan--> SQLite: local_tracks --embed (9x30 s windows)--> embeddings/local/*.npz
Spotify (likes, plays) --sync--> likes / plays / liked_map (ID tag -> ISRC -> fuzzy) --> positives
Deezer + ListenBrainz --harvest--> candidates (+ source) --preview, in memory only--> deezer_vectors (512/1280-d)
score = max cosine(candidate preview, any positive window) -> weekly batch -> Discoveries -> outcomes -> like rate
```

| Module | Role |
|---|---|
| `library.py` | read-only scan (tags, normalised artists/title, version suffix, duration, Spotify ID tag, audio-content hash); resumable multi-process embedding |
| `embed.py`, `windows.py` | EffNet wrapper (one forward pass -> 512-d similarity head + 1280-d); ~9 windows of 30 s, one every ~10 % |
| `spotify.py` | **the only code that talks to Spotify**: token refresh + rotation write-back, 429/Retry-After, saved tracks, recently-played, `/me/playlists`, `/playlists/{id}/items` (add/remove with `snapshot_id`), `/me/library/contains`, search (limit 10) |
| `secrets.py` | the secrets: 0600 files (default) or 1Password via `op`; a systemd credential overrides either |
| `matching.py`, `textnorm.py` | Spotify ID -> ISRC -> fuzzy (artist + title + ±3 s + version-suffix check) |
| `deezer.py`, `listenbrainz.py`, `harvest.py` | candidate sources (reach only, never a score), 1 req / 1.1 s cross-process throttle |
| `vectors.py`, `scoring.py` | positives assembly, nearest-positive scoring, batch selection (1 per artist, ~20 % exploration), like rates |
| `sync.py`, `batch.py`, `clear.py` | hourly sync, weekly batch, and the pre-batch Discoveries clear |
| `verdicts.py` | Your Verdicts (1-5 star Ratings, Pass, Dislike) on any track, append-only (ADR 0005), and the Spotify writes they imply: the heart, Discoveries, To Rip (ADR 0002) |
| `traits.py`, `baseline.py` | Readings of the seven Traits from the Deezer preview (ADR 0001), backfill and on-demand measuring; the Baseline and usual range |
| `review.py`, `api.py` | needle's private API (ADR 0004, ADR 0006): track cards, the Review queues, verdicts and plays in |
| `playlists.py` | Your playlists for needle's Playlist review: listed hourly, songs re-read when a playlist changes; Followed playlists' songs come from needle's engine (ADR 0008). Read-only towards Spotify |
| `status.py` | `status.json` in the state dir, `notify-send` (no-op without a desktop session) |
| `scheduler.py` | `haystack serve`'s job schedule (`[schedule]`), in place of the systemd timers |
| `alert.py` | optional webhook alerts (`haystack:` prefix) |


### Key rules
* **Library is read-only.** Files are opened read-only and tags are only read. The audio-content hash
  (length + 4 samples of the `mdat` payload) is the fallback identity when a path or size changes (moves keep
  their embedding); mtime is never used. A 22-char **Spotify track ID** in the `MusicBrainz Track Id` tag (some
  rippers write it there) is the primary mapping key, then the ISRC, then a fuzzy match.
* **Embeddings.** The primary vector is the 512-d track-similarity head; per-window 1280-d means are kept too
  (float16, same forward pass). A match is scored by the best window.
* **Deezer previews** (see the README's Deezer notice): at most 1 request per 1.1 s across all
  processes; preview audio is held only in memory and piped into ffmpeg, never written to disk (not even a temp
  file); only vectors are kept, locally.
* **Playlists.** Exactly one private "Discoveries" and one private "To Rip" (found by name or created once; ids
  cached). Every bot addition is recorded in `playlist_adds`. The hourly sync only removes tracks the bot added,
  so a hand edit during the week is not overwritten. A track you remove from Discoveries yourself counts as a
  dismissal. The exception is the clear step before each batch, which removes every track in Discoveries.
* **Outcomes.** Liked discovery -> removed from Discoveries at once, positive (higher weight), added to To Rip
  if not in the library. After 14 days unliked -> removed: played = negative; unplayed = back to the pool with no
  penalty, retired after 2 unplayed expiries.
* **Clear before each batch.** `batch` and `first-batch` empty Discoveries before adding tracks. The offer
  outcome is committed before the Spotify delete, so the hourly sync cannot mark the removal
  `manual_removed`. A track in Liked Songs keeps that positive: the offer is closed as `liked` and no verdict
  is written. Every other track that has been on the playlist for at least `batch.clear_min_age_hours`
  (default 24) gets a normal `no` in `feedback` (source `playlist`, weight 1) and the offer is closed as
  `negative`. Discoveries is meant to be heard through the playlist, so a track you did not like is a no, played
  or not.
  Younger tracks follow `batch.clear_recent`: `remove` (default; the weekly and first batch pin it) takes them off and closes
  the offer as `unplayed` (back in the pool, no negative weight), `keep` leaves them for a later run.
  `--no-penalty` closes every removed offer except likes that same unplayed way. A batch skipped because the
  library embedding is incomplete does not clear.
* **Unheard by design.** `recently-played` only returns the last 50 plays. The server is always on and syncs
  hourly, so plays are only missed after more than ~50 plays in one hour or a long server/network outage; such
  tracks count as "unplayed" (the conservative direction). Not a bug.
* **Metric.** Like rate, overall and per source (`haystack sources`), and the fuller `haystack report`. A source
  can't be retired before 30 offered tracks. No classifier and no skip inference in v1 (classifier after ~200
  played discoveries; skips later from the Extended Streaming History export).
* **Spotify dev mode (2026).** No audio-features / recommendations / browse / artist top tracks / batch track
  fetch; search capped at 10. None of these are used.

## Commands

In the container, put `docker compose exec haystack` in front (or `docker compose run --rm haystack` without the
leading `haystack` while the server is stopped).

```
haystack serve [--no-api]            needle's API plus the scheduled jobs ([schedule]); the container runs this
haystack auth [--timeout 900]        Spotify sign-in: open the printed URL anywhere, paste back the address
haystack library scan|embed|update   index / embed / both + To Rip check  [--workers N --limit N]
haystack harvest [--sources a,b] [--limit-per-source N]
haystack score                       re-score the candidate pool
haystack sync                        hourly job (needs auth)
haystack batch [--size N] [--dry-run] [--force] [--variant V] [--no-penalty]
                                              [--clear-min-age-hours H] [--clear-recent keep|remove]
                                              weekly job (needs auth): clear Discoveries, then add a batch
haystack batch --plan [--size N] [--variant V]   the would-be picks, scored in memory: no Spotify, no writes
haystack clear-discoveries [--dry-run] [--no-penalty] [--clear-min-age-hours H] [--clear-recent keep|remove]
                                              empty Discoveries and record nos (also run by batch)
haystack eval [--seeds 1,2,3] [--choose]  offline scoring-variant eval (no Spotify calls)
haystack genres [--top 15]           genre makeup of the liked songs (400 Discogs styles)
haystack harvest --broad [N] [--shard i/N] [--no-retry] [--styles N]   stratified broad harvest
haystack feedback yes|no ARTIST TITLE [--strength mild|normal|strong] [--note T] [--to-rip]
                                              record a manual verdict (never re-offered; yes = positive)
haystack feedback list|rip           list verdicts / add manual YES tracks to To Rip (bot-added)
haystack first-batch [--no-penalty] [--clear-min-age-hours H] [--clear-recent keep|remove]
                                              one-time first batch + taste sanity check; clears first
haystack sanity-check [-n 10 -k 10]  write sanity-check.md in the state dir
haystack status                      status.json + live counts
haystack review [list|accept ID..|reject ID..]   uncertain liked-song -> library matches
haystack sources [list|retire NAME]  like rate per source
haystack report [--tracks] [--json]  like rates, Verdict counts, the pool's make-up, latest eval (read-only)
haystack check-secrets [--write]     which secrets are set (lengths only); --write: the token record is writable
haystack secret set NAME             store a secret, from a hidden prompt or stdin
haystack token new [--force]|show    needle's API token: make one / print it for a client
haystack verdict give TRACK 1-5|pass|dislike [--no-apply]   a Verdict from the shell (needle is the usual way)
haystack verdict list|apply          current verdicts / make pending Spotify writes now
haystack verdict export FILE         every verdict, with title and artists, to a JSON file (0600)
haystack verdict import FILE [--pending]   add a file's verdicts (by id: importing it again adds nothing)
haystack models fetch|check          the Essentia models: download the missing ones (into models/)
haystack traits backfill [--limit N] Readings for stored previews (Discoveries and likes first)
haystack traits measure|show TRACK   Readings for one Spotify track (measure: on demand)
haystack baseline                    your Baseline per Trait (mean, usual range)
haystack api                         needle's API alone (the haystack-api unit runs it)
```
`feedback yes|no` still works: yes is a 3-star Rating, no a Pass (`--strength strong`: a Dislike).

**Moving verdicts.** `verdict export` writes every Verdict (history included) with the track's title, artists,
album, ISRC and duration, so the file is readable and can be re-matched if a Spotify id changes. `verdict import`
adds them by id, all or none. By default it makes no Spotify writes (a restore to the same account already has the
hearts and playlists). `--pending` leaves them for the next sync (or `verdict apply`), e.g. for a new account.
Spotify jobs exit with code **3** and `status.json` -> `auth.ok=false` when no refresh token exists.

Sources: `deezer_chart`, `deezer_editorial` (genre charts), `deezer_new_releases` (editorial releases),
`deezer_artist_radio`, `deezer_related` (related artists' top tracks) and `listenbrainz_similar` (labs
`recording-search` + `similar-recordings`; the keyless `api.listenbrainz.org` lookup/radio endpoints now need a
token). Seeds are the positives: recent likes first, then library tracks.

## Scoring and batches

The settings below are rarely changed; the README's configuration guide covers the common ones.

**Scoring variants** (`scoring.variant`): `raw` (max cosine to the nearest positive), `topk_mean` (mean cosine to
the k=10 nearest positive tracks, best window each), `genre` (mean cosine between the candidate's
genre_discogs400 style vector and its k=10 most similar liked songs' style vectors), `genre_blend` (50/50 z-scored
raw + genre), `centred` (subtract the
mean of library + pool vectors, re-normalise), `csls` (raw minus the candidate's mean cosine to its k=10 nearest
vectors in a fixed, seeded random reference set of library + pool vectors: penalises hubs that are close to
everything) and `csls_centred`. `auto` (default) uses the variant the first-batch eval chose (kv
`scoring_variant`), else `raw`; set an explicit name to pin it.

**Negative penalty** (`scoring.negative_penalty`, applied after whichever variant scores the pool, so the weekly
batch and the first batch both get it): candidates that sound like tracks you rejected rank lower.
`final = score - w * sd(score over the pool) * max_j(strength_j * excess_j)`, with
`excess_j = max(0, sim_j - margin_j) / (1 - margin_j)`: `sim_j` is the similarity to rejected track j in the
variant's own space (preview embedding, centred for the centred variants; style vector for `genre`; the average of
the two for `genre_blend`) and `margin_j` is the pool's 95th percentile of similarity to j, so anything no more
similar than that is untouched and an exact copy gets excess 1. Strength: manual "no" feedback at its weight (0.5
mild, 1 normal, 2 strong), a Discoveries-clear "no" at normal weight (source `playlist`), played-but-not-liked
expiries `weight_expired` (1), tracks removed from Discoveries by hand `weight_removed` (1). Default `w = 0.75`: a near-copy of a strong no loses ~1.5 score-sd (out of the
top 15), a normal one ~0.75 sd. `candidates.neg_penalty` / `nearest_negative` record the drop and its cause.
Turn it off with
```toml
[scoring.negative_penalty]
w = 0
```
The eval's held-out queries are unaffected; its sanity table on your verdicts also shows the penalised ranks
(leave-one-out for the rejected tracks).

**Styles.** Each preview also gets a 400-d Discogs style vector: the official `discogs-effnet-bs64-1`
backbone + `genre_discogs400-discogs-effnet-1` head on the same mel patches. The track-similarity model's
1280-d output is *not* compatible with the head (tested: correlation ~0 with the official pipeline), hence the
second backbone pass. Library files have no style vectors (a second full pass would take hours), so the genre
variants use the liked songs' previews as positives. Older previews are backfilled by the daily harvest.

**Broad harvest.** `harvest --broad 150` seeds artist radio and related artists from 150 artists stratified over
genre buckets (sqrt-weighted, liked songs first) and ListenBrainz similar-recordings from every liked song;
`--shard i/N` splits the seeds across parallel processes (the 1.1 s Deezer throttle is shared). Chart +
editorial-chart candidates are capped at `harvest.chart_max_share` (20 %) of the pool.

**Eval** (`haystack eval`): hold out 20 % of the liked songs (5 seeds), remove them (and any library
file of the same song) from the positives, and rank each among the harvested pool. Reports mean percentile
(50 = chance), recall@15, and the Deezer-chart share of the pool's top 15. Queries are the liked songs' Deezer
*preview* vectors, the same audio domain as the candidates (the sync fetches previews for all likes for this;
scoring still uses the local file for locally matched likes). Local-file-only queries are shown as a
diagnostic and never drive the choice: a local window is closer to the local-file positives than any preview,
and centring/CSLS amplify that domain gap. The first batch runs the eval, picks the best mean percentile (ties
within 0.5 points go to the simpler variant), and logs the table to `sanity-check.md` and `status.json`
(`scoring_eval`).

**Embedding guard.** A real batch skips (exit 0, `status.json` job message `skipped: library embedding
incomplete: ...`) until at least `batch.min_embedded_fraction` (default 0.98) of library files are embedded or
have given up; positives would be thin otherwise. A skipped batch does not clear Discoveries. `--dry-run` still
runs (with a warning) and makes no Spotify writes: it prints the clear plan (counts only) but removes nothing
and records no verdicts, no playlist is created, nothing is added, and no batch/offer rows are stored. `--force`
overrides the guard.

**Planning a batch.** `batch --plan` runs the batch's scoring and selection with no Spotify session and a read-only
database: scores are computed in memory (the stored scores are left alone), and no run is recorded. It prints each
would-be pick with its score, whether it is an exploration pick, its source, the positive it scored closest to and
the variant. A candidate with no Spotify id yet is marked `unmapped` instead of being looked up, so the picks can
differ from a real batch's only where a lookup would have failed or found a known track, or where Spotify reports a
pick already saved in Liked Songs (a real batch then drops it). Both paths share one selection function
(`batch.select_for_batch`), so with the same scores and seed they pick the same candidates. It ignores the
embedding guard and says so when the library is not fully embedded.

**Report.** `haystack report` is token-free and opens the database read-only. Offers: offered, decided (an outcome
is set), judged (decided, minus tracks that went unplayed and have no Verdict), kept (the current Verdict keeps the
heart, or the offer was liked and has no Verdict), with kept/decided, kept/judged and liked/decided (the
`haystack sources` rate), overall, exploit vs exploration, per batch and per source. Then the current Verdicts by
kind and stars and how many Liked Songs are rated; the pool (size, embedded, scored, active and scored variant); of
the top 100 by stored score, how many have a library window as their nearest positive and how many a preview (by owner
kind: like, discovery, manual, rated); the top 30's score range, distinct nearest positives and top-1 Discogs
style; and the latest eval's per-variant table from `status.json`, with judged tracks counted, never named. It
prints counts and shares only, so it can be pasted into an issue; `--tracks` adds the top 30 by name and `--json`
prints everything as JSON.

**Clear Discoveries** (`batch.clear_min_age_hours`, `batch.clear_recent`, `batch.clear_no_penalty`), run at the
start of every batch that actually proceeds, and on its own via `clear-discoveries`. Liked Songs are checked
live (`/me/library/contains`), not from the last sync. The minimum age defaults to 24 hours, measured from the
playlist item's `added_at` (falling back to the offer, then the bot's `playlist_adds` row). A missing timestamp
is treated as old enough to judge and counted as `age_unknown`. Outcomes are written and committed before the
playlist delete. The hourly sync only turns an *active* offer into `manual_removed` when the track is already
gone, so a clear never takes that path.

```toml
[batch]
clear_min_age_hours = 24
clear_recent = "remove"       # or "keep"
clear_no_penalty = false
```

`remove` takes a younger track off and closes its offer as `unplayed`: candidate `state` goes back to `pool`,
`unplayed_expiries` is not incremented, and `vectors.negatives` ignores it. `keep` leaves it in Discoveries.
`--no-penalty` uses that same unplayed close for every removed offer except likes, including tracks past the
age cutoff, and writes no `no` verdict. Likes stay `outcome='liked'` either way. The weekly and first batch
(units and `haystack serve` alike) pin `--clear-recent remove --clear-min-age-hours 24`. A one-off keep is
`haystack batch --clear-recent keep`; a one-off with no negatives is
`haystack batch --no-penalty` (or the same flags on `clear-discoveries`). `--dry-run` on either
command reads the playlist and Liked Songs and logs counts only. Logs never include tokens or track ids.
A clear's `no` is a Pass (verdict source `playlist`, effects `skipped`: the clear makes its own playlist delete),
which the negative penalty reads like any Pass; it does not read `unplayed`. A later clear that finds the track
liked or unplayed takes that Pass back. A track you already gave a verdict (needle, the CLI) keeps it and gets
no Pass, because its heart may simply not be written yet. Tracks with no candidate row are still removed; they
are counted as `unlinked` and get no verdict.

**Verdict weights** (`[scoring.verdicts]`, ADR 0002): every Rating is a positive whose bonus grows with its
stars, `star_bonus = [0.0, 0.0025, 0.005, 0.0075, 0.01]` for 1-5 stars (3 stars = `bonus_discovery_like`); an
Unrated like counts as `unrated_like_stars = 3`. A Pass is a negative of `pass_strength = 1.0` (the old
"normal" no), a Dislike `dislike_strength = 2.0` (the old "strong"), on the negative-penalty scale. Every variant
adds the bonus before taking the nearest / top-k.

## Running it

Two ways, the same jobs. Pick one per server: the container's scheduler doesn't know about the units.

| Job | Default schedule | Does | Needs auth |
|---|---|---|---|
| library | daily 03:15 | rescan + embed new/changed files (+ To Rip check if signed in) | no |
| harvest | daily 05:00 | harvest + embed candidates, re-score | no |
| sync | hourly | plays, likes, outcomes, expiry, To Rip, your playlists, token check | yes |
| batch | Mondays 08:00 | clear Discoveries (`--clear-recent remove --clear-min-age-hours 24`), then add a batch | yes |
| first batch | once | sanity check + scoring eval/variant choice + first batch (clears first, same flags); then writes `first-batch.done` in the state dir | yes |
| API | always on | needle's API, Spotify writes after verdicts, on-demand Readings, your playlists | yes |

Heavy jobs run at nice 19.

### `haystack serve` (the container)

One process: needle's API in a thread, and a scheduler that starts each job as a child process at the times in
`[schedule]` (`hourly`, `daily HH:MM`, `Mon HH:MM` with comma-separated days, or `off`), in local time (`TZ`).

* A run missed while haystack was down starts when it comes back. The last start of each job comes from the
  database's `runs` table, which the units write too, so moving from the units to the container doesn't replay a
  batch. A job that has never run starts at once, except the batch, which waits for its time.
* Until the first batch is done, it follows every good library or harvest run. Without a library, a new install
  gets it the day it is signed in; with one, it waits for the library embedding.
* A job still running at its next time is skipped, not doubled. Different jobs may overlap.
* A failing job (any exit but 3) sends an alert, if alerts are on.
* `docker compose stop` gives running jobs 20 s to stop; they resume or redo their work on the next run.
* `--no-api` runs only the scheduler. Without the `api-token` secret the API stays off with a warning (set it
  with `haystack token new`, then restart).

The image (`Dockerfile`) is `python:3.14-slim` with ffmpeg and tini, linux/amd64 only: the pinned
essentia-tensorflow has no ARM wheel. It runs as uid 1000. The Essentia models are not in it (see License in the
README); `haystack models fetch` puts them in `/data/models`. A version tag (`v0.2.0`) makes the `image` workflow
publish `ghcr.io/kurtlieber/haystack:0.2.0`, `:0.2` and `:latest`.

### systemd user units (no container)

`systemd/` holds the four oneshot services + timers above (all `Persistent=true`, `Nice=19`,
`IOSchedulingClass=idle`), the one-shot `haystack-first-batch.service` (via `OnSuccess=` of the library unit,
`ConditionPathExists=!` on `first-batch.done`) and the always-on `haystack-api.service`. They ignore `[schedule]`.

```
git clone https://github.com/kurtlieber/haystack ~/source/haystack
python3.14 -m venv ~/.local/share/haystack/venv
~/.local/share/haystack/venv/bin/pip install -e "$HOME/source/haystack[embed]"   # needs ffmpeg on the PATH
ln -s ~/.local/share/haystack/venv/bin/haystack ~/.local/bin/haystack
~/source/haystack/scripts/install-units.sh   # symlinks into ~/.config/systemd/user/, daemon-reload; enables nothing
haystack models fetch
systemctl --user enable --now haystack-library.timer haystack-harvest.timer
# after `haystack auth`:
systemctl --user enable --now haystack-sync.timer haystack-batch.timer
systemctl --user enable --now haystack-api.service   # needle
```
Turn on lingering once (`sudo loginctl enable-linger $USER`) so the timers run without a login session. Times
are the server's local time; a unit drop-in (`systemctl --user edit haystack-batch.timer`, override
`OnCalendar=`) moves them. Follow a job: `journalctl --user -u haystack-library -f`. A unit failure runs
`haystack-alert@%n.service` (`OnFailure=`).

## Secrets
`[secrets] backend` picks where the secrets live:

* `files` (the default): one 0600 file per secret in the config dir's `secrets/` (0700; `secrets.dir`). The
  Spotify refresh token and its non-secret dates and scopes share `spotify-token.json` there, replaced whole
  (a new file renamed over the old) on each rotation.
* `1password`: the `op` CLI with a service-account token from `onepassword.token_file` (0600) or a systemd
  credential (`LoadCredential=op-token:<file>` in a unit drop-in). The code ignores any `OP_*` variables in the
  environment and passes the token only to the `op` child process. `[onepassword]` names the vault, items and
  fields (by id, or a name that is valid in an `op://` reference). The image has no `op`; this backend is for
  installs without a container.

| Secret | What it is | Set with |
|---|---|---|
| `spotify-client-id`, `spotify-client-secret` | your Spotify app's credentials | `haystack secret set NAME` |
| `api-token` | needle's bearer token | `haystack token new` |
| `alert-webhook-url`, `alert-webhook-key` | optional, see Alerts | `haystack secret set NAME` |
| refresh token | Spotify's long-lived login | `haystack auth` |

`haystack secret set` takes the value from a hidden prompt or stdin, never the command line. A systemd
credential named like the secret (`$CREDENTIALS_DIRECTORY/<name>`) wins over the backend for all but the refresh
token, which the jobs rewrite. If the backend can't be written at the moment Spotify rotates the refresh token, a
0600 copy is kept in the state dir and stored on the next run. Secret values are never logged.

## Spotify sign-in
Spotify only allows `127.0.0.1` for an http redirect, so the redirect URI (`spotify.redirect_uri`, default
`http://127.0.0.1:8518/spotify/callback`) points at whatever machine your browser runs on, not at the server.
`haystack auth` handles both cases at once:

* It prints the authorize URL and listens on the redirect port. If the browser runs on the server itself (or you
  forwarded the port: `ssh -L 8518:127.0.0.1:8518 <server>`), the login finishes by itself.
* Otherwise the browser lands on a page that can't load. Its address holds the one-time code: copy the whole
  address from the address bar and paste it at the prompt. A pasted address must carry the same `state` as the
  URL just printed; an old or different address is refused and the prompt asks again.

Whichever arrives first wins. The prompt only appears when `haystack auth` runs in a terminal; with no terminal
(a script) it only listens. If the port is taken (the container's `serve` doesn't use it), it falls back to the
prompt alone.

Refresh tokens last ~6 months: from 5 months after the `issued at` date the hourly sync sends an alert (once a
day) and sets `auth.expiry_warning=true` in `status.json`. Re-auth is the same command; it overwrites the token
and issue date. Hearts need the `user-library-modify` scope: a token from before it was added leaves heart writes
waiting (`haystack verdict list`) until the next `haystack auth`.

## Alerts
Off by default. With `[alert] enabled = true` haystack POSTs JSON (`job`, `host`, `status`, `exit_code`,
`message`, `ts`) to the `alert-webhook-url` secret, with `Authorization: Bearer <alert-webhook-key>` when that
secret is set. Messages start with `haystack:`. Sent for: the token expiry warning (once a day),
`auth.ok=false` (once a day), and a failure of the sync, harvest, library or batch job (exit 3, needs auth, is
left to the daily auth alert). Alerting never fails a job. Dry run: `HAYSTACK_ALERT_DRY_RUN=1`. `host` is the
container's `hostname` (compose.yaml sets `haystack`).

## Development
```
git clone https://github.com/kurtlieber/haystack && cd haystack
python3.14 -m venv .venv && .venv/bin/pip install -e ".[embed,test]"
.venv/bin/python -m pytest -q
docker build -t haystack:local .     # the image; compose.yaml with image: haystack:local runs it
```
