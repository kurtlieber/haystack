# haystack

Personal music discovery for one Spotify account, running on an always-on Linux server as systemd *user* units
under linger.
Taste comes from the audio itself: every local track (and every liked song) is embedded with Essentia's
Discogs-EffNet **track** model, candidates from Deezer and ListenBrainz are embedded from their 30 s previews,
and the best-matching ones land in a private **Discoveries** playlist. Liked discoveries that aren't in the
local library go to a private **To Rip** playlist.

## Deezer notice
Every candidate's audio is its 30 s Deezer preview, and most candidates come from Deezer's charts, editorial
picks, artist radio and related artists, through Deezer's public API (no key). Deezer's terms may not allow this
kind of automated use, so decide for yourself before you run the harvest. haystack keeps it gentle: at most one
request per 1.1 s across all its processes, preview audio held only in memory (piped into ffmpeg, never written
to disk), and only the derived vectors kept, locally. `harvest.sources` drops the Deezer sources, but
ListenBrainz candidates still need their Deezer previews.

## Architecture

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
| `status.py` | `~/.local/state/haystack/status.json`, `notify-send` (no-op without a desktop session) |
| `alert.py` | optional webhook alerts (`haystack:` prefix) |

Data: `~/.local/share/haystack/` (`discovery.db`, `embeddings/`, `models/`, `venv/`).
State: `~/.local/state/haystack/` (`status.json`, throttle files, auth log). Logs: the journal.

### Key rules
* **Library is read-only.** Files are opened read-only and tags are only read. The audio-content hash
  (length + 4 samples of the `mdat` payload) is the fallback identity when a path or size changes (moves keep
  their embedding); mtime is never used. A 22-char **Spotify track ID** in the `MusicBrainz Track Id` tag (some
  rippers write it there) is the primary mapping key, then the ISRC, then a fuzzy match.
* **Embeddings.** The primary vector is the 512-d track-similarity head; per-window 1280-d means are kept too
  (float16, same forward pass). A match is scored by the best window.
* **Deezer previews** (see the Deezer notice): at most 1 request per 1.1 s across all
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
  Younger tracks follow `batch.clear_recent`: `remove` (default; both units pin it) takes them off and closes
  the offer as `unplayed` (back in the pool, no negative weight), `keep` leaves them for a later run.
  `--no-penalty` closes every removed offer except likes that same unplayed way. A batch skipped because the
  library embedding is incomplete does not clear.
* **Unheard by design.** `recently-played` only returns the last 50 plays. The server is always on and syncs
  hourly, so plays are only missed after more than ~50 plays in one hour or a long server/network outage; such
  tracks count as "unplayed" (the conservative direction). Not a bug.
* **Metric.** Like rate, overall and per source (`haystack sources`). A source can't be retired before
  30 offered tracks. No classifier and no skip inference in v1 (classifier after ~200 played discoveries; skips
  later from the Extended Streaming History export).
* **Spotify dev mode (2026).** No audio-features / recommendations / browse / artist top tracks / batch track
  fetch; search capped at 10. None of these are used.

## Commands

```
haystack auth [--timeout 900]        one-time login (prints URL, loopback callback)
haystack library scan|embed|update   index / embed / both + To Rip check  [--workers N --limit N]
haystack harvest [--sources a,b] [--limit-per-source N]
haystack score                       re-score the candidate pool
haystack sync                        hourly job (needs auth)
haystack batch [--size N] [--dry-run] [--force] [--variant V] [--no-penalty]
                                              [--clear-min-age-hours H] [--clear-recent keep|remove]
                                              weekly job (needs auth): clear Discoveries, then add a batch
haystack clear-discoveries [--dry-run] [--no-penalty] [--clear-min-age-hours H] [--clear-recent keep|remove]
                                              empty Discoveries and record nos (also run by batch)
haystack eval [--seeds 1,2,3] [--choose]  offline scoring-variant eval (no Spotify calls)
haystack genres [--top 15]           genre makeup of the liked songs (400 Discogs styles)
haystack harvest --broad [N] [--shard i/N] [--no-retry] [--styles N]   stratified broad harvest
haystack feedback yes|no ARTIST TITLE [--strength mild|normal|strong] [--note T] [--to-rip]
                                              record a manual verdict (never re-offered; yes = positive)
haystack feedback list|rip           list verdicts / add manual YES tracks to To Rip (bot-added)
haystack first-batch [--no-penalty] [--clear-min-age-hours H] [--clear-recent keep|remove]
                                              one-time first batch + taste sanity check (library unit); clears first
haystack sanity-check [-n 10 -k 10]  write ~/.local/state/haystack/sanity-check.md
haystack status                      status.json + live counts
haystack review [list|accept ID..|reject ID..]   uncertain liked-song -> library matches
haystack sources [list|retire NAME]  like rate per source
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
haystack api                         needle's API (the haystack-api unit runs it)
```
`feedback yes|no` still works: yes is a 3-star Rating, no a Pass (`--strength strong`: a Dislike).

**Moving verdicts.** `verdict export` writes every Verdict (history included) with the track's title, artists,
album, ISRC and duration, so the file is readable and can be re-matched if a Spotify id changes. `verdict import`
adds them by id, all or none. By default it makes no Spotify writes (a restore to the same account already has the
hearts and playlists). `--pending` leaves them for the next sync (or `verdict apply`), e.g. for a new account.
The CLI lives at `~/.local/share/haystack/venv/bin/haystack` (editable install of this repo).
Spotify jobs exit with code **3** and `status.json` -> `auth.ok=false` when no refresh token exists.

Sources: `deezer_chart`, `deezer_editorial` (genre charts), `deezer_new_releases` (editorial releases),
`deezer_artist_radio`, `deezer_related` (related artists' top tracks) and `listenbrainz_similar` (labs
`recording-search` + `similar-recordings`; the keyless `api.listenbrainz.org` lookup/radio endpoints now need a
token). Seeds are the positives: recent likes first, then library tracks. Last.fm is skipped (no key in
1Password). Soundalike is dropped from v1; ReccoBeats is out.

## Configuration
Defaults are in `src/haystack_music/config.py`; override any key in `~/.config/haystack/config.toml`
(`config.example.toml` has the common ones, with examples):
```toml
[batch]
size = 20            # weekly batch size
[harvest]
retired_sources = ["deezer_chart"]
```
Batch *schedule*: `systemctl --user edit haystack-batch.timer` (override `OnCalendar=`).

The settings most installs change:

* `library.root`: your music folder (MP4/M4A files, only ever read). Unset (the default): no local library, so
  your taste comes from Liked Songs and Ratings. A root that is set but missing, or empty while tracks are known
  (an unmounted disk), fails the scan instead of marking every track gone.
* `spotify.to_rip = true`: also keep a private **To Rip** playlist of the tracks you rate 3★ or more that aren't
  in the local library (to buy or rip). Off by default.
* `api.allow_networks`: who besides loopback may reach needle's API. The default is every private range; narrow
  it to your LAN (and VPN).
* `harvest.listenbrainz_contact`: an email or URL that ListenBrainz's User-Agent carries, as MetaBrainz asks.
* `[secrets]` and `[alert]`: see Secrets and Alerts below.

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
age cutoff, and writes no `no` verdict. Likes stay `outcome='liked'` either way. The weekly and first-batch
units pin `--clear-recent remove --clear-min-age-hours 24`. A one-off keep is
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

## needle

[needle](https://github.com/kurtlieber/needle) is the desktop app for Review, and a separate client of this
server: it knows the server only through the HTTP API, `docs/api.md`. It has its own repository, with its own
README, glossary, ADRs (0003, 0007 and 0008, the numbers missing here), build plan and mockups. The API runs as `haystack-api` on port 8517, open only to loopback and `api.allow_networks`.
Its bearer token comes from `haystack token new`; `haystack token show` prints it for each needle's token file.
Hearts need the `user-library-modify` scope: after updating to this
version, run `haystack auth` once (SSH tunnel as below); until then heart writes wait
(`haystack verdict list` shows them) and the rest is applied.

## Units
`systemd/` holds four oneshot services + timers, plus the one-shot `haystack-first-batch.service` (all `Persistent=true`, `Nice=19`, `IOSchedulingClass=idle`):

| Unit | Schedule | Needs auth |
|---|---|---|
| `haystack-library` | daily 03:15: rescan + embed new/changed files (+ To Rip check if authorised) | no |
| `haystack-harvest` | daily 05:00: harvest + embed candidates, re-score | no |
| `haystack-sync` | hourly: plays, likes, outcomes, expiry, To Rip, your playlists, token check | yes |
| `haystack-batch` | Mondays 08:00: clear Discoveries (`--clear-recent remove --clear-min-age-hours 24`), then add a batch | yes |
| `haystack-api` | always on (service, not a timer): needle's API, Spotify writes after verdicts, on-demand Readings, your playlists | yes |
| `haystack-first-batch` | once, via `OnSuccess=` of the library unit: sanity check + scoring eval/variant choice + first batch (clears Discoveries first, same flags); then writes `~/.local/state/haystack/first-batch.done` (`ConditionPathExists=!` keeps it from running again) | yes |

Install (symlinks into `~/.config/systemd/user/`, then daemon-reload; enables nothing):
```
~/source/haystack/scripts/install-units.sh
haystack models fetch          # the Essentia models (see NOTICE), about 45 MB
systemctl --user enable --now haystack-library.timer haystack-harvest.timer
# after `haystack auth`:
systemctl --user enable --now haystack-sync.timer haystack-batch.timer
systemctl --user enable --now haystack-api.service   # needle
```
Turn on lingering once (`sudo loginctl enable-linger $USER`) so the timers run without a login session. Times
are the server's local time; a unit drop-in (`systemctl --user edit haystack-library.timer`) moves them. Follow a job: `journalctl --user -u haystack-library -f`.

## Secrets
`[secrets] backend` picks where the secrets live:

* `files` (the default): one 0600 file per secret in `~/.config/haystack/secrets/` (0700; `secrets.dir`). The
  Spotify refresh token and its non-secret dates and scopes share `spotify-token.json` there, replaced whole
  (a new file renamed over the old) on each rotation.
* `1password`: the `op` CLI with a service-account token from `onepassword.token_file` (0600) or a systemd
  credential (`LoadCredential=op-token:<file>` in a unit drop-in). The code ignores any `OP_*` variables in the
  environment and passes the token only to the `op` child process. `[onepassword]` names the vault, items and
  fields (by id, or a name that is valid in an `op://` reference).

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

## Auth and re-auth
Create an app in the Spotify developer dashboard and add the redirect URI
`http://127.0.0.1:8518/spotify/callback` (`spotify.redirect_uri`; Spotify accepts only 127.0.0.1 for a loopback
redirect). Then, on the server:
```
haystack secret set spotify-client-id
haystack secret set spotify-client-secret
haystack auth
```
and open the printed URL in a browser. On a headless server, tunnel the port from your desktop first,
`ssh -L 8518:127.0.0.1:8518 <server>`, and run `haystack auth` in that session. Refresh tokens last ~6 months:
from 5 months after the `issued at` date the hourly sync sends an alert (once a day) and sets
`auth.expiry_warning=true` in `status.json`. Re-auth is the same command; it overwrites the token and issue date.

## Alerts
Off by default. With `[alert] enabled = true` haystack POSTs JSON (`job`, `host`, `status`, `exit_code`,
`message`, `ts`) to the `alert-webhook-url` secret, with `Authorization: Bearer <alert-webhook-key>` when that
secret is set. Messages start with `haystack:`. Sent for: the token expiry warning (once a day),
`auth.ok=false` (once a day), and a failure of the sync, harvest, library or batch unit
(`OnFailure=haystack-alert@%n.service`; exit 3, needs auth, is left to the daily auth alert). Alerting never
fails a job. Dry run: `HAYSTACK_ALERT_DRY_RUN=1`.

## Development
```
cd ~/source/haystack
~/.local/share/haystack/venv/bin/python -m pytest -q
```
Heavy jobs run at nice 19.

## License
AGPL-3.0-or-later (`LICENSE`). haystack downloads the Essentia models on first setup instead of shipping
them: they are NonCommercial (Creative Commons BY-NC), so commercial use needs a license from the MTG. `NOTICE`
lists what haystack uses and under which terms, Deezer's included. Security reports: `SECURITY.md`.
