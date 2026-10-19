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
| `deezer.py`, `listenbrainz.py`, `harvest.py` | candidate sources (reach only, never a score), seeds by Verdicts (ADR 0013), 1 req / 1.1 s cross-process throttle |
| `sources.py` | kept rate by source and by seed, and the harvest cap each source earns from it (ADR 0013) |
| `vectors.py`, `scoring.py` | positives assembly, nearest-positive scoring, batch selection (1 per artist, ~20 % exploration moved by the outcomes, spread across styles; ADR 0010), like rates |
| `align.py` | the preview alignment (ADR 0012): a linear map from preview vectors into the library's space, fitted on the liked songs that have both |
| `rerank.py` | the second stage (ADR 0011): a logistic regression trained on your Verdicts orders the top of the pool once it beats similarity alone, with a chance and reasons per pick |
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
  (float16, same forward pass). A match is scored by the best window. A Deezer preview and your own file of the
  same song land a little apart, so with a library every preview vector is first mapped into the library's space
  (see **Preview alignment**).
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
* **Metric.** Kept rate, overall, per source and per seed (`haystack sources`), and the fuller `haystack report`. A source
  can't be retired before 30 offered tracks. No skip inference yet (skips later from the Extended Streaming
  History export). The second stage (Re-ranking) learns from Verdicts and outcomes only, and switches itself on
  once it has 40 labels and beats similarity alone.
* **Spotify dev mode (2026).** No audio-features / recommendations / browse / artist top tracks / batch track
  fetch; search capped at 10. None of these are used.

## Commands

In the container, put `docker compose exec haystack` in front (or `docker compose run --rm haystack` without the
leading `haystack` while the server is stopped).

```
haystack serve [--no-api]            needle's API plus the scheduled jobs ([schedule]); the container runs this
haystack auth [--timeout 900]        Spotify sign-in: open the printed URL anywhere, paste back the address
haystack library scan|embed|update   index / embed / both + preview alignment refit + To Rip check
                                              [--workers N --limit N]
haystack harvest [--sources a,b] [--limit-per-source N]
haystack score                       re-score the candidate pool
haystack sync                        hourly job (needs auth)
haystack batch [--size N] [--dry-run] [--force] [--variant V] [--no-penalty]
                                              [--clear-min-age-hours H] [--clear-recent keep|remove]
                                              weekly job (needs auth): clear Discoveries, then add a batch
haystack batch --plan [--size N] [--variant V] [--stage1]
                                              the would-be picks, scored in memory: no Spotify, no writes
haystack clear-discoveries [--dry-run] [--no-penalty] [--clear-min-age-hours H] [--clear-recent keep|remove]
                                              empty Discoveries and record nos (also run by batch)
haystack eval [--seeds 1,2,3] [--choose] [--objective auc|mean_pct]
                                              offline scoring-variant eval (no Spotify calls); --choose stores its pick
haystack rerank fit|show [--limit N] fit the second stage on your Verdicts and record it / show the last fit
haystack align fit|show [--limit N]  fit the preview alignment and store it / show it, its fits and the eval's verdict
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
haystack sources [list|retire NAME]  kept rate per source and per seed, this harvest's caps
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
`deezer_artist_radio`, `deezer_related` (related artists' top tracks), `deezer_style_search` (Deezer playlists
named after your top styles) and `listenbrainz_similar` (labs `recording-search` + `similar-recordings`; the
keyless `api.listenbrainz.org` lookup/radio endpoints now need a token).

**Seeds** (ADR 0013). Artist radio, related artists and ListenBrainz start from seed tracks. With
`harvest.seed_policy = "verdicts"` (default) they come from three buckets, by `harvest.seed_quotas` (40 / 30 / 30
%; a share a bucket can't fill passes to the next): **top**, the tracks you Rated 4-5 stars and the discoveries you
kept in the last `seed_recent_days` (90), 5 stars first; **mid**, your other liked songs (Unrated or 3 stars or
more); **library**, your files. Seeds have distinct artists, and an artist seeded for a source in the last week is
skipped while there are others (`harvest_seeds` keeps the history). Each candidate's source detail ends in
`seed=<stars or origin>`, so `haystack sources` can show the kept rate by seed. With no Ratings yet this is about
70 % liked songs and 30 % library, as before; `"recent"` keeps the 0.3 rule (70 % from the 200 most recent liked
songs, the rest library, untagged).

**`deezer_style_search`** takes your `style_search_styles` (8) top Discogs styles by mean activation over the
liked songs (`haystack eval` shows them), searches Deezer's playlists for each style's name, and takes up to 25
new tracks, in playlist order, from each of the `style_search_playlists` (3) with the most fans among the first 6
results. Each style gets at most its share of the source's cap. It skips with a log line until liked songs have
style vectors.

**Caps.** Each source finds at most `harvest.max_new_per_source` (60) new candidates per run, or its own
`harvest.caps.<source>`. With `harvest.adaptive_caps` (default on) a source with at least
`sources.min_offers_before_retire` (30) decided offers has that cap scaled by its kept rate over the kept rate of
all sources, clipped to 0.25-2x; `haystack sources` shows the rates and the caps. `--limit-per-source` sets a
fixed cap for one run.

## Scoring and batches

The settings below are rarely changed; the README's configuration guide covers the common ones.

**Scoring variants** (`scoring.variant`): `raw` (max cosine to the nearest positive), `topk_mean` (mean cosine to
the k=10 nearest positive tracks, best window each), `genre` (mean cosine between the candidate's
genre_discogs400 style vector and its k=10 most similar liked songs' style vectors), `genre_blend` (50/50 z-scored
raw + genre), `centred` (subtract the
mean of library + pool vectors, re-normalise), `csls` (raw minus the candidate's mean cosine to its k=10 nearest
vectors in a fixed, seeded random reference set of library + pool vectors: penalises hubs that are close to
everything) and `csls_centred`. `auto` (default) uses the variant the eval chose (kv `scoring_variant`), else
`raw`: the first batch sets it and every later batch re-checks it (see **Eval**); set an explicit name to pin it,
and the eval then only reports.

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
The eval scores without the penalty: a turned-down offer would otherwise be pushed down by its own rejection.
Its sanity table on your verdicts also shows the penalised ranks (leave-one-out for the rejected tracks).

**Styles.** Each preview also gets a 400-d Discogs style vector: the official `discogs-effnet-bs64-1`
backbone + `genre_discogs400-discogs-effnet-1` head on the same mel patches. The track-similarity model's
1280-d output is *not* compatible with the head (tested: correlation ~0 with the official pipeline), hence the
second backbone pass. Library files have no style vectors (a second full pass would take hours), so the genre
variants use the liked songs' previews as positives. Older previews are backfilled by the daily harvest.

**Broad harvest.** `harvest --broad 150` seeds artist radio and related artists from 150 artists stratified over
genre buckets (sqrt-weighted, liked songs first) and ListenBrainz similar-recordings from every liked song (with
`seed_policy = "verdicts"`, 5-star ones first and Passed ones left out). It runs weekly (`harvest_broad`, Sundays);
the seed sample follows `harvest.broad_seed_rng` and the ISO week, so each week draws different artists and every
shard of one week draws the same. `--shard i/N` splits the seeds across parallel processes (the 1.1 s Deezer
throttle is shared). Chart + editorial-chart candidates are capped at `harvest.chart_max_share` (20 %) of the
pool, and of each batch (3 of 15).

**Eval** (`haystack eval`, ADR 0009): each variant scores the past offers like candidates, and the eval asks how
well it ranks the ones you **kept** (a Rating of 3★ or more, or liked without a Rating) above the ones you
**turned down** (a Pass or a Dislike, or expired after a play or removed by hand). A kept offer is weighted by its
stars (an Unrated like counts as 3★) and is left out of its own positives, with any library file of the same song;
a turned-down offer that is a liked or library song is skipped. Reported per variant:

- **AUC**: the chance that a kept offer outscores a turned-down one (0.5 = chance). The objective.
- **gap**: mean percentile of the kept offers minus that of the turned-down ones, among the pool.
- **top-50 spread**: how far the pool's top 50 scores span, in pool-score sd. Under 0.25 the variant is
  *degenerate* (its top picks are near-ties, so the batch is decided by noise) and is not chosen unless
  every variant is.

Only offers count: both sides were picked from the top of the pool the same way. Held-out liked songs were never
picked, and set against offers they make whichever variant did the picking look worst.

The choice is the best AUC among the variants that are not degenerate; within `eval.auc_tie` (0.02) is a tie and
the simpler variant wins (simplest first: raw, topk_mean, centred, csls, csls_centred, genre, genre_blend). It
needs `eval.min_verdicts` (15) kept and 15 turned-down offers with a preview vector; with fewer, the eval falls
back to the rule before 0.2 and says so (`objective_note`).

The fallback, and a diagnostic always: hold out 20 % of the liked songs (5 seeds), remove them (and any library
file of the same song) from the positives, and rank each among the harvested pool. Reports mean percentile
(50 = chance), recall@15 and @50, and the Deezer-chart share of the pool's top 15; the best mean percentile wins
(ties within `eval.tie_tolerance`, 0.5 points, go to the simpler variant). Queries are the liked songs' Deezer
*preview* vectors, the same audio domain as the candidates (the sync fetches previews for all likes for this;
scoring still uses the local file for locally matched likes). Local-file-only queries are shown as a
diagnostic and never drive the choice: a local window is closer to the local-file positives than any preview,
and centring/CSLS amplify that domain gap. `eval.objective = "mean_pct"` (or `--objective mean_pct`) uses this
rule always.

When it runs: the first batch (and `haystack eval --choose`) stores the eval's choice. Every later batch re-runs
it first (seconds, token-free) and switches only on the AUC objective, and only when the current variant is
degenerate or the winner's AUC is more than `eval.switch_margin` (0.03) higher; on the fallback it keeps the
current variant. `haystack eval` alone reports (and records the run) but keeps the variant; `batch --plan` and `--dry-run` say which
variant the batch would use and write nothing. A pinned `scoring.variant` is never changed. Each run is a row in
`eval_runs` (trigger, objective, chosen, used, offers per side, AUC per variant, switched) and the latest is in
`status.json` (`scoring_eval`); the first batch also logs its table to `sanity-check.md`.

```toml
[eval]
objective = "auc"      # or "mean_pct"
min_verdicts = 15      # kept and turned-down offers each, before the AUC is used
auc_tie = 0.02
switch_margin = 0.03
```

**Preview alignment** (`scoring.align`, ADR 0012). Candidates are Deezer previews, but with a library most of your
positives are your own files' windows, and the two land a little apart in the embedding space: a preview is often
nearer the other previews (liked songs without a file, liked discoveries, Rated tracks) than the file of the same
song. The **pairs** are the liked songs that have both a preview vector and a matched local file. A linear map is
fitted on them from the preview to the file, `x -> (x - mp) @ M + mt` on L2-normalised vectors: `M` is a ridge fit
shrunk toward no change (`lambda` from infinity, a pure mean shift, down to 1) or an orthogonal Procrustes rotation,
and the target is the file's best-matching window or the mean of its windows. They are chosen by 5-fold
cross-validation on **self-retrieval**: is a held-out pair's own file the nearest (top 1), or among the 10 nearest,
of the library's files and the other liked songs' previews (mapped by the same fold's map)? The best top 10 wins,
then the best top 1; ties go to the simpler fit. The map is stored in `embeddings/preview_align.npz` (kv
`preview_align_version`) and applied to every preview vector before scoring: candidates, positives, negatives, the
eval's queries and the re-ranker's features. The genre variants' style vectors are left alone, and so are the
batch's spread rule and the sanity check, which compare previews with previews.

`auto` (default) uses the map when it was fitted on at least `scoring.align_min_pairs` (100) pairs, its
cross-validated self-retrieval beat the unaligned vectors, and the latest eval did not find it worse. With a map,
the eval runs twice, aligned and unaligned. The variant is chosen on the aligned run; if, for the variant used, the
aligned objective (AUC, or the mean percentile on the fallback) is lower by more than its tie, or recall@15 by more
than one held-out song per seed, the alignment is turned off until the next eval (kv `preview_align_gate`) and the
choice is made again on the unaligned run. A pinned variant still gets this check at each batch. The eval's
markdown and `status.json` (`scoring_eval.align`, `scoring_eval.domain`) show both runs side by side, the pairs,
self-retrieval with and without the map, and the share of the pool's top 100 whose nearest positive is a preview.
`library update` (daily) refits when there is no map yet or the pair count moved by more than 10 %; too few pairs
remove the map. `off` never maps. Without a library there are no pairs and no map, and scoring is as before 0.3.1.

```toml
[scoring]
align = "auto"         # or "off"
align_min_pairs = 100
```

**Re-ranking** (`[rerank]`, ADR 0011). The score above is stage 1. Once you have judged enough offers, a second
stage orders the top `rerank.pool_top` (400) of the pool by stage-1 score: a logistic regression trained on your
Verdicts gives each of them a *confidence*, its chance that you keep the track, and they are ordered by
`alpha * z(logit) + (1 - alpha) * z(stage-1 score)` (`rerank.alpha`, 0.7), put back in stage-1 score units. The
rest of the pool keeps its stage-1 order below them, and exploration picks are still drawn by stage-1 score.

- **Labels**: one per track with a candidate row and a preview vector that was offered, or that carries a Verdict
  given outside the Unrated likes queue (those are all liked songs, so they say nothing about yes against no).
  The current Verdict decides: a Rating of 3★ or more is a yes, weighted 1.0, 1.25 or 1.5 for 3, 4 or 5★; a
  Rating of 1-2★ is a no of weight 0.75 (to stage 1 it is still a positive: you like the track a little, but it
  is not for the batch); a Pass is a no of 1.0 and a Dislike of 1.5. With no Verdict an offer you liked is a yes,
  and one that expired after a play or was removed by hand a no, both 1.0; `unplayed` is not a label.
- **Features**, computed the same way for a pool candidate and a labelled track (which is left out of its own
  positives and negatives): the stage-1 score; the nearest and top-k similarity and the genre top-k with no star
  bonus; hubness; the negative penalty and the largest excess over a turned-down track; the stars of the nearest
  positive, the similarity-weighted mean stars of the `rerank.neighbours` (10) nearest, the share of them at
  4-5★ and the share that are library files (a positive with no Rating counts as `rerank.prior_stars`, 3); per
  Trait, how far the track lies outside your usual range, in widths of the range; the source and whether it is a
  chart; whether the artist is among your positives or has a Dislike. From `rerank.min_labels_embed` (300)
  labels, 32 principal components of the preview vector join them. Each is standardised over the pool.
- **Fit**: L2-regularised logistic regression by Newton steps (numpy only), refitted at every batch in
  milliseconds. `rerank.l2` (1.0) is the penalty on the weighted mean log-loss, per unit of label weight.
- **Gate**: the model is used only with at least `rerank.min_labels` (40) labels and when its leave-one-out AUC
  (each label scored by a model fitted on all the others) beats stage 1's AUC on the same labels by
  `rerank.min_gain` (0.05). Otherwise the batch goes by stage 1 and the log says why. The check runs again at
  every batch, so the second stage switches itself on as labels grow. An error in it never stops a batch.
- **Reasons**: each re-ranked candidate keeps up to two phrases for the features that pushed its chance up the
  most (`close to your 4-5★ tracks`, `brighter than you usually like`) and up to two for what held it back. A
  feature counts when it moves the logit by at least 0.1, and is quoted only when its weight agrees with the
  feature's own direction over the labels and with the direction it should have (being close to your 4-5★
  tracks is never a reason to hold a track back); such a weight still orders the batch. A Trait is quoted only
  when the track lies outside your usual range on the side the phrase names, or as `<trait> in your usual range`.

Each fit is a row in `rerank_runs`, and the last model is kept in the `kv` table (`rerank_model`). When the model
is used, a batch writes the chance and reasons of the pool's top 400 to `candidates` (`rerank_p`,
`rerank_reason`), and the batch summary carries `rerank` (used, why, the labels, both AUCs). The API returns them
in a Discoveries card's `why` (features `rerank` and `why.reasons`). `haystack rerank fit` fits and records
without touching the pool; `rerank show` prints the last fit, its largest weights and the recent fits; `batch
--plan --stage1` shows what stage 1 alone would pick.

```toml
[rerank]
enabled = true         # false: every batch by stage 1 alone, as before 0.3.0
pool_top = 400
min_labels = 40
min_gain = 0.05
alpha = 0.7
```

**Picking the batch** (ADR 0010). About `batch.exploration_fraction` of the picks are exploration picks, taken at
random from the `batch.explore_band` (the 40th to 75th score percentile of the pool); the rest are the top picks,
by score or, once the re-ranker is used, in its order (in score units, so the diversity below works the same).
A batch has at most `batch.max_per_artist` songs per artist, and three rules spread the top picks:

- **Diversity** (`batch.diversity`, 0.5): after each pick, the rest are re-ranked by
  `score - diversity * sd(pool score) * max(0, max cosine to the picks so far)`. The cosine is between preview
  vectors centred on the pool (the pool's mean subtracted, then re-normalised): raw preview cosines are all close
  to 1, so only centred ones tell which songs are alike. A candidate unlike every pick, or with no vector, is not
  marked down. 0 picks by score alone.
- **Style cap** (`batch.max_style_share`, 0.4): at most that share of the top picks (rounded down, at least 1) may
  share a top-1 Discogs style. Exploration picks and candidates with no style vector don't count.
- **Nearest cap** (`batch.max_per_nearest`, 2): at most that many picks, exploration picks included, may share a
  nearest positive (`candidates.nearest`), so one liked song can't fill the batch with its neighbours. 0: no cap.

The caps give way when the pool can't fill the batch with them: the rest is filled without the style cap, then
without either. A pick that fails the Spotify lookup is replaced in a later round, and the picks already made
count toward the caps and the diversity.

**Exploration share.** `batch.exploration_fraction` (0.2) is the prior. Each batch takes the judged offers of the
last `batch.explore_window_batches` (8) batches (decided, minus tracks that went unplayed and have no Verdict, as
`haystack report` counts them) and the share of each kind that was kept. When both kinds have at least
`batch.explore_min_judged` (8), the batch's share is `prior + 0.5 * (exploration kept rate - top kept rate)`,
clipped to `batch.exploration_bounds` ([0.1, 0.4]); otherwise the prior. A window or prior of 0 turns this off.
The batch summary carries `exploration_fraction_used`, `exploration_share` (both rates and counts, and why) and
`spread` (the most top picks sharing a style, and the most picks sharing a nearest positive). Each offer stores
its score percentile in the pool its batch was picked from (`offers.score_percentile`), which the API returns in
a card's `why.band`.

```toml
[batch]
diversity = 0.5                  # 0: by score alone
max_style_share = 0.4            # 1: no style cap
max_per_nearest = 2              # 0: no cap
explore_window_batches = 8       # 0: always batch.exploration_fraction
explore_min_judged = 8
exploration_bounds = [0.1, 0.4]
```

With `diversity = 0`, `max_style_share = 1`, `max_per_nearest = 0` and `explore_window_batches = 0` a batch is
picked exactly as before 0.2.0.

**Embedding guard.** A real batch skips (exit 0, `status.json` job message `skipped: library embedding
incomplete: ...`) until at least `batch.min_embedded_fraction` (default 0.98) of library files are embedded or
have given up; positives would be thin otherwise. A skipped batch does not clear Discoveries. `--dry-run` still
runs (with a warning) and makes no Spotify writes: it prints the clear plan (counts only) but removes nothing
and records no verdicts, no playlist is created, nothing is added, and no batch/offer rows are stored. `--force`
overrides the guard.

**Planning a batch.** `batch --plan` runs the batch's scoring and selection with no Spotify session and a read-only
database: scores are computed in memory (the stored scores are left alone), and no run is recorded. It prints each
would-be pick with its score and score percentile, whether it is an exploration pick, its source, its top-1
style, the positive it scored closest to and the variant, with the batch's exploration share and spread. A candidate with no Spotify id yet is marked `unmapped` instead of being looked up, so the picks can
differ from a real batch's only where a lookup would have failed or found a known track, or where Spotify reports a
pick already saved in Liked Songs (a real batch then drops it). Both paths share one selection function
(`batch.select_for_batch`), so with the same scores and seed they pick the same candidates. With
`scoring.variant = auto` it first re-checks the variant as a batch does and shows the outcome (`variant_choice`),
recording nothing. It ignores the embedding guard and says so when the library is not fully embedded. When the
re-ranker is used, each top pick also shows its chance, up to two reasons and what held it back; `--stage1` leaves
the second stage out, to compare the two orders.

**Report.** `haystack report` is token-free and opens the database read-only. Offers: offered, decided (an outcome
is set), judged (decided, minus tracks that went unplayed and have no Verdict), kept (the current Verdict keeps the
heart, or the offer was liked and has no Verdict), with kept/decided (the `haystack sources` rate), kept/judged
and liked/decided (the like rate in `status.json`), overall, exploit vs exploration, per batch and per source. Then the current Verdicts by
kind and stars and how many Liked Songs are rated; the pool (size, embedded, scored, active and scored variant); of
the top 100 by stored score, how many have a library window as their nearest positive and how many a preview (by owner
kind: like, discovery, manual, rated); the top 30's score range, distinct nearest positives and top-1 Discogs
style; and the latest eval from `status.json`: the objective it chose by, the kept and turned-down offers it
counted, and per variant the AUC (marked when degenerate) and mean percentile, with tracks never named. It
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

**Verdict weights** (`[scoring.verdicts]`, ADR 0002, ADR 0011): every Rating is a positive whose bonus grows
with its stars. By default the bonus is one of five fixed cosine offsets, `star_bonus = [0.0, 0.0025, 0.005,
0.0075, 0.01]` for 1-5 stars (3 stars = `bonus_discovery_like`). `star_bonus = []` switches to
`star_weight_sd = 0.25` per star above 3 (a penalty per star below), in units of the sd of the pool's
nearest-positive similarity in the variant's space, measured at scoring time, so a 5★ track is worth the same
lift in every pool whatever its spread. That form is opt-in until `haystack eval` shows it is not worse (ADR
0011). An Unrated like counts as `unrated_like_stars = 3`; a library file nobody liked or rated gets 0. A Pass
is a negative of `pass_strength = 1.0` (the old "normal" no), a Dislike `dislike_strength = 2.0` (the old
"strong"), on the negative-penalty scale. Every variant adds the bonus before taking the nearest / top-k.

## Running it

Two ways, the same jobs. Pick one per server: the container's scheduler doesn't know about the units.

| Job | Default schedule | Does | Needs auth |
|---|---|---|---|
| library | daily 03:15 | rescan + embed new/changed files (+ To Rip check if signed in) | no |
| harvest | daily 05:00 | harvest + embed candidates, re-score | no |
| harvest_broad | Sundays 04:00 | `harvest --broad 150`: a wider harvest from the week's seeds | no |
| sync | hourly | plays, likes, outcomes, expiry, To Rip, your playlists, token check | yes |
| batch | Mondays 08:00 | clear Discoveries (`--clear-recent remove --clear-min-age-hours 24`), then add a batch | yes |
| first batch | once | sanity check + scoring eval/variant choice (later batches re-check it) + first batch (clears first, same flags); then writes `first-batch.done` in the state dir | yes |
| API | always on | needle's API, Spotify writes after verdicts, on-demand Readings, your playlists | yes |

Heavy jobs run at nice 19.

### `haystack serve` (the container)

One process: needle's API in a thread, and a scheduler that starts each job as a child process at the times in
`[schedule]` (`hourly`, `daily HH:MM`, `Mon HH:MM` with comma-separated days, or `off`), in local time (`TZ`).

* A run missed while haystack was down starts when it comes back. The last start of each job comes from the
  database's `runs` table, which the units write too, so moving from the units to the container doesn't replay a
  batch. A job that has never run starts at once, except the batch and the broad harvest, which wait for their
  time.
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

`systemd/` holds the five oneshot services + timers above (all `Persistent=true`, `Nice=19`,
`IOSchedulingClass=idle`), the one-shot `haystack-first-batch.service` (via `OnSuccess=` of the library unit,
`ConditionPathExists=!` on `first-batch.done`) and the always-on `haystack-api.service`. They ignore `[schedule]`.

```
git clone https://github.com/kurtlieber/haystack ~/source/haystack
python3.14 -m venv ~/.local/share/haystack/venv
~/.local/share/haystack/venv/bin/pip install -e "$HOME/source/haystack[embed]"   # needs ffmpeg on the PATH
ln -s ~/.local/share/haystack/venv/bin/haystack ~/.local/bin/haystack
~/source/haystack/scripts/install-units.sh   # symlinks into ~/.config/systemd/user/, daemon-reload; enables nothing
haystack models fetch
systemctl --user enable --now haystack-library.timer haystack-harvest.timer haystack-harvest-broad.timer
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
