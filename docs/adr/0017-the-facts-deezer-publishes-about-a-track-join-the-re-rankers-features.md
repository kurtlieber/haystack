# The facts Deezer publishes about a track join the re-ranker's features

The re-ranker (ADR 0011) knows what a track sounds like and how it relates to the listener's history. Deezer, which
haystack already asks for every candidate, publishes more: the album's label, release date, record type (album,
single, EP or compilation), genres and fan count, the track's rank (how popular it is on Deezer), and the artist's
fan count. A listener may keep most tracks from a handful of labels, prefer older records, or pass on anything from
a compilation. From 0.6.0 haystack keeps these facts and the re-ranker learns whether they matter.

1. **What is kept.** `deezer_albums` holds one row per album and `deezer_artist_fans` one per artist, by Deezer id.
   A candidate keeps its `album_id` and `deezer_rank`, and a liked song mapped to Deezer its `album_id`. An album
   id of 0 means Deezer no longer has the track; an album or artist row with no data means Deezer gave none.
2. **How it is fetched.** The harvest stores a new candidate's album id and rank from the track object it already
   has, at no cost. After the sources, each daily harvest fetches what is still missing for up to
   `harvest.metadata_per_run` (150) tracks, the judged ones and the liked songs first, then the pool oldest first:
   `/album/{id}` once per album, `/artist/{id}` once per artist, and `/track/{id}` first for a track whose album
   isn't known. `haystack metadata backfill --limit N` does the same at once, for an existing pool. Every request
   goes through the shared 1.1 s throttle, and a failed one is tried again next time.
3. **Seven features.** The log of the Deezer rank (`log_rank`, with `rank_missing` when there is none), the log of
   the artist's fans (`log_fans`), the album's age in years (`age_years`), whether it is a compilation
   (`compilation`), whether its label is that of one of the listener's positives (`label_known`), and the share of
   the judged tracks on its label that were kept (`label_kept_rate`). A missing number takes the pool's median.
   Like every feature, they are standardised over the pool and quoted as reasons: `on the label of a song you
   like`, `on a label you like`, `a very popular track` or `an obscure track`, `by a very popular artist` or `by a
   little-known artist`, `an older release`, `from a compilation`.
4. **The card shows the label.** A track card carries `label` (API feature `card.label`).
5. **`rerank show` compares.** Each fit also records the model's leave-one-out AUC without these seven features
   (`loo_auc_no_meta`), and `rerank show` prints both, so whether they help is plain to see.

**Why the label's kept rate is shrunk.** A label with one judged track, kept, would read as 100 %. The rate is
`(kept + n * mean) / (judged + n)`, where `mean` is the listener's overall kept rate and `n` is
`rerank.label_prior_n` (5), so a label needs several judged tracks before it moves far from the mean. A labelled
track's own verdict is left out of its label's rate, and its own positive out of `label_known`, as the artist
features leave a track out of its own positives; otherwise the leave-one-out AUC would count the answer as evidence.
The rate is worked out from the verdicts themselves, so each leave-one-out fold works it out again without the
held-out verdict: that verdict is then in no track's rate and not in the mean, and the model can't learn it from the
other tracks on the same label.

**Why only some features have an expected sign.** A label the listener keeps tracks from, or shares with a
positive, should only push a track up, and a compilation should only hold one back, so those three are quoted only
in their own direction (`EXPECTED_SIGN`). Popularity, fans and age have no right direction: one listener digs for
obscure records, another wants the hits. Their weights are quoted whichever way the listener's verdicts point.

**Why fans are kept by artist id.** `deezer_artists` is a cache from an artist's name to one Deezer id, and two
artists can share a name. The fan count belongs to the id the candidate came with, so it has its own table.

**Why it stays local.** These are public facts about public tracks, fetched gently through the same throttle as the
previews and kept on the listener's machine like the vectors (README, Deezer notice). Nothing about the listener is
sent: the requests name Deezer albums and artists, as the harvest's requests already do.

**Considered Options**:

- Deezer's album genres as features. The 400 Discogs styles from the audio already say more, and say it for every
  track; the genres are stored but not used.
- One feature per record label. Hundreds of them, most with one judged track: far too many weights for a few
  hundred Verdicts. One shrunk rate carries what the data can support.
- Labels and popularity from Spotify. Spotify is asked only for what the listener's account needs, its development
  mode has no batch track fetch, and Deezer is already being asked about every candidate.
- Labels from MusicBrainz. Another service and another throttle, and thin coverage of new releases.
- Deezer's facts as seven re-ranker features, fetched once per album and artist through the shared throttle, with a
  shrunk label kept rate (chosen).

**Consequences**:

- At most three more Deezer requests for a track, fewer as albums and artists repeat. The daily harvest asks for
  at most `harvest.metadata_per_run` tracks, about eight minutes at 1.1 s a request. A one-off backfill of an
  existing pool takes hours, so it runs outside the harvest's hours.
- The gate is unchanged: the model is still used only when it beats stage 1. If the facts only add noise, the
  model's AUC falls and `rerank show` says so.
- needle shows the label once it reads `label`; until then nothing changes for it.
- A fresh installation has no facts until its first harvest, then gathers them as it goes. With none known, the
  seven features are constant, the model is the same as before 0.6.0, and `rerank show` prints the same AUC twice.
