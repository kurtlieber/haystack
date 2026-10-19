# Harvest seeds follow the listener's Verdicts, and styles are searched by name

Artist radio, related artists and ListenBrainz find candidates from **seeds**: tracks the listener likes. Until
0.4.0 a seed was a random pick from the 200 most recent liked songs (70 %) and the library (30 %). A song rated 5★
was no more likely to be a seed than one liked years ago and never played since. Every source brought at most 40
new candidates a day whatever it found, and the charts, which the listener rarely keeps, could fill a fifth of a
batch.

**Seeds by Verdicts.** `harvest.seed_policy = "verdicts"` (the default) draws seeds from three buckets, by
`harvest.seed_quotas`:

| Bucket | What | Share |
|---|---|---|
| top | tracks Rated 4-5★, and discoveries kept, in the last `seed_recent_days` (90); 5★ first | 40 % |
| mid | the other liked songs (Unrated, or Rated 3★ or more) | 30 % |
| library | the listener's files | 30 % |

A share a bucket can't fill passes to the next, so with no Ratings yet it is about 70 % liked songs and 30 %
library, as before. A track Passed or Disliked since it was liked is never a seed. Seeds have distinct artists, and
an artist seeded for a source in the last 7 days is skipped while there are others; `harvest_seeds` keeps that
history for 90 days. `"recent"` keeps the 0.3 rule.

**Seeds are tagged.** Each candidate's source detail ends in `seed=<tag>`: the seed's stars, else where it came
from (`discovery`, `like`, `library`, `broad`). `haystack sources` shows the kept rate by seed next to the kept rate
by source, so the next change to the quotas can rest on outcomes.

**Styles searched by name.** A new source, `deezer_style_search`, takes the listener's `style_search_styles` (8)
top Discogs styles by mean activation over the liked songs, searches Deezer's playlists for each style's name,
reads the first 6 results' fan counts and takes the `style_search_playlists` (3) with the most fans. It takes up to
25 new tracks per playlist in playlist order, skipping tracks it already knows, rather than the first 25 tracks:
the first 25 of a popular playlist change slowly, and the source would dry up within days. Each style gets an equal
share of the source's cap.

**Adaptive caps.** `max_new_per_source` rises from 40 to 60, and `harvest.caps` can set one source's base. With
`harvest.adaptive_caps` (default on) a source with at least `sources.min_offers_before_retire` (30) decided offers
has its base scaled by its kept rate over the kept rate of all decided offers, clipped to 0.25-2x. A source that
keeps failing shrinks to a quarter but is never retired by itself: `retired_sources` stays the listener's switch.
`--limit-per-source` still sets a fixed cap.

**Charts per batch.** Chart and editorial-chart candidates were already held to `chart_max_share` (20 %) of the
pool; now a batch takes at most that share of its picks from them too (3 of 15). Unlike the style and
nearest-positive caps, this cap never gives way. The default editorials stay as they were (0 All, 106 Electro,
113 Dance, 116 Rap/Hip Hop, 152 Rock, 464 Metal), now named in `config.py`: which charts suit a listener is their
own config, and the adaptive caps and the per-batch cap shrink the ones that don't.

**Weekly broad harvest.** `harvest --broad 150` runs every Sunday (`haystack-harvest-broad.timer`, or
`[schedule] harvest_broad` in the container). Its seed sample follows `broad_seed_rng` and the ISO week: each week
draws different artists, and every shard of one week draws the same. Its seeded sources get
`broad_max_new_per_source` (1000), since their per-seed limits do the limiting; the other sources keep their
daily caps. A broad run's seeds count toward the weekly skip, so the next daily runs start from other artists.

**On the reference installation**, before the change:

- 65 offers, 50 decided and 18 kept: a kept rate of 36 %. By source, decided offers and kept rate: artist radio 27
  and 44 %, ListenBrainz 11 and 45 %, the editorial charts 6 and 17 %, related artists 4 and 0 %, the main chart 2
  and 0 %. No source had 30 decided offers yet, so every cap stays at its base for now.
- The top bucket held 59 tracks (12 Rated 5★, 34 Rated 4★, 13 discoveries kept) from 53 artists, the mid bucket 347
  liked songs and the library 3,734 files. At 4 top seeds a day per source, a week takes 28 top artists, so the
  weekly skip doesn't have to repeat one.
- A test run capped at 5 new candidates per source drew 20 seeds for the three seeded sources, 12 of them Rated 5★.
  The style search found 5 new tracks (7 sightings) from 5 of the 8 styles before its cap. ListenBrainz's 11
  sightings from its first seeds were all candidates already in the pool.

**Considered Options**:

- Seeding from 5★ tracks alone. There were too few to fill 10 seeds a day without repeating artists within the
  week; the quotas take them first and fill the rest from the wider likes.
- Weighting the seed draw by stars instead of buckets. It gives the same order on average, but the buckets make
  the share of each kind fixed and easy to report.
- Scaling caps by the like rate (liked over decided). It ignores Ratings: an offer closed by a Rating of 3★ or more
  rather than by a heart would count as a miss.
- Scaling against the plain mean of the sources' kept rates. A source with a handful of offers would move it as
  much as one with hundreds; the rate over all decided offers weights each source by what it brought.
- Retiring the charts outright. A fresh installation has no Verdicts and still needs candidates; the per-batch cap
  and the adaptive cap shrink them where they don't work.
- Searching Deezer tracks (not playlists) by style name. Track titles rarely name a style; playlist titles do.

**Consequences**:

- The daily harvest asks for more: up to 60 new candidates per source, and one Deezer search plus up to 6 playlist
  reads per style for the style search. A weekly broad run takes hours at nice 19.
- Candidates found before 0.4.0 are untagged and show as such in `haystack sources`.
- A source's cap moves only once it has 30 decided offers, so a new installation harvests as it did, with the new
  base of 60.
- The charts can fill at most 20 % of a batch, so a batch may come up short when little else is in the pool.
- A fresh installation and one with no Ratings harvest much as 0.3 did: 70 % liked songs and 30 % library seeds,
  plus the style search once liked songs have style vectors and the weekly broad harvest.
