# Sources follow the listener's scene: similar artists, style tags, fresh releases in those tags, small labels and the model's deep cuts; charts stay only where they earn their cap

The adaptive caps (ADR 0013) weigh a source by the kept rate of what it offered. That misses most of what goes
wrong with a chart. A chart's songs are popular, so a listener often has them already: the harvest counts them as
a dupe, a library file or a liked song and drops them, and they never reach an offer to be kept or not. The cap
sees a source that offers little, not one whose intake is mostly known. And what a chart offers follows a
country's or a genre's mainstream, not the listener's corner of it. Playlists found by a style's name have the
same trouble from another side: a name match on Deezer finds anyone's playlist, so "Bassline" found a funk
playlist and "Dubstep" a reggae one.

So from this release the harvest looks where the listener's scene is, measures how much each source brings that
is already known, and stops offering what a retired source alone found:

1. **The scene, from the listener's own data.** `scene.top_styles` gives the top Discogs styles by mean
   activation over the liked songs (as `deezer_style_search` used), each with its Discogs genres.
   `scene.positive_artists` gives the artists the listener has: liked songs, Ratings that keep their heart,
   present library files and kept discoveries.
2. **`lastfm_tag_artists`** (the Last.fm key). For each top style, Last.fm's top artists of the style as a tag,
   the first `harvest.tag_artists` the listener doesn't have, two top tracks each. A style is spelt as Last.fm's
   listeners tag it (`scene.TAG_SPELLING`, the spelling with the most taggings when probed, 2026-11; otherwise the
   style lower-cased with "&" as "and"). `tag.getTopTracks` is not used: its tracks drift to pop.
3. **`lastfm_similar` widens.** Its artist half takes `harvest.lastfm_similar_artists` (6) similar artists per
   seed instead of 3, and its seed artists come from the top bucket first. Its name stays, so its history stays
   comparable.
4. **`listenbrainz_fresh`** (keyless). ListenBrainz's fresh releases of the past week, kept when a release tag
   names a top style (`scene.tag_keys`, so `drum & bass`, `dnb` and `drum and bass` all count) or the release is
   by a positive artist. A release that new has no recordings in ListenBrainz's metadata, so it is found on Deezer
   by an album search, and its two most played tracks are offered.
5. **`listenbrainz_artist_similar`** (keyless). ListenBrainz Labs' similar artists of the seed artists (top bucket
   first), and two most listened recordings of each: a second artist graph, built from listening sessions,
   independent of Deezer's related artists.
6. **`llm_suggest`** (the language model, ADR 0019). Once an ISO week, in the first shard of the weekly broad
   harvest, the model is asked for `llm.seeds_per_week` deep cuts: songs in the listener's styles by artists not
   in the profile and not the best-known acts of those styles, from small labels, singles and EPs, the last ten
   years unless the profile points earlier. A suggestion by a profile artist is dropped (the prompt forbids it;
   the code checks), and one the pool already has is a dupe like any other. It is gated by `llm.enabled`,
   `"seeds"` in the new `llm.jobs` (default empty) and the key, and stops for the week once the jobs' recorded
   cost (`llm_runs`) reaches `llm.weekly_budget_usd`.
7. **`deezer_style_search` takes Deezer's genre editors only.** A playlist counts only when its creator is
   Deezer's editor for the style's Discogs genre (`scene.GENRE_EDITORS`: one Deezer account per Deezer genre,
   read once from Deezer's editorial playlists; it holds no listener data). The source stays on by default: for a
   listener whose styles are mainstream, the editors' playlists are good.
8. **The already-known share.** Each harvest's summary carries, per source, the share of its intake that was
   already a candidate, a library file or a liked song, and `deezer_editorial` the same per chart. `haystack
   sources` shows it over the last 14 days next to the kept rate.
9. **A retired source's songs stop being offered.** A pool candidate is eligible for a batch or Calibrate only
   while one of its `candidate_sources` rows is live: its source is not in `harvest.retired_sources` and, for an
   editorial chart or editorial selection, its id is still in `harvest.deezer_editorial_ids`. A song another live
   source also found stays eligible. Until now only the candidate's first source counted, so dropping a chart left
   its songs in every batch.

**Why the deep cuts are off by default, and on only after a dry run.** The model's ordinary suggestions are mostly
songs the listener knows: famous songs of famous artists in their styles. The deep-cut prompt is there to change
that, and whether it does is measured before the source is switched on: `haystack llm suggest --dry-run` must find
at least half of 100 suggestions new, on Deezer and with a preview. A source that costs money each week has to show
it brings something first.

**Why fresh releases are found by album.** ListenBrainz knows a fresh release's MBID and tags, but its recordings
reach the metadata API only later, so there are no recording names or ISRCs to resolve. Deezer's album search by
artist and title, with the title's version suffix ignored and the artist checked, finds the release; its track
ranks pick the two to offer.

**Why the country editors are left out.** Deezer's country editors (Brazil, France, Germany, the UK, Latin
America) make playlists for a country's listeners across every style, so one named after a style is often that
country's hits with the style's name. A genre editor's playlist is what a style search means.

**Why the known share and not the kept rate alone.** The kept rate counts only what was offered. A source whose
intake is three quarters known offers a quarter as much, and that quarter can be kept at a fair rate: the cap stays
high while most of the source's requests buy nothing. The two numbers together tell a source that brings little
that is new from one that brings new songs the listener doesn't keep. Which charts to retire is the listener's
call, made in their own config with both numbers in hand.

**Considered Options**:

- Another country's chart, for songs the listener hasn't met. Rejected: a probe of a US chart found the same
  mainstream, only from another market, and no closer to the listener's styles.
- Keep the charts and let the adaptive caps shrink them. Rejected: the caps see the kept rate of what is offered,
  and a chart's trouble is that its intake is known before it is offered; the cap can't see it.
- Scene feeds (chosen): similar artists, style tags, fresh releases in those tags, small labels and the model's
  deep cuts, each under the same caps, and the known share shown beside the kept rate.

**Consequences**:

- Three keyless sources join the defaults (`listenbrainz_fresh`, `listenbrainz_artist_similar`, and
  `lastfm_tag_artists` once a Last.fm key is set), and `llm_suggest` waits for `llm.jobs`. They share the existing
  throttles; no new service is asked.
- Sent to Last.fm: the names of the listener's top styles as tags. Sent to ListenBrainz: the artist and title of
  seed songs (to find their artists); the fresh-release list is the same for everyone and is filtered on the
  listener's server. Sent to the model: the profile, as for the dry run.
- A chart dropped from `harvest.deezer_editorial_ids`, or a source retired, takes its songs out of the batches and
  Calibrate at once, unless another live source found them too. The pool rows stay, and come back if the chart or
  source does.
- `haystack sources` is wider (a `known` column, the latest skip reason, a line per editorial chart).
