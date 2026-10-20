# More places name candidates, and Apple's previews fill in where Deezer has none

Until 0.7.0 every candidate came from Deezer (charts, editorial picks, artist radio, related artists, playlists
named after the listener's styles) or from ListenBrainz's similar recordings. Each of these looks outward from what
the listener already likes in much the same way, and a source that never sees a song can't offer it. From 0.7.0
four more sources can name candidates, and a candidate whose Deezer track has no preview can be heard through
Apple's instead.

1. **Four sources, on the list but idle until set up.** `lastfm_similar`, `listenbrainz_recs`,
   `listenbrainz_radio` and `deezer_label` join the default `harvest.sources`. A source that isn't set up (no
   key, no user name, no token, or no label known yet) logs one line in the run's stats and asks nothing of
   anyone. They earn their caps like the other sources (ADR 0013): `haystack sources` shows their kept rate, and
   one that finds nothing the listener keeps brings fewer.
2. **A candidate is still a Deezer track.** Last.fm and ListenBrainz name songs. The harvest finds each one on
   Deezer, by its ISRC when the source gives one and Deezer's track carries the same names, else by an artist and
   title search with the matching's version check (a remix or a live take is not the studio song). A song Deezer
   doesn't have is skipped. The pool, its metadata (ADR 0017) and its Spotify lookups stay keyed by Deezer id.
3. **`lastfm_similar`** (a Last.fm API key, the `lastfm-api-key` secret). Half the cap from `track.getSimilar` on
   seed tracks, half from `artist.getSimilar` on seed artists, taking the top 2 tracks of each seed's 3 most
   similar artists. Seeds come from the same buckets and history as the other seeded sources.
4. **`listenbrainz_recs`** (`harvest.listenbrainz_user`). The listener's own weekly recommendations, which
   ListenBrainz works out from the listening it records for them, read from `/1/cf/recommendation/user/<name>/recording`
   and named through `/1/metadata/recording`. Both answered without a token when checked (2026-10). A
   recommendation found once is not resolved again; one Deezer doesn't have is tried again next run.
5. **`listenbrainz_radio`** (a ListenBrainz user token, the `listenbrainz-token` secret). `harvest.lb_radio_prompts`
   LB Radio prompts a run, alternating the listener's top styles (`tag:(<style>)`) and seed artists
   (`artist:(<name>)::nosim`). `/1/explore/lb-radio` answered 401 without a token when checked (2026-10); with one,
   both prompt forms gave a JSPF playlist whose tracks carry `creator`, `title` and, in `identifier`, a list of
   MusicBrainz recording URLs.
6. **`deezer_label`** (nothing to set up). The record labels with the most kept mass, from the album rows that
   Deezer's metadata (ADR 0017) holds for the listener's Rated and liked songs: 3 for 5 stars, 2 for 4 stars, 1 for 3
   stars or a liked song with no Rating, nothing for a Pass, a Dislike or fewer stars. For each of the top
   `harvest.label_count` labels, Deezer's album search (`label:"<name>"`), the albums whose label really matches,
   newest first, and the 2 most played tracks of each of `harvest.label_albums` albums not seen before.
7. **A second preview provider, off by default.** `harvest.preview_fallback = "itunes"` looks up a candidate
   without a Deezer preview on Apple's iTunes Search API (no key) by artist and title, with the same version check
   and the durations within the matching tolerance, and embeds Apple's 30-second preview in its place. The vectors
   stay under the Deezer id, marked `deezer_vectors.provider = 'itunes'`. New candidates are tried as they arrive;
   the daily retry step tries 50 older `no_preview` pool candidates a run, and asks again after 30 days about one
   Apple didn't have (`preview_fallback` table). Apple's audio is held in memory only, at most 3 MB, like Deezer's.
8. **A throttle per service.** Each service has its own cross-process throttle file in the state dir: Deezer
   1.1 s, ListenBrainz 1.2 s, Last.fm 1 s (Last.fm's terms ask for no more than about 5 a second averaged over five
   minutes; haystack stays well under), iTunes 3 s (Apple allows about 20 requests a minute).

**Why the preview fallback is off by default.** Turning it on sends Apple the artist and title of every candidate
Deezer has no preview for. Those names come from the listener's taste, and Apple is a new party. haystack's rule is
that nothing about the listener goes to a new service until they turn it on or hand over a credential, and the
README names exactly what is sent. The four sources keep that rule by needing a key, a name or a token; the
fallback needs a setting.

**Why iTunes is searched by name.** Apple's `lookup?isrc=` returned nothing for tracks that its search finds
(checked 2026-10), so an ISRC can't pin the song; the version check and the duration do the work instead.

**Why the label's albums go by id.** Deezer's album search has no sort by release date. Its album ids grow as albums
are added, so the highest ids are the newest additions, which for a label's catalogue is nearly always its newest
releases. Each album's real label is checked after the fetch, because the search matches loosely.

**Why kept mass and not a count.** A label with one 5-star song should outrank one with two songs merely liked once
and never rated; the weights follow the Ratings scale, and a song the listener passed on counts for nothing rather
than against the label (the re-ranker's `label_kept_rate`, ADR 0017, already weighs the misses).

**Why each listener brings their own Last.fm key.** A Last.fm API key belongs to one account and its terms; an open
source project can't ship one. Making one is free and takes a minute.

**Considered Options**:

- Spotify's recommendations and related artists. Closed to new apps, and haystack asks Spotify only for what the
  listener's account needs.
- MusicBrainz for a label's releases (it can sort by date). Another service at 1 request a second, thinner on new
  releases, and Deezer already has the label of every album haystack has seen.
- The iTunes candidates as their own pool entries, keyed by Apple's id. Two keys for one pool, two metadata paths,
  and no gain: the song is already a Deezer track that only lacks a clip.
- The preview fallback on by default (as first sketched). Apple would learn of the listener's candidates without
  their say.
- Other sources of previews (video sites and the like). No official 30-second clips, and their terms forbid it.
- Four optional sources resolved on Deezer, an opt-in Apple preview fallback, one throttle per service (chosen).

**Consequences**:

- The harvest asks more services, each at its own gentle rate. A run with every source set up makes at most a few
  hundred extra requests, most of them Deezer searches through the shared 1.1 s throttle.
- `deezer_vectors` gains a `provider` column (`'deezer'` for every existing row); `haystack report` counts the
  pool's vectors by provider. An Apple preview is a different encode of the same song and is aligned (ADR 0012)
  like a Deezer one.
- needle sees new `source` values with their labels (`source_label`); `/v1` is unchanged.
- A fresh installation behaves as 0.6.0 did: the three outside sources wait for a key, a name or a token,
  `deezer_label` waits until some kept song's label is known, and the fallback stays off. Each says so in the
  harvest's log.
