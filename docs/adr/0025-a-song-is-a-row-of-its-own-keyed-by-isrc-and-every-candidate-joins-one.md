# A Song is a row of its own, keyed by ISRC, and every candidate joins one

ADR 0021 made the Song, one recording on however many releases, known by its ISRC, but only where it is read: a
Verdict's Song is worked out from `spotify_tracks.isrc`. The candidates had no Song. A Deezer list (a chart, a
radio, a playlist) gives track objects with no ISRC, and the harvest threw away the one `/track/{id}` it fetched
for the bpm, so a candidate had an ISRC only once the batch had looked it up on Spotify. Most of the pool had
none. So the harvest's dupe check (Deezer id, then `song_key`) let a best-of or a deluxe edition of a pool song in
as a candidate of its own, and `known.Known` could only use the names for most rows. The library took no part in
ISRC matching: few files carry an ISRC tag, though most carry a Spotify id whose track has one.

So from this release a Song is a row of its own:

- `songs(id, isrc UNIQUE, name_key, mbid)`. A track with an ISRC joins the Song of that code. A track without one
  joins the Song of its names: the first artist, the base title and the hard version markers (`name_key`), so a
  live take, a remix or an acoustic version is another Song and a remaster with no code of its own is not.
  `candidates`, `spotify_tracks` and `local_tracks` each have a `song_id`. A library file's ISRC is its own tag,
  else the ISRC of the Spotify track its Spotify id tag names, for the Song and for every match against the
  library (`library.library_refs`).
- The harvest fetches each new candidate's full track once, before it checks the library and Liked Songs, and
  keeps the ISRC with the bpm. A new track whose ISRC is already a candidate's, under another Deezer id, is a dupe.
- Older candidates get their ISRC from a backfill, `harvest.isrc_backfill_per_run` (300) a day, the pool first,
  and `haystack songs backfill` at once. A track Deezer has no ISRC for is asked once.
- The harvest, the sync and the batch join new rows to their Songs (`songs.assign`). When a name-keyed Song's
  track gets an ISRC, the Song takes the code, or folds into the Song that already has it.
- `known.Known` tries the Song first, then the ISRC, then the names rule (`known.same_song`), which still runs on
  every row.

**Why the names rule stays.** A remaster often gets an ISRC of its own, so two releases of one recording can be two
Songs here. They are never joined by their names: the ISRC is the exact key, and joining by names would make a
re-recording, or another song of the same name by the same artist, the same Song. Known's names rule is what
tells the listener has a remaster, and it is kept for every row.

**Why a Verdict's Song stays the ISRC.** ADR 0021 works out a Verdict's Song when it is read, from the track's
ISRC, and the API's `song` field is that ISRC. A name-keyed Song has no code to show needle, and reading the Song
row instead would change no answer where a track has an ISRC. So `verdicts.songs` is unchanged; only its claim
that a remaster shares the ISRC is corrected.

**Considered Options**: keep matching by ISRC column alone and backfill it (the dupe check and Known would still
have no key for a track without one, and the library would still sit out); merge Songs by their names (above);
key the liked songs' map (`liked_map`) by Song too (it goes through `spotify_tracks`, which has the Song already);
fetch ISRCs only at the backfill (every new candidate would wait a day for its dupe check).

**Consequences**: one more Deezer request for each new candidate the harvest would have excluded or found no
preview for; the others made it anyway for the bpm. The backfill of an existing pool takes one throttled request
per candidate, about 3 hours per 10,000, so a large pool takes weeks at the daily 300; run `haystack songs
backfill` outside the harvest's hours to finish sooner. A library file with a Spotify id tag now matches a
candidate of that ISRC whatever the names, so the harvest's intake, the sync's mapping of liked songs and To Rip
find more of the library. `haystack report` prints how much of the pool has an ISRC and how many Songs have more
than one candidate. `songs.mbid` is empty until MusicBrainz ids are kept.
