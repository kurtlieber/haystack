# haystack

Personal music discovery and review for one Spotify account: the server finds and measures songs the listener might like, a desktop app plays them for review, and the listener's verdicts refine what the server finds next.

These are the server's terms, and the HTTP API's (`docs/api.md`). needle, the desktop app, adds its own in [needle's glossary](https://github.com/kurtlieber/needle/blob/main/GLOSSARY.md).

**Listener**: the one person whose Spotify account a haystack serves.

## Language

### Verdicts

**Verdict**:
The listener's current judgment of a track: a Rating, a Pass or a Dislike. A track has at most one; a newer one replaces the older.
_Avoid_: feedback, opinion

**Song**:
One recording, on however many releases: the album track, the deluxe edition and the best-of are one Song, known by its ISRC. A remaster often gets an ISRC of its own and is then another Song, though still a song the listener has. Each release is a track of its own on Spotify. The server keeps one row per Song, by ISRC where it has one, else by its names, and every candidate, Spotify track and library file joins one (ADR 0025). A Song's verdict is the newest verdict on any of its tracks, and outside Discoveries and Calibrate it is the one that counts (ADR 0021).
_Avoid_: recording (fine in prose), release (one of its tracks)

**Rating**:
The listener's 1–5 star measure of how much they like a track. Every star count is a degree of liking: 1★ means they like it a little, 5★ means they love it.
_Avoid_: score (that is the recommender's number), thumbs-up, like (Spotify's heart)

**Dislike**:
The listener's verdict that they hate a track and never want to hear it again. It is not a rating and sits below 1★.
_Avoid_: 0 stars, 1 star, thumbs-down, skip

**Pass**:
The listener's verdict that a track is not for them, though they don't hate it. It is not a rating.
_Avoid_: meh, no, skip, 0 stars

**Possible cover**:
A track that may stand in for another recording, its Original: a cover, a karaoke or tribute version, or a live take. The server flags it from its title, a borrowed ISRC or its popularity next to another artist's track of the same title. A hint for the listener, never a verdict (ADR 0022).
_Avoid_: cover (the card's art), fake, duplicate

**Original**:
The recording a Possible cover stands in for: the original artist's studio recording, or for a live take the same artist's studio one.
_Avoid_: real song, master

**Swap**:
The listener's decision that a Possible cover should give way to its Original: the cover's verdicts are taken back, and in the listener's own playlists and Liked Songs the Original takes its place. The one edit haystack makes to the listener's own playlists.
_Avoid_: replace, dedupe

### Generated songs

Songs a client such as spindle has a music service make, kept in the optional generated-song store (ADR 0029). None of these is a Song or a Verdict, and none changes what haystack recommends.

**Idea**:
What a client asked a music service for: the idea in the client's own form (its style, its lyrics and who wrote them), the reason the client chose it, and the exact request it sent.
_Avoid_: prompt (only part of the request), generation

**Take**:
One piece of audio a music service made from an Idea, known by the service and the service's id for it. One Idea usually has several. It is not a Song: haystack never harvests, measures or offers it.
_Avoid_: Song, clip, track, generation

**Verdict on a take**:
The listener's judgment of a Take: 1–5 stars or a Dislike, with an optional note on what worked and what didn't. There is no Pass. Append-only; the latest by when it was given is current (ADR 0005). It is not a Verdict, and the re-ranker, the Baseline and the Review never read it.
_Avoid_: Verdict (on its own), rating (when a Dislike is meant), feedback

**Kept**:
A Take the listener has filed in their own collection. The first time counts.
_Avoid_: saved, liked, downloaded

### Liking

**Liked Songs**:
Spotify's heart list, seen by every Spotify client. It mirrors the listener's Ratings of 3★ and up.
_Avoid_: likes (when you mean Ratings), favorites, saved tracks

**Unrated like**:
A track in Liked Songs that has no verdict yet.
_Avoid_: liked song (ambiguous)

### Traits

**Trait**:
One of the eight measured qualities of a track's sound: tempo, valence, energy, danceability, acousticness, instrumentalness, loudness and harmony (harmonic movement).
_Avoid_: audio feature (Spotify's retired term), metric, stat

**Reading**:
A track's measured value for one Trait.
_Avoid_: score, value

**Baseline**:
The listener's typical Reading for a Trait: the mean over the tracks they like, weighted by their Ratings, with an Unrated like counting as 3★, or up to 5★ when the listener plays it often.
_Avoid_: average, taste profile, my rating

**Usual range**:
The middle half of the Readings of the tracks the listener likes for one Trait, weighted like the Baseline. A track whose Reading falls outside it is "slower", "darker" and so on than what they usually like.
_Avoid_: normal range, comfort zone

### Review

**Review**:
Listening to a track in order to give it a verdict.
_Avoid_: rating (that is the verdict, not the act)

**Review queue**:
The tracks waiting for a verdict, in the order the app plays them: Discoveries, Calibrate, Unrated likes, and a Playlist review once the listener starts one.
_Avoid_: playlist (a Spotify object), queue (Spotify's play queue)

**Followed playlist**:
A playlist in the listener's Spotify library that someone else made. The server can't list its songs (Spotify's developer mode), so needle's engine reads them.
_Avoid_: shared playlist, public playlist

**Discoveries**:
The private Spotify playlist where the server's weekly picks land for the listener to hear.
_Avoid_: recommendations

**Calibrate**:
The Review queue of the candidates the server is least sure about, drawn from across its ranking rather than only the top. A verdict there is a Rating, Pass or Dislike like any other; it teaches the re-ranker the most.
_Avoid_: test queue, training set, exploration (the batch's random picks)

**To Rip**:
The private Spotify playlist of tracks the listener rated 3★ or more that aren't in their music library yet (`spotify.to_rip`, off by default).
_Avoid_: wishlist, download list

### Picking

**Seed**:
A track a harvest source starts from to find candidates: a song the listener rated 4★ or 5★ or a discovery they kept lately, another liked song, or a file in their music library.
_Avoid_: positive (the songs a candidate is scored against), source (where a candidate came from)

**Kept rate**:
The share of decided Discoveries picks the listener kept (a Rating of 3★ or more, or liked with no Rating), by source or by seed. A source's harvest cap follows it.
_Avoid_: like rate (liked over decided, without the Ratings), hit rate

**Soft no**:
A short, unfinished play in needle of a Discoveries pick or a Calibrate track that has no verdict. It counts a little against the track for the re-ranker only; it is not a verdict and changes nothing in Spotify.
_Avoid_: skip, Pass, implicit verdict

**Unplayed**:
A Discoveries pick that left the playlist unliked with nothing learned about it: never played (from Spotify or needle) since it was added, or let off because it arrived less than a day before the clear. It is not a verdict and counts nothing against the song; it goes back to the pool. A never-played pick counts toward retirement, so a second time retires it, unless a Play gap means a play may be missing (ADR 0023).
_Avoid_: no, Pass, expired, skipped

**Play gap**:
A span of time in which some of the listener's Spotify plays may be missing, because Spotify returned a full page of its last 50 when the server asked. A never-played pick whose time in Discoveries overlaps one is not counted toward its retirement (ADR 0023).
_Avoid_: outage, missed sync, hole

**Confidence**:
The re-ranker's chance, 0–1, that the listener keeps a Discoveries pick (rates it 3★ or more, or likes it with no Rating). A top pick of a re-ranked batch has one, and so does a Calibrate track whenever the re-ranker has a fitted model; a client may show it as a percentage.
_Avoid_: score (the similarity the picks start from), match, certainty

**Taste profile**:
What a request to the optional language model says about the listener: the artists they rate and like most with their mean stars, the artists of songs they turned down, their top styles and their Usual ranges. It is built without the songs the request asks about, and never carries a Verdict on them.
_Avoid_: Baseline (the typical Reading per Trait), Taste brief (what a client reads), history, model of the listener

**Taste brief**:
What `GET /v1/taste` serves a client that writes song ideas (spindle): how often each style, tempo band and instrumental level gets 5★, over every Song with a Verdict and over those that were new to the listener, the recent 5★ and disliked new Songs, the most-played 5★ Songs, and the Baseline. Counts from stored Verdicts, styles and Readings, with no language model (ADR 0028).
_Avoid_: Taste profile (what haystack's own language model is told), taste summary, profile
