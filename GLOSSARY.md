# haystack

Personal music discovery and review for one Spotify account: the server finds and measures songs the listener might like, a desktop app plays them for review, and the listener's verdicts refine what the server finds next.

These are the server's terms, and the HTTP API's (`docs/api.md`). needle, the desktop app, adds its own in [needle's glossary](https://github.com/kurtlieber/needle/blob/main/GLOSSARY.md).

**Listener**: the one person whose Spotify account a haystack serves.

## Language

### Verdicts

**Verdict**:
The listener's current judgment of a track: a Rating, a Pass or a Dislike. A track has at most one; a newer one replaces the older.
_Avoid_: feedback, opinion

**Rating**:
The listener's 1–5 star measure of how much they like a track. Every star count is a degree of liking: 1★ means they like it a little, 5★ means they love it.
_Avoid_: score (that is the recommender's number), thumbs-up, like (Spotify's heart)

**Dislike**:
The listener's verdict that they hate a track and never want to hear it again. It is not a rating and sits below 1★.
_Avoid_: 0 stars, 1 star, thumbs-down, skip

**Pass**:
The listener's verdict that a track is not for them, though they don't hate it. It is not a rating.
_Avoid_: meh, no, skip, 0 stars

### Liking

**Liked Songs**:
Spotify's heart list, seen by every Spotify client. It mirrors the listener's Ratings of 3★ and up.
_Avoid_: likes (when you mean Ratings), favorites, saved tracks

**Unrated like**:
A track in Liked Songs that has no verdict yet.
_Avoid_: liked song (ambiguous)

### Traits

**Trait**:
One of the seven measured qualities of a track's sound: tempo, valence, energy, danceability, acousticness, instrumentalness and loudness.
_Avoid_: audio feature (Spotify's retired term), metric, stat

**Reading**:
A track's measured value for one Trait.
_Avoid_: score, value

**Baseline**:
The listener's typical Reading for a Trait: the mean over the tracks they like, weighted by their Ratings, with an Unrated like counting as 3★.
_Avoid_: average, taste profile, my rating

**Usual range**:
The middle half of the Readings of the tracks the listener likes for one Trait, weighted like the Baseline. A track whose Reading falls outside it is "slower", "darker" and so on than what they usually like.
_Avoid_: normal range, comfort zone

### Review

**Review**:
Listening to a track in order to give it a verdict.
_Avoid_: rating (that is the verdict, not the act)

**Review queue**:
The tracks waiting for a verdict, in the order the app plays them: Discoveries, Unrated likes, and a Playlist review once the listener starts one.
_Avoid_: playlist (a Spotify object), queue (Spotify's play queue)

**Followed playlist**:
A playlist in the listener's Spotify library that someone else made. The server can't list its songs (Spotify's developer mode), so needle's engine reads them.
_Avoid_: shared playlist, public playlist

**Discoveries**:
The private Spotify playlist where the server's weekly picks land for the listener to hear.
_Avoid_: recommendations

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

**Confidence**:
The re-ranker's chance, 0–1, that the listener keeps a Discoveries pick (rates it 3★ or more, or likes it with no Rating). Only a top pick of a re-ranked batch has one; a client may show it as a percentage.
_Avoid_: score (the similarity the picks start from), match, certainty
