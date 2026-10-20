# Roadmap

Where haystack is heading next, in plain language. It's a direction, not a schedule or a promise: plans change
as we learn what works. What has already shipped is in the [changelog](CHANGELOG.md), and how haystack works
today is in [How haystack works](docs/how-it-works.md).

## Where things stand

haystack can now learn from what you think of its picks, not only from how close they sound to your songs. It
tests its ways of measuring "close" against the songs you kept and the ones you turned down, it can train a small
model on your verdicts to reorder its best matches, and it adjusts how many stretches it takes and how much each
source brings by what you keep.

All of that needs verdicts, and a lot of them. Each part waits until it has enough to be sure it's helping, and
until then haystack carries on as before. A weekly batch of 15 songs adds only a handful of verdicts a week, so
it can take months before those parts switch on. Most of what's next is about getting there sooner, and then
giving haystack better things to learn from.

## Now: helping haystack learn faster

haystack 0.5.0 brings this. The Calibrate list shows in needle from needle 0.6.0, the next needle release.

**A Calibrate list in needle.** Next to Discoveries and Rate your likes, needle gets a third list: the songs
haystack can't make its mind up about. Some are close calls near the top of its ranking, and some come from
further down, where the weekly batch rarely reaches. Rating them works like rating anything else, but each one
teaches haystack much more. Rating songs you already like tells haystack how much you like them, but each of
them is already a yes. Calibrate asks about songs that could go either way, so haystack learns where your yes
ends and your no begins.

**How you listen counts a little.** If you skip a discovery in needle after the first few seconds but within 30
seconds, without rating it, haystack takes that as a quiet "probably not". It weighs far less than a real verdict
and never changes anything in your Spotify account. And a liked song you haven't rated but play often counts for
more than one you liked once and forgot.

**You can see the progress.** `haystack report` shows which of the learning parts have switched on, and how many
more verdicts each one still needs.

## Now: knowing more about each song

haystack 0.6.0 brings this.

The fingerprint captures how a song sounds, but not where it comes from. haystack now also notes a song's record
label, how old it is, how popular it is and whether it's from a compilation, and learns from your verdicts whether
any of that matters to you. If you love obscure releases on a few small labels, it will notice. If you don't
care, those facts carry little weight. Popularity still doesn't rank songs on its own: it counts only as far as
your verdicts say it should.

## Next: more places to look

- **Songs that other listeners pair with yours**, from Last.fm, which knows what a very large number of people
  listen to together.
- **Your own ListenBrainz recommendations**, if you have a ListenBrainz account that records your Spotify
  listening.
- **New releases from the labels you like most.** In a lot of music, electronic music above all, the label is one
  of the best guides to taste there is.

As today, these sources only suggest. Every candidate is still judged by how it sounds, and a source that finds
songs you keep gets to bring more. Each new source is optional and stays off until you set it up. Using one sends
artist and song names to that service, and the README will say exactly what.

**A backup for previews.** Every candidate is fingerprinted from its 30-second Deezer preview, so today haystack
depends on Deezer. When Deezer has no preview of a song, haystack will try Apple's preview instead. That recovers
the songs Deezer can't play, and it's a first step towards haystack not depending on Deezer alone.

## Experiments: only if they help

These start as tests against real verdicts. Each becomes a feature only if it clearly beats what haystack does
today. If it doesn't, it stays on the shelf.

**Better ears.** The AI model that makes the fingerprints learned by sorting records into styles. That makes it
very good at telling deep house from hip-hop, and less good at telling two deep house tracks apart, which is
often where taste lives. Newer models learned from huge amounts of music on their own, without style labels, and
may hear finer differences. If one clearly does better at predicting what you keep, it will become an option you
can choose. Today's model stays the default until the case is clear. A new model needs more computing power, and
one long re-listen of your songs.

**An optional AI that knows about music.** A language model, the kind of AI behind chat assistants, can't hear a
song, but it has read a great deal about artists, labels and scenes. It might know that an artist is mainstream
pop rather than the underground scene you follow, or that a remix is a cash-in. The idea is an add-on that reads
a song's details (artist, title, label, year, style) and a summary of your taste, says how likely you are to keep
the song, and suggests songs worth looking up. haystack fingerprints those suggestions like any other candidate,
and drops the ones it can't find. The add-on is off unless you turn it on. It never sends audio or anything about
your accounts. You choose the provider and set a weekly spending cap, or point it at a model running on your own
computer, so nothing leaves your network.

## Smaller changes

- **A Dislike will reach the artist.** Disliking a song will also count a little against the same artist's other
  songs for a while.
- **Old likes will fade a little.** Songs you liked years ago will count slightly less than recent ones, so
  haystack keeps up as your taste moves. Songs you've rated keep their weight.

## Later

- **haystack's settings in needle.** needle may get a screen for the common haystack settings.
- **Spotify sign-in from needle.** needle would sign in to Spotify on your computer and hand the sign-in to
  haystack, with no address to paste back.

## What won't change

- **How a song sounds stays at the heart of it.** New facts and new models can reorder haystack's picks, but
  every candidate is still fingerprinted and judged by its sound.
- **Your taste stays on your server.** Anything that would send your data to a new service is off until you turn
  it on, and the docs say exactly what it sends.
- **It works without the extras.** A new install with no music folder, no ratings and no extra accounts works just
  as it does today. Each new part switches itself on once it has the data it needs.
- **needle and haystack stay two programs, and `config.toml` stays how you set haystack up.** The
  [README's roadmap](README.md#roadmap) explains why.
