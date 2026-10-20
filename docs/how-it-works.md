# How haystack works

This page explains, without the jargon, how haystack decides what to put in your Discoveries playlist, and what
the seven numbers needle shows for every song mean. You don't need to know anything about audio to follow it.
(The technical version, with the maths, is in [reference.md](reference.md).)

## The short version

1. haystack **listens** to the songs you love and turns each one into a "sound fingerprint".
2. Every day it collects **candidates**, songs you probably haven't heard, and fingerprints a 30-second clip
   of each.
3. Once a week it picks the candidates whose fingerprints are **closest to your songs**, and puts them in
   Discoveries. Once you've judged enough of its picks, it also learns which close matches you tend to keep.
4. You listen. What you like, and what you don't, **teaches it** for next week.

There's no genre list to fill in and no popularity ranking. Other listeners' habits and the names of styles help
find candidates, but what decides is how the songs sound and, once you've judged enough, what you kept before.

## 1. Listening to the songs you love

haystack starts from the songs it knows you like:

- your Spotify **Liked Songs**,
- songs you **rate** in needle (a 5-star song counts for more than a 2-star one),
- and, if you set one up, every song in **your own music folder**.

For each song, an AI model listens to it and produces a long list of numbers that sums up how it sounds:
its instruments, its rhythm, its production, its mood, the "feel" that makes two songs sound alike even when
they're in different genres. That list is the song's **fingerprint**. Songs that sound alike have fingerprints
that are close together.

The model was trained on a huge catalogue of records labelled with their styles, so it's learned to hear the
difference between, say, shoegaze and dream pop, or deep house and tech house. It doesn't know any titles or
artists. It only hears sound.

For your own files, haystack listens to nine 30-second slices spread across the song, so a song with a quiet
intro and a loud chorus gets credit for both.

The new songs it finds are 30-second previews, and a preview and your own file of the same song are measured a
little differently, so haystack learns the difference from the songs you have both ways and corrects for it.

## 2. Finding candidates

Every day haystack collects new songs to consider from:

- **Deezer**: charts, editorial picks and new releases, "artist radio" and related artists starting from artists
  you like, and playlists named after the styles you like most.
- **ListenBrainz**: an open music database that knows which recordings people tend to listen to together.

Four more places wait until you set them up (the README's "More places to look" says how, and what each one
is told):

- **Last.fm**: songs similar to yours, and the best-known songs of artists similar to yours.
- **Your ListenBrainz recommendations**: if ListenBrainz records your listening, it works out songs for you every
  week, and haystack considers them.
- **LB Radio**: ListenBrainz's radio playlists for the styles and artists you like most.
- **Your record labels**: the newest albums from the labels whose songs you keep most. This one needs nothing
  from you, only a few kept songs.

Artist radio, related artists, ListenBrainz and Last.fm start from **seeds**: the songs you rated 4 or 5 stars and the
discoveries you kept lately come first, then your other liked songs, then your music folder. An artist is a seed
at most once a week per source, so the harvest keeps moving. Once a week a wider harvest draws artists from
across all the styles you like.

These sources only *suggest*. They never decide whether a song is good for you. Each candidate gets
fingerprinted from its 30-second preview, exactly like your own songs. The preview comes from Deezer, or, for the
few songs Deezer has none for, from Apple if you turn that on. Sources that find songs you keep get to
bring more each day, and those that don't get to bring fewer. At most a fifth of each batch comes from the charts.

## 3. Picking the week's songs

For every candidate, haystack finds the songs of yours it sounds **most** like, and how close they are. That's
the candidate's score. A song doesn't have to sound like your taste "on average": being close to a few of the
songs you love is enough. That's how a mostly-jazz listener with a soft spot for 80s synth-pop still gets
synth-pop. (There are a few ways to measure "close". haystack tests them on the songs it already offered you,
keeps the one that best puts the ones you kept above the ones you turned down, and checks again every week; until
you have turned down enough, it tests them on your own likes.)

Once you have rated enough, haystack also learns which of the close matches you actually like. Sounding close
isn't the whole story: you may keep the brighter songs among them and pass on the ones from the charts. When you
have judged about 40 of its picks, haystack trains a small model on your verdicts and lets it reorder its 400
closest matches, but only when the model would have predicted your past verdicts better than closeness alone;
otherwise it goes by closeness and checks again next week. Each song the model moved comes with its chance that
you keep it and a reason or two, such as "close to your 4-5★ tracks", which needle shows on the track. To the
model, 1 or 2 stars means "not for the batch": you like the song a little, but the batch is for the ones you keep.
haystack also notes the label, the year and how popular a track is, and learns whether those matter to you.

Then it fills the batch (15 songs, unless you changed `batch.size`):

- **Most picks are the closest matches.**
- **About one in five is a deliberate stretch**: a song that's a decent match but not a top one. Without
  these, haystack would only ever find more of exactly what you already have. The share follows what you keep:
  when the stretches do better than the closest matches, there are more of them (up to two in five), and fewer
  when they do worse (down to one in ten).
- **The picks are spread across the styles you like.** No more than about two in five of the closest matches
  share a style, and no more than two picks are closest to the same song of yours, so one corner of your taste
  can't fill the week. Among close matches, a song unlike the ones already picked goes first.
- **One song per artist**, so one band can't take over the week.
- **Songs that sound like ones you turned down rank lower.** A Pass or a Dislike in needle, or a discovery you
  heard and didn't like, pushes down candidates that sound very like it.

## 4. Learning from you

You don't have to do anything special. haystack watches what happens to each discovery:

| You... | haystack hears |
|---|---|
| Like it in Spotify (the heart) | **Yes.** It becomes one of "your songs" and shapes future picks. |
| Rate it in needle | **Yes, this much** (1 to 5 stars), or **no** (Pass), or **really no** (Dislike). |
| Leave it in Discoveries until the next batch | **No**, played or not: the playlist is meant to be heard. |
| Remove it from Discoveries yourself | **No.** |

Songs added less than a day before a new batch get a pass, so a late Sunday batch doesn't count against you.

How you listen counts a little too. If you play a discovery in needle for more than 5 seconds and skip it within 30,
haystack takes that as a gentle no, much weaker than a Pass, and only for working out which close matches you keep.
It never removes the song or touches Spotify because of it. And of the liked songs you haven't rated, the ones you
keep playing count as more loved than the ones you never play.

**Help it learn faster.** The Calibrate list in needle is the songs haystack can't make its mind up about: some
from near the top of its ranking, some from further down that the weekly batch would never reach. Rate a few and
the next batch learns more from them than from anything else. A rating there is the same as anywhere else in
needle; it just goes where haystack is most unsure.

haystack also keeps score of where its good finds come from. `haystack sources` shows how many of each
source's songs you kept, and how many of the songs found from each kind of seed, so you can turn off the sources
that don't work for you. `haystack report` gives the bigger picture: how
many offers you kept, week by week, how you've rated things, and what the next batch is drawing on. It shows
numbers only, never song names, so it's safe to share when asking for help.

## The seven Traits

needle shows seven numbers for every song, called **Traits**. Tempo is the big BPM number at the top of the song
card. The other six (valence, energy, danceability, acousticness, instrumentalness and loudness) are the scales
below it. If you remember Spotify's old "audio features", they're the same
idea. Spotify no longer offers those to apps like this, so haystack measures its own from the song's 30-second
preview.

Two things to know first:

- **Traits mostly describe a song.** Discoveries are chosen mainly by the fingerprint above, which captures far
  more than seven numbers can. The Traits are there so you can see, at a glance, *how* a song differs from what
  you usually like. Once haystack learns from your verdicts (see "Picking the week's songs"), it also notices if
  you tend to keep songs that are, say, darker or calmer than usual, and may give that as a reason for a pick.
- **They're estimates.** An AI model makes an educated guess from a 30-second clip. A song that changes a lot
  after its first minute may not match its numbers. Songs with no preview have no Traits.

needle compares each song with your **Baseline** (the typical value across the songs you like, weighted by your
ratings) and your **usual range** (the middle half of your songs). When a song falls outside your usual range,
needle says so in a short sentence, such as "Slower, darker and more instrumental than you usually like."

### Tempo

**How fast the beat is**, in beats per minute (BPM): the speed you'd tap your foot or nod your head to. It's the
large number on needle's song card, with your own typical tempo ("Yours: …") under it.

- 60 to 80: slow ballads, a lot of ambient and soul.
- 85 to 100: most hip-hop, a relaxed groove.
- 120 to 130: house and most dance pop.
- 160 to 180: drum & bass, a lot of punk.

Fast music isn't always *energetic* (see Energy), and slow music isn't always sad. Tempo is just the speed of
the pulse. Measuring it is tricky because a beat can be counted at half or double speed; haystack counts the way
people do, so drum & bass reads 170, not 85.

### Valence

**How happy or sad a song sounds**, from 0 (sad, dark, tense) to 1 (happy, bright, cheerful). needle calls a song
below your range "darker" and one above it "happier".

Valence is about the *sound*, not the lyrics: major keys, bright chords and bouncy rhythms read high, minor keys
and brooding textures read low. A breakup song with a sunny, upbeat arrangement can score high. A gentle song in a
minor key scores low even if its words are hopeful.

### Energy

**How intense a song feels**, from 0 (calm, soft, still) to 1 (intense, driving, loud and busy). needle says
"calmer" or "more intense".

Think of how a song makes your body feel: a solo piano piece sits near 0, a metal track or a festival anthem near
1. Energy and valence together place a song's mood: high energy plus high valence is euphoric, high energy plus low
valence is angry or tense, low energy plus high valence is content, and low energy plus low valence is melancholy.

### Danceability

**How easy a song is to dance to**, from 0 (loose) to 1 (danceable). needle says "looser" or "more danceable".

It mostly hears how steady and strong the beat is, and how regular the rhythm. Disco, house and funk score high.
Free jazz, rubato piano and songs that keep changing tempo score low. It's not a judgement of quality, and a
song can be very energetic without being danceable (a lot of punk), or danceable without being energetic (a slow
groove).

### Acousticness

**Acoustic or electronic**, from 0 (electronic: synths, drum machines, heavy production) to 1 (acoustic:
guitars, piano, strings, voices, played by people in a room). needle says "more electronic" or "more acoustic".

It's the model's confidence that the song is acoustic, not a measure of *how much* of it is. An unplugged folk
song is near 1, a techno track near 0, and a rock band with lots of distortion and effects is somewhere in the
middle.

### Instrumentalness

**Vocal or instrumental**, from 0 (someone is singing or rapping) to 1 (no vocals). needle says "more vocal" or
"more instrumental".

Like acousticness, it's how sure the model is that there's no voice, not the share of the song without one.
"Ooh"s and "aah"s often count as instrumental. A spoken word sample in an electronic track may or may not.

### Loudness

**How loud the recording itself is**, in LUFS, the unit streaming services use to measure loudness. The scale
runs below zero: closer to 0 is louder.

- Around -6 to -9: most modern pop, rock and EDM, mastered to sound as loud as possible.
- Around -10 to -14: a lot of indie, older records, and anything mastered with more room to breathe.
- -15 and below: classical, jazz and quiet acoustic music.

This is about how the record was made, not your volume knob. Streaming services even out the playback volume,
but a loud master still tends to sound denser and more compressed, and a quiet one more open and dynamic. needle
says "quieter" or "louder".

## What haystack doesn't do

- **It doesn't use Spotify's audio**, or Spotify's recommendations. Only your own files and the previews (Deezer's,
  or Apple's if you turn that on) are analysed, on your own machine.
- **It doesn't rank by popularity.** An obscure B-side and a chart hit get the same chance if they sound right.
  Chart songs fill at most a fifth of a batch, and if you keep passing on them, haystack learns that too.
- **It doesn't read lyrics or genre tags** to choose. The fingerprint is all sound.
- **It doesn't share your taste with anyone.** Your ratings and fingerprints stay in haystack's database on your
  server. The optional places to look are told only names: of songs, artists, styles or labels, never your
  ratings.
