# Traits are measured on the server from Deezer previews

Spotify's development-mode Web API (2026) has no audio-features endpoint, so needle's Readings for the seven Traits come from the server: Essentia models run on the same 30 s Deezer preview the harvest already decodes. The discovery classifiers (danceability, acousticness, instrumentalness) reuse the `discogs-effnet-bs64-1` backbone pass that the genre styles already need; valence and energy come from a valence/arousal regression model (energy is the arousal estimate); tempo from beat tracking; loudness from EBU R128 over the clip. The listener's liked songs are measured the same way from their previews, so a track's Reading and their Baseline are comparable. Tracks the server has never seen are measured on demand when needle asks for them.

Readings are model estimates from a 30 s clip, not Spotify's numbers, and a track with no Deezer preview has no Readings (about 8% of liked songs in 2026-10).

**Considered Options**: ReccoBeats (Spotify-lookalike values keyed by Spotify id; a third party whose values can't be checked, with coverage gaps; already ruled out for discovery); measuring the playing audio on the desktop (whole track, but TensorFlow on the desktop the server exists to spare, and Python dependencies on every desktop).

## Note (2026-10-11): tempo takes Deezer's count when it agrees

On 20 liked songs, beat tracking on the clip (RhythmExtractor2013, and TempoCNN as a check) matched the bpm Deezer publishes for the track within 4% 16 times, and up to an octave all 20 times: the misses counted drum & bass at 85 instead of 170. So the tempo Reading is Deezer's bpm when it is the same tempo as the measurement at some metrical level (x2, x1/2, x3/2, x2/3), and the measurement otherwise (Deezer often has no bpm). Valence and energy use `deam-msd-musicnn-2`: no valence/arousal head exists for the effnet backbone, so they need one extra, small `msd-musicnn-1` pass.
