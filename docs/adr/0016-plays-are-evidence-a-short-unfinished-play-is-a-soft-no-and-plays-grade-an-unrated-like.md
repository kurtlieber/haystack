# Plays are evidence: a short unfinished play is a soft no, and plays grade an Unrated like

needle posts its plays to haystack (`POST /v1/plays`), with how long each lasted (`ms_played`) and whether it
reached the end (`completed`). haystack kept neither before 0.5.0. Spotify's recently played, which the hourly
sync reads, has neither. From 0.5.0 haystack keeps both and uses plays in two ways.

1. **A soft no.** An offer or a Calibrate pick (ADR 0015) that has no Verdict, no decided outcome and is not in
   Liked Songs, and whose latest needle play stopped before 30 s without finishing, is a no of weight 0.5 for the
   re-ranker's fit (`rerank.soft_labels`, `rerank.soft_pass_ms`, `rerank.soft_pass_weight`). A Verdict, a like, or
   a later play that runs on takes it back.
2. **Unrated likes graded by plays.** An Unrated like counted as 3★ whatever the listener did with it. It now
   gains a star per doubling of its plays in the last 90 days, from any context, at most two:
   `3 + min(2, floor(log2(1 + plays)))`, so one or two plays make 4★ and three or more 5★. This grades the
   positives' star bonus (ADR 0002), the re-ranker's star features, the Baseline's weights and the eval's kept
   weights. A Rating always stands.

**Why a soft no is not a Verdict.** A Verdict is the listener's judgment (ADR 0005), and it acts: a Pass takes a
track out of Discoveries, a Rating of 3★ or more hearts it on Spotify. A skip is only a guess at a judgment. The
listener may have skipped the moment, not the song. So a soft no writes nothing to the verdicts and nothing to
Spotify, weighs half a Pass, and only joins the re-ranker's fit. It counts toward neither the gate's labels nor the
leave-one-out AUCs, and the eval leaves it out: the gates and the variant choice rest on what the listener said,
not on what haystack inferred. A long play without a Verdict says little either way and is no label.

**Why only needle plays carry length.** Spotify's recently played has no length and no completion, and a play there
may be a radio station, a shuffle, or someone else at the speaker. needle knows when it started a track and when it
stopped. So only a needle play can be a soft no. For grading an Unrated like, plays from any context count: there a
play means the listener chose to hear a song they had already hearted.

**Why a star per doubling, and at most 5★.** Repeat plays of a liked song say a lot at first and little later: the
tenth play says less than the second. A star per doubling keeps a song on heavy rotation from outweighing a 5★
Rating, and the cap keeps an Unrated like level with one at most. The 90-day window keeps it current: a song played
often a year ago and not since goes back to 3★.

**Considered Options**:

- A skip as a Pass. It would take tracks out of Discoveries and write to Spotify on a guess, and the listener would
  find Passes they never gave.
- Soft nos in the eval and the gate too. The AUC objective and the re-ranker's switch would then partly measure
  haystack's own guesses.
- Skips from Spotify's recently played. With no length and no completion, a track skipped after five seconds can't
  be told from one played through.
- Unrated likes graded by play count, linearly. One song on repeat would outweigh every Rating.
- Soft nos for the re-ranker's fit only, from needle plays, and Unrated likes graded by doublings of their recent
  plays (chosen).

**Consequences**:

- The fit gets more labels at no cost to the listener, and the gate still opens on hard labels alone.
- An Unrated like the listener plays often pulls the pool toward itself as a 4★ or 5★ Rating does. For 0.5.0,
  `haystack eval` prints each variant's AUC and mean percentile with and without the grading, so its effect is
  visible. `scoring.verdicts.unrated_like_plays = false` turns it off.
- Plays stored before 0.5.0 have no length and are never soft nos.
- A soft no is only as good as the play report: a play cut short by a crash or a laptop going to sleep looks like a
  skip. Half a Pass's weight, and the latest play winning, keep that small.
- A fresh installation has no plays: no soft nos, and every Unrated like is 3★, as before.
