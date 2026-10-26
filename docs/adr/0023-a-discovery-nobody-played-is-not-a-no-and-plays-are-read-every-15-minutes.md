# A discovery nobody played is not a no, and plays are read every 15 minutes so haystack can tell

The clear before each batch gave every unliked discovery older than `batch.clear_min_age_hours` a no (a Pass,
offer outcome `negative`), played or not: Discoveries was meant to be listened through. A week's batch often
isn't. A track nobody played says nothing about the song, yet its Pass weighed in the negative penalty and the
re-ranker as if the listener had turned it down, and pushed down candidates that sound like it. The hourly
14-day expiry already told the two apart (played: negative; unplayed: back to the pool, retired after
`batch.max_unplayed_expiries`), but the clear, which ends most offers, didn't (kurtlieber/dotfiles#361).

Telling the two apart needs a complete play log. Spotify's recently played returns the last 50 plays only, and the
hourly sync read it, so a busy hour or an outage could drop plays without a trace (kurtlieber/dotfiles#360).
reference.md called that "unheard by design": while every track was a no either way, a lost play changed nothing.
Once never played means unplayed, a lost play would turn a no into an unplayed and count toward retirement.

So:

1. **Plays every 15 minutes.** `haystack plays` polls recently played alone (`schedule.plays = "every 15m"`, a new
   `every Nm` form for N dividing 60; `haystack-plays.timer`, `OnCalendar=*:0/15`). Spotify logs a play after 30
   seconds, so 50 plays take at least 25 minutes: polls 15 minutes apart can't overflow while the server is up.
   The hourly sync still polls too; the stored cursor makes the second poll cheap.
2. **A full page is a Play gap.** A poll that gets a full page records the span from the cursor it asked from (0
   for none) to the oldest play on the page in `play_gaps`: plays in it may be missing. On the first start of
   this version, a database that already has a cursor gets one gap from 0 to that cursor, since nothing watched
   for a full page before. `haystack status` shows the count and the latest gap.
3. **The clear splits three ways.** It polls plays once more, then judges an older unliked track by its plays
   since it was added, from Spotify or needle:
   - played: a no, as before;
   - never played, and no Play gap overlaps its time on the playlist: unplayed (`never_played`), counted toward
     `batch.max_unplayed_expiries` like the 14-day expiry, so the second one retires it;
   - never played, but a gap overlaps: unplayed (`unknown`), not counted.

   Tracks younger than the minimum age, `--clear-recent keep` and `--no-penalty` keep their behaviour. The
   crash-recovery path (`clear_pending`) carries the new reasons, and the count is taken only when the clear
   closes the offer, so a retried clear never counts a track twice.

**Why a gap only withholds the count.** A track a gap covers might have been played (a no) or not (unplayed and
counted). Unplayed without the count writes no verdict and retires nothing on a guess: at worst the song is
offered again, and the next clear decides it with a complete log.

**Why all the older history is one gap.** Before this change nothing watched for a full page, so no older play
can be trusted to be complete. Marking it all unknown means the first clear after the upgrade counts no
never-played track toward retirement; every offer added after it is judged in full. Trusting the old log could
retire a song the listener did hear.

**Why a full page and nothing finer.** Spotify doesn't say whether plays were dropped, and a full page is the only
sign. It can be exactly 50 plays with nothing lost; the gap is then spurious, but it only withholds a count. If
Spotify returns the oldest 50 after the cursor rather than the newest, nothing is lost (the cursor moves on and
the next poll reads the rest), and the gap is again only cautious.

**Why the clear reads `plays`, not the offer's `played` flag.** The hourly sync sets the flag, so a play from the
last hour may not have reached it. The clear reads the plays themselves, after polling, so a play from the last
15 minutes counts too.

**Considered Options**: polling every 5 minutes (three times the calls; 15 minutes already can't overflow); a
fixed 15-minute cadence instead of a schedule form (`every Nm` costs a few lines and keeps `[schedule]` the one
place for times); keeping a no for every unliked track (an unheard song becomes the listener's judgment);
unplayed without ever counting toward retirement (a song nobody plays would be offered forever); trusting the play
log from before this change (above); `haystack sync --plays-only` instead of a `plays` command (a job of its own
reads better in the schedule, the units and `runs`).

**Consequences**: fewer nos. A week's unheard tracks no longer push down candidates like them or teach the
re-ranker a no, so its labels shrink to what was heard. The kept rate rises, since its judged count already
leaves out unplayed tracks without a verdict. A song that goes unplayed in two batches is retired. The plays job
writes a `runs` row every 15 minutes, and its unit alerts on failure like sync's, so a Spotify outage alerts up
to four times an hour. The 14-day expiry in the hourly sync still reads the offer's `played` flag and ignores
gaps; with a clear before each batch it rarely decides an offer. `play_gaps` grows only when a poll overflows.
