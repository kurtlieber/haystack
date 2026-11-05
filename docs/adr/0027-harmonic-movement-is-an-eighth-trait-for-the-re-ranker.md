# Harmonic movement is an eighth Trait, for the re-ranker

A listener can like a dance track's beat and still turn it down because the same four or five seconds repeat with
nothing changing: one chord stab or riff, looped over the beat. That shows in the first 30-45 seconds, so the
30-second preview holds it. The fingerprint (the effnet embedding) and the seven Traits say what a clip sounds
like, not whether anything in it moves.

An experiment (2026-11-05, dotfiles #450) measured five candidate measures on 446 previews the listener had
already Rated, kept or turned down, using no new verdicts:

- **Harmonic movement**, the median chroma difference between bars on the beat grid, separated kept from
  turned-down electronic picks at AUC 0.65 (95% CI 0.59-0.72, n=286). Within House it reached 0.72, within
  Bassline 0.73. It correlated at |r| ≤ 0.16 with the stage-1 score and every Trait, so it adds information.
  Adding it to stage 1 and the Traits moved a 10-fold AUC on electronic picks from 0.53 to 0.66.
- **Timbre loop**, loop error, block change and distinct bars, which measure the beat and the texture
  repeating, all had CIs spanning 0.5. Almost every dance track loops its beat; what the listener reacts to is
  that nothing harmonic moves over it.
- **The listener's ear check.** Of the five lowest-scoring electronic tracks, four were what they meant. The miss
  changes at 0:18 and 0:34 of the song; its preview likely starts after both.

So from 0.13.0:

1. **`harmony` is a Trait.** `traits.measure` reads it from the same clip and the same RhythmExtractor2013 beat
   ticks the tempo uses: HPCP chroma per sixteenth note, sixteen to a bar, and the median over every pair of
   bars of their mean absolute difference. Fewer than six whole bars gives no Reading. `models` gains
   `+harmony-1`.
2. **Older Readings get it without a full re-measure.** `traits.harmony_backfill` decodes the preview once and
   runs only the beat pass and the chroma, `harvest.harmony_backfill_per_run` (300) a day in the backfill's
   order (Discoveries, likes, Rated tracks, then the pool). A row it tried carries the new `models` version, so
   it is not asked again. `haystack traits backfill-harmony` runs it at once.
3. **The re-ranker decides whether it helps.** It becomes `trait:harmony` like the other Traits and passes the
   same leave-one-out and `rerank.min_gain` gate (ADR 0011). No sign is forced on it, as on the other Traits.
4. **needle doesn't show it yet.** `harmony` is in `baseline.traits` and a card's `readings`. needle skips a
   Trait it doesn't know (needle's ADR 0014), so a scale and its wording are a needle change of their own.
5. **The language model's profile leaves it out.** Plan 3 gates the model's blend on the judge's run 2 (dotfiles
   #309), measured with the seven-Trait prompt; the prompt stays as measured until then.

## Considered

- **A dedicated "repetitiveness" feature in the re-ranker only**, not a Trait. It would need the same storage
  and backfill, and a Trait also gets a Baseline and a usual range, which the re-ranker's trait features use.
- **The beat-level loop measures.** No signal, see above.
- **Measuring the full track.** haystack keeps no audio beyond the preview (ADR 0001), and the listener hears
  the loop in the first 30-45 seconds, which the preview mostly covers.

## Consequences

- One more chroma pass per measured clip: about 0.06 s for 30 seconds on archie, next to the effnet and musicnn
  passes.
- The 30-second slice can misjudge a song that changes just before or after it.
- The measure was chosen after looking at five, and on offered picks, where stage 1 is range-restricted (its own
  AUC there was ~0.48). The re-ranker's gate on the listener's own verdicts is the check that it holds up.
