# Calibrate: haystack asks the listener about the candidates it is least sure of

The re-ranker (ADR 0011) and the AUC objective (ADR 0009) wait behind gates: 40 labels and a gain of 0.05 for the
one, 15 kept and 15 turned-down offers for the other. Discoveries brings one batch a week, about 15 tracks, all
from the top of the pool, and some of them pass without a play or a Verdict. At that rate the gates open after
months. And the labels they wait for all come from one narrow place, the candidates stage 1 already likes most: a
model trained only there learns little about the rest of the pool, and the eval's turned-down offers are all close
calls.

A second Review queue, **Calibrate**, asks the listener about the candidates where a Verdict teaches the most. The
scored pool is split by stage-1 percentile into bands, each with its share of the picks: 50-75 0.2, 75-90 0.3,
90-97 0.3 and 97-100 0.2. Within a band the candidates the re-ranker gives the most even odds come first
(`|chance - 0.5|`). The model is fitted whenever it has two yes and two no labels, whether or not its gate lets it
order the batch, and its chance is now kept for every scored pool candidate. Without a model, the candidates the
ways of measuring closeness disagree about most come first: the standard deviation of a candidate's percentile
across the five embedding variants and `genre`.

The hourly sync tops the queue up to 30 open picks, at most 30 new a week. Picks are mapped to Spotify as a
batch's are. There is one per artist per ISO week, none by an artist with a current Dislike, nothing liked, offered,
judged or in the library, and chart sources stay within their usual share. Each pick carries a short phrase for
needle's card saying why haystack is unsure.

A Calibrate Verdict is an ordinary Verdict. It has the same effects as on a Discoveries pick (a Rating of 3★ or
more hearts the track, ADR 0002), it is a label for the re-ranker, and it is a query for the eval. A pick stays in
the pool until its Verdict moves it on, and the batch never offers it.

**Why bands, and nothing below the 50th percentile.** The batch takes its top picks from the top of the pool and
its exploration picks from the 40th to 75th percentile (ADR 0010). Labels where it draws tell the re-ranker how to
order the top and tell the exploration share where good music hides. Labels from the bottom half would mostly be
easy noes from a part of the ranking the batch never reaches. The top band keeps a fifth of the picks because the
closest calls there decide the batch. Within a band, most uncertain first is plain uncertainty sampling.

**Considered Options**:

- Lower the gates. The model would order batches on fewer labels, with a leave-one-out AUC whose standard error is
  larger than the gain it has to show. It would not change where the labels come from either.
- More Discoveries slots. A bigger batch brings more labels from the same top of the pool. It costs the listener a
  longer playlist of tracks haystack is already sure of, and an unplayed one is no label at all.
- Random extra picks from across the pool. Simple, but most of them are easy noes from the bottom half, and each
  costs the listener as much time as a close call.
- Uncertainty sampling across score bands (chosen). Each Verdict lands where the model, or the variants, are
  unsure, across the part of the ranking the batch draws from.

**Consequences**:

- The label rate follows how much the listener uses the queue rather than the weekly batch, so the gates can open
  within weeks.
- The re-ranker's labels are no longer only what the batch offered, so its leave-one-out AUC measures it over a
  wider range than the top it orders. The gate compares both stages on the same labels, so the comparison stays
  fair.
- The eval's turned-down queries now include Calibrate Passes and Dislikes from lower bands, which are easier to
  rank below the kept ones. Every variant's AUC rises with them; comparisons between variants still hold, but an
  AUC from 0.5.0 on is not comparable with an older one.
- Each top-up fits the re-ranker in memory for its own use. The fit recorded in `rerank_runs`, and the chance
  written to `candidates`, still come from the batch.
- A fresh installation has no scored pool, so the queue stays empty. With a pool and no Verdicts it asks across the
  bands by disagreement, which gives a new listener labels too.
- needle shows the queue only when the server lists `queue.calibrate` (ADR 0014). An older needle ignores it and
  the picks wait. `calibrate.enabled = false` stops new picks; the open ones stay in the queue.
