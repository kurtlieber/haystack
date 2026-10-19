# A batch is spread across the listener's styles and its exploration share follows the outcomes

After each top pick the rest of the pool is re-ranked by `score - batch.diversity * sd(pool score) * max(0, max
cosine to the picks so far)`, with `batch.diversity = 0.5`. At most `batch.max_style_share = 0.4` of the top picks
share a top-1 Discogs style, and at most `batch.max_per_nearest = 2` picks share a nearest positive. When the pool
can't fill the batch under the caps, the style cap gives way first, then the nearest cap. The exploration share
starts at `batch.exploration_fraction`. Once the last 8 batches hold 8 judged offers of each kind, each batch sets
it to `clip(prior + 0.5 * (exploration kept rate - top kept rate), 0.1, 0.4)`; until then it stays at the prior.
Each offer stores its score percentile in its batch's pool, and the API returns it as `why.band`.

On the reference installation the pool's top 30 had 6 distinct top styles, the two largest holding 21 of them,
and 3 liked songs were the nearest positive for 9 of them. In the third batch 3 of 16 top picks were kept.
Over the first three batches 12 of 28 judged top picks were kept, against 6 of 7 exploration picks. On the same
pool a plan with no spread had 7 of 12 top picks in one style and one liked song nearest to 3 picks. With the
spread it had 4 of 12 and 1, and the lowest top pick moved from the 99.5th to the 99.0th score percentile.

The diversity cosine is between preview vectors centred on the pool. Raw cosines among the pool's top 100 ran
from 0.85 to 0.94 (10th to 90th percentile), so a raw penalty would mark every candidate down about the same.
Centred, they ran from -0.15 to 0.49. A negative cosine counts as 0, so a pick never lifts a candidate unlike it.

The rates count judged offers, not every decided one as first planned. An offer cleared before it was played, with
no Verdict, closes with no word from the listener. On the reference installation all 15 of the second batch's
offers closed that way.

**Considered Options**: a fixed exploration share (it ignores 6 of 7 exploration picks kept); a bandit over score
bands (it needs `score_percentile` data first, and the column is there for it); the diversity penalty on raw
cosines (near-constant); diversity in the 400-d style space only (the cap covers styles, and the preview space
also tells songs of one style apart); hard caps (a narrow pool, such as a fresh installation's, would give short
batches).

**Consequences**: the top picks reach a little deeper into the ranking. With a few dozen judged offers the kept
rates are noisy, so the share can move by up to 0.3 from one batch to the next. The bounds limit how far it moves,
and the 8-batch window damps it. A fresh installation keeps the prior until it has 8 judged offers of each kind,
and candidates with no vectors or styles are picked as before. `diversity = 0`, `max_style_share = 1`,
`max_per_nearest = 0` and `explore_window_batches = 0` restore the old selection exactly. Offers made before 0.2.0
have no percentile.
