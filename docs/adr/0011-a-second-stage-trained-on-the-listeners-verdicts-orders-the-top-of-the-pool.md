# A second stage trained on the listener's Verdicts orders the top of the pool

Stage 1 stays the active variant's score (ADR 0009). A second stage takes the top `rerank.pool_top = 400` of the
pool by that score and orders them by `0.7 * z(logit p) + 0.3 * z(stage-1 score)`, where p is a logistic
regression's chance that the listener keeps the track. The rest of the pool and the exploration band (ADR 0010)
keep the stage-1 order. The labels are the judged tracks: a Rating of 3★ or more is a yes weighted by its stars
(1.0, 1.25, 1.5), a 1-2★ Rating is a no of 0.75, a Pass a no of 1.0 and a Dislike a no of 1.5. With no Verdict,
an offer liked is a yes, and one that expired after a play or was removed by hand is a no. About 30 features
describe each candidate the same way whether it is in the pool or labelled: the stage-1 score and its parts, the
stars of its nearest positives, the negative penalty, how far its Traits lie outside the usual range, its source
and its artist. The model is fitted with numpy alone (Newton steps, an L2 penalty of 1.0 on the weighted mean
log-loss) at every batch. It is used only with at least 40 labels, and only when its leave-one-out AUC beats
stage 1's on the same labels by 0.05. A re-ranked offer's card says so (`why.model`), with the chance
(`why.confidence`) and up to two reasons (`why.reasons`).

**Why 1-2★ are noes here.** A Rating is a degree of liking (ADR 0002), and stage 1 keeps every Rating as a
positive: a song rated 1★ is still music the listener likes a little, and what sounds like it is still worth
finding. The second stage answers a narrower question: of the close matches, which will the listener keep? A
batch has 15 slots, and a pick rated 1-2★ took one that a 3★ song could have had. So for the second stage "a
little" means "not for the batch". It is a no, weighted below a Pass because the listener did like it a bit.

**Stars in pool sd.** The star bonus is a fixed cosine offset, 0.0025 per star. How much that lifts a track
depends on the pool's spread, which differs between variants and between listeners. `star_bonus = []` now
switches to `star_weight_sd = 0.25` per star away from 3, in units of the sd of the pool's nearest-positive
similarity, measured at scoring time. 0.25 is a unit-based default, not a value searched for. Tuning it to the
reference installation's AUC would fit the noise of 31 offers, and the second stage's star features carry the rest
of what the stars say. The fixed list stays the default until the eval shows that the sd form is not worse.

On the reference installation it has not shown that. The eval's AUC with the fixed list and with the sd form:

| Variant | Fixed list | sd form |
|---|---|---|
| raw | 0.626 | 0.620 |
| topk_mean | 0.574 | 0.546 |
| centred | 0.612 | 0.628 |
| csls | 0.602 | 0.588 |
| csls_centred | 0.632 | 0.616 |
| genre (the active variant) | 0.498 | 0.498 |
| genre_blend | 0.568 | 0.569 |

These AUCs cover 18 kept and 13 turned-down offers, short of the 15 and 15 that the AUC objective needs. On the
mean-percentile fallback the sd form was lower too: genre_blend 60.83 against 60.53, raw 59.09 against 58.23. A
list still means fixed offsets, so an existing config keeps its behaviour.

**On the reference installation** there were 42 labels:

- 22 yes: 17 at 3★ and 5 at 4★;
- 20 no: 10 Dislikes, 6 Passes, 3 at 2★ and 1 at 1★.

Leave-one-out AUC of the model and of stage 1 alone, on those labels:

| Variant | Model | Stage 1 | Gate |
|---|---|---|---|
| raw | 0.636 | 0.610 | off |
| topk_mean | 0.635 | 0.571 | passes |
| centred | 0.644 | 0.614 | off |
| csls | 0.634 | 0.595 | off |
| csls_centred | 0.649 | 0.606 | off |
| genre | 0.696 | 0.598 | passes |
| genre_blend | 0.686 | 0.590 | passes |

Over 22 and 20 labels an AUC's standard error is about 0.08, so the gain under the active variant is not yet
beyond noise.

A planned batch compared the two stages:

- Only 1 of the 12 top picks was the same with and without the second stage. The re-ranked picks came from stage-1
  ranks 3 to 379.
- The 11 picks the second stage brought in had a mean chance of 0.83. The 11 it left out had 0.67.
- The reasons for the picks it brought in were `close to your 4-5★ tracks` (9), `brighter than you usually like`
  (8) and `louder than you usually like` (3).
- The picks it left out were held back for being typical of Deezer charts or found via a genre chart.
- The largest weights (per pool sd) were the negative penalty -0.21, valence's distance from the usual range +0.11
  (brighter is better), the neighbours' stars +0.11 and a chart source -0.10.

**Reasons.** A reason is a feature that moves the track's logit by at least 0.1 (its weight times the track's
standardised value), strongest first. A feature is quoted only when its weight agrees with the feature's own
direction over the labels and with the direction it should have. A Trait is quoted only when the track lies
outside the usual range on the side the phrase names, or as `<trait> in your usual range` when being inside the
range lifted it.

Without these rules, two kinds of wrong reason appeared on the reference installation. A sparse feature with a
weight of the unexpected sign moved picks up "because" they sounded like a track the listener turned down. And a
track was called darker than usual for being less dark than the pool. A weight that fails the rules still orders
the batch; it is only never quoted.

**Considered Options**:

- A bigger star bonus only. It lifts what sounds like 4-5★ tracks and nothing else: not Traits, sources, or what
  the listener turned down.
- A kNN star regression only. The neighbours' stars are one of the features, and among the strongest, but alone
  they ignore the noes.
- Gradient boosting with a new dependency. 42 labels are too few for trees not to overfit, and the gate would keep
  them off. Boosting brings a large dependency, while a linear model's reasons fall out of its weights.
- Clipping each standardised feature at ±3 to tame sparse features. It cost about 0.04 of leave-one-out AUC on
  every variant (genre fell from 0.696 to 0.655), and the reason rules already fix what clipping was for.

**Consequences**:

- Once the gate passes, a top pick can come from far down the top 400 (rank 379 above). The batch leans toward
  what the listener has kept, which narrows it; the exploration band and the spread rules (ADR 0010) still widen
  it.
- The model is refitted and re-checked at every batch, so it switches itself on and off as labels change. With a
  few dozen labels the check itself is noisy. `rerank.enabled = false`, or a higher `rerank.min_gain`, keeps the
  second stage off.
- The leave-one-out AUC is slightly optimistic. The usual range that a Trait feature measures against includes the
  labelled tracks the listener liked or rated, so a rated discovery widens it. A ripped discovery also stays in the
  hubness reference.
- The L2 penalty is on the mean loss, so it does not fade as labels grow. At a few hundred labels a lower
  `rerank.l2` may fit better.
- From 300 labels, 32 principal components of the preview vector join the features.
- A fresh installation, and every installation with under 40 labels, picks exactly as 0.2.0 did: the star bonus is
  unchanged and the second stage is not used.
