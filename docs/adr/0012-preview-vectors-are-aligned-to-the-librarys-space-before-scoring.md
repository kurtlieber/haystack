# Preview vectors are aligned to the library's space before scoring

Every candidate is a Deezer preview. With a library, most positives are the listener's own files, one row per 30 s
window, and a preview and a file of the same song land a little apart in the embedding space (a 30 s clip against a
whole song, and likely a different master and encoding). So a candidate tends to find its nearest positive among the
other previews (liked songs without a file, liked discoveries, Rated tracks) rather than among the files, and a few
previews become the nearest match for much of the top of the pool.

The liked songs that have both a preview vector and a confidently matched local file are **pairs**. A linear map
from the preview to the file is fitted on them and applied to every preview vector before it is compared with
anything: candidates, positives, negatives, the eval's queries and the re-ranker's features
(`vectors.preview_vectors`). On L2-normalised vectors it is `x -> (x - mp) @ M + mt`, with `mp` and `mt` the pairs'
mean preview and mean target. `M` is either a ridge fit shrunk toward no change, `min |XM - Y|^2 + lambda |M - I|^2`
(lambda = infinity is a pure mean shift), or an orthogonal Procrustes rotation. The target is either the file's
window that best matches the preview, or the mean of its windows. Eight lambdas (infinity, 1000, 300, 100, 30,
10, 3, 1), Procrustes and both targets are tried by 5-fold cross-validation on **self-retrieval**: is a held-out
pair's own file the nearest (top 1), or among the 10 nearest (top 10), of the library's files and the other liked
songs' previews, mapped by the same fold's map? The best top 10 wins, then the best top 1, and a tie goes to the
simpler fit. The map is used only when that cross-validated score beats the unaligned vectors.

`scoring.align = "auto"` uses the map from `scoring.align_min_pairs = 100` pairs. The eval then runs twice,
aligned and unaligned, and the alignment stays on only if it is not worse for the variant in use: the objective
(AUC, or the mean percentile on the fallback) may not fall by more than its tie (`eval.auc_tie`,
`eval.tie_tolerance`), nor recall@15 by more than one held-out song per seed. The verdict is kept in kv
`preview_align_gate`, and the eval's markdown and `status.json` show both runs. The library job refits daily when
there is no map yet or the pair count moved by more than 10 %. Too few pairs remove the map. `off` never maps.

**What it leaves alone.** The genre variants' style vectors (a second model, previews only, so no gap). The batch's
spread rule and the sanity check, which compare previews with previews. The stored vectors themselves: the map is
applied when scoring, so turning it off, or a refit, needs no backfill.

**On the reference installation** there were 337 pairs among 3,734 library files. Self-retrieval:

| | Top 1 | Top 10 | Top 1, files alone | Top 10, files alone |
|---|---|---|---|---|
| Unaligned | 0.537 | 0.804 | 0.828 | 0.973 |
| Mean shift alone (cross-validated) | 0.947 | 1.000 | | |
| Procrustes (cross-validated) | 0.967 | 0.997 | | |
| Chosen: ridge, lambda 1, window mean (cross-validated) | 0.967 | 1.000 | 0.991 | 1.000 |

Without the map, a preview found its own file first only half the time, and missed the top 10 one time in five,
because the other previews were nearer. Every fit closed nearly all of that gap, and the mean shift alone did most
of it.

The pool's top 100 by the raw nearest-positive score, by the kind of that nearest positive:

| | Liked song's preview | Library file | Rated track's preview | Preview share |
|---|---|---|---|---|
| Unaligned | 66 | 28 | 6 | 72 % |
| Aligned | 11 | 88 | 1 | 12 % |

The eval on the same data. Its objective was the mean-percentile fallback (18 kept and 13 turned-down offers,
short of 15 and 15), and the variant in use was `genre`:

| Variant | AUC aligned / unaligned | Mean percentile aligned / unaligned | recall@15 aligned / unaligned |
|---|---|---|---|
| raw | 0.637 / 0.626 | 59.97 / 59.09 | 0.027 / 0.032 |
| topk_mean | 0.602 / 0.574 | 60.14 / 60.22 | 0.035 / 0.019 |
| centred | 0.538 / 0.612 | 57.29 / 55.17 | 0.013 / 0.029 |
| csls | 0.617 / 0.602 | 53.24 / 54.15 | 0.013 / 0.003 |
| csls_centred | 0.603 / 0.632 | 52.63 / 53.23 | 0.008 / 0.005 |
| genre (in use) | 0.498 / 0.498 | 59.99 / 59.99 | 0.056 / 0.056 |
| genre_blend (the eval's choice) | 0.582 / 0.568 | 61.28 / 60.83 | 0.043 / 0.035 |

`genre` ranks by style vectors, so its stage-1 order is unchanged, and the check keeps the alignment. What changes under
it is which positive a pick "sounds like" (and so the batch's cap per nearest positive, ADR 0010) and the re-ranker's
similarity features. The centred variants lose AUC; neither is in use, and the check looks only at the variant in use,
by the eval's objective. Over 18 and 13 offers an AUC's standard error is about 0.1, so none of these differences is
beyond noise.

A fit took about 5 s (about a minute of CPU over all cores) and an eval with both runs about 12 s.

**Considered Options**:

- Sub-windows: store three 10 s window means per preview and score a candidate by its best sub-window. It needs
  about two more Deezer requests per stored preview (weeks of backfill at the harvest's pace). It was the fallback
  if the map moved the preview share of the top 100 by less than a third; it moved it from 72 % to 12 %.
- Centring each domain on its own mean. That is the mean shift, which the fit tries first and keeps when nothing
  richer cross-validates better.
- Choosing the fit by the eval's AUC instead of self-retrieval. Self-retrieval has a right answer for every pair
  (hundreds of them), while the AUC rests on a few dozen offers; the eval is the check, not the fit's objective.
- Adding the previews of the liked songs that have files as extra positives. It keeps both domains among the
  positives, so a candidate still finds a preview first.

**Consequences**:

- With a library, most candidates now find their nearest positive among the listener's own files, so "sounds
  like" names a library track far more often, and the cap per nearest positive spreads the batch over more of them.
- Every preview vector is mapped at scoring time, a 512x512 product: milliseconds for a pool of a few thousand.
- The eval runs twice while a map exists. A pinned `scoring.variant` gets the check at each batch too.
- A map fitted on few pairs could overfit; `align_min_pairs` and the cross-validation guard against that, and
  the eval's check turns it off when it hurts the variant in use.
- A fresh installation, and one without a library or with fewer than 100 pairs, scores exactly as 0.3.0 did.
