# Contribution 1 - Heterophily-aware backbone (H2GCN-style encoder)

Code: `models/h2gcn_backbone.py` (`H2GCNLayer`, `use_h2gcn_backbone`). Evaluated by `evaluate_team_contributions.py`.

## 1. Motivation

AnomalyGFM encodes a node with a 2-layer GCN, `H = act(A_hat * X * W)` with `A_hat = D^-1/2 A D^-1/2 + I`. The anomaly
score is not computed on `H` directly but on the **residual** `r_v = h_v - sum_u w_vu h_u` (the node's embedding minus
the mean embedding of its neighbours). The whole method rests on the assumption that normal nodes look like their
neighbours (small residual) and anomalies do not (large residual).

A GCN layer *averages the node's own features into its neighbours'* (the self-loop in `A_hat`). Stacking two layers
mixes the 2-hop neighbourhood in as well. In a heterophilous neighbourhood (an anomaly connected to mostly normal
nodes - the "camouflage" pattern of fraud) this smoothing pulls the anomaly's embedding towards its neighbours', which
shrinks the very residual the detector relies on. The hypothesis behind C1 was: *a backbone that keeps the ego
representation separate from the neighbour representation should preserve the residual signal and raise accuracy.*

## 2. Method

H2GCN (Zhu et al., 2020) proposes three design principles; the layer implements all three:

| Principle | What it means | In the layer |
|---|---|---|
| D1 ego/neighbour separation | do not mix a node's own features with its neighbours' | separate weight matrix `W_ego` for the node itself; propagation uses `A_n = A_hat - I` (neighbours only) |
| D2 higher-order neighbourhoods | use 1-hop and 2-hop aggregates | terms `A_n X W_1` and `A_n^2 X W_2` |
| D3 combine intermediate representations | do not discard lower-order information | ego, hop-1 and hop-2 terms are summed (summing separately projected terms is identical to H2GCN's concat + linear) |

```
out = act( X W_ego  +  (A_n X) W_1  +  (A_n^2 X) W_2  +  b ),      A_n = A_hat - I
```

`A_n z` is computed as `A_hat z - z` (sparse mat-mul then subtract), so no new adjacency is materialised.
Approximation to note: `A_n^2` is a 2-step *walk* operator, so it includes returns to the ego node; strict H2GCN uses
the exact 2-hop neighbour set. We chose the walk form because it is O(E) with sparse ops.

## 3. Implementation and integration

* `H2GCNLayer(in_ft, out_ft, act='prelu')` has the same constructor and `forward(seq, adj, sparse)` contract as the
  paper's `GCN`, so it is a **drop-in**: `use_h2gcn_backbone(model, 8, hidden)` replaces `model.gcn1`/`model.gcn2` in
  place after `model = Model(...)`. Supports the dense (`bmm`) and sparse (`spmm`) code paths.
* Parameter count (hidden = 400, input = 8): GCN backbone 164,002 -> H2GCN backbone 490,402 (3 weight matrices per
  layer instead of 1). Whole model 484,402 -> 810,802 (+67%).
* Everything else is unchanged: same loss, same residual, same prototypes, same optimiser (Adam, lr 1e-4, 301 epochs).

## 4. Experimental protocol

Train on Facebook (30% of nodes labelled, as in the paper), evaluate zero-shot on Amazon (11,944 nodes), Amazon-UPU
(10,224) and Reddit (10,984). 5 seeds. Metric: AUROC / AUPRC of `exp(cos(r, p_abnormal))`, computed on all target
nodes except the 100 labelled-normal nodes reserved for the adaptation experiments (identical eval set for every row).
Also reported: AUROC on the 70% held-out *Facebook* nodes ("FB-val") as an in-domain sanity check.

## 5. Results (single-source: trained on Facebook only; mean ± std, 5 seeds)

| Target | Baseline AUROC | **+C1 AUROC** | Baseline AUPRC | **+C1 AUPRC** | FB-val base / C1 |
|---|---|---|---|---|---|
| Amazon | 0.834 ± 0.022 | **0.668 ± 0.060** | 0.476 ± 0.121 | 0.174 ± 0.059 | 0.828 / 0.789 |
| Amazon-UPU | 0.792 ± 0.017 | **0.701 ± 0.051** | 0.388 ± 0.100 | 0.230 ± 0.086 | 0.828 / 0.789 |
| Reddit | 0.564 ± 0.026 | **0.589 ± 0.044** | 0.037 ± 0.002 | 0.050 ± 0.014 | 0.828 / 0.789 |

Cost: parameters 484,402 -> 810,802; peak GPU memory 508 -> 532 MB (Amazon), 139 -> 162 MB (UPU), 137 -> 160 MB (Reddit).

## 6. Analysis

* **The hypothesis was not supported on the Amazon graphs.** C1 loses 0.17 AUROC on Amazon and 0.09 on Amazon-UPU.
  On Reddit it gains 0.025, which is inside one standard deviation of either method (±0.026/±0.044), so we do not
  count it as a gain.
* **The module is not broken.** In-domain (held-out Facebook nodes) C1 reaches 0.789 versus 0.828 for the GCN: it
  learns, it is only slightly worse in-domain, and it generalises much worse to Amazon. The training graph has 1,081
  nodes and about 25 anomalies, and C1 has 67% more parameters, so over-fitting to Facebook is a plausible cause. We
  have **not** tested this directly (e.g. by weight decay or a smaller hidden size); it is an explanation consistent
  with the data, not a finding.
* Amazon's fraud signal is in how a reviewer *differs from co-reviewers*. A plain GCN already exposes a usable
  residual there (0.83), so the extra ego/neighbour capacity may simply add variance. The camouflage argument may matter
  more on graphs where the baseline is weak - Reddit is the only such target here and it shows a small, non-significant
  gain.
* Variance: C1's seed-to-seed std on Amazon (0.060) is nearly 3x the baseline's (0.022): less stable training.

## 7. Limitations / threats to validity

* One training graph (the paper's protocol); results may differ with multi-source pre-training (see the multi-source
  section in `WORK.md`).
* Hidden size, depth, learning rate and epochs were kept at the baseline's values - not re-tuned for the larger model.
  A tuned H2GCN could behave differently. We deliberately did not tune on the target graphs (that would leak test data).
* 2-hop operator is a walk approximation (Section 2).

## 8. Reproduce

```
python evaluate_team_contributions.py --seeds 0 1 2 3 4 --targets amazon amazon_upu reddit   # rows "+ C1 H2GCN backbone"
```
