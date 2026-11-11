# Contribution 3 - Label-free contrastive prototype pre-training

Code: `utils/ssl_pretrain.py` (`SSLConfig`, `augment`, `prototype_info_nce`, `view_info_nce`, `ssl_loss`, `build_ops`).
Evaluated by `evaluate_team_contributions.py` (`mode='ssl'` and `mode='sup+ssl'`).

## 1. Motivation

AnomalyGFM learns its normal/abnormal prototypes with **labelled** anomalies from the training graph (BCE on node
logits plus an alignment loss between labelled-abnormal residuals and the abnormal prompt). Labelled anomalies are
scarce, expensive, and exactly what is missing in the settings a foundation model targets. The hypothesis behind C3:
*we can learn the same prototypes with no labels at all, by corrupting the graph ourselves and treating corrupted
nodes as pseudo-anomalies.*

## 2. Method

Each epoch the training graph is corrupted **on the fly, training only** (the target graph is never touched):

* **Structural corruption (5% of nodes):** a node's neighbourhood is replaced by random neighbours of equal expected
  degree (rewiring: destroys the node's real local structure).
* **Contextual corruption (5% of nodes + 5% partners):** features are *swapped* between random node pairs (a node now
  carries another node's attributes).
* Corrupted nodes (about 15% of the graph, because swap partners are also corrupted) are **pseudo-abnormal**; the rest
  **pseudo-normal**. No ground-truth label is read.

Two InfoNCE terms are optimised on the corrupted view:

1. **Prototype InfoNCE.** With residual `r_v` and prototypes `p_n, p_a` (the paper's mechanism: Gaussian noise through a
   learned linear+ReLU layer), `logits_v = [cos(r_v,p_n), cos(r_v,p_a)] / tau`, loss = cross-entropy against the
   pseudo-label, class-re-weighted by `1/frac` because corruptions are the minority class. `tau = 0.2`.
2. **View InfoNCE** (weight 0.5). An *untouched* node's embedding in the clean graph must retrieve itself among 512
   sampled nodes' embeddings in the corrupted graph (positive = same node, negatives = other nodes). This anchors
   normal nodes' embeddings while corrupted ones are pushed away.

`build_ops` rebuilds the sparse GCN operators (`A_hat`, neighbour-mean `W_n`) from the corrupted dense adjacency
(Facebook is small enough for dense augmentation; larger sources use sampled subgraphs of at most 4,000 nodes).

## 3. Integration

In the training loop replace (or add to) `loss = loss_bce + loss_alignment` with
`loss, _ = ssl_loss(encode, A_dense, X, SSLConfig(), gen, normal_prompt, abnormal_prompt)`. `encode(A_hat, Wn, X)`
must return `(embeddings, residuals)`. The logit head `fc1` is unused in SSL mode (484,002 vs 484,402 trainable
parameters). Modes in the pipeline: `ssl` (label-free) and `sup+ssl` (supervised loss + SSL loss).

## 4. Protocol

Train Facebook with **no labels** (mode `ssl`), zero-shot on Amazon, Amazon-UPU, Reddit; 5 seeds; same evaluation set
and in-domain "FB-val" check as C1. For FB-val, the labels are used only to *evaluate*, never to train.

## 5. Results (single-source; mean ± std, 5 seeds)

| Target | Baseline (supervised) AUROC | **C3 (no labels) AUROC** | Baseline AUPRC | **C3 AUPRC** | FB-val base / C3 |
|---|---|---|---|---|---|
| Amazon | 0.834 ± 0.022 | **0.620 ± 0.082** | 0.476 ± 0.121 | 0.100 ± 0.020 | 0.828 / 0.701 |
| Amazon-UPU | 0.792 ± 0.017 | **0.588 ± 0.056** | 0.388 ± 0.100 | 0.087 ± 0.017 | 0.828 / 0.701 |
| Reddit | 0.564 ± 0.026 | **0.585 ± 0.010** | 0.037 ± 0.002 | 0.041 ± 0.001 | 0.828 / 0.702 |

## 6. Analysis

* **Learning anomaly structure without labels works, but does not match supervision on Amazon.** With zero labels C3
  reaches 0.701 AUROC on held-out Facebook nodes (supervised: 0.828) and 0.62 / 0.59 on Amazon / Amazon-UPU (supervised:
  0.83 / 0.79). Both are far above chance (0.5), so the corruption-based pseudo-labels carry real signal.
* **Reddit:** 0.585 vs 0.564 - a small difference, but with the lowest variance of any method here (±0.010); we treat it
  as parity with the supervised baseline at best, not a win (the baseline's own std is ±0.026).
* The gap to supervision is expected: synthetic corruptions (random rewiring, random feature swaps) are *different*
  from real fraud, so the prototypes learn "looks randomly corrupted" rather than "looks fraudulent". Real fraud is
  often *camouflaged* (carefully mimics normal behaviour), which random corruption does not model.
* The paper's fusion score `exp(cos_abn) + beta * exp(-cos_normal)` is *worse* than the abnormal-only score for C3
  (Amazon 0.547 vs 0.620): the normal prototype in C3 is not informative on Amazon (consistent with the earlier finding
  that the two prototypes are nearly collinear).
* **Why it is still the most useful contribution:** it removes the labelled-anomaly requirement, so it can use *any*
  unlabelled graph for pre-training - which the multi-source experiment exploits.

## 7. Limitations

* Corruption types/rates (5%+5%, tau 0.2) are fixed, not tuned. Natural next step: learned or camouflage-aware
  corruptions (e.g. rewiring towards *similar* nodes rather than random ones).
* Pseudo-anomaly rate (about 15%) is far above real anomaly rates (2-7%); the class re-weighting compensates only
  partly.
* Only single-graph pre-training in the main table.

## 8. Reproduce

```
python evaluate_team_contributions.py --seeds 0 1 2 3 4 --targets amazon amazon_upu reddit   # rows "+ C3 SSL prototypes (NO labels)"
```
