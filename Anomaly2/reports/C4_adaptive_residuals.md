# C4 - Adaptive multi-hop (PPR) residuals

Code: `models/adaptive_residuals.py` (`MultiHopResidual`), hook in `models/core.py` (`encode`); ablation `scripts/ablation_residuals.py`.

## Idea
The paper compares a node with its **1-hop** neighbour mean. Deeper anomalous structures (laundering rings) may only differ from a larger neighbourhood.
`r_v = h_v - sum_{k=1..K} w_vk (P^k h)_v`, computed with K sparse mat-muls (no N x N matrix; works on 46k nodes).
* `hop1`: the paper. `hop2`: 0.5/0.5 of hops 1-2. `ppr`: `w_k ~ (1 - alpha)^(k-1)` (truncated Personalised-PageRank walk distribution, alpha 0.3, K=4, normalised; the `k=0` teleport term is the node itself).
* `adaptive`: `w_v = softmax(log PPR prior + MLP(structure_v))`, structure = [log degree, log mean-neighbour degree, log 2-hop mass] (z-scored); last MLP layer zero-initialised so training starts at `ppr`. 132 parameters, lr 1e-2 (selected on the source).
`P^k` is a random-walk operator, so even hops contain returns to the node.

## Results (5 seeds; train Facebook)
| AUROC | Amazon | UPU | Reddit | Elliptic | Tolokers | Average | Avg AUPRC |
|---|---|---|---|---|---|---|---|
| 1-hop (paper) | 0.834 | 0.792 | 0.573 | 0.394 | 0.538 | 0.626 | 0.249 |
| 2-hop | 0.819 | 0.792 | 0.531 | 0.380 | 0.544 | 0.613 | 0.233 |
| PPR (alpha 0.3) | 0.815 | 0.790 | 0.534 | 0.372 | 0.540 | 0.610 | 0.225 |
| **Adaptive PPR** | 0.834 | 0.795 | 0.558 | 0.395 | 0.532 | 0.623 | 0.254 |
Master table (4 targets): adaptive average AUROC 0.646 vs 0.648, AUPRC 0.251 vs 0.244.

**Learned hop weights** (mean over nodes; prior 0.40/0.28/0.19/0.14): Facebook 0.785 / 0.003 / 0.003 / 0.210; Amazon 0.801 / 0.006 / 0.005 / 0.188; Reddit 0.788 / 0.006 / 0.006 / 0.200; Elliptic 0.853 / 0.002 / 0.003 / 0.142; within-graph std 0.27-0.30.

## Analysis
* Fixed multi-hop references are **worse** than the paper's 1-hop on 4 of 5 targets (2-hop and PPR both about -0.015 on average): smearing the reference over a larger neighbourhood dilutes the contrast.
* The adaptive gate **learns a per-node, bimodal choice** - most nodes use hop 1, a minority hop 4 - and the same pattern appears on all graphs; it recovers the 1-hop accuracy (0.623 vs 0.626; slightly higher AUPRC) but **does not beat it**, including on Elliptic (laundering structures): 0.395 vs 0.394.
* With a lr of 1e-4 (round 1) the gate never moved and adaptive == fixed PPR to 3 decimals - the lesson in WORK.md Section 3.

## Limitations / next
Four hops, three structural features, one alpha; non-backtracking walks would stop deeper hops from being dominated by returns; the gate sees only structure, not embeddings.

Reproduce: `python main.py ablation`.
