# C1 - Mixture-of-Experts abnormal prototypes

Code: `models/moe_prototypes.py`, hooks in `models/core.py` (`prototypes`, `scores`) and `train.pretrain`; plots/statistics `utils/viz.py`.

## Idea
The paper represents every anomaly by one prototype `p_abn`. If anomalies are heterogeneous, one vector averages different patterns ("modality collapse").
C1 uses K expert prototypes `p_k = ReLU(W_abn eps_k + delta_k)` (the paper's shared abnormal layer, per-expert noise `eps_k` and learnable offset `delta_k`) and a gate
`g_v = softmax(Linear(r_v/|r_v|)/T)` that routes a node's residual to experts. Zero-shot score: `s_v = sum_k g_vk cos(r_v, p_k)`.

## Loss (replaces the single-prototype alignment, in `MoEPrototypes.loss`)
`sum_k g_vk ||p_k - r_v||` (soft assignment) `+ 0.1 * K * sum_k f_k P_k` (Switch-style load balancing: f = hard routing share, P = mean gate probability)
`+ 0.1 * mean_{k!=l} cos(p_k, p_l)` (diversity, targets the near-collinear prototypes found in the first study).
Parameters: +3,204 (offsets 1,600 + gate 1,604).

## Results (5 seeds; train Facebook; learning rate of the new parameters 1e-3, chosen on Facebook held-out nodes)
| AUROC | Amazon | UPU | Reddit | Elliptic |
|---|---|---|---|---|
| K=1 (paper) | 0.834 | 0.792 | 0.573 | 0.394 |
| K=2 | 0.827 | 0.796 | 0.563 | 0.405 |
| K=4 | 0.820 | 0.793 | 0.556 | 0.399 |
| K=8 | 0.812 | 0.791 | 0.547 | 0.406 |
Master table (K=4): average AUROC 0.642 vs 0.648, AUPRC 0.226 vs 0.244.

**Routing** (seed 0, K=4, anomalies of the target): Amazon [689, 117, 2, 13] (silhouette 0.145), UPU [438, 220, 7, 28] (0.076), Reddit [366, 0, 0, 0] (collapse), Elliptic [1086, 908, 1864, 687] (0.035).
`results/moe_tsne_*.png`: experts 1 and 3 sit in separate regions of residual space; expert 0 takes most anomalies. Assignment correlates with a degree-based *proxy* type (chi-square p < 1e-4).

## Analysis
* The MoE does not improve accuracy; AUROC falls slightly with K on Amazon and Reddit, is flat on UPU and Elliptic.
* Experts collapse despite load balancing. Likely cause (untested): the Facebook split has only about 7 labelled anomalies (25 x 30%) for 4 experts, so the gate sees almost no signal.
* **We cannot show experts = anomaly types:** no dataset has type labels (Reddit's structural/attribute columns equal its main label, checked). The t-SNE shows the experts are *separable regions*, not that they are semantically distinct fraud types; the association with degree suggests they partly encode node degree.

## Limitations / next
Needs more labelled anomalies (multi-source pre-training); gate temperature, K and the loss weights were not tuned; gated scoring is used (the max-over-experts variant is logged as `auroc_max` in `results/moe_ksweep_raw.csv`).

Reproduce: `python main.py moe`.
