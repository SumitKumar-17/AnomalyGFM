# C3 - Adversarial edge purification

Code: `models/edge_purification.py` (`EdgePurifier`, `common_neighbors`, `run_attack_benchmark`), `utils/attacks.py`, `train.training_attack`.

## Idea
The paper trusts the adjacency matrix. An anomaly that wires itself to normal nodes (camouflage) pulls its neighbour mean toward itself and its residual vanishes. The purifier gives every undirected edge a keep-weight *before* message passing.

## Method
* Edge features (symmetric, so `w_uv = w_vu`): sign-log of `|x_u - x_v|` and `x_u * x_v`, log-degree sum/difference, **common-neighbour count and Salton overlap** (20 dims) -> MLP(32) -> logit. 705 parameters.
* Training: binary-concrete relaxation `w = sigmoid((logit + logistic_noise)/0.5)`; inference: hard drop `logit <= 0`. Weights re-weight both the GCN operator and the neighbour mean; rows left with almost no weight keep a small self-mass (their residual -> ~0, not `h`).
* Adversarial training on Facebook: on 50% of epochs the training graph is attacked (10% random noise edges + 10 camouflage edges per anomaly). The injected edges are known, so the purifier also gets an edge-level BCE target (keep = 0 for injected edges; un-balanced BCE). Nothing from a target graph is used at inference.
* Attacks for evaluation (`utils/attacks.py`): random rewiring of a fraction of edges; camouflage = each anomaly wired to m random normal nodes (labels used only to build the attack).

## Design history (source graph only; details in WORK.md Section 3)
node-feature-only scorer: edge-AUROC 0.70, dropped 48% of real edges -> + common-neighbour features and un-balanced loss: edge-AUROC **0.96**, drops 75-81% of injected / 3-4% of real edges; learning rate 1e-2 and augmentation strength selected on held-out Facebook nodes. Before this the purifier was a **no-op (0% edges dropped)**.

## Results
Master table: average AUROC 0.645 (baseline 0.648; **control without purifier 0.617**), AUPRC 0.252 vs 0.244. Per target AUROC: Amazon 0.824 (0.834), UPU 0.758 (0.792), Reddit 0.500 (0.573), **Elliptic 0.499 (0.394; control 0.372)**.

Attack benchmark (AUROC, 5 seeds, Amazon-UPU): baseline 0.792 -> 0.669 / 0.244 / **0.085** under camouflage 5 / 20 / 50 edges per anomaly, edge-purified 0.760 -> 0.734 / 0.701 / **0.678**; under random rewiring the purifier is worse
(30%: 0.667 vs 0.822; 50%: 0.617 vs 0.819). Reddit @ camouflage 50: 0.300 vs 0.029 (both < 0.5). Amazon (dense, avg degree 736): attacks hardly matter (0.841 vs 0.826).
Edge-level diagnostic (`results/purifier_diagnostic.md`): UPU edge-AUROC 0.99 (drops 99.5% injected / 4.7% real); **Reddit 0.12 (drops 96% of real edges)**.

## Analysis
* The baseline's collapse under camouflage is real and large (the ranking inverts); the purifier removes it on graphs with clustering.
* **It fails where its structural signal does not exist:** Reddit has zero common neighbours on every edge, so Facebook-learned "no shared neighbours = fake" deletes nearly all real edges.
* Random rewiring hurts the purified model because it also deletes many real edges when 30-50% of the graph is noise, while the baseline is oddly unaffected.
* Cost: peak GPU memory 1.7 GB vs 0.63 GB on average (common-neighbour computation); 2.5 GB on Amazon-UPU.

## Limitations / next
Not graph-agnostic (see Reddit); trained on one source graph; attack strength matters (selected on the source); structural scorer, not a learned attention over embeddings.

Reproduce: `python main.py attack`; `python scripts/diagnose_purifier.py`.
