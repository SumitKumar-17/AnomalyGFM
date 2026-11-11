# WORK.md - everything we did, why we did it, and what happened

Project: term project on **AnomalyGFM** (Qiao, Niu, Chen, Pang - *Graph Foundation Model for Zero/Few-shot Anomaly
Detection*, KDD'25, arXiv 2502.09254). Fork: `SumitKumar-17/AnomalyGFM`, branch `team-contributions`.
All experiments ran on the lab server (RTX 4500 Ada **24 GB** - not 40 GB, 32 CPU cores, 62 GB RAM, PyTorch 2.5.1+cu121);
the code lives locally and is synchronised with `sync.sh` (rsync push; results pulled back).

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR (read this first)

* We reproduced the paper's zero-shot pipeline (Facebook -> Amazon: AUROC about 0.83-0.86 over seeds; the paper's
  Amazon number is about 0.84).
* We built **4 extensions** (H2GCN backbone, hypernetwork feature alignment, label-free contrastive prototype
  learning, Graph-LoRA) and a fair evaluation pipeline on **real data only**.
* **Honest outcome: none of the four improved accuracy over the baseline** in the single-source (train-on-Facebook)
  setting. Three findings *are* solid and presentable:
  1. **C3 learns useful anomaly structure with zero labels** (0.62 AUROC on Amazon vs 0.83 supervised; far above 0.5).
  2. **LoRA delivers the parameter-efficiency claim** (4,832 trainable params = 2.9% of full fine-tuning) but does not
     beat the paper's 80-parameter prompt in accuracy.
  3. **The released few-shot code does no real tuning** (the prompt is commented out of `forward`).
* Earlier benchmark (Exp H/I) found: the paper's *normal*-prototype score is inverted (AUROC 0.40, worse than chance);
  the two prototypes are nearly collinear; the paper's row-sum feature normalisation is numerically unsafe.
* A **multi-source pre-training** experiment (Section 7) tests whether the failures come from training on a single
  1,081-node graph.

---------------------------------------------------------------------------------------------------------------------

## 1. The paper in 60 seconds

* **Task:** graph anomaly detection (GAD): flag nodes that differ from the majority (fraud accounts, fake reviews...).
* **Problem:** classical GAD trains and tests on the same graph; it does not transfer. Generic graph foundation models
  do not transfer to GAD either, because anomalies are rare and irregular.
* **Idea:** learn *graph-agnostic* normal/abnormal **prototypes**. A node's **residual** `r_v = h_v - mean_neighbors(h)`
  is small for normal nodes in *any* graph and large for anomalies, so residuals live in a shared space. The prototypes
  are Gaussian-noise vectors passed through a learned linear+ReLU layer, trained so that labelled-abnormal residuals
  align with the abnormal prototype.
* **Inference (zero-shot):** score = `exp(cos(r_v, p_abnormal))`, no tuning. **Few-shot:** tune a prompt with a few
  labelled normal nodes. **Features:** every graph is projected to 8 dims with a fixed SVD.
* **Training:** one graph (Facebook), 30% of nodes labelled; Adam lr 1e-4, 301 epochs; 2-layer GCN, hidden 400.

---------------------------------------------------------------------------------------------------------------------

## 2. Workflow and environment

| Item | Detail |
|---|---|
| Local | code, reports, slides (this repo) |
| Server | `ssh` with key; conda env `gfm` (python 3.10, torch 2.5.1+cu121, torch_geometric, scikit-learn) |
| `sync.sh` | `push` (rsync code, excludes data/results/logs), `run "<cmd>"`, `bg <name> "<cmd>"` (detached, writes `logs/<name>.done`), `wait <name>`, `pull`, `sudo "<cmd>"` |
| Git | fork cloned locally; branch `team-contributions`; commits are local then pushed to the fork |
| Data (server) | `data/datasets for AnomalyGFM/*_svd.mat` (the authors' SVD-aligned graphs, Google Drive) and `data/raw/*.mat` (raw features from the same lab's UNPrompt repo) |

Raw/SVD consistency check: for Facebook, Amazon, Amazon-UPU and Reddit the node counts and labels of the raw and SVD files
are identical. YelpChi raw (23,831 nodes) does **not** match the SVD file (45,941 nodes), so YelpChi is used only as a
source graph with an SVD we compute ourselves (the paper's `svd_feature_trans.py` recipe).

### Datasets used

| Graph | Nodes | Edges (nnz) | Anomaly % | Raw feat. dim | Role |
|---|---|---|---|---|---|
| Facebook | 1,081 | 55,104 | 2.31 | 576 | training graph (paper protocol) |
| Amazon (all) | 11,944 | 8,796,784 | 6.87 | 25 | test target |
| Amazon-UPU | 10,224 | 351,216 | 6.78 | 25 | test target |
| Reddit | 10,984 | 168,016 | 3.33 | 64 | test target |
| YelpChi | 23,831 | 98,630 | 5.11 | 32 | multi-source training only |
| T-Finance | 39,357 | 42,445,086 | 4.58 | 10 | multi-source training only |
| Elliptic / Questions / Tolokers / Yelp-svd / Disney | 46,564 / 48,921 / 11,758 / 45,941 / 124 | - | 9.8 / 3.0 / 21.8 / 14.5 / 4.8 | (SVD only) | Exp H/I only |

---------------------------------------------------------------------------------------------------------------------

## 3. Phase 0 - baseline reproduction

**What:** ran the authors' `zero_shot/run_abnormal.py` (Facebook -> Amazon) on the server.
**Why:** a baseline that we can trust is the foundation of every later comparison.
**Result:** 13 s wall-clock on 16 CPU cores, 4.5 GB RAM; abnormal-prototype AUROC 0.909 for the first seed.
**Observation:** the paper's "normal" score and the code's `paper_mid` fusion came out *inverted* (AUROC 0.09) - a first
hint of problem P1 below.
**Fix needed to run:** `dgl` and `torch_sparse` imports were unused/uninstallable - made them optional.

## 4. Phase 1 - explainability prototype (and why we dropped it)

**What:** `zero_shot/explain.py` trains the model, scores a target graph, and for the top-5% flagged nodes writes a
plain-English explanation: score rank, how far the node deviates from its neighbours, similarity to each prototype,
leave-one-out neighbour attribution, feature z-scores, and a domain-aware "business decision" (e.g. suspend a reviewer
account).
**Result:** Amazon AUROC 0.903, **79.6% of the top-5% flagged accounts are true fraud** (base rate 6.9%); Reddit AUROC 0.595,
precision only 1.8% (base rate 3.3%).
**Limitation found:** the shared `*_svd.mat` files contain only 8 SVD components, the edges and a label - no product IDs
and no original features - so we could not say "anomalous *because of product X*". Entity/edge meanings came from the
datasets' published descriptions, not the files.
**Decision:** the team judged a text-template explainer too simple for a research contribution, so we moved to ideas with
measurable accuracy / efficiency gains. The code is kept (`explain.py`) as an auxiliary result.

## 5. Phase 2 - finding weaknesses worth attacking

Reading the released code and our runs gave six concrete weaknesses:
P1 inference uses random-noise prompts (score depends on the random draw); P2 score uses cosine only (ignores residual
magnitude); P3 a single abnormal prototype for heterogeneous anomalies; P4 residual = plain neighbour mean (camouflage);
P5 nothing adapts to the target graph at test time (Reddit about 0.57); P6 hand-tuned fusion weight (`ab + 6*normal`).

### Exp I - scoring function (hypothesis: magnitude-aware scoring helps) and Exp H - feature alignment

**Setup (`zero_shot/bench.py`, `summarize.py`):** train Facebook, zero-shot on 9 graphs, **5 seeds**, 10 scorers
and 16 feature configs (dimension {2,4,6,8} x normalisation {paper row-sum, z-score, row-L2, none}) = 80 trained models,
7,200 result rows. Inference is sparse + GPU; the sparse path reproduces the dense results exactly (Amazon 0.9026,
Reddit 0.5952 at seed 0) - that is how we know the harness is correct.

**Exp I result (AUROC, mean over 9 targets and 5 seeds; features 8d/row-sum):**

| Scorer | Mean AUROC | vs paper's abnormal score |
|---|---|---|
| `cos_abn` (paper) | **0.603** | - |
| `cos_norm_pos` (sign-flipped normal score) | 0.602 | same |
| rank-fuse(abnormal, residual norm) | 0.592 | -0.011 (n.s.) |
| Mahalanobis of residual | 0.558 | -0.045 (p=0.16) |
| rank-fuse(diff, Mahalanobis) | 0.550 | -0.053 |
| residual magnitude only | 0.536 | -0.067 |
| `cos_abn - cos_norm` | 0.510 | -0.093 |
| Euclidean to prototypes | 0.498 | -0.105 |
| **paper's normal score, and the code's fusion** | **0.398** | -0.205 (inverted) |

**Conclusion:** no alternative scorer beats the paper's abnormal score (all Wilcoxon p > 0.05). The *normal*-prototype
score is **worse than random on 7 of 9 graphs** and `cos_norm_pos` (the same quantity, sign-flipped) ties the abnormal
score: the two prototypes point in nearly the same direction (an inference from AUROC behaviour; we did not measure the
angle directly).

**Exp H result (mean AUROC over targets, scorer = abnormal):**

| dim | row-L2 | none | row-sum (paper) | z-score |
|---|---|---|---|---|
| 2 | 0.576 | 0.571 | 0.550 | 0.509 |
| 4 | 0.616 | 0.596 | 0.571 | 0.564 |
| 6 | 0.615 | 0.598 | 0.548 | 0.549 |
| 8 | **0.622** | 0.609 | 0.603 | 0.573 |

**Conclusion:** more than 4 dimensions barely helps; z-score is consistently worst; row-L2 is best on average but **not
significantly** better than the paper's row-sum (4/9 graphs improve, p = 0.91; over all 45 seed-graph pairs p = 0.41) -
the whole gain is on Elliptic (0.41 -> 0.60). Diagnostic: SVD features are *signed*, so the paper's division by the row
sum flips signs for 52.6% (Elliptic) and about 99.5% (Amazon) of rows and produces values up to about 10^4 - numerically
unsafe even though it happens to work on Amazon. Dimensions above 8 could not be tested (raw features not in the
shared download).

**Why this phase mattered:** it ruled out the cheap ideas (re-scoring, re-normalising) with evidence, and pointed to
the real lever - *what the model learns*, not how we score it.

## 6. Phase 3 - the four team contributions

Detailed per-contribution reports: `reports/C1..C4_*.md`. Pipeline: `evaluate_team_contributions.py`.

**Protocol (identical for every row):** pre-train on Facebook; zero-shot on Amazon / Amazon-UPU / Reddit; **5 seeds**;
the same fixed prompts per seed; evaluate on all target nodes except the 100 labelled-normal nodes reserved for
adaptation; report AUROC, AUPRC, trainable parameters, peak GPU memory (`torch.cuda.max_memory_allocated`), plus the
in-domain AUROC on 70% held-out Facebook nodes ("FB-val") to separate *"the module does not learn"* from *"the module
does not transfer"*.

### Results (mean ± std over 5 seeds; AUROC)

| Method | Params | Amazon | Amazon-UPU | Reddit |
|---|---|---|---|---|
| **Baseline AnomalyGFM** (GCN + SVD, supervised) | 484,402 | **0.834 ± .022** | **0.792 ± .017** | 0.564 ± .026 |
| + C1 H2GCN backbone | 810,802 | 0.668 ± .060 | 0.701 ± .051 | 0.589 ± .044 |
| + C2 Hypernetwork alignment | 507,514 | 0.418 ± .092 | 0.398 ± .055 | 0.584 ± .043 |
| + C3 SSL prototypes, **no labels** | 484,002 | 0.620 ± .082 | 0.588 ± .056 | 0.585 ± .010 |
| C1+C2+C3 combined (sup+SSL) | 833,914 | 0.529 ± .105 | 0.543 ± .030 | 0.550 ± .016 |

In-domain FB-val AUROC: baseline 0.828, C1 0.789, C2 0.75, C3 0.70 (no labels), combined 0.68 - *all modules learn*;
the failure is transfer.

Adaptation (k = 100 normal nodes; frozen baseline underneath):

| Method | Trainable | Amazon | Amazon-UPU | Reddit |
|---|---|---|---|---|
| Prompt tuning, paper protocol (5 ep, lr 1e-4) | 80 | 0.834 | 0.792 | 0.564 |
| Full GNN fine-tune, paper protocol | 164,002 | 0.839 | 0.795 | 0.563 |
| **C4 Graph-LoRA (r=4), paper protocol** | 4,832 | 0.834 | 0.793 | 0.563 |
| Combined + LoRA, paper protocol | 14,496 | 0.529 | 0.543 | 0.549 |
| [sens.] prompt, 30 ep lr 1e-3 | 80 | 0.851 ± .049 | 0.797 ± .010 | 0.568 ± .019 |
| [sens.] full fine-tune, 30 ep lr 1e-3 | 164,002 | 0.637 ± .116 | 0.586 ± .054 | 0.596 ± .025 |
| [sens.] **Graph-LoRA**, 30 ep lr 1e-3 | 4,832 | 0.651 ± .070 | 0.693 ± .044 | 0.608 ± .017 |

GPU memory (peak, MB): baseline 508 / 139 / 137 (Amazon / UPU / Reddit); +C1 532 / 162 / 160; +C2 512 / 144 / 142;
+C3 511 / 143 / 141; adaptation rows 1,060-1,085 / 208-234 / 208-228 (dominated by graph + model copy).

### Why each extension did what it did (full reasoning in the reports)

* **C1 (H2GCN, +67% params):** learns in-domain (0.789 vs 0.828) but transfers worse to Amazon; Reddit +0.025 is within noise.
* **C2 (hypernetwork):** inverted (<0.5) on Amazon. A hypernetwork trained on one graph sees one metadata vector, so it
  cannot learn how the projection should vary with metadata - our leading, untested explanation.
* **C3 (SSL):** the one clearly useful result: label-free, 0.62 on Amazon.
* **C4 (LoRA):** real parameter saving (2.9% of full fine-tuning), no reliable accuracy gain; the normal-only objective
  is the bottleneck.

### Fairness corrections we made (and why)

1. The prompt baseline is **identity-initialised**; the paper's random init perturbs the input before any tuning, which
   made the first prompt baseline look 0.09 AUROC worse than it is.
2. The main adaptation budget is the **paper's** (5 epochs, lr 1e-4); our larger budget is reported only as a labelled
   sensitivity check - choosing a budget on test AUROC would be leakage.
3. All rows are evaluated on the **same node set** (target minus the 100 tuned normals).
4. The in-domain FB-val column was added to prove the modules are not simply buggy.

