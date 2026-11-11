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
* A **multi-source pre-training** experiment (Section 7) tested whether the failures come from training on a single
  1,081-node graph. **Result: naive multi-source pooling did not fix it - it made the supervised baseline worse on
  Amazon (0.83 -> 0.61)** - but it removed C2's inversion (0.36 -> 0.62) and C3 is best on Reddit (0.61, within noise).

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


---------------------------------------------------------------------------------------------------------------------

## 7. Phase 4 - multi-source pre-training (does more training data fix transfer?)

**Why:** every extension learned in-domain but transferred badly from one 1,081-node graph. A hypernetwork conditioned on
dataset metadata (C2) cannot learn *how* to change with the metadata from a single metadata point, and a label-free
method (C3) can use any unlabelled graph. So the natural test was to pre-train on several graphs.

**Design (`evaluate_team_contributions.py --multisource`):**
* Source pool = Facebook, Reddit, Amazon, YelpChi, T-Finance **minus the target's domain** (the target domain is never
  seen: Amazon and Amazon-UPU both exclude Amazon; Reddit excludes Reddit) -> 4 sources per target.
* Large sources are represented by 12 pre-sampled **BFS subgraphs of at most 4,000 nodes** (SSL augmentation is dense);
  each epoch uses one random view of each source and a single optimiser step on the mean per-source loss.
* Supervised loss uses 30% labelled nodes per view (source labels only; the target is never used); SSL uses no labels.
* YelpChi/T-Finance 8-dim features come from our own SVD (the paper's recipe), Facebook/Reddit/Amazon from the authors' files.
* Same 301 epochs, lr 1e-4; same evaluation protocol as Section 6; **3 seeds** (about 5 minutes per seed per target on the GPU).
* The single-source Facebook rows were re-run inside the same job as a control: 0.832 / 0.785 / 0.572 (Amazon / UPU / Reddit)
  vs 0.834 / 0.792 / 0.564 in Section 6 - the pipeline reproduces itself.

**Results (AUROC mean ± std, 3 seeds; full tables in `results/ms/multisource.md`):**

| Method | Amazon | Amazon-UPU | Reddit |
|---|---|---|---|
| 1-source (Facebook) baseline | **0.832 ± .029** | **0.785 ± .020** | 0.572 ± .032 |
| 4-source baseline | 0.611 ± .089 | 0.612 ± .070 | 0.591 ± .018 |
| 4-source + C1 H2GCN | 0.548 ± .116 | 0.548 ± .066 | 0.590 ± .014 |
| 4-source + C2 Hypernet | 0.620 ± .012 | 0.617 ± .018 | 0.467 ± .004 |
| 4-source + C3 SSL (no labels) | 0.573 ± .122 | 0.576 ± .086 | **0.610 ± .035** |
| 4-source C1+C2+C3 | 0.530 ± .012 | 0.508 ± .025 | 0.476 ± .081 |

| Adaptation on the 4-source baseline | Trainable | Amazon | Amazon-UPU | Reddit |
|---|---|---|---|---|
| prompt / LoRA / (paper protocol) | 80 / 4,832 | 0.610 / 0.610 | 0.612 / 0.612 | 0.590 / 0.590 |
| [sens.] prompt, 30 ep lr 1e-3 | 80 | 0.576 | 0.593 | 0.582 |
| [sens.] full fine-tune | 164,002 | 0.449 | 0.485 | 0.608 |
| [sens.] **LoRA** | 4,832 | 0.448 | 0.527 | **0.608 ± .007** |

**What happened, honestly:**
1. **Naive pooling hurt the supervised baseline on both Amazon graphs** (-0.22 and -0.17 AUROC); on Reddit it is +0.019
   (inside the noise). More data did not help; it is a *negative transfer* result.
2. **C2's inversion disappeared** (0.36 -> 0.62 on Amazon; 0.36 -> 0.62 on UPU) once the hypernetwork saw four different
   metadata points. This is the one result that supports our single-metadata-point explanation. But it only *matches* the
   (degraded) 4-source baseline, and on Reddit C2 got worse (0.57 -> 0.47), so the picture is mixed.
3. **C3 is the best method on Reddit** (0.610; vs 0.572 for the 1-source baseline = +0.038, about 1.2 of that baseline's
   std - suggestive only with 3 seeds).
4. **Adaptation behaves as before:** no-ops at the paper's budget; at the larger budget LoRA and full fine-tuning hurt on
   Amazon (about 0.45) and are slightly positive on Reddit; the 80-parameter prompt is the safest.

**Why might pooling hurt? (all untested hypotheses):** (a) anomaly semantics differ across sources (spam reviews vs fraud
accounts vs Facebook accounts), so averaging supervised prototype alignment across them blurs the prototypes; (b) BFS-sampled
4,000-node subgraphs have different degree/neighbourhood statistics than the full target graph, so the *residuals* seen in
training differ from those at test time; (c) the SVD we computed for YelpChi/T-Finance may use different sign/scale
conventions than the authors' files, and the paper's row-sum normalisation is numerically unstable (Section 5); (d) all
sources are weighted equally although they differ in size and quality. These are confounds of *our* implementation, so this
result should be read as "naive multi-source pooling with our sampling does not help", not "multi-source pre-training cannot help".

## 8. Mistakes, bugs and corrections (an honest log)

| What went wrong | How we caught it | Fix |
|---|---|---|
| `G['X8'][0]` took one row instead of the matrix (IndexError) | first smoke test crashed | removed the stray index |
| Hypernet feature-range descriptor used `X.min(0).min()` (a scalar) | code review while writing the next file | per-feature `X.min(0).values` |
| Prompt baseline looked 0.09 AUROC worse | in-domain/fairness check | identity initialisation of the prompt |
| First adaptation budget (30 ep, lr 1e-3) was chosen without basis | self-review | paper protocol is now the main result; larger budget is a labelled sensitivity check |
| `rsync --delete` removed the `zero_shot/data` symlink | run failed with FileNotFound | exclude `data` in `sync.sh` and re-link |
| Monitor `pgrep -f` matched itself, never reported | job had finished long before | `bg`/`wait` with a `.done` marker file |
| `bg` mode held the SSH session open | tool call timed out | `setsid nohup ... < /dev/null` |
| `RuntimeWarning: divide by zero in power` in logs | user noticed | benign (isolated nodes; next line zeroes the infs); now wrapped in `np.errstate` |
| Network to the server dropped twice mid-run | `No route to host` | jobs run detached on the server and survived; we reconnected and read the results |
| Assumed a 40 GB GPU | `nvidia-smi` | actual RTX 4500 Ada is 24 GB |

## 9. What we can and cannot claim

**Can claim (measured):** the baseline reproduces; no re-scoring/re-normalisation beats the paper (Exp H/I); the normal
prototype score is inverted on 7 of 9 graphs; C1/C2 hurt transfer to Amazon from a single training graph; C3 reaches 0.62
AUROC on Amazon with no labels; LoRA uses 2.9% of full fine-tuning's parameters at equal accuracy under the paper's
protocol; naive multi-source pooling hurt the baseline on Amazon; the released few-shot code leaves its prompt disabled.

**Cannot claim:** that any extension *improves* accuracy over the baseline; any statement about *why* things fail beyond
the labelled hypotheses; results beyond 3 target graphs, one training protocol, 5 (single-source) / 3 (multi-source)
seeds; generality to other datasets (Elliptic, T-Finance, Yelp as *targets* were only used in Exp H/I with SVD features).

## 10. How to reproduce

```
# data: data/datasets for AnomalyGFM/*_svd.mat  (Google Drive link in the repo README)
#       data/raw/{Facebook,Amazon,Amazon-all,Reddit,YelpChi,t_finance}.mat  (UNPrompt Drive folder)
python zero_shot/bench.py --exp both --seeds 0 1 2 3 4 && python zero_shot/summarize.py      # Exp H + I
python evaluate_team_contributions.py --seeds 0 1 2 3 4 --targets amazon amazon_upu reddit  # Section 6
python evaluate_team_contributions.py --multisource --seeds 0 1 2 --targets amazon amazon_upu reddit --out results/ms   # Section 7
python zero_shot/explain.py --dataset_test amazon_svd                                       # explanation prototype
```
Remote workflow: `./sync.sh push`, `./sync.sh bg <name> "<cmd>"`, `./sync.sh wait <name>`, `./sync.sh pull`.

## 11. File map

| Path | What |
|---|---|
| `models/h2gcn_backbone.py` | C1 |
| `models/hypernet_alignment.py` | C2 |
| `utils/ssl_pretrain.py` | C3 |
| `models/graph_lora.py` | C4 |
| `evaluate_team_contributions.py` | unified evaluation (single-source and `--multisource`) |
| `zero_shot/bench.py`, `summarize.py` | Exp H and I harness / aggregation |
| `zero_shot/explain.py` | explanation prototype (Phase 1) |
| `reports/C1..C4_*.md` | detailed per-contribution reports |
| `results/team/`, `results/ms/` | final tables and raw CSVs |
| `slides/main.tex`, `slides/main.pdf` | Beamer deck |
| `TEAM_CONTRIBUTIONS.md` | where each module hooks into the original scripts |

## 12. Next steps (ordered by expected value)

1. **Find out why pooling hurts:** ablate sources one at a time; replace BFS subgraphs by full-graph sparse training; weight
   sources by size; re-do YelpChi/T-Finance SVD to match the authors' conventions.
2. **Fix the prototype collinearity** (add a separation/margin loss between normal and abnormal prototypes) - the one
   defect found independently in three experiments.
3. **Camouflage-aware corruptions for C3** (rewire towards *similar* nodes) so the pseudo-anomalies look like real fraud.
4. **Better few-shot objective for C4** (use the abnormal prototype and pseudo-anomalies, early stopping) - the objective, not the
   adapter, is the bottleneck; then a LoRA rank sweep.
5. 5 seeds for the multi-source runs.
