# Contribution 2 - Non-linear, dataset-conditioned feature alignment (hypernetwork)

Code: `models/hypernet_alignment.py` (`make_prep`, `FeatureHyperAlign`). Evaluated by `evaluate_team_contributions.py`.

## 1. Motivation

A graph foundation model must accept graphs whose node features have different meaning and width (Facebook 576
dims, Amazon 25, Reddit 64, YelpChi 32, T-Finance 10). The paper aligns them by a *fixed linear SVD* to `d' = 8`
dimensions (`U[:, :8] * S[:8]`, then row-normalisation). SVD is linear, fitted separately per graph, and its component
`k` has no consistent meaning across graphs. The hypothesis behind C2: *a learned, non-linear projection whose weights
are generated from dataset statistics could align graphs better than a fixed linear SVD.*

## 2. Method

A hypernetwork generates the weights of a 2-layer non-linear projection `R^D -> R^32 -> R^8` from descriptors, so it
works for any raw width `D` and for graphs never seen in training.

**Per-feature descriptors** `z_j` (8 numbers per raw feature column `j`): signed log-mean, log-std, skewness,
kurtosis, non-zero fraction, **graph smoothness** (normalised Dirichlet energy `x^T L x / (x^T D x)` - how much the
feature varies across edges), correlation with log-degree, log-range.
**Per-graph metadata** `m` (8 numbers): log N, log E, log mean degree, density, degree coefficient of variation,
log D, mean feature smoothness, mean non-zero fraction.

```
W1[j, :] = g1([z_j ; m]) / sqrt(D)          # D x 32, one row per feature (permutation-equivariant)
b1       = g_b1(m)
W2       = g2(m).view(32, 8) / sqrt(32)     # dataset-conditioned second layer
b2       = g_b2(m)
out      = L2norm( GELU( standardise(X) W1 + b1 ) W2 + b2 )         # N x 8
```
`standardise` is a per-graph z-score clipped to [-5, 5]. Each `g` is a 2-layer MLP (width 64). The hypernetwork is
trained **end-to-end** with the anomaly objective on the training graph; on an unseen graph it computes that graph's
descriptors and generates its projection - no SVD, no fitting on the target.

## 3. Implementation and integration

* `make_prep(X, A)` computes the constants (mean, std, descriptors, metadata) once per graph (no gradients).
* `FeatureHyperAlign(X_raw, prep)` returns the `N x 8` aligned features; in `evaluate_team_contributions.py` it
  replaces `preprocess_features`/SVD (`Net(align='hyper')`). To hook into the original scripts: compute `prep` once per
  graph, call the module inside the forward pass, and add its parameters to the optimiser.
* Parameters: 23,112 (model total 484,402 -> 507,514). Raw `.mat` files are required (the shared `*_svd.mat` files
  contain only the SVD output); we downloaded raw Facebook/Amazon/Reddit/YelpChi/T-Finance from the authors' sibling
  repository (UNPrompt) and verified the labels equal the SVD versions' labels exactly.

## 4. Protocol

As in C1 (train Facebook, 5 seeds, zero-shot on Amazon / Amazon-UPU / Reddit, same eval nodes, plus in-domain "FB-val").

## 5. Results (single-source; mean ± std, 5 seeds)

| Target | Baseline AUROC | **+C2 AUROC** | Baseline AUPRC | **+C2 AUPRC** | FB-val base / C2 |
|---|---|---|---|---|---|
| Amazon | 0.834 ± 0.022 | **0.418 ± 0.092** | 0.476 ± 0.121 | 0.058 ± 0.013 | 0.828 / 0.750 |
| Amazon-UPU | 0.792 ± 0.017 | **0.398 ± 0.055** | 0.388 ± 0.100 | 0.053 ± 0.005 | 0.828 / 0.752 |
| Reddit | 0.564 ± 0.026 | **0.584 ± 0.043** | 0.037 ± 0.002 | 0.043 ± 0.003 | 0.828 / 0.753 |

## 6. Analysis

* **Worse than random on both Amazon graphs** (AUROC 0.42 / 0.40 < 0.5): the model's ranking is *inverted* there. On
  Reddit it is on par with the baseline (+0.02, inside the noise).
* **In-domain it learns** (0.750 on held-out Facebook nodes vs 0.828 baseline), so the failure is in *transfer*.
* **Likely cause (reasoned, not verified):** a hypernetwork learns a mapping *dataset metadata -> projection*. Trained
  on a single graph it sees exactly one metadata vector, so nothing teaches it how the projection should change with the
  metadata; at test time Amazon's descriptors push it into a regime it has never been trained on. The multi-source
  experiment (see `WORK.md`) is the direct test of this explanation.
* Raw features have graph-specific semantics (Facebook's 576 attributes vs Amazon's 25 features - per the dataset's published
  description, hand-crafted reviewer-behaviour statistics; we did not inspect their meaning); a learned projection from the first can encode Facebook-specific structure that has no
  counterpart in the second. SVD, being unsupervised, avoids that failure mode - which may be why it is hard to beat.

## 7. Limitations

* Single training graph = a single metadata point (the central weakness above).
* Descriptors are hand-designed; a learned set encoder (e.g. DeepSets over feature columns) might be more expressive.
* We did not tune the hypernetwork size or add a reconstruction/auto-encoding objective (a graph auto-encoder variant
  of C2 is a natural extension).

## 8. Reproduce

```
python evaluate_team_contributions.py --seeds 0 1 2 3 4 --targets amazon amazon_upu reddit   # rows "+ C2 Hypernet alignment"
```

## 9. Multi-source pre-training: the direct test of the explanation (4 sources, target held out; 3 seeds)

| Target | 1-src C2 | 4-src baseline | **4-src + C2** |
|---|---|---|---|
| Amazon | 0.360 ± .031 | 0.611 ± .089 | **0.620 ± .012** |
| Amazon-UPU | 0.363 ± .022 | 0.612 ± .070 | **0.617 ± .018** |
| Reddit | 0.573 ± .058 | 0.591 ± .018 | 0.467 ± .004 |

* **The inversion disappears.** With four training graphs (four different metadata vectors) C2 goes from 0.36 to 0.62 on
  Amazon and from 0.36 to 0.62 on Amazon-UPU, and its variance collapses (±0.012). This is consistent with - and the only
  evidence for - the claim that a single training graph cannot teach a hypernetwork how to adapt to metadata.
* **But it is not a win.** It only matches the 4-source baseline (0.620 vs 0.611), which itself is far below the
  single-source baseline (0.832). And on **Reddit, C2 gets worse with more sources** (0.573 -> 0.467, below chance), so the
  gain is not general.
* Reading: more metadata diversity removes the *catastrophic* failure of C2, but we have not shown that learned alignment
  beats SVD. Four metadata points are still very few for learning a metadata -> weights map; the pool also contains
  graphs whose raw features (10-32 dims) are small and heterogeneous.
