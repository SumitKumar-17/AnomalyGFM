# Anomaly2 / WORK.md - what we did, why, and what happened

Second study on **AnomalyGFM** (Qiao, Niu, Chen, Pang, KDD'25; paper recap in `../WORK.md` Section 1). Same infrastructure as the first study (GPU server, `sync.sh`),
different ideas: instead of changing the encoder/feature alignment, these four extensions change **how anomalies are represented (C1), how the
model is adapted without labels (C2), which edges it trusts (C3) and what it compares a node against (C4)**.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

* **No extension improves average zero-shot AUROC** over the paper's model: baseline 0.648, C1 0.642, C2 0.644, C3 0.645, C4 0.646
  (4 targets x 5 seeds; differences are within noise).
* The robust, *interesting* findings are narrower:
  1. **Camouflage breaks the paper's model.** Wiring each anomaly to 50 random normal nodes drops Amazon-UPU AUROC 0.792 -> **0.085** and Reddit 0.573 -> **0.029**
     (the ranking inverts). **C3 (edge purification) largely prevents it** (UPU 0.678; Reddit 0.300, still < 0.5) and lifts Elliptic 0.394 -> 0.499
     (control without purifier: 0.372). It does **not** help under random rewiring (UPU @50%: 0.617 vs 0.819) and fails on Reddit (clean 0.50).
  2. **C4 learns a genuinely adaptive receptive field** (about 80% weight on hop 1, about 15-20% on hop 4, strongly node-dependent) **but it only ties the 1-hop residual** (0.623 vs 0.626);
     fixed PPR / 2-hop are worse (0.610 / 0.613).
  3. **C1 experts collapse**: one expert gets 84% of Amazon's anomalies (689/821) and 100% on Reddit; AUROC decreases slowly with K.
  4. **C2 label-free adaptation does not work here**: it lowers AUROC on Amazon (-0.028 at lr 1e-3, -0.113 at 1e-2); its +0.14 on Elliptic is matched by *random* pseudo-labels (+0.127), so it is not learning from pseudo-labels.
* **Process lesson (my mistake, fixed):** the first round trained all new modules with the paper's lr 1e-4 and they were barely trained (purifier edge-AUROC 0.30 on the
  *training* graph; adaptive gate == fixed PPR). Round 1 looked like "everything collapses"; after source-only selection of learning rates and a structural edge feature,
  round 2 shows parity-with-the-baseline plus the specific effects above. Both rounds are kept (`results/round1_lr1e-4_untuned/`).

---------------------------------------------------------------------------------------------------------------------

## 1. The four contributions

| # | Idea | Why it might help | Implementation (file) |
|---|---|---|---|
| C1 | K abnormal sub-prototypes `p_k = ReLU(W_abn eps_k + delta_k)`, gate `softmax(Linear(r/|r|))`, load-balancing + diversity losses | one abnormal prototype cannot cover heterogeneous anomalies (paper's own premise) | `models/moe_prototypes.py` |
| C2 | at test time: robust-z score -> `p = sigmoid((z - q)/T)`, pseudo-label `p>0.95 / <0.05`, BCE + entropy, update only an 80-param input prompt | adapt to the target with no labels | `utils/unsupervised_tta.py` |
| C3 | Gumbel-sigmoid edge keep-weights from symmetric edge features (feature difference/product, degrees, **common neighbours**), hard drop at inference, adversarial training graph with known injected edges | camouflage edges drag a neighbour mean toward the anomaly and erase its residual | `models/edge_purification.py`, `utils/attacks.py`, `train.training_attack` |
| C4 | residual vs `sum_k w_vk (P^k h)_v`: hop1 / hop2 / PPR (truncated, alpha 0.3) / adaptive (`w_v = softmax(log PPR prior + MLP(structure_v))`) | deeper structures (laundering rings) are invisible to a 1-hop reference | `models/adaptive_residuals.py` |

Integration layer: `train.py` (training/evaluation), `main.py` (entry point), `models/core.py` (`GFM` = the paper's `Model` + pluggable parts). With defaults `GFM` **is** the paper's model:
seed-0 Amazon AUROC 0.8603, identical to the first study's baseline.

## 2. Protocol

Train on Facebook (30% of nodes labelled, 301 epochs, Adam, lr 1e-4 - the paper's setting); zero-shot on **all nodes** of Amazon (11,944 nodes), Amazon-UPU (10,224), Reddit (10,984), Elliptic (46,564);
5 seeds; fixed per-seed prompt draw; metrics AUROC / AUPRC; parameters; `torch.cuda.max_memory_allocated`. All graphs are real (authors' SVD-aligned files); only the *attack benchmark* perturbs
the edge set of real graphs (labels and features untouched; labels are used only to construct the attacker). GPU: RTX 4500 Ada **24 GB**.

## 3. Process: round 1 -> diagnosis -> fixes -> round 2

**Round 1 (all new modules at the paper's lr 1e-4).** Everything got worse (average AUROC: C1 0.639, C3 0.559, C4 0.626, combined 0.494). Controls showed C3's damage came from the adversarial augmentation, not the
purifier, and an edge-level diagnostic (`scripts/diagnose_purifier.py`) showed the purifier **dropped 0% of edges** - it was a no-op. A check on the *training* graph showed edge-AUROC 0.22-0.35 (< 0.5):
it had not learned. Root cause: new, freshly initialised modules were trained for 301 steps at 1e-4. Also adaptive PPR == fixed PPR to three decimals (its zero-initialised gate had not moved).

**Fixes (selection on the SOURCE graph only - Facebook held-out nodes; never the targets):**
1. *Learning rate of the new modules* (`scripts/select_lr_ext.py`, 3 seeds, held-out Facebook AUROC; baseline 0.828): MoE best at 1e-3 (0.820), adaptive-PPR best at 1e-2 (0.828 - matches baseline), purifier needs >= 1e-2 to learn at all. Stored in `train.LR_EXT`.
2. *Purifier features*: with node features only, edge-AUROC on a fresh attack of Facebook was 0.70 and it dropped 48% of real edges. Adding **common-neighbour counts / Salton overlap** (random edges share almost no neighbours) raised it to **0.96**, and removing class-balancing in the edge loss cut real-edge drops 48% -> 3%.
3. *Attack-augmentation strength* (`scripts/select_attack_cfg.py`): rule fixed in advance = maximise mean(clean, attacked) held-out Facebook AUROC. Chosen: 10% noise edges, 10 camouflage edges per anomaly, attacked graph on 50% of epochs
   (purified 0.839 clean / 0.735 attacked vs control without purifier 0.741 / 0.610). Tables: `results/select_*.md`.
4. A technical fix: sparse `A @ A` overflowed the GPU's sparse-matmul workspace on Amazon; a blocked dense fallback was added (verified identical to the sparse path, max difference 0).

**Honesty note:** the diagnostics that exposed round 1's flaws were run on target graphs (edge-level statistics, and I saw round-1 target AUROCs) before I redesigned the purifier. Hyper-parameters were then chosen on the source only, but the *design decision* to add a structural feature was informed by what I saw on targets, so round 2 is not a fully blind evaluation.

## 4. Round-2 results (final)

**Master table - AUROC (mean over 5 seeds; ± in `results/team_results.md`)**

| Method | Amazon | Amazon-UPU | Reddit | Elliptic | Average | Avg AUPRC | Params | Peak GPU MB (avg) |
|---|---|---|---|---|---|---|---|---|
| Baseline AnomalyGFM | 0.834 | 0.792 | 0.573 | 0.394 | **0.648** | 0.244 | 484,402 | 626 |
| + C1 MoE (K=4) | 0.820 | 0.793 | 0.556 | 0.399 | 0.642 | 0.226 | 487,606 | 630 |
| + C2 TTA on baseline (30 steps, lr 1e-3) | 0.812 | 0.796 | 0.551 | 0.418 | 0.644 | 0.211 | +80 updated | 860 |
| ctrl: TTA with shuffled pseudo-labels | 0.840 | 0.795 | 0.556 | 0.417 | 0.652 | 0.247 | +80 updated | 860 |
| + C3 Edge purification | 0.824 | 0.758 | 0.500 | **0.499** | 0.645 | 0.252 | 485,107 | 1,675 |
| ctrl: baseline + attack-aug (no purifier) | 0.799 | 0.771 | 0.528 | 0.372 | 0.617 | 0.213 | 484,402 | 640 |
| + C4 Adaptive PPR residual | 0.834 | 0.795 | 0.558 | 0.395 | 0.646 | 0.251 | 484,534 | 675 |
| C1+C3+C4 combined | 0.825 | 0.753 | 0.482 | 0.499 | 0.640 | 0.220 | 488,443 | 1,706 |
| Combined + C2 TTA | 0.819 | 0.750 | 0.484 | 0.525 | 0.644 | 0.202 | 488,523 | 1,764 |

Seed std of the 4-target average is 0.009-0.037, so none of the average differences is significant. The rows marked *ctrl* isolate what each contribution actually adds.

**C3 under attack (AUROC; Amazon-UPU; full tables in `results/attack_benchmark.md`)**

| | clean | camouflage 5 | camouflage 20 | camouflage 50 | rewire 30% | rewire 50% |
|---|---|---|---|---|---|---|
| Baseline | 0.792 | 0.669 | 0.244 | **0.085** | 0.822 | 0.819 |
| Baseline + attack-aug (ctrl) | 0.771 | 0.632 | 0.164 | 0.046 | 0.799 | 0.793 |
| **Edge-purified** | 0.760 | **0.734** | **0.701** | **0.678** | 0.667 | 0.617 |

Reddit @ camouflage 50: baseline 0.029, purified 0.300 (still inverted); Amazon (average degree 736): attacks barely matter (baseline 0.841, purified 0.826). Edge-level diagnostic (`results/purifier_diagnostic.md`): on Amazon-UPU the
purifier separates injected from real edges with **AUROC 0.99**, dropping 99.5% of injected and 4.7% of real edges; on **Reddit it fails** (edge-AUROC 0.12, drops 96% of real edges) because Reddit has **zero common neighbours on every edge** (triangle-free),
so the structural feature learned on Facebook mislabels all of them as injected.

**C2 learning curves (50 steps, mean of 5 seeds; `results/tta_curves_*.png`, `results/tta.md`)**

| Target | AUROC step 0 | lr 1e-3 (delta) | lr 1e-2 (delta) | random-pseudo-label control, lr 1e-2 (delta) | pseudo-positive precision (lr 1e-3) | base rate |
|---|---|---|---|---|---|---|
| Amazon | 0.834 | -0.028 | -0.113 | -0.338 | 0.748 | 0.069 |
| Amazon-UPU | 0.792 | +0.005 | -0.016 | -0.230 | 0.648 | 0.068 |
| Reddit | 0.573 | -0.030 | -0.085 | -0.055 | 0.007 | 0.033 |
| Elliptic | 0.394 | +0.042 | **+0.140** | **+0.127** | 0.002 | 0.098 |

Pseudo-labels are only trustworthy when the zero-shot ranking already works (Amazon 0.75, UPU 0.65). On Reddit/Elliptic the baseline ranking is weak/inverted, so pseudo-positives are *worse than chance* (0.7% and 0.2% precision vs 3.3% / 9.8% base rate).
Elliptic's +0.14 is therefore not label-free learning: random pseudo-labels give nearly the same gain (the prompt perturbation shifts an inverted ranking), and on Amazon, where pseudo-labels are good, adaptation *hurts* (entropy minimisation sharpens the score distribution).

**C1 (MoE)** K sweep (AUROC, K=1 is the paper): Amazon 0.834 / 0.827 / 0.820 / 0.812 for K = 1, 2, 4, 8; Reddit 0.573 -> 0.547; Amazon-UPU and Elliptic flat. Routing (seed 0, K=4): Amazon expert usage over 821 anomalies [689, 117, 2, 13],
silhouette of the expert assignment 0.145; UPU [438, 220, 7, 28], silhouette 0.076; Reddit [366, 0, 0, 0] (full collapse); Elliptic [1086, 908, 1864, 687], silhouette 0.035. The t-SNE (`results/moe_tsne_*.png`) shows experts 1 and 3 occupying separate regions of residual space while expert 0 absorbs most anomalies.
Expert assignment is significantly associated with a *degree-based proxy* type (chi-square p < 1e-4), i.e. experts partly encode node degree.
**Important:** none of the datasets carries anomaly-type labels (Reddit's "structural"/"attribute" label columns are identical to its main label), so we cannot show that experts correspond to real anomaly *types*. The Facebook training split holds only about 7 labelled anomalies for K experts - data starvation is a likely cause of collapse (untested).

**C4 ablation (AUROC; `results/ablation_residuals.md`)**

| Residual | Amazon | UPU | Reddit | Elliptic | Tolokers | Average | Avg AUPRC |
|---|---|---|---|---|---|---|---|
| 1-hop (paper) | 0.834 | 0.792 | 0.573 | 0.394 | 0.538 | 0.626 | 0.249 |
| 2-hop | 0.819 | 0.792 | 0.531 | 0.380 | 0.544 | 0.613 | 0.233 |
| PPR (alpha 0.3, K=4) | 0.815 | 0.790 | 0.534 | 0.372 | 0.540 | 0.610 | 0.225 |
| **Adaptive PPR** | 0.834 | 0.795 | 0.558 | 0.395 | 0.532 | 0.623 | 0.254 |

Learned adaptive hop weights (mean over nodes; prior was 0.40/0.28/0.19/0.14): Facebook 0.785 / 0.003 / 0.003 / 0.210; Amazon 0.801 / 0.006 / 0.005 / 0.188; Reddit 0.788 / 0.006 / 0.006 / 0.200; Elliptic 0.853 / 0.002 / 0.003 / 0.142, with within-graph std about 0.27-0.30: most nodes use the 1-hop reference and a minority a 4-hop one. On Elliptic (laundering structures) deeper references bring nothing (0.395 vs 0.394).
Note that `P^k` is a random-walk operator, so even hops include returns to the node itself.

## 5. Mistakes and corrections (log)

| Mistake | How found | Fix |
|---|---|---|
| New modules trained at the paper's lr 1e-4 | purifier edge-AUROC < 0.5 *on the source*; adaptive == PPR | per-module lr chosen on source (`LR_EXT`) |
| Purifier used only node features (edge-AUROC 0.70, dropped 48% real edges) | source diagnostic | + common-neighbour features, un-balanced edge BCE |
| Attack augmentation too strong (clean AUROC 0.54-0.69 in-domain) | source diagnostic | strength selected on source |
| Sparse `A@A` ran out of GPU workspace on Amazon | CUDA error | blocked dense fallback, verified identical |
| `main.py ablation/attack` saved raw CSV but no tables | missing files | shared `report()` functions |
| Attack-table labels `5.0` vs `5` | KeyError | formatted with `{:g}` |
| Overlapping t-SNE titles | looked at the figure | smaller fonts |

## 6. What we can and cannot claim

**Can claim:** the paper's residual is *fragile to camouflage* (AUROC inverts); C3 with structural edge features detects injected edges (edge-AUROC 0.99 on UPU) and keeps AUROC at 0.68 under 50-edge camouflage where the baseline falls to 0.085;
C3 improves Elliptic over both baseline and control; an adaptive PPR gate learns a per-node, bimodal hop choice; MoE routing collapses with the paper's tiny labelled-anomaly set; pseudo-label self-training is unreliable unless the zero-shot ranking is already good.

**Cannot claim:** any average accuracy improvement; that MoE experts match anomaly types; that C3 helps on graphs without triangles or under random rewiring; generality beyond four target graphs and Facebook as the single source; a blind evaluation of round 2 (see honesty note, Section 3).

## 7. Reproduce

`cd Anomaly2 && python main.py all --seeds 0 1 2 3 4` (about 15 min) after `python scripts/select_lr_ext.py` and `python scripts/select_attack_cfg.py` if you want to re-derive the source-selected settings.
Remote workflow: `./sync.sh push`, `./sync.sh bg <name> "<cmd>"`, `./sync.sh wait <name>`.

## 8. Next steps (by expected value)

1. Make the purifier **graph-agnostic**: normalise structural features per graph / use learned embeddings so it does not misfire on triangle-free graphs (Reddit).
2. **Pre-train MoE on more labelled anomalies** (multi-source, as in the first study) to test the data-starvation explanation.
3. **Better pseudo-labels for C2** (use the model's agreement across prompt draws / augmentations instead of one score) and a stopping rule; test only where the zero-shot ranking is good.
4. Apply C4's gate with *non-backtracking* walks so deeper hops are not dominated by returns to the node.
