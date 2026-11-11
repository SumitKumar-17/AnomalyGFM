<div align="center">

<img src="logo.png" width="200">

## AnomalyGFM: Graph Foundation Model for Zero/Few-shot Anomaly Detection 

![](https://img.shields.io/github/last-commit/mala-lab/AnomalyGFM?color=green)
![](https://img.shields.io/github/stars/mala-lab/AnomalyGFM?color=yellow)
![](https://img.shields.io/github/forks/mala-lab/AnomalyGFM?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)
[**📜 Datasets**](https://drive.google.com/drive/folders/1SSWgFRdth3U44_IMRnW775B1l-bjQATW?usp=sharing) [**📝 arxiv**](https://arxiv.org/abs/2502.09254)
</div>

> **This is a term-project fork of [mala-lab/AnomalyGFM](https://github.com/mala-lab/AnomalyGFM)** (Qiao, Niu, Chen, Pang - KDD'25, [arXiv:2502.09254](https://arxiv.org/abs/2502.09254)).
> The upstream paper README is kept **unchanged** further down (section *Original README*). Everything added by us lives on the branch
> **`team-contributions`**, in the folders and files listed below; the upstream code is only touched where noted in [Changes to upstream files](#changes-to-upstream-files).

---

# Extensions in this fork (two studies)

We ask: *which parts of AnomalyGFM are weak, and can zero/few-shot anomaly detection be improved?* - answered by **two independent studies**, each with
4 team contributions, controls, 5-seed benchmarks on **real data only**, detailed reports and slides. **The honest summary: no extension improves the
paper's average accuracy**; the reports explain what *does* hold up (and why the rest fails).

## Quick links

| | Study 1 (repo root) | Study 2 ([`Anomaly2/`](Anomaly2/)) |
|---|---|---|
| **Slides (PDF)** | [`slides/main.pdf`](slides/main.pdf) - 19 pp (source [`main.tex`](slides/main.tex)) | [`Anomaly2/slides/main.pdf`](Anomaly2/slides/main.pdf) - 18 pp (source [`main.tex`](Anomaly2/slides/main.tex)) |
| **Full write-up** (what we did, why, what happened, mistakes log, claims we can/can't make) | [`WORK.md`](WORK.md) | [`Anomaly2/WORK.md`](Anomaly2/WORK.md) |
| **Hooking guide** into the official code | [`TEAM_CONTRIBUTIONS.md`](TEAM_CONTRIBUTIONS.md) | [`Anomaly2/README.md`](Anomaly2/README.md) |
| **Detailed reports** | [C1 H2GCN](reports/C1_h2gcn_backbone.md) · [C2 Hypernet](reports/C2_hypernet_alignment.md) · [C3 SSL](reports/C3_ssl_prototypes.md) · [C4 LoRA](reports/C4_graph_lora.md) | [C1 MoE](Anomaly2/reports/C1_moe_prototypes.md) · [C2 TTA](Anomaly2/reports/C2_unsupervised_tta.md) · [C3 Purification](Anomaly2/reports/C3_edge_purification.md) · [C4 PPR](Anomaly2/reports/C4_adaptive_residuals.md) |
| **Result tables** | [`results/team/`](results/team/team_results.md) · [`results/ms/`](results/ms/multisource.md) | [`Anomaly2/results/`](Anomaly2/results/) ([master](Anomaly2/results/team_results.md), [attack](Anomaly2/results/attack_benchmark.md), [ablation](Anomaly2/results/ablation_residuals.md), [MoE](Anomaly2/results/moe.md), [TTA](Anomaly2/results/tta.md)) |
| **Entry point** | [`evaluate_team_contributions.py`](evaluate_team_contributions.py) | [`Anomaly2/main.py`](Anomaly2/main.py) |

## Study 1 - encoder, feature alignment, label-free prototypes, parameter-efficient tuning

| # | Contribution | Code | Idea |
|---|---|---|---|
| C1 | Heterophily-aware backbone | [`models/h2gcn_backbone.py`](models/h2gcn_backbone.py) | H2GCN-style ego/neighbour/2-hop layer, drop-in for the paper's GCN |
| C2 | Non-linear feature alignment | [`models/hypernet_alignment.py`](models/hypernet_alignment.py) | a hypernetwork generates the projection from feature/graph descriptors (replaces the static SVD) |
| C3 | Contrastive prototype learning, **no labels** | [`utils/ssl_pretrain.py`](utils/ssl_pretrain.py) | corrupt the training graph (rewire + feature swap) -> pseudo-anomalies -> InfoNCE to the prototypes |
| C4 | Graph-LoRA | [`models/graph_lora.py`](models/graph_lora.py) | low-rank adapters in the GNN weights for few-shot tuning (toggle on/off) |

Train on Facebook, zero-shot on Amazon / Amazon-UPU / Reddit, 5 seeds (AUROC):

| Method | Amazon | Amazon-UPU | Reddit | Params |
|---|---|---|---|---|
| **Baseline AnomalyGFM** | **0.834** | **0.792** | 0.564 | 484,402 |
| + C1 H2GCN | 0.668 | 0.701 | 0.589 | 810,802 |
| + C2 Hypernetwork | 0.418 | 0.398 | 0.584 | 507,514 |
| + C3 SSL, **no labels** | 0.620 | 0.588 | 0.585 | 484,002 |
| C4 Graph-LoRA (few-shot) | = baseline at the paper's budget | | | **4,832** (2.9% of full fine-tuning) |

Take-aways: every module learns in-domain but **transfers worse from one 1,081-node training graph**; C3 learns anomaly structure with **zero labels** (0.62 on Amazon);
LoRA delivers the parameter saving but no accuracy gain; naive multi-source pooling hurt the baseline (Amazon 0.83 -> 0.61) while removing C2's inversion (0.36 -> 0.62).
Also: the released few-shot code leaves its prompt commented out of `forward`, so it does no real tuning (details in [`WORK.md`](WORK.md)).

## Study 2 - prototypes, test-time adaptation, trusted edges, receptive field ([`Anomaly2/`](Anomaly2/))

| # | Contribution | Code | Idea |
|---|---|---|---|
| C1 | Mixture-of-Experts prototypes | [`models/moe_prototypes.py`](Anomaly2/models/moe_prototypes.py) | K abnormal experts + gate + load balancing; t-SNE of routing |
| C2 | Unsupervised test-time adaptation | [`utils/unsupervised_tta.py`](Anomaly2/utils/unsupervised_tta.py) | entropy + confident pseudo-labels, updates only an 80-parameter prompt; per-step AUROC/AUPRC curves |
| C3 | Adversarial edge purification | [`models/edge_purification.py`](Anomaly2/models/edge_purification.py) | Gumbel-sigmoid edge dropping (structural + feature scorer) before message passing; attack benchmark |
| C4 | Adaptive PPR residuals | [`models/adaptive_residuals.py`](Anomaly2/models/adaptive_residuals.py) | multi-hop reference with per-node learned hop weights; 1-hop vs 2-hop vs PPR vs adaptive ablation |

Train on Facebook, zero-shot on Amazon / Amazon-UPU / Reddit / Elliptic, 5 seeds:

| Method | Avg AUROC | Avg AUPRC | Params | Peak GPU MB |
|---|---|---|---|---|
| **Baseline AnomalyGFM** | **0.648** | 0.244 | 484,402 | 626 |
| + C1 MoE (K=4) | 0.642 | 0.226 | 487,606 | 630 |
| + C2 TTA | 0.644 | 0.211 | +80 updated | 860 |
| + C3 Edge purification | 0.645 | 0.252 | 485,107 | 1,675 |
| + C4 Adaptive PPR | 0.646 | 0.251 | 484,534 | 675 |

What is robust (see [`Anomaly2/WORK.md`](Anomaly2/WORK.md)): the paper's residual is **fragile to camouflage** - wiring each anomaly to 50 random normal nodes drops Amazon-UPU AUROC
0.792 -> **0.085**, and **C3 holds 0.678**; C3 also lifts Elliptic 0.394 -> 0.499 (control without purifier: 0.372) but fails on triangle-free Reddit and under random rewiring.
MoE experts collapse (one expert takes 84% of Amazon's anomalies); label-free TTA does not help (its Elliptic gain is matched by random pseudo-labels); adaptive PPR learns a per-node hop choice but only ties 1-hop.

## Honesty notes
* Results are **negative on average accuracy**; each report separates measured facts from untested hypotheses.
* Study 2 round 1 trained new modules at the paper's learning rate and was flawed; it was diagnosed and corrected (per-module learning rates, structural edge feature, all chosen on the **source** graph). Round 1 is kept in `Anomaly2/results/round1_lr1e-4_untuned/`. Round 2 is not fully blind (round-1 target results were seen before the purifier redesign).
* Study 1's multi-source run used 3 seeds (runtime); everything else uses 5.
* Only the graphs' **edge sets** are ever perturbed (attack benchmark); no synthetic graphs are used. No dataset carries anomaly-type labels, so claims about MoE experts matching anomaly *types* are not made.
* Hardware: one RTX 4500 Ada with **24 GB** (not 40 GB).

## Setup and reproduce

```bash
# environment used (python 3.10): torch 2.5.1+cu121, torch_geometric, scipy, scikit-learn, pandas, matplotlib, tabulate, gdown
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric scipy scikit-learn pandas matplotlib tabulate gdown numpy
# (the upstream requirements.txt pins very old versions - torch 1.11 - and is not needed for our scripts)
```

Data (not stored in git; `data/` is git-ignored):
* `data/datasets for AnomalyGFM/*_svd.mat` - the authors' SVD-aligned graphs ([Drive](https://drive.google.com/drive/folders/1SSWgFRdth3U44_IMRnW775B1l-bjQATW)).
* `data/raw/{Facebook,Amazon,Amazon-all,Reddit,YelpChi,t_finance}.mat` - raw features, from the authors' sibling repo [UNPrompt](https://github.com/mala-lab/UNPrompt) ([Drive](https://drive.google.com/drive/folders/1qcDBcVdcfAr_q5VOXBYagtnhA_r3Mm3Z)); needed only by Study 1's hypernetwork (C2). Their labels were verified identical to the SVD files'.

```bash
# Study 1 (repo root)
python evaluate_team_contributions.py --seeds 0 1 2 3 4 --targets amazon amazon_upu reddit             # single-source table
python evaluate_team_contributions.py --multisource --seeds 0 1 2 --targets amazon amazon_upu reddit   # multi-source pre-training
python zero_shot/bench.py --exp both --seeds 0 1 2 3 4 && python zero_shot/summarize.py                # scoring / feature-alignment benchmark

# Study 2
cd Anomaly2 && python main.py all --seeds 0 1 2 3 4        # ~15 min on one RTX 4500 Ada: master table, ablation, attack, MoE, TTA
```
Neither study can run from the upstream scripts alone: the official repo has **no `train.py` or `main.py`** (training lives in `zero_shot/run_abnormal.py`
and `few_shot/*.py`); `Anomaly2/train.py`/`main.py` and `evaluate_team_contributions.py` are our integration layers, and the two hooking guides show where each
module plugs into the original scripts.

## Repository map

```
README.md  WORK.md  TEAM_CONTRIBUTIONS.md        this file; Study 1 write-up; Study 1 hooking guide
models/ utils/ evaluate_team_contributions.py     Study 1 code (C1-C4 + evaluation)
reports/  slides/  results/{team,ms}/             Study 1 reports, deck, tables
Anomaly2/                                         Study 2: code, README, WORK.md, reports/, slides/, results/
zero_shot/ few_shot/ large_scale/ ...             upstream code (+ our bench.py, explain.py, summarize.py in zero_shot/)
```

## Changes to upstream files
* `zero_shot/*.py`, `few_shot/*.py`: `dgl` / `torch_sparse` imports made optional (unused by the code paths we run).
* `zero_shot/utils.py`: the benign `divide by zero` warning for isolated nodes in `normalize_adj` is suppressed (the next line already zeroes the infinities).
* Added: `zero_shot/bench.py`, `zero_shot/summarize.py` (scoring / feature-alignment benchmark), `zero_shot/explain.py` (explanation prototype).

---

# Original README (upstream, unchanged)

## Promotional Video

📢 We are excited to share that our paper presentation video is now available on YouTube!

🎬 Watch it here: [[YouTube Link](https://youtu.be/OrY9epUwh0M?si=izAi9xjQuNLQmaMX)]

📌 Your feedback and support are greatly appreciated!


## Main Contributions


> 1️⃣ AnomalyGFM is the **first** GAD-oriented GFM with strong zero-shot and few-shot generalization abilities. 

> 2️⃣ AnomalyGFM is pre-trained to learn discriminative, graph-agnostic class prototypes with normal and abnormal residual features, and it supports few-shot graph prompt tuning for better adaptation.

> 3️⃣ A comprehensive benchmark on both zero-shot and few-shot settings using 11 real-world GAD datasets is established, on which i) AnomalyGFM performs significantly
better the state-of-the-art unsupervised, supervised, and generalist GAD methods, and ii) AnomalyGFM can scale up to very large graphs


![Framework of AnomalyGFM](framework.png)



The GAD datasets after feature alignment can be obtained from   [google drive link](https://drive.google.com/drive/folders/1SSWgFRdth3U44_IMRnW775B1l-bjQATW?usp=sharing). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training and Zero-shot Inference  

To run the model(s), run this command:
```
python run_abnormal.py 
```

## Training and Few-shot Inference  

Run this command:
```
python run_finetune_normal.py 
```
For few-shot labeled normal nodes fine-tuning.

Run this command:
```
python run_finetune_abnormal.py 
```
for few-shot labeled abnormal nodes fine-tuning.

## Large-scale Graph Generalization 

Run this command:
```
python run_inference.py 
```

## Citation
If you find this repo useful, please cite our paper.

```bibtex
@inproceedings{qiao2025anomalygfm,
  title={AnomalyGFM: Graph foundation model for zero/few-shot anomaly detection},
  author={Qiao, Hezhe and Niu, Chaoxi and Chen, Ling and Pang, Guansong},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={2326--2337},
  year={2025}
}
```
