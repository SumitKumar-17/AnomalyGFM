# Extending AnomalyGFM - term project

Two studies on improving zero/few-shot graph anomaly detection with a graph foundation model. All experiments use **real graphs only**
(train: Facebook; zero-shot targets: Amazon, Amazon-UPU, Reddit, Elliptic), 5 seeds, and report AUROC, AUPRC, trainable parameters and peak GPU memory.

**Reference:** this is a fork of [mala-lab/AnomalyGFM](https://github.com/mala-lab/AnomalyGFM) - Qiao, Niu, Chen, Pang,
*AnomalyGFM: Graph Foundation Model for Zero/Few-shot Anomaly Detection*, KDD'25 ([arXiv:2502.09254](https://arxiv.org/abs/2502.09254)).
The original README is in the upstream repository. Our work is on branch `team-contributions`.

## Study 1 - encoder, feature alignment, label-free prototypes, parameter-efficient tuning

| # | Contribution | Code | Report |
|---|---|---|---|
| C1 | Heterophily-aware backbone (H2GCN) | [`models/h2gcn_backbone.py`](models/h2gcn_backbone.py) | [report](reports/C1_h2gcn_backbone.md) |
| C2 | Hypernetwork feature alignment | [`models/hypernet_alignment.py`](models/hypernet_alignment.py) | [report](reports/C2_hypernet_alignment.md) |
| C3 | Label-free contrastive prototype learning | [`utils/ssl_pretrain.py`](utils/ssl_pretrain.py) | [report](reports/C3_ssl_prototypes.md) |
| C4 | Graph-LoRA few-shot adaptation | [`models/graph_lora.py`](models/graph_lora.py) | [report](reports/C4_graph_lora.md) |

Write-up: [`WORK.md`](WORK.md) · Hooks into the original code: [`TEAM_CONTRIBUTIONS.md`](TEAM_CONTRIBUTIONS.md) · Results: [`results/team/`](results/team/team_results.md), [`results/ms/`](results/ms/multisource.md) · Slides: [`slides/main.pdf`](slides/main.pdf)

## Study 2 - prototypes, test-time adaptation, trusted edges, receptive field (`Anomaly2/`)

| # | Contribution | Code | Report |
|---|---|---|---|
| C1 | Mixture-of-Experts abnormal prototypes | [`Anomaly2/models/moe_prototypes.py`](Anomaly2/models/moe_prototypes.py) | [report](Anomaly2/reports/C1_moe_prototypes.md) |
| C2 | Unsupervised test-time adaptation | [`Anomaly2/utils/unsupervised_tta.py`](Anomaly2/utils/unsupervised_tta.py) | [report](Anomaly2/reports/C2_unsupervised_tta.md) |
| C3 | Adversarial edge purification | [`Anomaly2/models/edge_purification.py`](Anomaly2/models/edge_purification.py) | [report](Anomaly2/reports/C3_edge_purification.md) |
| C4 | Adaptive PPR residuals | [`Anomaly2/models/adaptive_residuals.py`](Anomaly2/models/adaptive_residuals.py) | [report](Anomaly2/reports/C4_adaptive_residuals.md) |

Write-up: [`Anomaly2/WORK.md`](Anomaly2/WORK.md) · Hooks and layout: [`Anomaly2/README.md`](Anomaly2/README.md) · Results: [`Anomaly2/results/`](Anomaly2/results/) · Slides: [`Anomaly2/slides/main.pdf`](Anomaly2/slides/main.pdf)

## Results at a glance

**No extension improves the paper's average accuracy.** What does hold up:

* **Study 1:** every module learns in-domain but transfers worse from the single 1,081-node training graph. The label-free C3 reaches 0.62 AUROC on Amazon
  (supervised baseline 0.83); LoRA trains 4,832 parameters (2.9% of full fine-tuning) at unchanged accuracy.
* **Study 2** (average AUROC over 4 targets: baseline 0.648, C1 0.642, C2 0.644, C3 0.645, C4 0.646): the paper's residual is **fragile to camouflage**
  (Amazon-UPU AUROC 0.792 -> 0.085 with 50 camouflage edges per anomaly) and **C3 holds 0.678**; C3 also lifts Elliptic 0.394 -> 0.499 (control without purifier 0.372).
  MoE experts collapse, label-free test-time adaptation does not help, adaptive PPR only ties the 1-hop residual.

Caveats are stated in each write-up: Study 2's first round was flawed and is kept for the record (`Anomaly2/results/round1_lr1e-4_untuned/`); its second round is not fully blind;
Study 1's multi-source run used 3 seeds.

## Setup and run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121          # torch 2.5.1+cu121, python 3.10
pip install torch_geometric scipy scikit-learn pandas matplotlib tabulate gdown numpy
```
Data (git-ignored):
`data/datasets for AnomalyGFM/*_svd.mat` ([authors' Drive](https://drive.google.com/drive/folders/1SSWgFRdth3U44_IMRnW775B1l-bjQATW)) and, for Study 1's hypernetwork,
`data/raw/{Facebook,Amazon,Amazon-all,Reddit,YelpChi,t_finance}.mat` ([UNPrompt Drive](https://drive.google.com/drive/folders/1qcDBcVdcfAr_q5VOXBYagtnhA_r3Mm3Z)).

```bash
python evaluate_team_contributions.py --seeds 0 1 2 3 4 --targets amazon amazon_upu reddit            # Study 1
python evaluate_team_contributions.py --multisource --seeds 0 1 2 --targets amazon amazon_upu reddit  # Study 1, multi-source
cd Anomaly2 && python main.py all --seeds 0 1 2 3 4                                                   # Study 2 (~15 min, one 24 GB GPU)
```

## Notes on the upstream code
* The upstream repo has no `train.py`/`main.py`; `evaluate_team_contributions.py` and `Anomaly2/train.py`, `Anomaly2/main.py` are our integration layers.
* Small edits to upstream files: optional `dgl`/`torch_sparse` imports, and a suppressed benign divide-by-zero warning in `zero_shot/utils.py`.
  Added `zero_shot/bench.py`, `summarize.py`, `explain.py` (scoring benchmark and explanation prototype).
