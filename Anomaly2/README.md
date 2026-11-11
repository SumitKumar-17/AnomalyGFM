# Anomaly2 - second extension study of AnomalyGFM (KDD'25)

Four new extensions of the paper's zero-shot model, each with its own experiment, all on **real graphs** (train: Facebook; zero-shot targets:
Amazon, Amazon-UPU, Reddit, Elliptic; 5 seeds). Everything lives in this folder; the first study is at the repo root (`../WORK.md`).
Full story, numbers and caveats: **[WORK.md](WORK.md)**. Per-contribution reports: `reports/`. Slides: `slides/main.pdf` (source `slides/main.tex`, 18 pages).

| # | Contribution | Code | Experiment |
|---|---|---|---|
| C1 | Mixture-of-Experts abnormal prototypes (K experts + gate + load balancing) | `models/moe_prototypes.py` | `python main.py moe` -> K sweep + t-SNE (`results/moe_tsne_*.png`) |
| C2 | Unsupervised test-time adaptation (entropy + confident pseudo-labels, prompt-only) | `utils/unsupervised_tta.py` | `python main.py tta` -> learning curves (`results/tta_curves_*.png`) |
| C3 | Adversarial edge purification (Gumbel-sigmoid edge dropping) | `models/edge_purification.py` | `python main.py attack` -> AUROC under attack table |
| C4 | Adaptive multi-hop / PPR residuals | `models/adaptive_residuals.py` | `python main.py ablation` -> 1-hop vs 2-hop vs PPR vs adaptive |
| all | master comparison (AUROC, AUPRC, params, peak GPU memory) | `evaluate_team_contributions.py` | `python main.py eval` |

## Headline (honest) result
No extension improves the average AUROC over the paper's model (0.648; extensions 0.640-0.646, all within noise). What *is* robust:
C3 protects against **camouflage** attacks (Amazon-UPU at 50 camouflage edges/anomaly: 0.678 vs 0.085 for the baseline) and lifts Elliptic
0.394 -> 0.499 (control without purifier: 0.372); C4's learned gate matches the 1-hop residual while using per-node hop weights; C1 experts
collapse (one expert takes 84% of anomalies); C2's label-free adaptation does not help (its Elliptic "gain" is matched by random pseudo-labels).

## Layout
```
Anomaly2/
  main.py                              entry point: python main.py {eval|ablation|attack|moe|tta|all} [--seeds ..] [--targets ..]
  train.py                             pretrain() / evaluate() / build_model()   <- where the four extensions plug into training
  evaluate_team_contributions.py       master comparison table
  models/{core,moe_prototypes,edge_purification,adaptive_residuals}.py
  utils/{data,attacks,unsupervised_tta,viz,experiment}.py
  scripts/{ablation_residuals,benchmark_attacks,diagnose_purifier,select_lr_ext,select_attack_cfg}.py
  slides/main.{tex,pdf}                Beamer deck (built with pdflatex on the server)
  results/                             final tables/plots; results/round1_lr1e-4_untuned/ = first (flawed) round, kept on purpose
```

## Where to hook into the official code (step by step)
The official repo has **no `train.py` or `main.py`**: training is `zero_shot/run_abnormal.py` (pre-train + zero-shot) and `few_shot/*.py`.
`Anomaly2/train.py` and `Anomaly2/main.py` are the integration layer; the paper's `Model` (GCN, prompt layers) is reused unchanged via
`zero_shot/model.py`. To port a contribution into the original script instead:

1. **C4 residual** - in `Model.forward` replace the `emb_neighbors = torch.bmm(adj_normalized, emb)` / `emb_residual = emb - emb_neighbors` pair by
   `emb_residual = MultiHopResidual('adaptive')(emb, P, struct)` with `P` the row-normalised neighbour operator (sparse).
2. **C3 purifier** - before `h_1 = self.gcn1(seq1, adj, sparse)` compute `adj, P, aux = EdgePurifier()(X, deg, edge_index, N)` and use those operators for both
   the GCN and the neighbour mean; add `EdgePurifier.edge_loss(aux['logit'], injected)` to the loss and train on attacked graphs (`train.training_attack`).
3. **C1 MoE** - replace `abnormal_prompt = act(fc_abnormal_prompt(raw))` by `moe.experts(fc_abnormal_prompt, act, raw_K)` ([K,H] noise), use
   `moe.loss(r_abn, P)` instead of `dif_abnormal` in `loss_alignment`, and score with `moe.score(r, P)`.
4. **C2 TTA** - replace the `evaluate` block of the script by `test_time_adapt(model, G, seed, steps, lr)`; it adds an identity-initialised input prompt and
   updates only its 80 parameters.
Learning rates: **new modules need their own learning rate** (the paper's 1e-4 left them untrained, see WORK.md); `train.LR_EXT` holds the values selected on the source graph.

## Reproduce
```
# from the repo root (data/ must hold 'datasets for AnomalyGFM/*_svd.mat', see the root README)
cd Anomaly2 && python main.py all --seeds 0 1 2 3 4        # about 15 min on one RTX 4500 Ada (24 GB)
python scripts/select_lr_ext.py && python scripts/select_attack_cfg.py      # source-only hyper-parameter selection
python scripts/diagnose_purifier.py                                          # edge-level purifier diagnostic
```
Note: the GPU is a 24 GB RTX 4500 Ada (not 40 GB); peak memory of every row is reported by `evaluate_team_contributions.py`.
