# C2 - Unsupervised test-time adaptation (TTA)

Code: `utils/unsupervised_tta.py` (`test_time_adapt`, `pseudo_probability`, `plot_tta_curves`).

## Method
Freeze the pre-trained model; add an identity-initialised graph prompt on the input features (`x + a(x) + g`, 80 parameters). On the **unlabeled** target graph, every step:
1. anomaly score `s_v = cos(r_v, p_abn)` -> robust z-score `z` (median/MAD) -> `p_v = sigmoid((z_v - q)/T)` with `q` the `(1 - pi)` quantile of `z`, `pi = 5%` assumed contamination (an assumption, not a label), `T = 0.25`;
2. pseudo-labels: `p > 0.95` -> anomaly, `p < 0.05` -> normal;
3. loss = BCE on confident nodes + 0.1 x entropy of `p` on all nodes; only the prompt is updated (Adam).
Ground-truth labels are used **only for logging** AUROC/AUPRC and pseudo-label purity per step. Step 0 = zero-shot. A control shuffles the pseudo-labels among the confident nodes.

## Results (5 seeds; baseline model underneath)
Master table (30 steps, lr 1e-3): average AUROC 0.644 vs 0.648 baseline; shuffled-label control 0.652.
Learning curves (50 steps; `results/tta_curves_*.png`, `results/tta.md`):

| Target | step-0 AUROC | final delta, lr 1e-3 | final delta, lr 1e-2 | shuffled-label control (lr 1e-2) | pseudo-positive precision (lr 1e-3) | anomaly base rate |
|---|---|---|---|---|---|---|
| Amazon | 0.834 | -0.028 | -0.113 | -0.338 | 0.748 | 0.069 |
| Amazon-UPU | 0.792 | +0.005 | -0.016 | -0.230 | 0.648 | 0.068 |
| Reddit | 0.573 | -0.030 | -0.085 | -0.055 | 0.007 | 0.033 |
| Elliptic | 0.394 | +0.042 | +0.140 | +0.127 | 0.002 | 0.098 |

## Analysis
* **No label-free improvement.** Accuracy falls on Amazon and Reddit; UPU is flat.
* **Pseudo-label quality is the bottleneck, and it is only good where it is not needed:** on Amazon/UPU, where the zero-shot ranking is good, 75%/65% of the pseudo-positives are true anomalies, yet adapting still hurts
  (sharpening an already-ranked score distribution adds noise); on Reddit/Elliptic, where the ranking is poor/inverted, pseudo-positives are *worse than chance* (0.7% / 0.2%).
* **Elliptic's +0.14 is not learning:** random pseudo-labels give +0.127 (5/5 seeds both) - the prompt perturbation shifts an inverted ranking. This is why the control row exists.
* The curves are monotone-ish in the wrong direction for the good graphs, which is the main message of the figure.

## Limitations / next
Contamination `pi`, temperature, thresholds, entropy weight were fixed a priori (no tuning on targets); one pseudo-label source (a single score) - consensus across augmentations/prompt draws and a stopping rule are the natural next step. Memory: about +230 MB peak during adaptation.

Reproduce: `python main.py tta`.
