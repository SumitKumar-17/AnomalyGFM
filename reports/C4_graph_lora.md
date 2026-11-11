# Contribution 4 - Parameter-efficient adaptation with Graph LoRA

Code: `models/graph_lora.py` (`LoRALinear`, `inject_graph_lora`, `set_lora_enabled`, `freeze_base_train_lora`,
`count_trainable`). Evaluated by `evaluate_team_contributions.py` (`adapt(..., method='lora')`).

## 1. Motivation

In the few-shot setting a handful of labelled **normal** nodes (k = 100) from the new graph are used to adapt the
pre-trained model. The paper does this with a small *graph prompt*. We investigate Low-Rank Adaptation (LoRA): freeze
every pre-trained weight and learn only low-rank updates to the GNN's linear layers, so the adapted model is the
pre-trained one plus a tiny delta that can be switched off.

## 2. Method

For each linear layer `W` in the GNN (`gcn1.fc`, `gcn2.fc`, or the three matrices of each H2GCN layer):

```
W' = W + (alpha / r) * B A,        A in R^{r x in}  (Kaiming init),   B in R^{out x r}  (zero init)
```
`B = 0` at init so the adapted network equals the pre-trained network at step 0. `r = 4`, `alpha = 2r`. Only `A, B`
are trained. `set_lora_enabled(model, False)` restores the exact zero-shot model; `merge()` folds the delta into `W`
for zero inference overhead.

**Adaptation objective** (the paper's few-shot normal-alignment loss, used for every adaptation method so the
comparison isolates *what is tuned*): `L = mean_i || p_normal - r_i ||_2` over the k labelled normal nodes.

## 3. What we compare (all on the same frozen pre-trained baseline)

| Method | What is trained | Trainable params |
|---|---|---|
| Prompt tuning (paper's idea) | `SimplePrompt(8)` on input features: `x + a(x) + g` | **80** |
| Full GNN fine-tune (reference) | both GCN layers | **164,002** |
| **Graph-LoRA, r=4 (C4)** | A, B in `gcn1.fc`, `gcn2.fc` | **4,832** (2.9% of full) |
| Combined (H2GCN backbone) + LoRA | A, B in all 6 H2GCN matrices | 14,496 |

Fairness notes: (i) in the **released** `few_shot/model.py` the line `seq1 = self.prompt(seq1)` is commented out, so the
repository's tuned prompt parameters never reach the output (read from the code; not run). We therefore apply the prompt
to the input as the paper *describes*, and initialise it to the identity (the paper's random init would perturb the
model before any tuning). (ii) Two budgets: the **paper protocol** (5 epochs, lr 1e-4) and a **larger-budget
sensitivity check** (30 epochs, lr 1e-3) that we looked at *after* seeing the first results - it is a sensitivity
analysis, not a tuned selection.

## 4. Results (single-source baseline; mean ± std, 5 seeds; k = 100 normal nodes excluded from evaluation)

AUROC (AUPRC in the full tables of `results/team/team_results.md`):

| Method | Params | Amazon | Amazon-UPU | Reddit |
|---|---|---|---|---|
| Zero-shot baseline (no adaptation) | - | 0.834 ± 0.022 | 0.792 ± 0.017 | 0.564 ± 0.026 |
| Prompt - paper protocol | 80 | 0.834 | 0.792 | 0.564 |
| Full fine-tune - paper protocol | 164,002 | 0.839 | 0.795 | 0.563 |
| **LoRA - paper protocol** | 4,832 | 0.834 | 0.793 | 0.563 |
| Prompt - 30 ep, lr 1e-3 | 80 | 0.851 ± 0.049 | 0.797 ± 0.010 | 0.568 ± 0.019 |
| Full fine-tune - 30 ep, lr 1e-3 | 164,002 | 0.637 ± 0.116 | 0.586 ± 0.054 | 0.596 ± 0.025 |
| **LoRA - 30 ep, lr 1e-3** | 4,832 | 0.651 ± 0.070 | 0.693 ± 0.044 | 0.608 ± 0.017 |

Peak GPU memory during adaptation+evaluation: about 1,080 MB (Amazon), 230 MB (UPU / Reddit) for LoRA and full
fine-tuning - dominated by the resident graph and a copy of the model, not by the adapters.

## 5. Analysis

* **Parameter efficiency is real:** LoRA trains 2.9% of the parameters of full fine-tuning (4,832 vs 164,002).
* **Accuracy gains are not.** At the paper's budget every adaptation method is within noise of the zero-shot baseline -
  the updates are simply too small to matter (5 steps, lr 1e-4, and LoRA's `B` starts at zero).
* **At the larger budget the picture splits by graph:** on Amazon both LoRA (0.651) and full fine-tuning (0.637) *lose*
  ~0.18-0.20 AUROC; on Amazon-UPU LoRA loses less than full fine-tuning (0.693 vs 0.586; baseline 0.792); on Reddit
  both gain (LoRA +0.044, full +0.032; baseline std 0.026, i.e. about 1.2-1.7 standard deviations - suggestive, not conclusive). The 80-parameter prompt is the
  safest (Amazon 0.851, UPU 0.797), and the only method that does not degrade anywhere.
* **Why does weight adaptation hurt on Amazon? (hypothesis, not verified):** the objective only ever sees *normal*
  nodes and pulls their residuals towards the normal prototype. Earlier benchmarks (`WORK.md`, Exp I) showed the normal
  and abnormal prototypes are nearly collinear, so pulling normal residuals towards one also shifts the ranking of
  anomalies. A method that can change the encoder (LoRA, full) can exploit this more than an input prompt can.
* **Honest conclusion for C4:** LoRA delivers the *efficiency* claim relative to full fine-tuning and loses clearly
  less than it on Amazon-UPU (and is equal within noise on the other two), but it does **not** beat the paper's (80-parameter) prompt approach in accuracy
  or parameter count, and no adaptation method gives a reliable improvement with this normal-only objective.

## 6. Limitations / next steps

* The objective is the bottleneck, not the adapter. Using the abnormal prototype too, a contrastive/pseudo-anomaly
  objective (reusing C3's corruptions on the *target* graph), or early stopping would be the natural next steps.
* Rank `r` and `alpha` were fixed (4, 8); a rank sweep (r = 1, 2, 8, 16) is not yet run.
* LoRA was applied to the GNN's linear weights only (not to the prototype layers).
* Memory column mostly reflects graph size; the adapters' own footprint is negligible.

## 7. Reproduce

```
python evaluate_team_contributions.py --seeds 0 1 2 3 4 --targets amazon amazon_upu reddit   # rows "C4 Graph-LoRA" and "[sens.]"
```
