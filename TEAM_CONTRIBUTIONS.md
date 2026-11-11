# Team extensions to AnomalyGFM

The repo has no `train.py`; training lives in `zero_shot/run_abnormal.py` (pre-train + zero-shot) and
`few_shot/run_finetune_normal.py` (few-shot). `evaluate_team_contributions.py` is a self-contained pipeline that
reuses the repo's `Model`, `GCN`, `SimplePrompt` and `normalize_adj`; the hooks below show how to wire each
module into the original scripts.

| # | File | Hook |
|---|------|------|
| C1 | `models/h2gcn_backbone.py` | in `run_abnormal.py`, right after `model = Model(...)`: `from models.h2gcn_backbone import use_h2gcn_backbone; use_h2gcn_backbone(model, 8, args.embedding_dim)` |
| C2 | `models/hypernet_alignment.py` | replace `preprocess_features(feat_*)` by `prep = make_prep(X_raw, A_bin)` once per graph and `X8 = align(X_raw, prep)` inside the forward pass; the raw `.mat` files are needed (see `data/raw`) and `align` joins the optimiser's parameters |
| C3 | `utils/ssl_pretrain.py` | in the epoch loop replace/add to `loss = loss_bce + loss_alignment` with `loss, _ = ssl_loss(encode, A_dense, X, SSLConfig(), gen, normal_prompt, abnormal_prompt)` - no labels required |
| C4 | `models/graph_lora.py` | in `few_shot/run_finetune_normal.py` replace the `requires_grad` block with `inject_graph_lora(model); params = freeze_base_train_lora(model)`; `set_lora_enabled(model, False)` restores the zero-shot model |

Run everything: `python evaluate_team_contributions.py --seeds 0 1 2 3 4 --targets amazon amazon_upu reddit`
(data layout: `data/datasets for AnomalyGFM/*_svd.mat` and `data/raw/{Facebook,Amazon,Amazon-all,Reddit}.mat`).

Notes
* Released `few_shot/model.py` has `seq1 = self.prompt(seq1)` commented out, so its tuned prompt parameters never enter
  the forward pass; the pipeline therefore applies the prompt to the input features as the paper describes, initialised
  to the identity so it is a fair baseline.
* Peak GPU memory is `torch.cuda.max_memory_allocated()` around each stage (pre-train+eval for zero-shot rows,
  adaptation+eval for adaptation rows).
