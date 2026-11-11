"""Contribution 2 - Unsupervised test-time adaptation (entropy minimisation + confident pseudo-labels), prompt-only.

Replaces the standard evaluate() on an UNLABELED target graph:
  1. Freeze the whole pre-trained model; add an identity-initialised graph prompt on the input features (x + a(x) + g, 80 params).
  2. Turn the anomaly score s_v into a probability  p_v = sigmoid((z_v - q)/T), z = robust z-score of s, q = the (1-pi) quantile of z
     (pi = ASSUMED contamination, default 5% - an assumption, not a label).
  3. Pseudo-labels: p_v > 0.95 -> anomaly, p_v < 0.05 -> normal.  Loss = BCE(p, pseudo) on confident nodes + lambda * entropy(p) on all.
  4. Update ONLY the prompt parameters; repeat for `steps` steps.
Ground-truth labels are used only to LOG AUROC/AUPRC/pseudo-label purity per step (never for any gradient).
"""
import copy
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.data import zs_model, dev


def _prompts(model, seed):
    gen = torch.Generator().manual_seed(seed + 1)
    return (torch.randn(model.hid, generator=gen).to(dev), torch.randn(model.K, model.hid, generator=gen).to(dev))


def _score(model, G, nraw, araw):
    _, r, _ = model.encode(G)
    pn, pa = model.prototypes(nraw, araw)
    return model.scores(r, pn, pa)[0]


def pseudo_probability(s, contamination=0.05, T=0.25):
    sd = s.detach()
    med = sd.median()
    mad = (sd - med).abs().median() * 1.4826 + 1e-6
    z = (s - med) / mad
    q = torch.quantile(z.detach(), 1 - contamination)
    return torch.sigmoid((z - q) / T)


def test_time_adapt(model, G, seed, steps=30, lr=1e-3, hi=0.95, lo=0.05, contamination=0.05, T=0.25,
                    entropy_w=0.1, shuffle_pseudo=False):
    """Returns (adapted_model, history DataFrame). history row 0 = zero-shot (before any update)."""
    m = copy.deepcopy(model).eval()
    for p in m.parameters():
        p.requires_grad = False
    m.input_prompt = zs_model.SimplePrompt(8).to(dev)
    for p in m.input_prompt.parameters():
        torch.nn.init.zeros_(p); p.requires_grad = True              # identity at step 0
    opt = torch.optim.Adam(m.input_prompt.parameters(), lr=lr)
    nraw, araw = _prompts(m, seed)
    y = G['y']
    hist = []
    for step in range(steps + 1):
        s = _score(m, G, nraw, araw)
        sc = s.detach().cpu().numpy()
        p = pseudo_probability(s, contamination, T)
        pos, neg = p.detach() > hi, p.detach() < lo
        pl = pos.cpu().numpy()
        hist.append(dict(step=step, auroc=roc_auc_score(y, sc), auprc=average_precision_score(y, sc),
                         n_pos=int(pos.sum()), n_neg=int(neg.sum()),
                         pos_precision=float(y[pl].mean()) if pl.any() else float('nan'),
                         neg_purity=float(1 - y[neg.cpu().numpy()].mean()) if neg.any() else float('nan')))
        if step == steps:
            break
        conf = pos | neg
        if conf.any():
            tgt = pos[conf].float()
            if shuffle_pseudo:                                         # control: same label counts, random assignment
                tgt = tgt[torch.randperm(len(tgt), device=dev)]
            pc = p[conf].clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(pc, tgt)
        else:
            loss = 0.0 * p.sum()
        pe = p.clamp(1e-6, 1 - 1e-6)
        loss = loss + entropy_w * (-(pe * pe.log() + (1 - pe) * (1 - pe).log())).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m, pd.DataFrame(hist)


def plot_tta_curves(curves, path, title=''):
    """curves: {label: DataFrame(step, auroc, auprc)} -> learning-curve PNG (AUROC left, AUPRC right)."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for label, df in curves.items():
        for a, col in zip(ax, ('auroc', 'auprc')):
            a.plot(df.step, df[col], marker='o', ms=3, label=label)
    for a, col in zip(ax, ('AUROC', 'AUPRC')):
        a.set_xlabel('adaptation step (no labels)'); a.set_ylabel(col); a.grid(alpha=.3)
    ax[0].legend(fontsize=8); fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
