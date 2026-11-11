"""Shared helpers for the experiment scripts."""
import os, sys
import numpy as np, pandas as pd, torch

ANOMALY2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ANOMALY2, 'results')


def peak_mb():
    return torch.cuda.max_memory_allocated() / 2 ** 20


def ms(x):
    return f'{np.mean(x):.3f}±{np.std(x, ddof=1) if len(x) > 1 else 0:.3f}'


def table(df, index, cols=('auroc', 'auprc'), extra=None):
    """Mean±std per `index` group (order of first appearance)."""
    g = df.groupby(index, sort=False)
    t = pd.DataFrame({c.upper(): g[c].apply(lambda s: ms(s.values)) for c in cols})
    for name, col, fmt in (extra or []):
        t[name] = g[col].mean().map(fmt)
    return t


def save(df, name):
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(f'{RESULTS}/{name}.csv', index=False)
