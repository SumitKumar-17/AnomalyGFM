#!/usr/bin/env python
"""Contribution 4 ablation: 1-hop (paper) vs 2-hop vs fixed PPR vs adaptive PPR residuals.  Each residual type is trained from
scratch (it changes the training signal too) and evaluated zero-shot; Elliptic is included because its anomalies are laundering
structures that a 1-hop reference can miss."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, torch
from train import TrainCfg, build_model, pretrain, evaluate, seed_all
from utils.data import load_graph
from utils.experiment import table, save, peak_mb, RESULTS

KINDS = [('1-hop (paper)', 'hop1'), ('2-hop', 'hop2'), ('PPR (alpha=0.3, K=4)', 'ppr'), ('Adaptive PPR (learned per-node hops)', 'adaptive')]


def run(targets, seeds, epochs=301):
    Gtr = load_graph('train')
    graphs = {t: load_graph(t) for t in targets}
    rows = []
    for seed in seeds:
        for label, kind in KINDS:
            seed_all(seed)
            m = build_model(residual=kind)
            pretrain(m, Gtr, seed, TrainCfg(epochs=epochs))
            for t, G in graphs.items():
                torch.cuda.reset_peak_memory_stats()
                rows.append(dict(target=t, seed=seed, residual=label, params=m.n_params(), peak_mb=peak_mb(), **evaluate(m, G, seed)))
    return pd.DataFrame(rows)


def report(df, targets, seeds):
    md = []
    for t in list(targets) + ['AVERAGE']:
        d = df if t == 'AVERAGE' else df[df.target == t]
        if t == 'AVERAGE':
            d = d.groupby(['residual', 'seed'], sort=False)[['auroc', 'auprc']].mean().reset_index()
        s = f'\n### {t} ({len(seeds)} seeds)\n' + table(d, 'residual').to_markdown()
        print(s); md.append(s)
    open(f'{RESULTS}/ablation_residuals.md', 'w').write('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--targets', nargs='+', default=['amazon', 'amazon_upu', 'reddit', 'elliptic', 'tolokers'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--from_csv', action='store_true', help='only (re)build the markdown from results/ablation_residuals_raw.csv')
    a = p.parse_args()
    df = pd.read_csv(f'{RESULTS}/ablation_residuals_raw.csv') if a.from_csv else run(a.targets, a.seeds)
    if not a.from_csv: save(df, 'ablation_residuals_raw')
    report(df, list(dict.fromkeys(df.target)), sorted(df.seed.unique()))
