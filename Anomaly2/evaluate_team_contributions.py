#!/usr/bin/env python
"""Master evaluation: baseline AnomalyGFM vs the 4 extensions on REAL graphs (train: Facebook; zero-shot targets: Amazon, ...).

Prints, per target and averaged over targets: AUROC, AUPRC (mean±std over seeds), total parameters, parameters updated at test
time, and peak GPU memory (torch.cuda.max_memory_allocated).  Rows labelled 'ctrl:' are controls that isolate what each
extension really contributes (attack augmentation without the purifier; TTA with shuffled pseudo-labels).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from train import TrainCfg, build_model, pretrain, evaluate, seed_all
from utils.data import load_graph
from utils.experiment import table, save, peak_mb, RESULTS
from utils.unsupervised_tta import test_time_adapt

VARIANTS = [                                                   # (label, GFM kwargs, adversarial training-graph augmentation)
    ('Baseline AnomalyGFM', dict(), False),
    ('+ C1 MoE prototypes (K=4)', dict(moe_k=4), False),
    ('+ C3 Edge purification', dict(purify=True), True),
    ('ctrl: baseline + attack-aug (no purifier)', dict(), True),
    ('+ C4 Adaptive PPR residual', dict(residual='adaptive'), False),
    ('C1+C3+C4 combined', dict(moe_k=4, purify=True, residual='adaptive'), True),
]
ORDER = ['Baseline AnomalyGFM', '+ C1 MoE prototypes (K=4)', '+ C2 Unsup. TTA (on baseline)', 'ctrl: TTA, shuffled pseudo-labels',
         '+ C3 Edge purification', 'ctrl: baseline + attack-aug (no purifier)', '+ C4 Adaptive PPR residual',
         'C1+C3+C4 combined', 'Combined + C2 TTA']


def run(targets, seeds, epochs=301, tta_steps=30, tta_lr=1e-3):
    Gtr = load_graph('train')
    graphs = {t: load_graph(t) for t in targets}
    rows = []
    for seed in seeds:
        models = {}
        for label, kw, atk in VARIANTS:
            seed_all(seed); torch.cuda.reset_peak_memory_stats()
            m = build_model(**kw)
            pretrain(m, Gtr, seed, TrainCfg(epochs=epochs, attack_aug=atk))
            train_peak, models[label] = peak_mb(), m
            for t, G in graphs.items():
                torch.cuda.reset_peak_memory_stats()
                r = evaluate(m, G, seed)
                rows.append(dict(target=t, seed=seed, method=label, params=m.n_params(), tta_params=0,
                                 peak_mb=max(train_peak, peak_mb()), **r))
        for src, label, shuf in (('Baseline AnomalyGFM', '+ C2 Unsup. TTA (on baseline)', False),
                                 ('Baseline AnomalyGFM', 'ctrl: TTA, shuffled pseudo-labels', True),
                                 ('C1+C3+C4 combined', 'Combined + C2 TTA', False)):
            for t, G in graphs.items():
                torch.cuda.reset_peak_memory_stats()
                _, h = test_time_adapt(models[src], G, seed, steps=tta_steps, lr=tta_lr, shuffle_pseudo=shuf)
                rows.append(dict(target=t, seed=seed, method=label, params=models[src].n_params() + 80, tta_params=80,
                                 peak_mb=peak_mb(), auroc=h.auroc.iloc[-1], auprc=h.auprc.iloc[-1]))
        print(f'seed {seed} done', flush=True)
    return pd.DataFrame(rows)


def report(df, seeds):
    extra = [('Params', 'params', '{:,.0f}'.format), ('Updated @test', 'tta_params', '{:.0f}'.format), ('Peak GPU MB', 'peak_mb', '{:.0f}'.format)]
    md = []
    for t in list(dict.fromkeys(df.target)) + ['AVERAGE over targets']:
        d = df if t.startswith('AVERAGE') else df[df.target == t]
        if t.startswith('AVERAGE'):          # average per seed first, so ± reflects seed variation of the target-mean
            d = d.groupby(['method', 'seed'], sort=False).agg(auroc=('auroc', 'mean'), auprc=('auprc', 'mean'), params=('params', 'mean'),
                                                              tta_params=('tta_params', 'mean'), peak_mb=('peak_mb', 'mean')).reset_index()
        tb = table(d, 'method', extra=extra).reindex([o for o in ORDER if o in set(d.method)])
        s = f'\n### {t}  (train: Facebook; zero-shot; mean±std over {len(seeds)} seeds; all target nodes evaluated)\n' + tb.to_markdown()
        print(s); md.append(s)
    return '\n'.join(md)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--targets', nargs='+', default=['amazon', 'amazon_upu', 'reddit', 'elliptic'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--epochs', type=int, default=301)
    p.add_argument('--tta_steps', type=int, default=30)
    p.add_argument('--tta_lr', type=float, default=1e-3)
    a = p.parse_args()
    df = run(a.targets, a.seeds, a.epochs, a.tta_steps, a.tta_lr)
    save(df, 'team_results_raw')
    open(f'{RESULTS}/team_results.md', 'w').write(report(df, a.seeds))
