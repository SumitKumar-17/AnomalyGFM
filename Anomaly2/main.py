#!/usr/bin/env python
"""Anomaly2 entry point (the repo has no main.py).   python main.py {eval|ablation|attack|moe|tta|all} [--seeds 0 1 2 3 4]

  eval      master comparison table: baseline + C1..C4  (evaluate_team_contributions.py)
  ablation  C4: 1-hop vs 2-hop vs PPR vs adaptive-PPR residuals (scripts/ablation_residuals.py)
  attack    C3: AUROC under rewiring / camouflage attacks (scripts/benchmark_attacks.py)
  moe       C1: K sweep + t-SNE of expert routing (results/moe_tsne_*.png)
  tta       C2: label-free test-time-adaptation learning curves (results/tta_curves_*.png)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from train import TrainCfg, build_model, pretrain, evaluate, seed_all
from utils.data import load_graph
from utils.experiment import RESULTS, save, table
from utils.unsupervised_tta import test_time_adapt, plot_tta_curves
from utils.viz import extract_moe_embeddings, plot_moe_tsne


def exp_moe(targets, seeds):
    Gtr = load_graph('train'); graphs = {t: load_graph(t) for t in targets}
    rows, stats = [], {}
    for seed in seeds:
        for K in (0, 2, 4, 8):                                    # K=0: the paper's single prototype
            seed_all(seed); m = build_model(moe_k=K); pretrain(m, Gtr, seed, TrainCfg())
            for t, G in graphs.items():
                r = evaluate(m, G, seed)
                rows.append(dict(target=t, seed=seed, K=K if K else 1, **r))
                if K == 4 and seed == seeds[0]:
                    emb = extract_moe_embeddings(m, G, seed)
                    stats[t] = plot_moe_tsne(emb, f'{RESULTS}/moe_tsne_{t}.png', seed=seed, title=f'MoE (K=4) routing on {t} (train: Facebook, seed {seed})')
    df = pd.DataFrame(rows); save(df, 'moe_ksweep_raw')
    md = ['## C1 - MoE: K sweep (K=1 is the paper\'s single prototype; AUROC/AUPRC mean±std)']
    for t in targets:
        md += [f'\n### {t}', table(df[df.target == t], 'K', cols=('auroc', 'auprc')).to_markdown()]
    md += ['\n## Expert usage / cluster statistics (seed %d, K=4; t-SNE PNGs in results/)' % seeds[0], '```', json.dumps(stats, indent=1), '```']
    open(f'{RESULTS}/moe.md', 'w').write('\n'.join(md)); print('\n'.join(md))


def exp_tta(targets, seeds, steps=50):
    Gtr = load_graph('train'); graphs = {t: load_graph(t) for t in targets}
    cfgs = {'lr 1e-3': dict(lr=1e-3), 'lr 1e-2': dict(lr=1e-2), 'ctrl: shuffled pseudo-labels (lr 1e-2)': dict(lr=1e-2, shuffle_pseudo=True)}
    allh = []
    for seed in seeds:
        seed_all(seed); m = build_model(); pretrain(m, Gtr, seed, TrainCfg())
        for t, G in graphs.items():
            for name, kw in cfgs.items():
                _, h = test_time_adapt(m, G, seed, steps=steps, **kw)
                allh.append(h.assign(target=t, seed=seed, cfg=name))
    H = pd.concat(allh); save(H, 'tta_curves_raw')
    md = ['## C2 - label-free test-time adaptation (prompt only; %d steps; mean±std over %d seeds)' % (steps, len(seeds))]
    for t in targets:
        d = H[H.target == t]
        curves = {c: d[d.cfg == c].groupby('step')[['auroc', 'auprc']].mean().reset_index() for c in cfgs}
        plot_tta_curves(curves, f'{RESULTS}/tta_curves_{t}.png', f'Unsupervised TTA on {t} (no labels used for adaptation)')
        s0, sT = d[d.step == 0], d[d.step == steps]
        rows = []
        for c in cfgs:
            a, b = s0[s0.cfg == c].set_index('seed'), sT[sT.cfg == c].set_index('seed')
            rows.append(dict(config=c, AUROC_step0=f'{a.auroc.mean():.3f}', AUROC_final=f'{b.auroc.mean():.3f}±{b.auroc.std():.3f}',
                             delta=f'{(b.auroc - a.auroc).mean():+.4f}', AUPRC_delta=f'{(b.auprc - a.auprc).mean():+.4f}',
                             seeds_improved=f'{int(((b.auroc - a.auroc) > 0).sum())}/{len(b)}',
                             pseudo_pos_n=f'{b.n_pos.mean():.0f}', pseudo_pos_precision=f'{b.pos_precision.mean():.3f}',
                             pseudo_neg_purity=f'{b.neg_purity.mean():.3f}'))
        md += [f'\n### {t}', pd.DataFrame(rows).to_markdown(index=False)]
    open(f'{RESULTS}/tta.md', 'w').write('\n'.join(md)); print('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['eval', 'ablation', 'attack', 'moe', 'tta', 'all'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--targets', nargs='+', default=None)
    a = p.parse_args()
    T = lambda d: a.targets or d
    if a.cmd in ('eval', 'all'):
        import evaluate_team_contributions as E
        df = E.run(T(['amazon', 'amazon_upu', 'reddit', 'elliptic']), a.seeds); save(df, 'team_results_raw')
        open(f'{RESULTS}/team_results.md', 'w').write(E.report(df, a.seeds))
    if a.cmd in ('ablation', 'all'):
        from scripts import ablation_residuals as A
        tg = T(['amazon', 'amazon_upu', 'reddit', 'elliptic', 'tolokers']); df = A.run(tg, a.seeds); save(df, 'ablation_residuals_raw'); A.report(df, tg, a.seeds)
    if a.cmd in ('attack', 'all'):
        from scripts import benchmark_attacks as B
        tg = T(['amazon', 'amazon_upu', 'reddit']); df = B.run(tg, a.seeds); save(df, 'attack_benchmark_raw'); B.report(df, tg, a.seeds)
    if a.cmd in ('moe', 'all'):
        exp_moe(T(['amazon', 'amazon_upu', 'reddit', 'elliptic']), a.seeds)
    if a.cmd in ('tta', 'all'):
        exp_tta(T(['amazon', 'amazon_upu', 'reddit', 'elliptic']), a.seeds)
