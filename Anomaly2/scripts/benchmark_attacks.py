#!/usr/bin/env python
"""Contribution 3 benchmark: AUROC under structural attacks on the real target graphs.
Models: baseline | baseline trained with the SAME attack augmentation but no purifier (control) | edge-purified.
Attacks: random rewiring of a fraction of edges; camouflage (each anomaly wired to m random normal nodes)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from train import TrainCfg, build_model, pretrain, evaluate, seed_all
from utils.data import load_graph
from utils.experiment import save, ms, RESULTS
from models.edge_purification import run_attack_benchmark

ATTACKS = [('rewire', 0.0), ('rewire', 0.1), ('rewire', 0.3), ('rewire', 0.5),
           ('camouflage', 5), ('camouflage', 20), ('camouflage', 50)]


def run(targets, seeds, epochs=301):
    Gtr = load_graph('train')
    models = {'Baseline': {}, 'Baseline + attack-aug (ctrl)': {}, 'Edge-purified (C3)': {}}
    for seed in seeds:
        for name, kw, atk in (('Baseline', {}, False), ('Baseline + attack-aug (ctrl)', {}, True), ('Edge-purified (C3)', dict(purify=True), True)):
            seed_all(seed); m = build_model(**kw); pretrain(m, Gtr, seed, TrainCfg(epochs=epochs, attack_aug=atk)); models[name][seed] = m
    graphs = {t: load_graph(t) for t in targets}
    return run_attack_benchmark(models, graphs, ATTACKS, seeds, evaluate)


def report(df, targets, seeds):
    md = []
    for t in targets:
        d = df[df.target == t]
        d = d.assign(attack_=d.attack + ' ' + d.level.map('{:g}'.format))
        tb = d.pivot_table(index='model', columns='attack_', values='auroc', aggfunc=lambda s: ms(s.values),
                           sort=False)[[f'{k} {l:g}' for k, l in ATTACKS]]
        s = f'\n### AUROC under attack - {t} ({len(seeds)} seeds; rewire = fraction of edges replaced, camouflage = edges per anomaly)\n' + tb.to_markdown()
        print(s); md.append(s)
    open(f'{RESULTS}/attack_benchmark.md', 'w').write('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--targets', nargs='+', default=['amazon', 'amazon_upu', 'reddit'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--from_csv', action='store_true')
    a = p.parse_args()
    df = pd.read_csv(f'{RESULTS}/attack_benchmark_raw.csv') if a.from_csv else run(a.targets, a.seeds)
    if not a.from_csv: save(df, 'attack_benchmark_raw')
    report(df, list(dict.fromkeys(df.target)), sorted(df.seed.unique()))
