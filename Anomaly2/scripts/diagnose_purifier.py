#!/usr/bin/env python
"""Does the edge purifier detect injected edges at all? Edge-level diagnostic, independent of the anomaly score.
For an attacked REAL target graph we know exactly which edges we injected: report AUROC of the purifier's drop-score for
injected-vs-original edges, the share of injected / original edges it drops, and per-seed AUROC of the (a) purified model and
(b) attack-aug control, to expose seed instability."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from train import TrainCfg, build_model, pretrain, evaluate, seed_all
from utils.data import load_graph, graph_from_edges, dev
from utils.attacks import _canon, _unique
from utils.experiment import RESULTS

Gtr = load_graph('train')
rows = []
for target, kind, level in (('amazon_upu', 'camouflage', 20), ('reddit', 'camouflage', 20), ('amazon_upu', 'rewire', 0.3)):
    G = load_graph(target)
    N, eu, y = G['N'], G['eu'], torch.as_tensor(G['y'], device=dev)
    for seed in range(5):
        g = torch.Generator(device=dev).manual_seed(777 + seed)
        if kind == 'camouflage':
            an, nm = torch.nonzero(y == 1).squeeze(1), torch.nonzero(y == 0).squeeze(1)
            src = an.repeat_interleave(level); dst = nm[torch.randint(0, len(nm), (len(src),), device=dev, generator=g)]
            new = _canon(torch.stack([src, dst]))
        else:
            new = _canon(torch.randint(0, N, (2, int(level * eu.shape[1])), device=dev, generator=g))
        allp = _unique(torch.cat([eu, new], 1), N)
        key_o = set(((eu[0] * N + eu[1]).cpu().numpy()).tolist())
        inj = torch.tensor([k not in key_o for k in (allp[0] * N + allp[1]).cpu().numpy().tolist()], device=dev).float()
        Ga = graph_from_edges(G, allp, with_struct=True)
        seed_all(seed); m = build_model(purify=True); pretrain(m, Gtr, seed, TrainCfg(attack_aug=True)); m.eval()
        with torch.no_grad():
            logit = m.purifier.mlp(m.purifier.edge_features(Ga['X8'], Ga['deg'], allp)).squeeze(-1)
        drop = (logit <= 0).cpu().numpy(); ij = inj.cpu().numpy() > 0
        rows.append(dict(target=target, attack=f'{kind} {level:g}', seed=seed, n_injected=int(ij.sum()), n_edges=len(ij),
                         edge_auroc=roc_auc_score(ij, -logit.cpu().numpy()), drop_injected=drop[ij].mean(), drop_original=drop[~ij].mean(),
                         model_auroc_attacked=evaluate(m, Ga, seed)['auroc']))
df = pd.DataFrame(rows)
g = df.groupby(['target', 'attack'], sort=False)
out = g[['edge_auroc', 'drop_injected', 'drop_original', 'model_auroc_attacked']].agg(['mean', 'std']).round(3)
s = '### Purifier edge-level diagnostic (5 seeds)\n' + out.to_markdown() + '\n\n### per seed\n' + df.round(3).to_markdown(index=False)
print(s); open(f'{RESULTS}/purifier_diagnostic.md', 'w').write(s)
