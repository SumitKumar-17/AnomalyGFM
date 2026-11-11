#!/usr/bin/env python
"""Choose the learning rate of the NEW modules using ONLY the source graph (Facebook) - never the target graphs.
Criteria: (a) AUROC on the 70% of Facebook nodes that were not used for training (all variants);
          (b) for the purifier: edge AUROC / drop rates on a FRESH attack of Facebook (can it tell injected edges from real ones?)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from train import TrainCfg, build_model, pretrain, evaluate, seed_all, training_attack
from utils.data import load_graph, dev
from utils.experiment import RESULTS

Gtr = load_graph('train')
VAR = [('MoE K=4', dict(moe_k=4), False), ('Purifier', dict(purify=True), True), ('Adaptive PPR', dict(residual='adaptive'), False),
       ('Baseline (reference)', dict(), False)]
only = sys.argv[1:]
rows = []
for name, kw, atk in [v for v in VAR if not only or v[0] in only]:
    for lr_ext in (1e-4, 1e-3, 1e-2, 3e-2):
        for seed in range(3):
            seed_all(seed); m = build_model(**kw)
            pretrain(m, Gtr, seed, TrainCfg(attack_aug=atk, lr_ext=lr_ext))
            held = np.random.RandomState(seed).permutation(Gtr['N'])[int(0.3 * Gtr['N']):]
            r = dict(variant=name, lr_ext=lr_ext, seed=seed, fb_heldout_auroc=evaluate(m, Gtr, seed, held)['auroc'])
            if m.purifier is not None:
                V = training_attack(Gtr, TrainCfg(), torch.Generator(device=dev).manual_seed(999 + seed))
                with torch.no_grad():
                    lg = m.purifier.mlp(m.purifier.edge_features(V['X8'], V['deg'], V['eu'])).squeeze(-1).cpu().numpy()
                inj = V['injected'].cpu().numpy() > 0
                r.update(edge_auroc=roc_auc_score(inj, -lg), drop_injected=(lg[inj] <= 0).mean(), drop_original=(lg[~inj] <= 0).mean())
            rows.append(r)
df = pd.DataFrame(rows)
out = df.groupby(['variant', 'lr_ext'], sort=False).mean(numeric_only=True).drop(columns='seed').round(3)
s = '### Source-only selection of lr_ext (Facebook held-out nodes; 3 seeds)\n' + out.to_markdown()
print(s); open(f'{RESULTS}/select_lr_ext' + ('_' + '_'.join(only).replace(' ', '') if only else '') + '.md', 'w').write(s)
