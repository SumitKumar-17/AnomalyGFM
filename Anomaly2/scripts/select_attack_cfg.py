#!/usr/bin/env python
"""Choose the adversarial-augmentation strength for the purifier on the SOURCE graph only (3 seeds).
Reports, for the purified model and the no-purifier control: held-out Facebook AUROC on the CLEAN graph and on a FRESH attacked graph,
plus the purifier's edge AUROC / drop rates.  Selection rule (fixed in advance): maximise mean(clean AUROC, attacked AUROC)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from train import TrainCfg, build_model, pretrain, evaluate, seed_all, training_attack
from utils.data import load_graph, dev
from utils.experiment import RESULTS

Gtr = load_graph('train')
CFGS = [(0.20, 10, 1.0), (0.10, 5, 0.5), (0.05, 2, 0.5), (0.10, 10, 0.5), (0.05, 5, 0.3), (0.02, 2, 0.3)]
rows = []
for noise, camo, prob in CFGS:
    for name, kw in (('purified', dict(purify=True)), ('control (no purifier)', {})):
        for seed in range(3):
            cfg = TrainCfg(attack_aug=True, atk_noise=noise, atk_camo=camo, atk_prob=prob, lr_ext=1e-2)
            seed_all(seed); m = build_model(**kw); pretrain(m, Gtr, seed, cfg)
            held = np.random.RandomState(seed).permutation(Gtr['N'])[int(0.3 * Gtr['N']):]
            V = training_attack(Gtr, TrainCfg(), torch.Generator(device=dev).manual_seed(999 + seed))   # fixed evaluation attack
            r = dict(noise=noise, camo=camo, prob=prob, model=name, seed=seed, clean=evaluate(m, Gtr, seed, held)['auroc'],
                     attacked=evaluate(m, V, seed, held)['auroc'])
            if m.purifier is not None:
                with torch.no_grad():
                    lg = m.purifier.mlp(m.purifier.edge_features(V['X8'], V['deg'], V['eu'])).squeeze(-1).cpu().numpy()
                inj = V['injected'].cpu().numpy() > 0
                r.update(edge_auroc=roc_auc_score(inj, -lg), drop_inj=(lg[inj] <= 0).mean(), drop_orig=(lg[~inj] <= 0).mean())
            rows.append(r)
df = pd.DataFrame(rows)
out = df.groupby(['noise', 'camo', 'prob', 'model'], sort=False).mean(numeric_only=True).drop(columns='seed')
out['score'] = (out.clean + out.attacked) / 2
s = '### Source-only selection of attack-augmentation strength (Facebook held-out nodes; 3 seeds)\n' + out.round(3).to_markdown()
print(s); open(f'{RESULTS}/select_attack_cfg.md', 'w').write(s)
