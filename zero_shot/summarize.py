"""Aggregate ../results/bench.csv into the tables for Exp I (scoring) and Exp H (feature alignment)."""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon

d = pd.read_csv('../results/bench.csv')
d['cfg'] = d.cfg_dim.astype(str) + 'd/' + d.cfg_norm
out = []
fmt = lambda df: df.to_markdown(floatfmt='.3f')

# ---------- Exp I: scoring functions at the paper's default features (8d/rownorm) ----------
I = d[(d.cfg_dim == 8) & (d.cfg_norm == 'rownorm')]
per_seed = I.groupby(['scorer', 'seed', 'target']).auroc.mean().reset_index()
mean_t = I.groupby(['scorer', 'target']).auroc.mean().unstack()
std_t = I.groupby(['scorer', 'target']).auroc.std().unstack()
tbl = mean_t.round(3).astype(str) + '±' + std_t.round(3).astype(str)
tbl['MEAN'] = mean_t.mean(1).round(3)
tbl['AUPRC_mean'] = I.groupby('scorer').auprc.mean().round(3)
tbl['P@5%_mean'] = I.groupby('scorer').p_at_5pct.mean().round(3)
tbl = tbl.sort_values('MEAN', ascending=False)
out += ['## Exp I - scoring function (AUROC mean±std over 5 seeds; features 8d/rownorm)', fmt(tbl), '']

base = I[I.scorer == 'cos_abn'].groupby(['seed', 'target']).auroc.mean()
rows = []
for sc in I.scorer.unique():
    if sc == 'cos_abn': continue
    a = I[I.scorer == sc].groupby(['seed', 'target']).auroc.mean()
    dm = (a - base).groupby('target').mean()
    try: pv = wilcoxon(dm.values).pvalue
    except Exception: pv = np.nan
    rows.append(dict(scorer=sc, mean_delta_vs_cos_abn=dm.mean(), targets_improved=int((dm > 0).sum()), of=len(dm), wilcoxon_p=pv))
out += ['### Paired comparison vs the paper\'s abnormal score (cos_abn), over targets',
        fmt(pd.DataFrame(rows).set_index('scorer').sort_values('mean_delta_vs_cos_abn', ascending=False)), '']

# ---------- Exp H: feature alignment, scorer fixed to cos_abn ----------
H = d[d.scorer == 'cos_abn']
g = H.groupby(['cfg_dim', 'cfg_norm']).auroc.mean().unstack()
out += ['## Exp H - feature dimension x normalisation (mean AUROC over targets & seeds, scorer=cos_abn)', fmt(g), '']
pt = H.groupby(['cfg', 'target']).auroc.mean().unstack()
pt['MEAN'] = pt.mean(1)
out += ['### Per target', fmt(pt.sort_values('MEAN', ascending=False)), '']
open('../results/summary.md', 'w').write('\n'.join(out))
print('\n'.join(out))
