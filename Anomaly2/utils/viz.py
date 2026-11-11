"""MoE visualisation helpers (Contribution 1 deliverable)."""
import numpy as np, torch
from utils.data import dev


@torch.no_grad()
def extract_moe_embeddings(model, G, seed):
    """Residual embeddings, gate routing and expert prototypes of a trained MoE model on graph G (CPU numpy)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed + 1)
    nraw = torch.randn(model.hid, generator=gen).to(dev)
    araw = torch.randn(model.K, model.hid, generator=gen).to(dev)
    _, r, _ = model.encode(G)
    pn, pa = model.prototypes(nraw, araw)
    s, _, ex = model.scores(r, pn, pa)
    g = ex['gate']
    return dict(res=r.cpu().numpy(), gate=g.cpu().numpy(), assign=g.argmax(1).cpu().numpy(), experts=pa.cpu().numpy(),
                normal_proto=pn.cpu().numpy(), score=s.cpu().numpy(), y=G['y'], deg=G['deg'].cpu().numpy())


def plot_moe_tsne(emb, path, n_normal=1500, seed=0, title=''):
    """t-SNE of anomaly residuals (+ a normal sample + the K expert prototypes) coloured by routed expert.
    Also reports: expert usage, silhouette of the expert assignment among anomalies, and association with a PROXY anomaly type
    (low- vs high-degree anomaly; datasets carry no real anomaly-type labels)."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import chi2_contingency
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    rng = np.random.RandomState(seed)
    y, K = emb['y'], emb['experts'].shape[0]
    an = np.where(y == 1)[0]
    nm = rng.choice(np.where(y == 0)[0], min(n_normal, int((y == 0).sum())), replace=False)
    idx = np.concatenate([an, nm])
    X = np.vstack([emb['res'][idx], emb['experts']])
    Z = TSNE(n_components=2, perplexity=30, init='pca', random_state=seed).fit_transform(X)
    Zi, Zp = Z[:len(idx)], Z[len(idx):]
    a_mask = np.arange(len(idx)) < len(an)
    asg = emb['assign'][idx]
    cols = plt.cm.tab10(np.arange(K))
    proxy = (emb['deg'][an] > np.median(emb['deg'][an])).astype(int)          # PROXY type, not ground truth
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(*Zi[~a_mask].T, c='lightgrey', s=6, alpha=.5, label='normal (sample)')
    for k in range(K):
        mk = a_mask & (asg == k)
        ax[0].scatter(*Zi[mk].T, c=[cols[k]], s=26, marker='x', label=f'anomaly -> expert {k} (n={int(mk.sum())})')
        ax[0].scatter(*Zp[k], c=[cols[k]], s=260, marker='*', edgecolors='k')
    ax[0].legend(fontsize=7); ax[0].set_title('residual space, coloured by routed expert (* = prototype)', fontsize=9)
    for t, mk_ in ((0, 'o'), (1, '^')):
        for k in range(K):
            sel = (asg[a_mask] == k) & (proxy == t)
            ax[1].scatter(*Zi[a_mask][sel].T, c=[cols[k]], marker=mk_, s=28, edgecolors='k', linewidths=.3)
    ax[1].set_title('anomalies: colour = expert, marker = PROXY type (o low / ^ high degree)', fontsize=9)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    a_asg = emb['assign'][an]
    stats = dict(usage=np.bincount(a_asg, minlength=K).tolist(),
                 silhouette=float(silhouette_score(emb['res'][an], a_asg)) if len(set(a_asg)) > 1 else float('nan'))
    table = np.array([[np.sum((a_asg == k) & (proxy == t)) for t in (0, 1)] for k in range(K)])
    nz = table[table.sum(1) > 0]
    stats['proxy_type_table'] = table.tolist()
    stats['proxy_chi2_p'] = float(chi2_contingency(nz)[1]) if nz.shape[0] > 1 and (nz.sum(0) > 0).all() else float('nan')
    return stats
