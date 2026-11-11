#!/usr/bin/env python
"""Unified evaluation of the baseline AnomalyGFM and 4 team extensions on REAL data (no synthetic graphs).

Protocol (as in the paper): pre-train on Facebook, evaluate on an unseen target graph.
  * zero-shot rows : no target labels at all
  * adaptation rows: K labelled NORMAL target nodes (paper's few-shot setting); the K tuned nodes are excluded
                     from evaluation for EVERY row so all rows are scored on the identical node set.
Primary score = the paper's abnormal-prototype measure  exp(cos(residual, abnormal prototype)).

Hook points in the existing repo (there is no train.py; training lives in zero_shot/run_abnormal.py and
few_shot/run_finetune_*.py - see TEAM_CONTRIBUTIONS.md):
  C1  models.h2gcn_backbone.use_h2gcn_backbone(model)         after `model = Model(...)`
  C2  models.hypernet_alignment.FeatureHyperAlign             instead of preprocess_features()/SVD
  C3  utils.ssl_pretrain.ssl_loss(...)                         replaces/adds to the supervised loss in the epoch loop
  C4  models.graph_lora.inject_graph_lora(model) + freeze_base_train_lora(model)   in the few-shot phase
"""
import argparse, copy, importlib.util, os, random, sys, time
import numpy as np, scipy.io as sio, scipy.sparse as sp, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


zs_model = _load('zs_model', f'{ROOT}/zero_shot/model.py')      # the paper's Model / GCN / SimplePrompt
zs_utils = _load('zs_utils', f'{ROOT}/zero_shot/utils.py')      # the paper's normalize_adj
from models.h2gcn_backbone import use_h2gcn_backbone
from models.hypernet_alignment import FeatureHyperAlign, make_prep
from models.graph_lora import inject_graph_lora, freeze_base_train_lora, count_trainable, set_lora_enabled
from utils import ssl_pretrain as ssl

p = argparse.ArgumentParser()
p.add_argument('--svd_dir', default=f'{ROOT}/data/datasets for AnomalyGFM')
p.add_argument('--raw_dir', default=f'{ROOT}/data/raw')
p.add_argument('--targets', nargs='+', default=['amazon', 'amazon_upu', 'reddit'])
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
p.add_argument('--epochs', type=int, default=301)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--hid', type=int, default=400)
p.add_argument('--k_shot', type=int, default=100, help='labelled normal target nodes for adaptation')
p.add_argument('--adapt_lr', type=float, default=1e-3)
p.add_argument('--adapt_epochs', type=int, default=30)
p.add_argument('--lora_rank', type=int, default=4)
p.add_argument('--ssl_weight', type=float, default=1.0, help='weight of SSL loss when combined with supervised loss')
p.add_argument('--multisource', action='store_true', help='pre-train on several source graphs (target domain excluded)')
p.add_argument('--pool', nargs='+', default=['train', 'reddit', 'amazon', 'yelp', 'tfinance'])
p.add_argument('--cap', type=int, default=4000, help='max nodes per sampled source subgraph')
p.add_argument('--n_views', type=int, default=12)
p.add_argument('--out', default=f'{ROOT}/results')
args = p.parse_args()
dev = torch.device('cuda')
FILES = {'train': ('Facebook.mat', 'Facebook_svd.mat'),
         'amazon': ('Amazon-all.mat', 'amazon_svd.mat'),
         'amazon_upu': ('Amazon.mat', 'Amazon_upu_svd.mat'),
         'reddit': ('Reddit.mat', 'reddit_svd.mat'),
         'yelp': ('YelpChi.mat', None),            # source-only: no shared SVD file, SVD is computed (paper's recipe)
         'tfinance': ('t_finance.mat', None)}
DOMAIN = {'train': 'facebook', 'reddit': 'reddit', 'amazon': 'amazon', 'amazon_upu': 'amazon',
          'yelp': 'yelp', 'tfinance': 'tfinance'}


# ----------------------------------------------------------------------------- data
def rownorm(X):                                    # paper's preprocess_features: divide by row sum (inf -> 0)
    rs = X.sum(1, keepdims=True)
    with np.errstate(divide='ignore'):
        inv = 1.0 / rs
    inv[~np.isfinite(inv)] = 0.0
    return np.nan_to_num(X * inv).astype(np.float32)


def sp_t(m):
    m = sp.coo_matrix(m)
    return torch.sparse_coo_tensor(np.vstack([m.row, m.col]), m.data.astype(np.float32), m.shape).coalesce().to(dev)


def load_graph(name, ops=True):
    raw, svd = FILES[name]
    r = sio.loadmat(f'{args.raw_dir}/{raw}')
    g = lambda d, ks: next(d[k] for k in ks if k in d)
    adj = sp.csr_matrix(g(r, ['Network', 'A']))
    y = np.asarray(g(r, ['Label', 'gnd'])).ravel().astype(int)
    dense = lambda m: np.asarray(m.todense() if sp.issparse(m) else m, dtype=np.float64)
    Xraw = dense(g(r, ['Attributes', 'X']))
    if svd is not None:
        s_ = sio.loadmat(f'{args.svd_dir}/{svd}')
        assert (y == np.asarray(g(s_, ['Label', 'gnd'])).ravel().astype(int)).all(), f'{name}: raw/SVD labels differ'
        Xsvd = dense(g(s_, ['Attributes', 'X']))
    else:                                           # paper's svd_feature_trans.py: U[:, :d] * S[:d]
        U, S, _ = np.linalg.svd(Xraw, full_matrices=False)
        Xsvd = U[:, :8] * S[:8]
    N = adj.shape[0]
    A_bin = (adj > 0).astype(np.float32).tolil(); A_bin.setdiag(0); A_bin = sp.csr_matrix(A_bin); A_bin.eliminate_zeros()
    G = dict(name=name, N=N, y=y, A_bin=A_bin, X8=torch.FloatTensor(rownorm(Xsvd)).to(dev),
             Xraw=torch.FloatTensor(Xraw).to(dev))
    G['prep'] = make_prep(G['Xraw'], sp_t(A_bin))
    if ops:
        Wn = adj.tolil(); Wn.setdiag(0); Wn = sp.csr_matrix(Wn)
        rs = np.asarray(Wn.sum(1)).ravel(); rs[rs == 0] = 1
        G['A_hat'] = sp_t(zs_utils.normalize_adj(adj) + sp.eye(N))
        G['Wn'] = sp_t(sp.diags(1.0 / rs) @ Wn)
        G['A_dense'] = torch.FloatTensor(np.asarray(A_bin.todense())).to(dev) if N < 3000 else None
    return G


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    """The paper's Model (GCN/prompt fc layers) + optional extensions. Only the zero-shot path is used."""

    def __init__(self, backbone='gcn', align='svd'):
        super().__init__()
        self.core = zs_model.Model(8, 8, args.hid, 'prelu', 1, 'avg')
        if backbone == 'h2gcn':
            use_h2gcn_backbone(self.core, 8, args.hid)                   # Contribution 1
        self.align = FeatureHyperAlign(8) if align == 'hyper' else None  # Contribution 2
        self.input_prompt, self.use_prompt = None, False

    def active_modules(self, with_logit_head=True):
        mods = [self.core.gcn1, self.core.gcn2, self.core.fc_normal_prompt, self.core.fc_abnormal_prompt]
        if with_logit_head: mods.append(self.core.fc1)
        if self.align is not None: mods.append(self.align)
        return mods

    def raw_features(self, G):
        return G['Xraw'] if self.align is not None else G['X8']

    def features(self, G, X=None):
        X = self.raw_features(G) if X is None else X
        return (self.align(X, G['prep']) if self.align is not None else X)[None]

    def encode(self, G, A_hat, Wn, X=None):
        x = self.features(G, X)
        if self.use_prompt: x = self.input_prompt(x)
        c = self.core
        emb = c.gcn2(c.gcn1(x, A_hat, True), A_hat, True).squeeze(0)
        return emb, emb - torch.sparse.mm(Wn, emb)

    def protos(self, nraw, araw):
        return self.core.act(self.core.fc_normal_prompt(nraw)), self.core.act(self.core.fc_abnormal_prompt(araw))


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def n_params(mods):
    return sum(p.numel() for m in mods for p in m.parameters())


# ----------------------------------------------------------------------------- pre-training
def pretrain(net, Gtr, mode, seed):
    """mode: 'sup' (paper) | 'ssl' (no labels) | 'sup+ssl'."""
    rng = np.random.RandomState(seed)
    idx_all = rng.permutation(Gtr['N'])
    idx = torch.LongTensor(idx_all[:int(0.3 * Gtr['N'])]).to(dev)              # paper: train_rate = 0.3
    y = torch.FloatTensor(Gtr['y']).to(dev)[idx]
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=args.lr)
    gen = torch.Generator(device=dev).manual_seed(seed)
    cfg = ssl.SSLConfig()
    X_orig = net.raw_features(Gtr)
    for ep in range(args.epochs):
        net.train(); opt.zero_grad()
        nraw, araw = torch.randn(args.hid, device=dev), torch.randn(args.hid, device=dev)
        nprm, aprm = net.protos(nraw, araw)
        loss = 0.0
        if 'sup' in mode:
            emb, res = net.encode(Gtr, Gtr['A_hat'], Gtr['Wn'])
            logits = net.core.fc1(emb).squeeze(-1)
            loss = loss + F.binary_cross_entropy_with_logits(logits[idx], y)
            r = res[idx]
            if (y == 1).any():
                loss = loss + (aprm - r[y == 1]).norm(dim=1).mean()
            loss = loss + 0.1 * (nprm - r[y == 0]).norm(dim=1).mean()
        if 'ssl' in mode:
            enc = lambda A_hat, Wn, X: net.encode(Gtr, A_hat, Wn, X)
            l, _ = ssl.ssl_loss(enc, Gtr['A_dense'], X_orig, cfg, gen, nprm, aprm)   # NO labels used here
            loss = loss + (args.ssl_weight if 'sup' in mode else 1.0) * l
        loss.backward(); opt.step()
    return net.eval(), idx_all[int(0.3 * Gtr['N']):]


# ----------------------------------------------------------------------------- adaptation (few-shot, normal labels only)
def adapt(net0, Gte, normal_idx, method, epochs, lr, seed):
    net = copy.deepcopy(net0)
    for q in net.parameters(): q.requires_grad = False
    if method == 'prompt':                                         # paper's input prompt (SimplePrompt on features)
        net.input_prompt, net.use_prompt = zs_model.SimplePrompt(8).to(dev), True
        for q in net.input_prompt.parameters(): nn.init.zeros_(q)      # identity at init (paper's random init perturbs the input)
        params = list(net.input_prompt.parameters())
        for q in params: q.requires_grad = True
    elif method == 'lora':                                         # Contribution 4
        inject_graph_lora(net.core, ('gcn1', 'gcn2'), r=args.lora_rank, alpha=2.0 * args.lora_rank)
        params = freeze_base_train_lora(net)
    elif method == 'full':                                         # reference: tune the whole GNN
        params = [q for m in (net.core.gcn1, net.core.gcn2) for q in m.parameters()]
        for q in params: q.requires_grad = True
    opt = torch.optim.Adam(params, lr=lr)
    torch.manual_seed(seed + 7)
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        nprm, _ = net.protos(torch.randn(args.hid, device=dev), torch.randn(args.hid, device=dev))
        _, res = net.encode(Gte, Gte['A_hat'], Gte['Wn'])
        loss = (nprm[None, :] - res[normal_idx]).norm(dim=1).mean()    # paper's few-shot normal alignment loss
        loss.backward(); opt.step()
    return net.eval(), count_trainable(params)


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(net, G, eval_idx, seed, beta=0.5):
    net.eval()
    gen = torch.Generator().manual_seed(seed + 1)                  # fixed prompt draw per seed -> reproducible
    nraw, araw = torch.randn(args.hid, generator=gen).to(dev), torch.randn(args.hid, generator=gen).to(dev)
    _, res = net.encode(G, G['A_hat'], G['Wn'])
    nprm, aprm = net.protos(nraw, araw)
    rn = F.normalize(res, dim=1)
    ca, cn = (rn @ F.normalize(aprm, dim=0)).cpu().numpy(), (rn @ F.normalize(nprm, dim=0)).cpu().numpy()
    y = G['y'][eval_idx]
    fus = np.exp(ca) + beta * np.exp(-cn)
    return dict(auroc=roc_auc_score(y, ca[eval_idx]), auprc=average_precision_score(y, ca[eval_idx]),
                auroc_fusion=roc_auc_score(y, fus[eval_idx]), auprc_fusion=average_precision_score(y, fus[eval_idx]))


def timed(fn):
    """Returns (result, peak GPU MB, seconds). In --multisource mode the peak is INCREMENTAL (peak minus what was
    already resident, e.g. cached source subgraphs), so rows stay comparable."""
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated(); t = time.time()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - (base if args.multisource else 0)
    return out, peak / 2 ** 20, time.time() - t


# ----------------------------------------------------------------------------- multi-source pre-training
def bfs_nodes(adj, cap, rng, fanout=20):
    """Collect `cap` nodes by BFS from random seeds (keeps local structure; hubs are fan-out limited)."""
    from collections import deque
    N = adj.shape[0]; seen = np.zeros(N, bool); order = []
    while len(order) < cap:
        st = rng.randint(N)
        if seen[st]: continue
        seen[st] = True; q = deque([st])
        while q and len(order) < cap:
            u = q.popleft(); order.append(u)
            nb = adj.indices[adj.indptr[u]:adj.indptr[u + 1]]
            nb = nb[~seen[nb]]
            if len(nb) > fanout: nb = rng.choice(nb, fanout, replace=False)
            seen[nb] = True; q.extend(nb.tolist())
    return np.array(order[:cap])


def make_view(G, idx_np, rng):
    adj = G['A_bin'][idx_np][:, idx_np].tocsr()
    N = len(idx_np); it = torch.LongTensor(idx_np).to(dev)
    Wn = adj.tolil(); Wn.setdiag(0); Wn = sp.csr_matrix(Wn)
    rs = np.asarray(Wn.sum(1)).ravel(); rs[rs == 0] = 1
    tr = torch.LongTensor(rng.permutation(N)[:int(0.3 * N)]).to(dev)          # paper: 30% labelled (source graphs only)
    y = torch.FloatTensor(G['y'][idx_np]).to(dev)
    return dict(name=G['name'], N=N, A_hat=sp_t(zs_utils.normalize_adj(adj) + sp.eye(N)),
                Wn=sp_t(sp.diags(1.0 / rs) @ Wn), X8=G['X8'][it], Xraw=G['Xraw'][it], prep=G['prep'],
                A_dense=torch.FloatTensor(np.asarray(adj.todense())).to(dev), tr_idx=tr, tr_y=y[tr])


def make_views(G, seed):
    rng = np.random.RandomState(10_000 + seed)
    if G['N'] <= args.cap:
        return [make_view(G, np.arange(G['N']), rng)]
    return [make_view(G, bfs_nodes(G['A_bin'], args.cap, rng), rng) for _ in range(args.n_views)]


def pretrain_multi(net, srcs, mode, seed):
    """One optimiser step per epoch on the mean loss over all source graphs (a random view of each)."""
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=args.lr)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.RandomState(seed + 5); cfg = ssl.SSLConfig()
    for ep in range(args.epochs):
        net.train(); opt.zero_grad()
        nprm, aprm = net.protos(torch.randn(args.hid, device=dev), torch.randn(args.hid, device=dev))
        total = 0.0
        for name, views in srcs.items():
            V = views[rng.randint(len(views))]
            if 'sup' in mode:
                emb, res = net.encode(V, V['A_hat'], V['Wn'])
                idx, y = V['tr_idx'], V['tr_y']
                total = total + F.binary_cross_entropy_with_logits(net.core.fc1(emb).squeeze(-1)[idx], y)
                r = res[idx]
                if (y == 1).any(): total = total + (aprm - r[y == 1]).norm(dim=1).mean()
                if (y == 0).any(): total = total + 0.1 * (nprm - r[y == 0]).norm(dim=1).mean()
            if 'ssl' in mode:
                enc = lambda A_hat, Wn, X, V=V: net.encode(V, A_hat, Wn, X)
                l, _ = ssl.ssl_loss(enc, V['A_dense'], net.raw_features(V), cfg, gen, nprm, aprm)   # no labels
                total = total + (args.ssl_weight if 'sup' in mode else 1.0) * l
        (total / len(srcs)).backward(); opt.step()
    return net.eval()


def main_multi():
    os.makedirs(args.out, exist_ok=True)
    print(f'GPU: {torch.cuda.get_device_name(0)}  MULTI-SOURCE pool={args.pool} cap={args.cap} seeds={args.seeds}', flush=True)
    pool_all = {n: load_graph(n, ops=(n == 'train')) for n in args.pool}
    Gtr = pool_all['train']
    rows = []
    for tgt in args.targets:
        Gte = load_graph(tgt)
        srcs_names = [n for n in args.pool if DOMAIN[n] != DOMAIN[tgt]]
        print(f'\n=== target {tgt} | multi-source pool (target domain excluded): {srcs_names}', flush=True)
        for seed in args.seeds:
            rs = np.random.RandomState(1000 + seed)
            normals = np.where(Gte['y'] == 0)[0]; rs.shuffle(normals)
            few = torch.LongTensor(normals[:args.k_shot]).to(dev)
            eval_idx = np.setdiff1d(np.arange(Gte['N']), normals[:args.k_shot])
            rec = lambda method, stage, m, params, mem, secs: rows.append(
                dict(target=tgt, seed=seed, method=method, stage=stage, params=params, peak_mb=mem, secs=secs, **m))
            views = {n: make_views(pool_all[n], seed) for n in srcs_names}

            def fit(label, backbone, align, mode, multi):
                def go():
                    seed_all(seed)
                    net = Net(backbone, align).to(dev)
                    pretrain_multi(net, views, mode, seed) if multi else pretrain(net, Gtr, mode, seed)
                    return net, evaluate(net, Gte, eval_idx, seed)
                (net, m), mem, secs = timed(go)
                rec(label, 'zero-shot', m, n_params(net.active_modules(with_logit_head='sup' in mode)), mem, secs)
                return net

            for tag, multi in (('[1-src FB]', False), (f'[{len(srcs_names)}-src]', True)):
                b = fit(f'{tag} Baseline (GCN+SVD, sup.)', 'gcn', 'svd', 'sup', multi)
                fit(f'{tag} + C1 H2GCN', 'h2gcn', 'svd', 'sup', multi)
                fit(f'{tag} + C2 Hypernet align', 'gcn', 'hyper', 'sup', multi)
                fit(f'{tag} + C3 SSL (NO labels)', 'gcn', 'svd', 'ssl', multi)
                c = fit(f'{tag} C1+C2+C3 (sup+SSL)', 'h2gcn', 'hyper', 'sup+ssl', multi)
                if multi:
                    for label, src, method, ep, lr in [
                        (f'{tag} Base + prompt tuning (paper protocol)', b, 'prompt', 5, 1e-4),
                        (f'{tag} Base + C4 Graph-LoRA (paper protocol)', b, 'lora', 5, 1e-4),
                        (f'{tag} Combined + C4 Graph-LoRA (paper protocol)', c, 'lora', 5, 1e-4),
                        (f'{tag} [sens.] prompt ({args.adapt_epochs} ep, lr {args.adapt_lr:g})', b, 'prompt', args.adapt_epochs, args.adapt_lr),
                        (f'{tag} [sens.] full GNN FT ({args.adapt_epochs} ep, lr {args.adapt_lr:g})', b, 'full', args.adapt_epochs, args.adapt_lr),
                        (f'{tag} [sens.] Graph-LoRA ({args.adapt_epochs} ep, lr {args.adapt_lr:g})', b, 'lora', args.adapt_epochs, args.adapt_lr)]:
                        def go():
                            a, n = adapt(src, Gte, few, method, ep, lr, seed)
                            return n, evaluate(a, Gte, eval_idx, seed)
                        (n, m), mem, secs = timed(go)
                        rec(label, f'adapt (k={args.k_shot})', m, n, mem, secs)
            del views; torch.cuda.empty_cache()
            print(f'  seed {seed} done', flush=True)
        del Gte; torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(f'{args.out}/multisource_raw.csv', index=False)
    order = list(dict.fromkeys(df.method)); md = []
    for tgt in args.targets:
        d = df[df.target == tgt]; g = d.groupby('method', sort=False)
        t = pd.DataFrame({'Method': order, 'Stage': [d[d.method == m].stage.iloc[0] for m in order],
            'AUROC': [f'{g.auroc.mean()[m]:.3f}±{g.auroc.std()[m]:.3f}' for m in order],
            'AUPRC': [f'{g.auprc.mean()[m]:.3f}±{g.auprc.std()[m]:.3f}' for m in order],
            'Params': [f'{int(g.params.mean()[m]):,}' for m in order],
            'GPU MB (incr.)': [f'{g.peak_mb.mean()[m]:.0f}' for m in order]})
        s_ = f'\n### Target: {tgt}  (mean±std over {len(args.seeds)} seeds; target domain never in the source pool)\n' + t.to_markdown(index=False)
        print(s_); md.append(s_)
    open(f'{args.out}/multisource.md', 'w').write('\n'.join(md))


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(args.out, exist_ok=True)
    print(f'GPU: {torch.cuda.get_device_name(0)}  seeds={args.seeds}  targets={args.targets}', flush=True)
    Gtr = load_graph('train')
    rows = []
    for tgt in args.targets:
        Gte = load_graph(tgt)
        print(f'\n=== target {tgt}: N={Gte["N"]}, raw feat dim={Gte["Xraw"].shape[1]}, anomaly rate={Gte["y"].mean():.3f}', flush=True)
        for seed in args.seeds:
            rs = np.random.RandomState(1000 + seed)
            normals = np.where(Gte['y'] == 0)[0]; rs.shuffle(normals)
            few = torch.LongTensor(normals[:args.k_shot]).to(dev)
            eval_idx = np.setdiff1d(np.arange(Gte['N']), normals[:args.k_shot])
            rec = lambda method, stage, m, params, mem, secs: rows.append(
                dict(target=tgt, seed=seed, method=method, stage=stage, params=params, peak_mb=mem, secs=secs, **m))

            def run_variant(label, backbone, align, mode):
                def go():
                    seed_all(seed)
                    net = Net(backbone, align).to(dev)
                    _, fb_val = pretrain(net, Gtr, mode, seed)
                    m = evaluate(net, Gte, eval_idx, seed)
                    m['fb_val_auroc'] = evaluate(net, Gtr, fb_val, seed)['auroc']   # in-domain sanity check
                    return net, m
                (net, m), mem, secs = timed(go)
                params = n_params(net.active_modules(with_logit_head='sup' in mode))
                rec(label, 'zero-shot', m, params, mem, secs)
                return net

            base = run_variant('Baseline AnomalyGFM (GCN+SVD, sup.)', 'gcn', 'svd', 'sup')
            run_variant('+ C1 H2GCN backbone', 'h2gcn', 'svd', 'sup')
            run_variant('+ C2 Hypernet alignment', 'gcn', 'hyper', 'sup')
            run_variant('+ C3 SSL prototypes (NO labels)', 'gcn', 'svd', 'ssl')
            comb = run_variant('C1+C2+C3 combined (sup+SSL)', 'h2gcn', 'hyper', 'sup+ssl')
            P = dict(ep=5, lr=1e-4)                                   # paper's few-shot protocol (small graphs)
            B = dict(ep=args.adapt_epochs, lr=args.adapt_lr)          # larger-budget sensitivity check
            for label, src, method, bud in [
                ('Base + prompt tuning (paper protocol)', base, 'prompt', P),
                ('Base + full GNN fine-tune (paper protocol)', base, 'full', P),
                ('Base + C4 Graph-LoRA (paper protocol)', base, 'lora', P),
                ('Combined + C4 Graph-LoRA (paper protocol)', comb, 'lora', P),
                (f'[sens.] prompt tuning ({B["ep"]} ep, lr {B["lr"]:g})', base, 'prompt', B),
                (f'[sens.] full GNN fine-tune ({B["ep"]} ep, lr {B["lr"]:g})', base, 'full', B),
                (f'[sens.] Graph-LoRA ({B["ep"]} ep, lr {B["lr"]:g})', base, 'lora', B)]:
                ep, lr = bud['ep'], bud['lr']
                def go():
                    a, n = adapt(src, Gte, few, method, ep, lr, seed)
                    return n, evaluate(a, Gte, eval_idx, seed)
                (n, m), mem, secs = timed(go)
                rec(label, f'adapt (k={args.k_shot})', m, n, mem, secs)
            print(f'  seed {seed} done', flush=True)
        del Gte; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(f'{args.out}/team_results_raw.csv', index=False)
    order = list(dict.fromkeys(df.method))
    md = []
    for tgt in args.targets:
        d = df[df.target == tgt]
        g = d.groupby('method', sort=False)
        t = pd.DataFrame({
            'Method': order, 'Stage': [d[d.method == m].stage.iloc[0] for m in order],
            'AUROC': [f'{g.auroc.mean()[m]:.3f}±{g.auroc.std()[m]:.3f}' for m in order],
            'AUPRC': [f'{g.auprc.mean()[m]:.3f}±{g.auprc.std()[m]:.3f}' for m in order],
            'AUROC(fusion)': [f'{g.auroc_fusion.mean()[m]:.3f}' for m in order],
            'FB-heldout AUROC': [(f'{g.fb_val_auroc.mean()[m]:.3f}' if pd.notna(g.fb_val_auroc.mean()[m]) else '-') for m in order],
            'Trainable params': [f'{int(g.params.mean()[m]):,}' for m in order],
            'Peak GPU MB': [f'{g.peak_mb.mean()[m]:.0f}' for m in order]})
        s = f'\n### Target: {tgt}  (train: Facebook; mean±std over {len(args.seeds)} seeds; eval excludes the {args.k_shot} tuned normal nodes)\n'
        s += t.to_markdown(index=False) if hasattr(t, 'to_markdown') else t.to_string(index=False)
        print(s); md.append(s)
    open(f'{args.out}/team_results.md', 'w').write('\n'.join(md))


if __name__ == '__main__':
    main_multi() if args.multisource else main()
