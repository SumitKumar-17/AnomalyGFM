"""Benchmark harness: Exp H (feature alignment) and Exp I (scoring function), zero-shot, train on Facebook.

Training mirrors run_abnormal.py (dense, CPU). Inference on target graphs is sparse (GPU if available) so
the 40k+ node graphs do not need dense N x N matrices. Every (config, seed, target) is scored with ALL
scoring functions, so one pass serves both experiments. Appends rows to ../results/bench.csv.
"""
import argparse, os, random, time, math
import numpy as np, scipy.sparse as sp, scipy.io as sio, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from model import Model
from utils import load_mat, normalize_adj

ALL_TARGETS = ['amazon_svd', 'Amazon_upu_svd', 'Disney_svd', 'elliptic_svd', 'questions_svd',
               'reddit_svd', 't_finance_svd', 'tolokers_svd', 'yelp_svd']
p = argparse.ArgumentParser()
p.add_argument('--exp', default='both', choices=['I', 'H', 'both'])
p.add_argument('--train', default='Facebook_svd')
p.add_argument('--targets', nargs='+', default=ALL_TARGETS)
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
p.add_argument('--epochs', type=int, default=301)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--hid', type=int, default=400)
p.add_argument('--out', default='../results/bench.csv')
p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = p.parse_args()
dev = torch.device(args.device)

# ------------- configs -------------
DEFAULT = (8, 'rownorm')
if args.exp == 'I':
    CFGS = [DEFAULT]
else:  # H grid (includes the default config, which also serves Exp I)
    CFGS = [(d, n) for d in (2, 4, 6, 8) for n in ('rownorm', 'zscore', 'l2row', 'none')]


def featprep(X, dim, norm):
    X = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)[:, :dim]
    if norm == 'rownorm':     # exactly what the paper's code does (divide by row sum; inf -> 0)
        rs = X.sum(1, keepdims=True)
        with np.errstate(divide='ignore'):
            inv = 1.0 / rs
        inv[~np.isfinite(inv)] = 0.0
        X = X * inv
    elif norm == 'zscore':
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    elif norm == 'l2row':
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    if X.shape[1] < 8:        # model input is fixed to 8: zero-pad truncated SVD features
        X = np.pad(X, ((0, 0), (0, 8 - X.shape[1])))
    return np.nan_to_num(X).astype(np.float32)


# ------------- training (dense, mirrors the paper's code) -------------
def train_one(cfg, seed, D):
    torch.manual_seed(seed)
    X = torch.FloatTensor(featprep(D['feat'], *cfg)[None])
    model = Model(8, 8, args.hid, 'prelu', 1, 'avg')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([1.0]))
    idx, y = D['idx'], D['y']
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        logits, _, _, emb_res, nprm, aprm, _ = model(X, D['A'], D['R'], torch.randn(args.hid), torch.randn(args.hid))
        l_bce = torch.mean(bce(torch.squeeze(logits[:, idx]), y))
        res = emb_res[:, idx, :]
        d_n = torch.sqrt(torch.sum((nprm - res[:, y == 0, :]) ** 2, dim=2))
        d_a = torch.sqrt(torch.sum((aprm - res[:, y == 1, :]) ** 2, dim=2))
        (l_bce + torch.mean(d_a) + 0.1 * torch.mean(d_n)).backward(); opt.step()
    return model.eval()


# ------------- sparse inference + scorers -------------
def to_t(m):
    m = sp.coo_matrix(m)
    return torch.sparse_coo_tensor(np.vstack([m.row, m.col]), m.data.astype(np.float32), m.shape).coalesce().to(dev)


def embed(model, X, Ahat):
    h = X
    for layer in (model.gcn1, model.gcn2):
        h = layer.act(torch.sparse.mm(Ahat, layer.fc(h)) + layer.bias)
    return h


def scorers(r, pa, pn):
    """All candidate anomaly scores; higher = more anomalous."""
    rn = F.normalize(r, dim=1)
    ca, cn = rn @ pa, rn @ pn
    rnorm = r.norm(dim=1)
    rc = (r - r.mean(0)).double()
    cov = rc.T @ rc / (len(r) - 1)
    cov += torch.eye(cov.shape[0], device=r.device, dtype=cov.dtype) * (1e-3 * torch.trace(cov) / cov.shape[0])
    md = ((rc @ torch.linalg.inv(cov)) * rc).sum(1).sqrt().float()
    S = {
        'cos_abn': ca,                                  # paper's abnormal measure (exp() is monotone)
        'cos_norm_neg': -cn,                            # paper's "normal" measure
        'cos_norm_pos': cn,                             # diagnostic: sign-flipped
        'cos_diff': ca - cn,
        'paper_mid': torch.exp(ca) + 6 * torch.exp(-cn),  # hand-tuned fusion in the released code
        'res_norm': rnorm,                              # magnitude only, no prototypes
        'euc_diff': (r - pn).norm(dim=1) - (r - pa).norm(dim=1),
        'maha': md,                                     # Mahalanobis of residual vs target-graph residuals
    }
    S = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in S.items()}
    rk = lambda v: rankdata(v) / len(v)
    S['rank_fuse_diff_maha'] = rk(S['cos_diff']) + rk(S['maha'])      # direction + magnitude
    S['rank_fuse_abn_norm'] = rk(S['cos_abn']) + rk(S['res_norm'])
    return S


def metrics(y, s):
    s = np.nan_to_num(s, nan=0.0, posinf=1e9, neginf=-1e9)
    k = int(math.ceil(0.05 * len(y)))
    top = np.argsort(-s)[:k]
    return roc_auc_score(y, s), average_precision_score(y, s), float(y[top].mean())


# ------------- main -------------
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fresh = not os.path.exists(args.out)
fout = open(args.out, 'a')
if fresh:
    fout.write('cfg_dim,cfg_norm,seed,target,scorer,auroc,auprc,p_at_5pct\n')

models, t0 = {}, time.time()
for seed in args.seeds:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    adj, _, feat, _, y, _, _, idx, _ = load_mat(args.train, args.train)
    A = torch.FloatTensor(np.asarray((normalize_adj(adj) + sp.eye(adj.shape[0])).todense())[None])
    D = dict(feat=feat, A=A, R=torch.FloatTensor(np.asarray(adj.todense())), idx=idx, y=torch.FloatTensor(y))
    for cfg in CFGS:
        models[(cfg, seed)] = train_one(cfg, seed, D)
    print(f'[{time.time() - t0:6.0f}s] trained seed {seed} x {len(CFGS)} cfgs', flush=True)

for tgt in args.targets:
    d = sio.loadmat(f'data/{tgt}.mat')
    lab = np.asarray(d['Label'] if 'Label' in d else d['gnd']).ravel().astype(int)
    attr = d['Attributes'] if 'Attributes' in d else d['X']
    net = sp.csr_matrix(d['Network'] if 'Network' in d else d['A'])
    Ahat = to_t(normalize_adj(net) + sp.eye(net.shape[0]))
    Wn = net.tolil(); Wn.setdiag(0); Wn = sp.csr_matrix(Wn)
    rs = np.asarray(Wn.sum(1)).ravel(); rs[rs == 0] = 1
    Wn = to_t(sp.diags(1.0 / rs) @ Wn)
    for cfg in CFGS:
        X = torch.FloatTensor(featprep(attr, *cfg)).to(dev)
        for seed in args.seeds:
            m = models[(cfg, seed)].to(dev)
            with torch.no_grad():
                H = embed(m, X, Ahat)
                r = H - torch.sparse.mm(Wn, H)
                g = torch.Generator().manual_seed(seed + 1)          # fixed prompt draw per seed
                pn = F.normalize(F.relu(m.fc_normal_prompt(torch.randn(args.hid, generator=g).to(dev))), dim=0)
                pa = F.normalize(F.relu(m.fc_abnormal_prompt(torch.randn(args.hid, generator=g).to(dev))), dim=0)
                S = scorers(r, pa, pn)
            for name, s in S.items():
                try:
                    au, ap, p5 = metrics(lab, s)
                except Exception as e:
                    au = ap = p5 = float('nan')
                fout.write(f'{cfg[0]},{cfg[1]},{seed},{tgt},{name},{au:.5f},{ap:.5f},{p5:.5f}\n')
            fout.flush(); m.cpu()
    print(f'[{time.time() - t0:6.0f}s] done target {tgt} (N={len(lab)})', flush=True)
fout.close()
