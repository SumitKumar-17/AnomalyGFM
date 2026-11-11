"""Data loading + sparse graph operators for Anomaly2 (real graphs only; SVD-aligned features from the authors).

Every graph is symmetrised and binarised (no self loops) and stored as an UNDIRECTED edge list `eu` (2,E) with u<v, from which
all operators are rebuilt on the GPU. This makes edge-level methods (purification, attacks) first-class.
"""
import importlib.util, os, sys
import numpy as np, scipy.io as sio, scipy.sparse as sp, torch

ANOMALY2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(ANOMALY2)                      # repo root (contains zero_shot/, data/)
SVD_DIR = os.environ.get('SVD_DIR', f'{ROOT}/data/datasets for AnomalyGFM')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


zs_model = _load('zs_model', f'{ROOT}/zero_shot/model.py')     # the paper's Model / GCN / SimplePrompt

FILES = {'train': 'Facebook_svd.mat', 'amazon': 'amazon_svd.mat', 'amazon_upu': 'Amazon_upu_svd.mat',
         'reddit': 'reddit_svd.mat', 'elliptic': 'elliptic_svd.mat', 'tolokers': 'tolokers_svd.mat',
         'questions': 'questions_svd.mat', 'yelp': 'yelp_svd.mat', 'tfinance': 't_finance_svd.mat'}


def rownorm(X):
    """The paper's preprocess_features: divide each row by its (signed) sum; inf -> 0."""
    rs = X.sum(1, keepdims=True)
    with np.errstate(divide='ignore'):
        inv = 1.0 / rs
    inv[~np.isfinite(inv)] = 0.0
    return np.nan_to_num(X * inv).astype(np.float32)


def ops_from_edges(eu, N, w=None, eps=0.0):
    """Differentiable sparse operators from an undirected edge list.

    eu: (2,E) long (u<v). w: (E,) edge weights (None -> all ones).
    Returns A_hat = D^-1/2 A D^-1/2 + I (GCN operator), P = row-normalised neighbour operator (the paper's neighbour mean),
    and the weighted degree. eps>0 gives rows with (almost) no weight a self-mass so their residual is ~0 instead of h.
    """
    u, v = eu
    if w is None:
        w = torch.ones(u.shape[0], device=u.device)
    row, col, val = torch.cat([u, v]), torch.cat([v, u]), torch.cat([w, w])
    deg = torch.zeros(N, device=u.device).index_add(0, row, val)
    dinv = deg.clamp(min=1e-12).pow(-0.5) * (deg > 0)
    ar = torch.arange(N, device=u.device)
    A_hat = torch.sparse_coo_tensor(torch.stack([torch.cat([row, ar]), torch.cat([col, ar])]),
                                    torch.cat([dinv[row] * val * dinv[col], torch.ones(N, device=u.device)]), (N, N)).coalesce()
    p_val = val / (deg[row] + eps) if eps > 0 else val / deg[row]
    idx, pv = torch.stack([row, col]), p_val
    if eps > 0:
        idx, pv = torch.cat([idx, torch.stack([ar, ar])], 1), torch.cat([p_val, eps / (deg + eps)])
    P = torch.sparse_coo_tensor(idx, pv, (N, N)).coalesce()
    return A_hat, P, deg


def struct_features(A):
    """Cheap per-node structural descriptors (used by the adaptive-PPR gate): log deg, log mean-neighbour deg, log 2-hop mass."""
    d = np.asarray(A.sum(1)).ravel()
    nd = A @ d
    f = np.stack([np.log1p(d), np.log1p(nd / np.maximum(d, 1)), np.log1p(nd)], 1)
    return ((f - f.mean(0)) / (f.std(0) + 1e-6)).astype(np.float32)


def make_graph(name, A, X8, y, with_struct=True):
    """A: scipy csr symmetric binary (no diag); X8: [N,8] float tensor on dev; y: np int labels."""
    N = A.shape[0]
    U = sp.triu(A, 1).tocoo()
    eu = torch.LongTensor(np.vstack([U.row, U.col])).to(dev)
    A_hat, P, deg = ops_from_edges(eu, N)
    G = dict(name=name, N=N, y=np.asarray(y).astype(int), y_t=torch.FloatTensor(np.asarray(y)).to(dev), A=A, X8=X8,
             eu=eu, A_hat=A_hat, Wn=P, deg=deg)
    if with_struct:
        G['struct'] = torch.FloatTensor(struct_features(A)).to(dev)
    return G


def load_graph(name, with_struct=True):
    s = sio.loadmat(f'{SVD_DIR}/{FILES[name]}')
    g = lambda ks: next(s[k] for k in ks if k in s)
    adj = sp.csr_matrix(g(['Network', 'A']))
    A = ((adj + adj.T) > 0).astype(np.float32).tolil(); A.setdiag(0); A = sp.csr_matrix(A); A.eliminate_zeros()
    X = g(['Attributes', 'X']); X = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)
    y = np.asarray(g(['Label', 'gnd'])).ravel().astype(int)
    return make_graph(name, A, torch.FloatTensor(rownorm(X)).to(dev), y, with_struct)


def graph_from_edges(G, eu, with_struct=False):
    """New graph dict sharing features/labels with G but using edge list eu (2,E) long on dev (u<v, unique)."""
    N = G['N']
    A_hat, P, deg = ops_from_edges(eu, N)
    H = dict(G, eu=eu, A_hat=A_hat, Wn=P, deg=deg)
    if with_struct:
        e = eu.cpu().numpy()
        A = sp.coo_matrix((np.ones(e.shape[1], np.float32), (e[0], e[1])), shape=(N, N)); A = (A + A.T).tocsr()
        H['A'] = A; H['struct'] = torch.FloatTensor(struct_features(A)).to(dev)
    return H
