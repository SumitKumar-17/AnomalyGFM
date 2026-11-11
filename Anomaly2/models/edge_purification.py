"""Contribution 3 - Adversarial edge purification (differentiable edge dropping).

The paper trusts the adjacency matrix: an anomaly that wires itself to normal nodes (camouflage) drags its neighbour mean toward
itself and its representation residual vanishes. EdgePurifier scores every undirected edge from SYMMETRIC edge features
(|x_u-x_v|, x_u*x_v, degree terms - so w_uv = w_vu) and gives it a keep-weight BEFORE message passing:

  training : binary-concrete (Gumbel-sigmoid) relaxation  w = sigmoid((logit + logistic_noise)/tau)   -> differentiable
  inference: hard drop  w = 1[logit > 0]

The weights re-weight both the GCN operator and the neighbour-mean operator used by the residual. Rows left with ~no weight get a
small self-mass (eps), so their residual is ~0 instead of h (avoids flagging nodes merely because their edges were dropped).

Adversarial training: each epoch the training graph is attacked (random noise edges + camouflage edges from anomalies to normals,
see train.py); those injected edges are KNOWN, so the purifier also gets a direct edge-level BCE target (keep=0 for injected
edges). No label of the target graph is used at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data import ops_from_edges


def _slog(t):
    return torch.sign(t) * torch.log1p(t.abs())


def _cn_dense_chunked(eu, N, block=2048):
    """Common neighbours via blocked DENSE matmuls (for dense graphs where sparse A@A exhausts GPU workspace)."""
    u, v = eu
    A = torch.zeros(N, N, device=u.device)
    A[u, v] = 1.0
    A[v, u] = 1.0
    order = torch.argsort(u)
    us, vs = u[order], v[order]
    out = torch.empty(len(u), device=u.device)
    for s in range(0, N, block):
        e = min(s + block, N)
        C = A[s:e] @ A                                                     # exact fp32 counts, [block, N]
        lo, hi = int(torch.searchsorted(us, s)), int(torch.searchsorted(us, e))
        out[order[lo:hi]] = C[us[lo:hi] - s, vs[lo:hi]]
    return out


@torch.no_grad()
def common_neighbors(eu, N):
    """|N(u) & N(v)| for every undirected edge (u<v). Random/camouflage edges share (almost) no neighbours, real edges in
    clustered graphs usually do - a structural signal that does not depend on the graph's feature semantics.
    Sparse A@A for sparse graphs; blocked dense matmul for dense graphs (avg degree > 100, N <= 27k)."""
    u, v = eu
    if 2 * u.shape[0] / N > 100 and N <= 27000:
        return _cn_dense_chunked(eu, N)
    A = torch.sparse_coo_tensor(torch.stack([torch.cat([u, v]), torch.cat([v, u])]), torch.ones(2 * u.shape[0], device=u.device), (N, N)).coalesce()
    A2 = torch.sparse.mm(A, A).coalesce()
    idx = A2.indices()
    key2, val2 = idx[0] * N + idx[1], A2.values()
    q = u * N + v
    pos = torch.searchsorted(key2, q).clamp(max=len(key2) - 1)
    return torch.where(key2[pos] == q, val2[pos], torch.zeros_like(val2[pos]))


class EdgePurifier(nn.Module):
    def __init__(self, hidden=32, tau=0.5, eps=0.05):
        super().__init__()
        self.tau, self.eps = tau, eps
        self.mlp = nn.Sequential(nn.Linear(20, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.mlp[2].bias, 2.0)                       # start by keeping almost every edge (sigmoid(2)=.88)

    def edge_features(self, X8, deg, eu):
        u, v = eu
        xu, xv = X8[u], X8[v]
        du, dv = torch.log1p(deg[u]), torch.log1p(deg[v])
        cn = common_neighbors(eu, X8.shape[0])
        salton = cn / torch.sqrt(deg[u] * deg[v] + 1e-6)                     # cosine-style overlap in [0,1]
        return torch.cat([_slog((xu - xv).abs()), _slog(xu * xv), (du + dv)[:, None], (du - dv).abs()[:, None],
                          torch.log1p(cn)[:, None], salton[:, None]], 1)

    def forward(self, X8, deg, eu, N, hard=None):
        logit = self.mlp(self.edge_features(X8, deg, eu)).squeeze(-1)
        hard = (not self.training) if hard is None else hard
        if hard:
            w = (logit > 0).float()
        else:
            uni = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
            w = torch.sigmoid((logit + torch.log(uni) - torch.log1p(-uni)) / self.tau)
        A_hat, P, wdeg = ops_from_edges(eu, N, w, eps=self.eps)
        return A_hat, P, dict(logit=logit, w=w)

    @staticmethod
    def edge_loss(logit, injected):
        """Plain BCE on 'keep' (1 for original edges, 0 for injected attack edges). Deliberately NOT class-balanced: balancing made
        the scorer drop ~half of the real edges (see WORK.md); the prior favours keeping an edge unless there is evidence."""
        return F.binary_cross_entropy_with_logits(logit, 1.0 - injected)


def run_attack_benchmark(models, graphs, attacks, seeds, evaluate_fn):
    """Table of AUROC under attack.  models: {name: {seed: model}}; graphs: {target: G}; attacks: [(kind, level)].
    evaluate_fn(model, G_attacked, seed) -> dict(auroc, auprc).  Returns a pandas DataFrame (long format)."""
    import pandas as pd
    from utils.attacks import ATTACKS
    rows = []
    for tname, G in graphs.items():
        for kind, level in attacks:
            for seed in seeds:
                Ga = G if level == 0 else ATTACKS[kind](G, level, seed)
                for mname, per_seed in models.items():
                    m = evaluate_fn(per_seed[seed], Ga, seed)
                    rows.append(dict(target=tname, attack=kind, level=level, seed=seed, model=mname, **m))
            del Ga
    return pd.DataFrame(rows)
