"""Contribution 2 - Non-linear, dataset-conditioned feature alignment (replaces the static SVD).

Problem: graphs have different raw feature widths (Facebook 576, Amazon 25, Reddit 64) and the paper maps every
graph to d'=8 with a *fixed linear SVD*. Here a hypernetwork generates the weights of a 2-layer non-linear
projection from dataset metadata:

  * per-FEATURE descriptors z_j  (mean, spread, skew, kurtosis, sparsity, graph-smoothness (Dirichlet energy),
    correlation with degree, range)                      -> permutation-equivariant, works for any width D
  * per-GRAPH metadata m        (size, density, degree skew, width, mean smoothness, ...)
  * W1[j,:] = g1([z_j ; m])  (D x hidden),   b1 = g_b1(m)
  * W2     = g2(m)           (hidden x d'),   b2 = g_b2(m)
  * out    = L2norm( GELU(X_std W1 + b1) W2 + b2 )

The hypernetwork is trained end-to-end with the anomaly objective on the training graph, and at test time it
generates a projection for the unseen graph from that graph's own descriptors - no SVD, no per-graph fitting.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def make_prep(X, A):
    """X: [N,D] raw features (tensor); A: [N,N] sparse binary adjacency (tensor). Returns constants for this graph."""
    N, D = X.shape
    mean, std = X.mean(0), X.std(0)
    Xs = ((X - mean) / (std + 1e-6)).clamp(-5, 5)
    deg = torch.sparse.sum(A, 1).to_dense()
    E = deg.sum()
    dx2 = (deg[:, None] * Xs ** 2).sum(0)
    dirichlet = dx2 - (Xs * torch.sparse.mm(A, Xs)).sum(0)
    smooth = (dirichlet / (dx2 + 1e-6)).clamp(0, 2)             # ~0: smooth over edges, ~1: uncorrelated
    skew = (Xs ** 3).mean(0).clamp(-5, 5) / 5
    kurt = ((Xs ** 4).mean(0) - 3).clamp(-3, 30) / 10
    nz = (X != 0).float().mean(0)
    ld = torch.log1p(deg)
    corr = (Xs * ((ld - ld.mean()) / (ld.std() + 1e-6))[:, None]).mean(0).clamp(-1, 1)
    desc = torch.stack([torch.sign(mean) * torch.log1p(mean.abs()), torch.log1p(std), skew, kurt, nz,
                        smooth, corr, torch.log1p(X.max(0).values - X.min(0).values)], 1)
    meta = torch.tensor([math.log(N) / 10, math.log(float(E) + 1) / 15, math.log1p(float(E) / N) / 5,
                         -math.log10(float(E) / (N * N) + 1e-9) / 10, float(deg.std() / (deg.mean() + 1e-6)) / 5,
                         math.log(D) / 7, float(smooth.mean()), float(nz.mean())], device=X.device)
    return dict(mean=mean, std=std, desc=torch.nan_to_num(desc), meta=torch.nan_to_num(meta))


class FeatureHyperAlign(nn.Module):
    def __init__(self, d_out=8, hidden=32, width=64, desc_dim=8, meta_dim=8):
        super().__init__()
        self.hidden, self.d_out = hidden, d_out
        mlp = lambda i, o: nn.Sequential(nn.Linear(i, width), nn.GELU(), nn.Linear(width, o))
        self.row_gen = mlp(desc_dim + meta_dim, hidden)     # generates W1 row-by-row from feature descriptors
        self.b1_gen = mlp(meta_dim, hidden)
        self.w2_gen = mlp(meta_dim, hidden * d_out)
        self.b2_gen = nn.Linear(meta_dim, d_out)
        self.act = nn.GELU()

    def forward(self, X, prep):
        D = X.shape[1]
        Xs = ((X - prep['mean']) / (prep['std'] + 1e-6)).clamp(-5, 5)
        z = torch.cat([prep['desc'], prep['meta'][None].expand(D, -1)], 1)
        W1 = self.row_gen(z) / math.sqrt(D)
        h = self.act(Xs @ W1 + self.b1_gen(prep['meta']))
        W2 = self.w2_gen(prep['meta']).view(self.hidden, self.d_out) / math.sqrt(self.hidden)
        return F.normalize(h @ W2 + self.b2_gen(prep['meta']), dim=1)       # [N, d']
