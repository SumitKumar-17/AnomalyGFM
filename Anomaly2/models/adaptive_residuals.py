"""Contribution 4 - multi-hop / PPR / adaptive residuals.

The paper: r_v = h_v - sum_{u in N(v)} P_vu h_u   (1-hop neighbour mean, P = row-normalised adjacency).
Here the reference ("what a normal neighbourhood looks like") is a weighted multi-hop mean:

    ref_v = sum_{k=1..K} w_vk (P^k h)_v ,   r_v = h_v - ref_v

  hop1     : w = (1,0,0,...)                         -> the paper
  hop2     : w = (.5,.5,0,...)                       -> 1- and 2-hop average
  ppr      : w_k ~ alpha (1-alpha)^(k-1), sum to 1   -> truncated Personalised PageRank walk distribution from v (k>=1; the
                                                        k=0 teleport term is the node itself, i.e. the thing being compared)
  adaptive : w_v = softmax_k( log ppr_prior_k + MLP(structure_v) )   -> per-node receptive field sized from local topology
             (MLP's last layer is zero-initialised, so training starts exactly at `ppr`).

P^k h is computed with K sparse mat-muls; no N x N matrix is ever materialised (works for 46k-node graphs).
Note: P^k is a k-step random-WALK operator, so even hops include returns to the node itself.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHopResidual(nn.Module):
    def __init__(self, kind='hop1', hops=4, alpha=0.3, struct_dim=3, width=16):
        super().__init__()
        assert kind in ('hop1', 'hop2', 'ppr', 'adaptive')
        self.kind, self.alpha = kind, alpha
        self.hops = {'hop1': 1, 'hop2': 2}.get(kind, hops)
        prior = torch.tensor([(1 - alpha) ** k for k in range(self.hops)])
        self.register_buffer('log_prior', torch.log(prior / prior.sum()))
        if kind == 'adaptive':
            self.gate = nn.Sequential(nn.Linear(struct_dim, width), nn.ReLU(), nn.Linear(width, self.hops))
            nn.init.zeros_(self.gate[2].weight); nn.init.zeros_(self.gate[2].bias)

    def weights(self, struct=None):
        """[N,K] per-node hop weights (adaptive) or [K] shared weights."""
        if self.kind == 'hop1':
            return torch.tensor([1.0], device=self.log_prior.device)
        if self.kind == 'hop2':
            return torch.tensor([0.5, 0.5], device=self.log_prior.device)
        if self.kind == 'ppr':
            return F.softmax(self.log_prior, 0)
        return F.softmax(self.log_prior[None] + self.gate(struct), 1)

    def forward(self, h, P, struct=None):
        w = self.weights(struct)
        z, ref = h, 0.0
        for k in range(self.hops):
            z = torch.sparse.mm(P, z)
            ref = ref + (w[:, k:k + 1] if w.dim() == 2 else w[k]) * z
        return h - ref
