"""Contribution 1 - Mixture-of-Experts abnormal prototypes.

The paper learns ONE abnormal prototype p_a (a noise vector pushed through a learned linear+ReLU layer). Heterogeneous anomalies
(different fraud patterns) cannot all sit near one vector. Here there are K abnormal experts

    p_k = ReLU( W_abn eps_k + delta_k )      (shared paper layer W_abn, per-expert noise eps_k and learnable offset delta_k)

and a light gate routes a node's (L2-normalised) residual to the experts:   g_v = softmax( Linear(r_v / |r_v|) / T ).
Score (zero-shot):  s_v = sum_k g_vk * cos(r_v, p_k)     (the paper's cosine score, gated).

Training loss for labelled anomalies (replaces the single-prototype alignment):
    L = sum_k g_vk * ||p_k - r_v||                       soft assignment to experts (EM-like)
      + lb * K * sum_k f_k P_k                           Switch-Transformer load balancing (f: hard share, P: mean gate prob)
      + div * mean_{k!=l} cos(p_k, p_l)                  keeps experts from collapsing onto one direction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEPrototypes(nn.Module):
    def __init__(self, hid=400, K=4, temp=1.0, lb_weight=0.1, div_weight=0.1):
        super().__init__()
        self.K, self.temp, self.lb_w, self.div_w = K, temp, lb_weight, div_weight
        self.delta = nn.Parameter(torch.randn(K, hid) * 0.1)
        self.gate = nn.Linear(hid, K)

    def experts(self, fc_abn, act, eps):
        """eps [K,H] noise -> expert prototypes [K,H] using the paper's shared abnormal layer."""
        return act(fc_abn(eps) + self.delta)

    def route(self, r):
        return F.softmax(self.gate(F.normalize(r, dim=1)) / self.temp, dim=1)

    def score(self, r, P):
        """Gated cosine score. Returns (score [N], gate [N,K], cos [N,K])."""
        cos = F.normalize(r, dim=1) @ F.normalize(P, dim=1).t()
        g = self.route(r)
        return (g * cos).sum(1), g, cos

    def loss(self, r_abn, P):
        """r_abn [M,H] residuals of labelled anomalies, P [K,H] experts."""
        g = self.route(r_abn)
        dist = torch.cdist(r_abn, P)                               # [M,K]
        align = (g * dist).sum(1).mean()
        f = F.one_hot(g.argmax(1), self.K).float().mean(0)
        lb = self.K * (f * g.mean(0)).sum()
        Pn = F.normalize(P, dim=1)
        off = Pn @ Pn.t() - torch.eye(self.K, device=P.device)
        div = off.sum() / (self.K * (self.K - 1))
        return align + self.lb_w * lb + self.div_w * div, dict(align=float(align), lb=float(lb), div=float(div))
