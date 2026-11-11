"""Contribution 3 - Label-free contrastive prototype pre-training.

No ground-truth labels are used. Each epoch the training graph is corrupted ON THE FLY (training only):
  * structural corruption : a node's neighbourhood is replaced by random neighbours of equal degree (rewiring)
  * contextual corruption : a node's features are swapped with another node's
Corrupted nodes become *pseudo-abnormal*, untouched nodes *pseudo-normal*. Two InfoNCE terms are optimised:
  (a) prototype InfoNCE : the node's residual embedding must pick its pseudo-class prototype (normal / abnormal)
      against the other prototype, with temperature tau and class re-weighting (corruptions are rare);
  (b) view InfoNCE      : an untouched node's embedding in the clean view must retrieve itself among all other
      nodes in the corrupted view (keeps normal-node structure stable while anomalies are pushed away).
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class SSLConfig:
    struct_frac: float = 0.05
    ctx_frac: float = 0.05
    tau: float = 0.2
    view_weight: float = 0.5
    view_nodes: int = 512


def build_ops(A):
    """Dense 0/1 adjacency -> (A_hat = D^-1/2 A D^-1/2 + I, Wn = row-normalised neighbour mean), both sparse."""
    N = A.shape[0]
    eye = torch.eye(N, device=A.device)
    A = A * (1 - eye)
    deg = A.sum(1)
    dinv = deg.clamp(min=1e-12).pow(-0.5) * (deg > 0)
    return (dinv[:, None] * A * dinv[None, :] + eye).to_sparse(), (A / deg.clamp(min=1)[:, None]).to_sparse()


def augment(A, X, cfg: SSLConfig, gen):
    """A: dense 0/1 [N,N]; X: [N,F]. Returns (A_aug, X_aug, pseudo_abnormal_mask[bool N])."""
    N, dev = A.shape[0], A.device
    perm = torch.randperm(N, device=dev, generator=gen)
    n_s, n_c = int(N * cfg.struct_frac), int(N * cfg.ctx_frac)
    s_idx, c_idx, c_partner = perm[:n_s], perm[n_s:n_s + n_c], perm[n_s + n_c:n_s + 2 * n_c]
    A2, X2 = A.clone(), X.clone()
    deg = A.sum(1)
    rows = (torch.rand(n_s, N, device=dev, generator=gen) < (deg[s_idx] / N)[:, None]).float()
    A2[s_idx, :] = rows
    A2[:, s_idx] = rows.t()
    A2 = ((A2 + A2.t()) > 0).float()
    A2.fill_diagonal_(0)
    X2[c_idx], X2[c_partner] = X[c_partner], X[c_idx]            # swap features between node pairs
    mask = torch.zeros(N, dtype=torch.bool, device=dev)
    mask[s_idx] = mask[c_idx] = mask[c_partner] = True
    return A2, X2, mask


def prototype_info_nce(res, pseudo, p_normal, p_abnormal, tau):
    z = F.normalize(res, dim=1)
    logits = torch.stack([z @ F.normalize(p_normal, dim=0), z @ F.normalize(p_abnormal, dim=0)], 1) / tau
    frac = pseudo.float().mean().clamp(1e-3, 1 - 1e-3)
    w = torch.stack([1 / (1 - frac), 1 / frac]) / 2               # balance rare pseudo-anomalies
    return F.cross_entropy(logits, pseudo.long(), weight=w)


def view_info_nce(h_clean, h_aug, keep_mask, tau, max_nodes, gen):
    idx = torch.nonzero(~keep_mask).squeeze(1)
    idx = idx[torch.randperm(len(idx), device=idx.device, generator=gen)[:max_nodes]]
    a, b = F.normalize(h_clean[idx], dim=1), F.normalize(h_aug[idx], dim=1)
    sim = a @ b.t() / tau
    lab = torch.arange(len(idx), device=sim.device)
    return 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.t(), lab))


def ssl_loss(encode, A_dense, X, cfg: SSLConfig, gen, nprm, aprm):
    """encode(A_hat, Wn, X) -> (emb, residual). Returns (loss, pseudo_abnormal_fraction)."""
    A2, X2, mask = augment(A_dense, X, cfg, gen)
    A_hat2, Wn2 = build_ops(A2)
    emb_a, res_a = encode(A_hat2, Wn2, X2)
    A_hat0, Wn0 = build_ops(A_dense)
    emb_o, _ = encode(A_hat0, Wn0, X)
    loss = prototype_info_nce(res_a, mask, nprm, aprm, cfg.tau) \
        + cfg.view_weight * view_info_nce(emb_o, emb_a, mask, cfg.tau, cfg.view_nodes, gen)
    return loss, float(mask.float().mean())
