"""Training + evaluation entry points for Anomaly2 (the repo has no train.py; this is the integration layer).

Hook points for the four extensions (everything else is the paper's pipeline):
  C1  build_model(moe_k=K)      -> loss in pretrain(): model.moe.loss(...) replaces the single-prototype alignment
  C2  utils/unsupervised_tta.py -> test_time_adapt() replaces evaluate() on the unlabeled target graph
  C3  build_model(purify=True) + TrainCfg(attack_aug=True)  -> edge purifier sits in GFM.encode before message passing
  C4  build_model(residual='ppr'|'adaptive'|'hop2')          -> GFM.encode computes the residual with the chosen operator
"""
from dataclasses import dataclass
import random
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from models.core import GFM
from models.edge_purification import EdgePurifier
from utils.data import dev, graph_from_edges
from utils.attacks import _canon, _unique


LR_EXT = dict(moe=1e-3, purifier=1e-2, resid=1e-2)     # per-module learning rates chosen on Facebook held-out nodes only


@dataclass
class TrainCfg:
    epochs: int = 301
    lr: float = 1e-4                  # paper's learning rate (backbone + the paper's own prompt layers)
    lr_ext: float = None              # learning rate of the NEW modules; None -> LR_EXT (selected on the SOURCE graph, scripts/select_lr_ext.py)
    hid: int = 400
    attack_aug: bool = False          # adversarial augmentation of the TRAINING graph each epoch (needed by the purifier)
    atk_noise: float = 0.1            # random edges added, as a fraction of |E|   (selected on the source graph)
    atk_camo: int = 10                # camouflage edges added per labelled anomaly
    atk_prob: float = 0.5             # probability that an epoch trains on the attacked graph (else the clean one)
    edge_w: float = 1.0               # weight of the purifier's edge-level BCE


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_model(moe_k=0, purify=False, residual='hop1', hid=400):
    return GFM(hid, moe_k=moe_k, purify=purify, residual=residual).to(dev)


def training_attack(G, cfg, gen):
    """Attack the training graph: random noise edges + anomaly->normal camouflage edges. Returns a graph view whose
    G['injected'] marks the attack edges (known to us, hence usable as a target for the purifier)."""
    N, eu = G['N'], G['eu']
    E = eu.shape[1]
    noise = _canon(torch.randint(0, N, (2, int(cfg.atk_noise * E)), device=dev, generator=gen))
    y = G['y_t']
    an, nm = torch.nonzero(y == 1).squeeze(1), torch.nonzero(y == 0).squeeze(1)
    src = an.repeat_interleave(cfg.atk_camo)
    camo = _canon(torch.stack([src, nm[torch.randint(0, len(nm), (len(src),), device=dev, generator=gen)]]))
    allp = torch.cat([eu, noise, camo], 1)
    is_orig = torch.cat([torch.ones(E, device=dev), torch.zeros(allp.shape[1] - E, device=dev)])
    key = allp[0] * N + allp[1]
    uniq, inv = torch.unique(key, return_inverse=True)
    orig = torch.zeros(len(uniq), device=dev).index_add(0, inv, is_orig) > 0
    V = graph_from_edges(G, torch.stack([uniq // N, uniq % N]))
    V['injected'] = (~orig).float()
    return V


def pretrain(model, Gtr, seed, cfg: TrainCfg):
    """Paper's supervised pre-training on the source graph (30% labelled nodes) with the extension losses added."""
    N = Gtr['N']
    idx = torch.LongTensor(np.random.RandomState(seed).permutation(N)[:int(0.3 * N)]).to(dev)
    y = Gtr['y_t'][idx]
    ext = [('moe', model.moe), ('purifier', model.purifier), ('resid', model.resid)]
    ext = [(n, m) for n, m in ext if m is not None and sum(p.numel() for p in m.parameters()) > 0]
    ext_ids = {id(p) for _, m in ext for p in m.parameters()}
    groups = [dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in ext_ids], lr=cfg.lr)]
    groups += [dict(params=list(m.parameters()), lr=cfg.lr_ext if cfg.lr_ext is not None else LR_EXT[n]) for n, m in ext]
    opt = torch.optim.Adam(groups)
    gen = torch.Generator(device=dev).manual_seed(seed)
    last = {}
    for ep in range(cfg.epochs):
        model.train(); opt.zero_grad()
        nraw = torch.randn(model.hid, device=dev)
        araw = torch.randn(model.K, model.hid, device=dev)
        G = training_attack(Gtr, cfg, gen) if (cfg.attack_aug and torch.rand(1, device=dev, generator=gen).item() < cfg.atk_prob) else Gtr
        h, r, aux = model.encode(G)
        loss = F.binary_cross_entropy_with_logits(model.core.fc1(h).squeeze(-1)[idx], y)
        pn, pa = model.prototypes(nraw, araw)
        ri = r[idx]
        if (y == 1).any():
            if model.moe is not None:                                  # C1: MoE-aware prototype loss
                la, last = model.moe.loss(ri[y == 1], pa); loss = loss + la
            else:                                                      # paper's single-prototype alignment
                loss = loss + (pa[0] - ri[y == 1]).norm(dim=1).mean()
        if (y == 0).any():
            loss = loss + 0.1 * (pn - ri[y == 0]).norm(dim=1).mean()
        if model.purifier is not None and 'injected' in G:             # C3: edge-level supervision from the known attack
            loss = loss + cfg.edge_w * EdgePurifier.edge_loss(aux['logit'], G['injected'])
        loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def evaluate(model, G, seed, idx=None):
    """Zero-shot evaluation on all nodes of G (or only the nodes `idx`); fixed per-seed prompt draw, as in the earlier benchmarks."""
    model.eval()
    gen = torch.Generator().manual_seed(seed + 1)
    nraw = torch.randn(model.hid, generator=gen).to(dev)
    araw = torch.randn(model.K, model.hid, generator=gen).to(dev)
    _, r, _ = model.encode(G)
    pn, pa = model.prototypes(nraw, araw)
    s, _, ex = model.scores(r, pn, pa)
    y, sc = G['y'], s.cpu().numpy()
    sel = slice(None) if idx is None else np.asarray(idx)
    out = dict(auroc=roc_auc_score(y[sel], sc[sel]), auprc=average_precision_score(y[sel], sc[sel]))
    if 's_max' in ex:
        out['auroc_max'] = roc_auc_score(y[sel], ex['s_max'].cpu().numpy()[sel])
    return out
