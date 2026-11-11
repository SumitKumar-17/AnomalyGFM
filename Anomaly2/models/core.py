"""GFM: the paper's AnomalyGFM (its own `Model` supplies gcn1/gcn2/fc1/prompt layers) with pluggable extensions.

  moe_k > 0     -> C1  K abnormal expert prototypes + gate           (models/moe_prototypes.py)
  purify        -> C3  differentiable edge dropping before message passing (models/edge_purification.py)
  residual      -> C4  'hop1' (paper) | 'hop2' | 'ppr' | 'adaptive'  (models/adaptive_residuals.py)
  input_prompt  -> C2  graph prompt on the input features, the only thing updated by test-time adaptation (utils/unsupervised_tta.py)
With defaults (moe_k=0, purify=False, residual='hop1') this IS the paper's zero-shot model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data import zs_model
from models.moe_prototypes import MoEPrototypes
from models.edge_purification import EdgePurifier
from models.adaptive_residuals import MultiHopResidual


class GFM(nn.Module):
    def __init__(self, hid=400, moe_k=0, purify=False, residual='hop1', hops=4, alpha=0.3):
        super().__init__()
        self.hid = hid
        self.core = zs_model.Model(8, 8, hid, 'prelu', 1, 'avg')          # created first -> identical init to the paper's
        self.moe = MoEPrototypes(hid, moe_k) if moe_k > 0 else None
        self.purifier = EdgePurifier() if purify else None
        self.resid = MultiHopResidual(residual, hops, alpha)
        self.input_prompt = None

    @property
    def K(self):
        return self.moe.K if self.moe is not None else 1

    # ---- forward pieces -------------------------------------------------------------------
    def encode(self, G, X8=None):
        """-> (embeddings h [N,H], residuals r [N,H], aux)"""
        X8 = G['X8'] if X8 is None else X8
        x = self.input_prompt(X8) if self.input_prompt is not None else X8
        aux = None
        if self.purifier is not None:
            A_hat, P, aux = self.purifier(X8, G['deg'], G['eu'], G['N'])
        else:
            A_hat, P = G['A_hat'], G['Wn']
        c = self.core
        h = c.gcn2(c.gcn1(x[None], A_hat, True), A_hat, True).squeeze(0)
        return h, self.resid(h, P, G.get('struct')), aux

    def prototypes(self, nraw, araw):
        """nraw [H], araw [K,H] noise -> (p_normal [H], p_abnormal [K,H])"""
        c = self.core
        pn = c.act(c.fc_normal_prompt(nraw))
        pa = self.moe.experts(c.fc_abnormal_prompt, c.act, araw) if self.moe is not None else c.act(c.fc_abnormal_prompt(araw))
        return pn, pa

    def scores(self, r, pn, pa):
        """-> (anomaly score [N] (higher = more anomalous), cos to normal prototype [N], extras)"""
        rn = F.normalize(r, dim=1)
        cn = rn @ F.normalize(pn, dim=0)
        if self.moe is not None:
            s, g, cos = self.moe.score(r, pa)
            return s, cn, dict(gate=g, cos=cos, s_max=cos.max(1).values)
        return rn @ F.normalize(pa[0], dim=0), cn, {}

    def n_params(self):
        c = self.core
        mods = [c.gcn1, c.gcn2, c.fc1, c.fc_normal_prompt, c.fc_abnormal_prompt, self.moe, self.purifier, self.resid]
        return sum(p.numel() for m in mods if m is not None for p in m.parameters())
