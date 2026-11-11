"""Contribution 1 - Heterophily-aware backbone (H2GCN / ego-neighbour-separated).

The paper's GCN layer computes  act(A_hat X W)  with A_hat = D^-1/2 A D^-1/2 + I, i.e. the node's own
features are *averaged into* its neighbours'. For structural anomalies (heterophilous neighbourhoods) that
smoothing erases exactly the signal the residual  r_v = h_v - mean(h_nbrs)  is trying to measure.

H2GCN's three design principles are used here:
  D1  ego / neighbour separation : the node's own features get their own weight matrix (no self-loop mixing)
  D2  higher-order neighbourhoods: 1-hop and 2-hop aggregates, each with its own weights
  D3  combination of intermediate reps: ego + hop-1 + hop-2 are combined (summing three separately
      projected terms is mathematically identical to H2GCN's concat followed by a linear layer).

Layer:  out = act( X W_ego + (A_n X) W_1 + (A_n^2 X) W_2 + b ),   A_n = A_hat - I  (neighbours only).
Note: A_n^2 is a 2-step *walk* operator (it includes returns to the ego), a cheap approximation of the strict
2-hop neighbour set used in the original H2GCN.

Drop-in for the paper's `GCN(in_ft, out_ft, act)`: same constructor, same forward(seq, adj, sparse) contract
(seq: [1,N,d]; adj: [1,N,N] dense or an [N,N] sparse tensor when sparse=True).
"""
import torch
import torch.nn as nn


class H2GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True, hops=2):
        super().__init__()
        self.fc_ego = nn.Linear(in_ft, out_ft, bias=False)
        self.fc_hop = nn.ModuleList([nn.Linear(in_ft, out_ft, bias=False) for _ in range(hops)])
        self.act = nn.PReLU() if act == 'prelu' else act
        self.bias = nn.Parameter(torch.zeros(out_ft)) if bias else None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _neighbour_prop(adj, z, sparse):
        """(A_hat - I) z  - neighbour-only propagation (removes the self-loop that GCN mixes in)."""
        if sparse:
            return torch.unsqueeze(torch.spmm(adj, torch.squeeze(z, 0)), 0) - z
        return torch.bmm(adj, z) - z

    def forward(self, seq, adj, sparse=False):
        out = self.fc_ego(seq)
        for k, fc in enumerate(self.fc_hop):
            z = fc(seq)
            for _ in range(k + 1):
                z = self._neighbour_prop(adj, z, sparse)
            out = out + z
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)


def use_h2gcn_backbone(model, in_ft=None, hidden=None):
    """In-place swap of the paper's 2-layer GCN (model.gcn1 / model.gcn2) for H2GCN layers."""
    in_ft = in_ft or model.gcn1.fc.in_features
    hidden = hidden or model.gcn1.fc.out_features
    model.gcn1 = H2GCNLayer(in_ft, hidden, 'prelu')
    model.gcn2 = H2GCNLayer(hidden, hidden, 'prelu')
    return model
