"""Structural attacks on REAL graphs (the graphs are real; only the edge set is perturbed, labels/features untouched).

random_rewire : replace a fraction of existing edges with uniformly random node pairs  (global structural noise)
camouflage    : connect every anomaly to `m` random NORMAL nodes (the evasion pattern the paper's residual cannot see)
Ground-truth labels are used ONLY to build the attack (the adversary's knowledge), never by the detector.
"""
import numpy as np, torch
from utils.data import graph_from_edges, dev


def _canon(pairs):
    pairs = pairs[:, pairs[0] != pairs[1]]
    lo, hi = pairs.min(0).values, pairs.max(0).values
    return torch.stack([lo, hi])


def _unique(eu, N):
    key = torch.unique(eu[0] * N + eu[1])
    return torch.stack([key // N, key % N])


def random_rewire(G, frac, seed):
    g = torch.Generator(device=dev).manual_seed(10_000 + seed)
    eu, N = G['eu'], G['N']
    E = eu.shape[1]
    keep = torch.rand(E, device=dev, generator=g) >= frac
    n_new = int((~keep).sum())
    new = _canon(torch.randint(0, N, (2, n_new), device=dev, generator=g))
    return graph_from_edges(G, _unique(torch.cat([eu[:, keep], new], 1), N), with_struct=True)


def camouflage(G, m, seed):
    g = torch.Generator(device=dev).manual_seed(20_000 + seed)
    eu, N, y = G['eu'], G['N'], torch.as_tensor(G['y'], device=dev)
    if m <= 0:
        return G
    an, nm = torch.nonzero(y == 1).squeeze(1), torch.nonzero(y == 0).squeeze(1)
    src = an.repeat_interleave(m)
    dst = nm[torch.randint(0, len(nm), (len(src),), device=dev, generator=g)]
    new = _canon(torch.stack([src, dst]))
    return graph_from_edges(G, _unique(torch.cat([eu, new], 1), N), with_struct=True)


ATTACKS = {'rewire': random_rewire, 'camouflage': camouflage}
