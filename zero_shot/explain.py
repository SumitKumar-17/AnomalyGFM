"""Explainable zero-shot GAD on top of AnomalyGFM.

Trains AnomalyGFM on --dataset_train, scores --dataset_test zero-shot, and for the top flagged
nodes writes an exact decomposition of *why* each was flagged plus a plain-English explanation.
Output: ../results/explanations_<train>_to_<test>.json  (consumed by the web app).

Score used (same as the paper's abnormal measure): s_v = exp(cos(r_v, p_abn)), where
r_v = h_v - sum_u w_vu h_u is the residual of node v from its neighbours.
"""
import argparse, json, os, random, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from model import Model
from utils import load_mat, preprocess_features, normalize_adj

p = argparse.ArgumentParser()
p.add_argument('--dataset_train', default='Facebook_svd')
p.add_argument('--dataset_test', default='amazon_svd')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--epochs', type=int, default=301)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--embedding_dim', type=int, default=400)
p.add_argument('--flag_pct', type=float, default=5.0, help='flag the top X%% highest-scoring nodes')
p.add_argument('--max_nodes', type=int, default=300, help='max flagged nodes to explain in detail')
p.add_argument('--top_neighbors', type=int, default=8)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
FEATURE_NAMES = [f'component {i + 1}' for i in range(8)]  # SVD-aligned features are abstract

# ---------------- data (mirrors run_abnormal.py) ----------------
adj_tr, adj_te, f_tr, f_te, y_tr, y_val, y_te, idx_tr, idx_val = load_mat(args.dataset_train, args.dataset_test)
f_tr, _ = preprocess_features(f_tr)
f_te, _ = preprocess_features(f_te)
raw_tr, raw_te = sp.csr_matrix(adj_tr), sp.csr_matrix(adj_te)


def to_dense_inputs(adj, feat):
    nadj = (normalize_adj(adj) + sp.eye(adj.shape[0])).todense()
    return (torch.FloatTensor(np.asarray(feat)[np.newaxis]),
            torch.FloatTensor(np.asarray(nadj)[np.newaxis]),
            torch.FloatTensor(np.asarray(adj.todense())))


X_tr, A_tr, R_tr = to_dense_inputs(adj_tr, f_tr)
X_te, A_te, R_te = to_dense_inputs(adj_te, f_te)
y_tr_t = torch.FloatTensor(y_tr)

# ---------------- train (same objective as the paper's code) ----------------
model = Model(X_tr.shape[2], 8, args.embedding_dim, 'prelu', 1, 'avg')
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([1.0]))
t0 = time.time()
for ep in range(args.epochs):
    model.train(); opt.zero_grad()
    np_raw, ap_raw = torch.randn(args.embedding_dim), torch.randn(args.embedding_dim)
    logits, _, emb, emb_res, nprm, aprm, _ = model(X_tr, A_tr, R_tr, np_raw, ap_raw)
    loss_bce = torch.mean(bce(torch.squeeze(logits[:, idx_tr]), y_tr_t))
    res = emb_res[:, idx_tr, :]
    d_n = torch.sqrt(torch.sum((nprm - res[:, y_tr_t == 0, :]) ** 2, dim=2))
    d_a = torch.sqrt(torch.sum((aprm - res[:, y_tr_t == 1, :]) ** 2, dim=2))
    loss = loss_bce + torch.mean(d_a) + 0.1 * torch.mean(d_n)
    loss.backward(); opt.step()
print(f'trained {args.epochs} epochs in {time.time() - t0:.1f}s', flush=True)

# ---------------- zero-shot inference on the target graph ----------------
model.eval()
g = torch.Generator().manual_seed(args.seed + 1)  # fixed prompt draw -> reproducible scores
with torch.no_grad():
    _, _, emb, emb_res, nprm, aprm, _ = model(
        X_te, A_te, R_te, torch.randn(args.embedding_dim, generator=g), torch.randn(args.embedding_dim, generator=g))
H = emb.squeeze(0)            # [N, h] node embeddings
Rres = emb_res.squeeze(0)     # [N, h] residuals
pa, pn = F.normalize(aprm, dim=0), F.normalize(nprm, dim=0)
Rn = F.normalize(Rres, dim=1)
cos_a, cos_n = (Rn @ pa).numpy(), (Rn @ pn).numpy()
score = np.exp(cos_a)
y = np.asarray(y_te).ravel()
auc, ap = roc_auc_score(y, score), average_precision_score(y, score)
print(f'{args.dataset_test}: zero-shot AUROC={auc:.4f} AUPRC={ap:.4f}', flush=True)

N = len(y)
order = np.argsort(-score)
rank_pct = np.empty(N); rank_pct[order] = 100.0 * (np.arange(N) + 1) / N       # 0.5 => top 0.5%
res_norm = Rres.norm(dim=1).numpy()
res_norm_pct = 100.0 * (np.argsort(np.argsort(res_norm)) + 1) / N               # higher = deviates more
deg = np.asarray(raw_te.sum(1)).ravel()
deg_pct = 100.0 * (np.argsort(np.argsort(deg)) + 1) / N

# row-normalised neighbour weights (self-loops removed), exactly as in model.forward
Wn = raw_te.tolil(); Wn.setdiag(0); Wn = sp.csr_matrix(Wn)
rs = np.asarray(Wn.sum(1)).ravel(); rs[rs == 0] = 1
Wn = sp.diags(1.0 / rs) @ Wn
Xf = np.asarray(f_te)
feat_diff = Xf - np.asarray(Wn @ Xf)                    # node features minus neighbour mean
feat_std = feat_diff.std(0) + 1e-9


# ---------------- domain knowledge ----------------
# NOTE: entity/edge meanings come from the datasets' published descriptions (general knowledge), NOT from the .mat
# files, which only hold SVD-compressed 8-dim features. Product/merchant IDs are not available in this data.
DOMAINS = {
    'amazon': dict(entity='reviewer account', group='accounts that reviewed the same products', n_group='co-reviewer',
        anomaly='a fraudulent / fake-review account', signal='coordinated review behaviour',
        pattern='its review behaviour is out of line with the accounts it shares products with',
        hub='accounts that review very many products are typical of paid-review farms',
        actions={'high': ['Suspend the account\'s ability to post reviews pending manual review',
                          'Down-weight or hide its reviews in product ratings until verified',
                          'Audit the products it reviewed for rating inflation / suppression'],
                 'medium': ['Queue for trust-and-safety review', 'Require verified-purchase for its next reviews'],
                 'review': ['Add to watchlist; re-score after new activity']},
        ring='Several of its co-reviewers are flagged too, which suggests a coordinated review ring rather than a lone account — review the whole group together.'),
    'yelp': dict(entity='review', group='reviews linked by the same user / business / time', n_group='related review',
        anomaly='a spam / fake review', signal='review-spam behaviour',
        pattern='it looks unlike the reviews it is linked to',
        hub='a review linked to unusually many others often belongs to a campaign',
        actions={'high': ['Filter the review from the business\'s public rating', 'Investigate the reviewer and the business for incentivised reviews'],
                 'medium': ['Send to moderation queue'], 'review': ['Monitor; do not remove automatically']},
        ring='Related reviews are flagged too, which is typical of a campaign of fake reviews for one business.'),
    'reddit': dict(entity='user', group='users with linked posting activity', n_group='linked user',
        anomaly='a user likely to be banned (abusive / spam)', signal='abusive or spam posting behaviour',
        pattern='its activity differs from the users it interacts with',
        hub='an account active across many threads is typical of spam bots',
        actions={'high': ['Send to moderators for review / rate-limit posting', 'Check for ban-evasion linked accounts'],
                 'medium': ['Add to moderation queue'], 'review': ['Watchlist']},
        ring='Linked users are flagged too — possible coordinated spam or brigading.'),
    'tolokers': dict(entity='crowd worker', group='workers who completed the same tasks', n_group='co-worker',
        anomaly='a low-quality or bot-like worker (likely to be banned)', signal='low-quality annotation behaviour',
        pattern='its work profile differs from the workers on the same tasks',
        hub='a worker touching very many tasks may be automated',
        actions={'high': ['Pause task assignment; run gold-question quality checks', 'Re-label tasks it completed using trusted workers'],
                 'medium': ['Increase quality-check frequency'], 'review': ['Monitor accuracy on gold tasks']},
        ring='Co-workers are flagged too — possible collusion or a shared bot farm.'),
    'questions': dict(entity='user', group='users who answered each other\'s questions', n_group='connected user',
        anomaly='an inactive / suspicious user', signal='unusual Q&A participation',
        pattern='its activity differs from the users it interacts with',
        hub='an unusually well-connected account may be spam',
        actions={'high': ['Review for spam / promotional answers'], 'medium': ['Moderation queue'], 'review': ['Watchlist']},
        ring='Connected users are flagged too — possible answer-farming group.'),
    'elliptic': dict(entity='Bitcoin transaction', group='transactions linked by payment flow', n_group='linked transaction',
        anomaly='a likely illicit transaction', signal='money-laundering / illicit-flow pattern',
        pattern='its flow pattern is out of line with the transactions it is linked to',
        hub='a transaction fanning in/out of many others can indicate mixing or layering',
        actions={'high': ['Escalate to compliance / AML analysts; consider holding or flagging the funds',
                          'Trace linked transactions and file a suspicious-activity report if confirmed'],
                 'medium': ['Enhanced due diligence on linked addresses'], 'review': ['Monitor downstream flows']},
        ring='Linked transactions are flagged too — consistent with a layering chain; trace the whole flow.'),
    't_finance': dict(entity='finance account', group='accounts with transaction relationships', n_group='linked account',
        anomaly='a likely fraudulent account', signal='fraudulent transaction behaviour',
        pattern='its transaction behaviour differs from the accounts it deals with',
        hub='an account transacting with unusually many others can indicate a mule or fraud hub',
        actions={'high': ['Freeze or limit transactions; trigger KYC re-verification', 'Review linked accounts as a group'],
                 'medium': ['Step-up authentication on risky transactions'], 'review': ['Add to risk watchlist']},
        ring='Linked accounts are flagged too — possible mule network.'),
    'facebook': dict(entity='user', group='friends / connected users', n_group='connected user',
        anomaly='a suspicious (likely fake or abusive) account', signal='fake-account behaviour',
        pattern='its profile differs from the users it is connected to',
        hub='an account with an unusually large friend list is typical of fake or spam accounts',
        actions={'high': ['Challenge the account (CAPTCHA / phone verification)', 'Limit friend requests and messaging'],
                 'medium': ['Add to review queue'], 'review': ['Watchlist']},
        ring='Its connections are flagged too — possible fake-account cluster.'),
}
DOMAIN = next((v for k, v in DOMAINS.items() if k in args.dataset_test.lower()),
              dict(entity='node', group='connected nodes', n_group='neighbour', anomaly='an anomalous node',
                   signal='anomalous behaviour', pattern='it looks unlike its neighbours',
                   hub='unusually high connectivity can be a warning sign',
                   actions={'high': ['Manual review'], 'medium': ['Queue for review'], 'review': ['Watchlist']},
                   ring='Connected nodes are flagged too — possible coordinated group.'))


def fmt_pct(x): return f'{x:.2f}%' if x < 1 else f'{x:.1f}%'


def plural(n, w): return f'{n} {w}' + ('' if n == 1 else 's')


k_flag = int(np.ceil(N * args.flag_pct / 100.0))
flag_set = np.zeros(N, bool); flag_set[order[:k_flag]] = True


def explain(v):
    row = Wn.getrow(v); nb, w = row.indices, row.data
    flagged_nb = float(flag_set[nb].mean()) if len(nb) else 0.0
    prio = 'high' if rank_pct[v] <= 1 else 'medium' if rank_pct[v] <= 3 else 'review'
    out = dict(node=int(v), score=float(score[v]), rank_pct=float(rank_pct[v]), is_anomaly=int(y[v]),
               cos_abnormal=float(cos_a[v]), cos_normal=float(cos_n[v]), degree=int(deg[v]),
               residual_pct=float(res_norm_pct[v]), degree_pct=float(deg_pct[v]), flagged_neighbor_frac=flagged_nb,
               entity=DOMAIN['entity'], priority=prio)
    # leave-one-out neighbour attribution: how much does the abnormal similarity drop if neighbour u is ignored?
    contrib = []
    if len(nb) >= 2:
        Hn = H[nb]                                                       # [d, h]
        wt = torch.tensor(w, dtype=torch.float32)[:, None]
        nbmean = (wt * Hn).sum(0)
        r_loo = H[v][None, :] - (nbmean[None, :] - wt * Hn) / (1 - wt)   # residual with each neighbour dropped
        c_loo = (F.normalize(r_loo, dim=1) @ pa).numpy()
        delta = cos_a[v] - c_loo                                          # >0: neighbour pushes toward "abnormal"
        for i in np.argsort(-delta)[:args.top_neighbors]:
            contrib.append(dict(node=int(nb[i]), effect=float(delta[i]), is_anomaly=int(y[nb[i]]),
                                weight=float(w[i])))
    out['neighbors'] = contrib
    z = feat_diff[v] / feat_std
    top_f = [dict(feature=FEATURE_NAMES[i], z=float(z[i])) for i in np.argsort(-np.abs(z))[:3] if abs(z[i]) >= 2]
    out['features'] = top_f
    # ---- plain-English explanation ----
    s = []
    s.append(f'This {DOMAIN["entity"]} (node {v}) is flagged as possibly {DOMAIN["anomaly"]}: its anomaly score ranks in the top {fmt_pct(rank_pct[v])} of the {N:,} in this graph.')
    if deg[v] == 0:
        s.append('It has no neighbours, so the comparison below is based on its own profile only.')
    else:
        if res_norm_pct[v] >= 60:
            s.append(f'Compared with its {plural(int(deg[v]), DOMAIN["n_group"])}, it deviates more than {res_norm_pct[v]:.0f}% of all nodes do: {DOMAIN["pattern"]}. '
                     f'Normal nodes tend to resemble their neighbours; this one does not.')
        else:
            s.append(f'Compared with its {plural(int(deg[v]), DOMAIN["n_group"])}, the size of its deviation is moderate (larger than {res_norm_pct[v]:.0f}% of nodes), '
                     f'but the direction of that deviation is what the model treats as the warning sign: {DOMAIN["pattern"]}.')
    s.append(f'Its deviation pattern matches the learned "abnormal" profile (similarity {cos_a[v]:+.2f}) '
             + ('far more' if cos_a[v] - cos_n[v] > 0.3 else 'more' if cos_a[v] > cos_n[v] else 'less')
             + f' than the learned "normal" profile ({cos_n[v]:+.2f}).')
    if contrib and contrib[0]['effect'] > 0.005:
        top = [c for c in contrib[:3] if c['effect'] > 0.005]
        s.append(f'{DOMAIN["n_group"].capitalize()}s that most increase its suspicion: ' + ', '.join(
            f'#{c["node"]} (+{c["effect"]:.3f})' for c in top) + '. Ignoring them would make the node look more ordinary.')
    elif len(nb) >= 2:
        s.append('No single neighbour dominates; the deviation is spread across its neighbourhood.')
    if top_f:
        s.append('On the raw input side, ' + '; '.join(
            f'{f["feature"]} is {abs(f["z"]):.1f} standard deviations {"above" if f["z"] > 0 else "below"} its neighbours'
            for f in top_f) + '.')
    if len(nb) >= 3 and flagged_nb >= 0.2:
        s.append(f'{flagged_nb * 100:.0f}% of its {DOMAIN["n_group"]}s are flagged as well' + ('. ' + DOMAIN['ring'] if flagged_nb >= 0.3 else '; worth checking as a group.'))
    if deg_pct[v] > 90: s.append(f'It is also unusually well-connected (top {100 - deg_pct[v]:.0f}% by degree); ' + DOMAIN['hub'] + '.')
    elif deg_pct[v] < 10 and deg[v] > 0: s.append('It is also weakly connected compared with most nodes.')
    out['explanation'] = ' '.join(s)
    out['business_decision'] = dict(priority=prio, actions=DOMAIN['actions'][prio],
        caveat='A model flag is a lead, not proof: have a human confirm before taking irreversible action.')
    return out


flagged = order[:k_flag]
detail = flagged[:args.max_nodes]
t1 = time.time()
nodes = [explain(int(v)) for v in detail]
print(f'explained {len(nodes)} nodes in {time.time() - t1:.1f}s', flush=True)

prec = float(y[flagged].mean())
meta = dict(train=args.dataset_train, test=args.dataset_test, seed=args.seed, n_nodes=int(N),
            n_edges=int(raw_te.nnz), anomaly_rate=float(y.mean()), auroc=float(auc), auprc=float(ap),
            flag_pct=args.flag_pct, n_flagged=int(k_flag), precision_at_flag=prec,
            note='Neighbour effects are leave-one-out changes in cosine similarity to the abnormal prototype, '
                 'holding node embeddings fixed. Input features are SVD-aligned components, hence abstract names.')
os.makedirs('../results', exist_ok=True)
fn = f'../results/explanations_{args.dataset_train}_to_{args.dataset_test}.json'
json.dump(dict(meta=meta, nodes=nodes), open(fn, 'w'))
print(f'precision among top {args.flag_pct}% flagged = {prec:.3f}; wrote {fn}')
