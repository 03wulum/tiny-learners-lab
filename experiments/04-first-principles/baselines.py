"""Baselines: they get far more data than the composer, and the same few-shot interface.

ICLTransformer  - MLC-style meta-learner (Lake & Baroni 2023): reads k worked examples + a query as
                  tokens and answers in context. Meta-trained on 14 arithmetic operations.
PathGNN         - Neural Bellman-Ford style path network (Zhu et al. 2021): propagates a query from the
                  start person along parent and child edges (both directions, so 'undo' is free).
                  Meta-trained on hundreds of fresh families; a new relation = a new relation vector
                  fitted to the k example people, everything else frozen."""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import worlds as W
from worlds import N, UNDEF

BLANK, ARROW, SEP = N, N + 1, N + 2
VOCAB = N + 3
MAX_K = 8


# ============================================================================ arithmetic
class ICLTransformer(nn.Module):
    def __init__(self, d=32, layers=2, heads=2, ff=64):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(5 * MAX_K + 3, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N))

    def forward(self, seq):
        h = self.tok(seq) + self.pos(torch.arange(seq.shape[1], device=seq.device))
        return self.out(self.enc(h)[:, -1])


SIZES_ICL = dict(small=dict(d=32, layers=2, heads=2, ff=64), large=dict(d=128, layers=4, heads=4, ff=384))


class OpBank:
    """Valid (a, b, y) triples of each operation, padded, on device, for fast episode sampling."""

    def __init__(self, names, device):
        xs, ys, cnt = [], [], []
        for name in names:
            arity, t = W.arith_table(name)
            v = W.valid_inputs(arity, t)
            xs.append([(a, BLANK if b is None else b) for a, b in v])
            ys.append([W.lookup(arity, t, x) for x in v])
            cnt.append(len(v))
        m = max(cnt)
        self.X = torch.zeros(len(names), m, 2, dtype=torch.long)
        self.Y = torch.zeros(len(names), m, dtype=torch.long)
        for i in range(len(names)):
            self.X[i, :cnt[i]] = torch.tensor(xs[i])
            self.Y[i, :cnt[i]] = torch.tensor(ys[i])
        self.X, self.Y = self.X.to(device), self.Y.to(device)
        self.cnt = torch.tensor(cnt, device=device)

    def batch(self, B, k):
        dev = self.X.device
        op = torch.randint(len(self.cnt), (B,), device=dev)
        idx = (torch.rand(B, k + 1, device=dev) * self.cnt[op, None]).long()
        x = self.X[op[:, None], idx]                     # B, k+1, 2
        y = self.Y[op[:, None], idx]                     # B, k+1
        return encode_batch(x, y), y[:, -1]


def encode_batch(x, y):
    """x: (B, k+1, 2) with BLANK for unary; y: (B, k+1). Query answer is not shown."""
    B, kp1, _ = x.shape
    dev = x.device
    ar = torch.full((B, kp1, 1), ARROW, device=dev)
    sep = torch.full((B, kp1, 1), SEP, device=dev)
    ex = torch.cat([x[:, :-1], ar[:, :-1], y[:, :-1, None], sep[:, :-1]], -1).reshape(B, -1)
    q = torch.cat([x[:, -1], ar[:, -1]], -1)
    return torch.cat([ex, q], 1)


def encode_episodes(episodes, device):
    """episodes from worlds.arith_episodes (same k) -> token tensor, answers."""
    x = torch.tensor([[(a, BLANK if b is None else b) for (a, b), _ in ex] + [(q[0], BLANK if q[1] is None else q[1])]
                      for ex, q, _ in episodes], device=device)
    y = torch.tensor([[v for _, v in ex] + [ans] for ex, _, ans in episodes], device=device)
    return encode_batch(x, y), y[:, -1]


def _cosine(opt, base, frac):
    for g in opt.param_groups:
        g["lr"] = base * (min(1.0, frac * 20) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))


def train_icl(op_names, size, device, seconds, seed=0, batch=1024, lr=None, log=print):
    torch.manual_seed(seed)
    m = ICLTransformer(**SIZES_ICL[size]).to(device)
    bank = OpBank(op_names, device)
    base = lr or (2e-3 if size == "small" else 6e-4)
    opt = torch.optim.AdamW(m.parameters(), lr=base, weight_decay=0.01)
    use_amp = device.type == "cuda"
    t0, step = time.time(), 0
    while True:
        frac = (time.time() - t0) / seconds
        if frac >= 1:
            break
        _cosine(opt, base, frac)
        k = int(torch.randint(1, MAX_K + 1, (1,)))
        seq, ans = bank.batch(batch, k)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(m(seq).float(), ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 500 == 0:
            log(f"  icl-{size} step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
    return m, step


@torch.no_grad()
def icl_answer(m, episodes, device, chunk=2048):
    out = []
    for i in range(0, len(episodes), chunk):
        seq, _ = encode_episodes(episodes[i:i + chunk], device)
        out.append(m(seq).argmax(-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, int)


# ============================================================================ family
REL_ALL = W.REL_TRAIN + W.REL_TEST
SIZES_GNN = dict(small=dict(d=24, h=48), large=dict(d=128, h=256))


class PathGNN(nn.Module):
    def __init__(self, n_rel=len(REL_ALL), d=24, h=48, layers=5):
        super().__init__()
        self.rel = nn.Embedding(n_rel, d)
        self.gender = nn.Embedding(2, d)
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(4 * d, h), nn.ReLU(), nn.Linear(h, d)) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.out = nn.Sequential(nn.Linear(2 * d, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, P, G, mask, src, r):
        """P: (B, n, n) float parent adjacency, G: (B, n) long, mask: (B, n) bool, src: (B,), r: (B, d).
        Returns logits (B, n): is node y in relation r to src?"""
        B, n, _ = P.shape
        h = torch.zeros(B, n, r.shape[-1], device=P.device)
        h[torch.arange(B), src] = r
        g = self.gender(G)
        for f, ln in zip(self.layers, self.norms):
            up = torch.bmm(P, h)                      # node y gathers from its parents
            down = torch.bmm(P.transpose(1, 2), h)    # node y gathers from its children
            h = ln(h + f(torch.cat([h, up, down, g], -1)))
        logit = self.out(torch.cat([h, r[:, None].expand(-1, n, -1)], -1)).squeeze(-1)
        return logit.masked_fill(~mask, -1e4)


class FamilyBank:
    """Many generated families, padded, with all relation tables, on device."""

    def __init__(self, seeds, device, n_max=200):
        F_, R = len(seeds), len(REL_ALL)
        self.P = torch.zeros(F_, n_max, n_max)
        self.G = torch.zeros(F_, n_max, dtype=torch.long)
        self.mask = torch.zeros(F_, n_max, dtype=torch.bool)
        self.T = torch.zeros(F_, R, n_max, n_max, dtype=torch.bool)
        for i, s in enumerate(seeds):
            P, G = W.family(s, max_people=n_max)
            n = len(P)
            rel = W.relations(P, G)
            self.P[i, :n, :n] = torch.tensor(P, dtype=torch.float32)
            self.G[i, :n] = torch.tensor(G.astype(np.int64))
            self.mask[i, :n] = True
            for j, name in enumerate(REL_ALL):
                self.T[i, j, :n, :n] = torch.tensor(rel[name])
        self.P, self.G, self.mask, self.T = (t.to(device) for t in (self.P, self.G, self.mask, self.T))

    def batch(self, B, rel_ids):
        dev = self.P.device
        f = torch.randint(len(self.P), (B,), device=dev)
        r = rel_ids[torch.randint(len(rel_ids), (B,), device=dev)]
        # pick a start person; half the time one with a non-empty answer
        rows = self.T[f, r]                                         # B, n, n
        nonempty = rows.any(-1) & self.mask[f]
        w = self.mask[f].float() + 3.0 * nonempty.float()
        src = torch.multinomial(w, 1).squeeze(1)
        return f, r, src, rows[torch.arange(B, device=dev), src].float()


def gnn_loss(m, bank, f, r_vec, src, target):
    logit = m(bank.P[f], bank.G[f], bank.mask[f], src, r_vec)
    pw = torch.tensor(5.0, device=logit.device)
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw, reduction="none")[bank.mask[f]].mean()


def train_gnn(rel_names, size, device, seconds, n_families=300, seed=0, batch=128, log=print):
    torch.manual_seed(seed)
    bank = FamilyBank(list(range(1000, 1000 + n_families)), device)
    m = PathGNN(**SIZES_GNN[size]).to(device)
    rel_ids = torch.tensor([REL_ALL.index(r) for r in rel_names], device=device)
    base = 3e-3 if size == "small" else 1e-3
    opt = torch.optim.AdamW(m.parameters(), lr=base, weight_decay=0.0)
    t0, step = time.time(), 0
    while True:
        frac = (time.time() - t0) / seconds
        if frac >= 1:
            break
        _cosine(opt, base, frac)
        f, r, src, tgt = bank.batch(batch, rel_ids)
        loss = gnn_loss(m, bank, f, m.rel(r), src, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 300 == 0:
            log(f"  gnn-{size} step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    return m, step


def _graph(P, G, device):
    n = len(P)
    return (torch.tensor(P, dtype=torch.float32, device=device)[None], torch.tensor(G.astype(np.int64), device=device)[None],
            torch.ones(1, n, dtype=torch.bool, device=device))


@torch.no_grad()
def gnn_predict(m, P, G, rel_vecs, device):
    """rel_vecs: (E, d). Returns (E, n, n) bool: predicted answer rows for every start person."""
    Pg, Gg, mk = _graph(P, G, device)
    n = len(P)
    out = []
    for e in range(len(rel_vecs)):
        src = torch.arange(n, device=device)
        logit = m(Pg.expand(n, -1, -1), Gg.expand(n, -1), mk.expand(n, -1), src, rel_vecs[e:e + 1].expand(n, -1))
        out.append((logit > 0).cpu())
    return torch.stack(out).numpy()


def gnn_fewshot(m, P, G, R_list, people_list, device, steps=200, lr=0.05):
    """Fit one new relation vector per episode to that episode's k example people (all else frozen).
    R_list[e]: true relation table, people_list[e]: example people. Returns (E, d) vectors."""
    for p in m.parameters():
        p.requires_grad_(False)
    Pg, Gg, mk = _graph(P, G, device)
    E = len(R_list)
    init = m.rel.weight.mean(0, keepdim=True).repeat(E, 1)
    vec = nn.Parameter(init.clone())
    src = torch.tensor([x for ps in people_list for x in ps], device=device)
    eid = torch.tensor([e for e, ps in enumerate(people_list) for _ in ps], device=device)
    tgt = torch.tensor(np.stack([R_list[e][x] for e, ps in enumerate(people_list) for x in ps]), dtype=torch.float32, device=device)
    B = len(src)
    opt = torch.optim.Adam([vec], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            logit = m(Pg.expand(B, -1, -1), Gg.expand(B, -1), mk.expand(B, -1), src, vec[eid])
            loss = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=torch.tensor(5.0, device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    for p in m.parameters():
        p.requires_grad_(True)
    return vec.detach()
