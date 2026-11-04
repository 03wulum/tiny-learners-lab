"""The first-principles composer.

1. Learned building blocks (tiny nets): next-number, or parent + gender.
2. Snap each block to a lookup table: every later step is a constant-time pointer lookup.
3. Bottom-up program search over five moves, smallest first, merging programs that give the same
   table (observational equivalence), then pick the smallest program that fits the k examples.

Arithmetic moves:  undo = inv / inv1 · repeat = iter / acc · chain = compose · same-input-twice = diag
                   · fix-an-input = bind(B, small constant), plus swap.
Family moves:      undo = inv · chain = compose · filters male / female · nonself (not the start person).
The composer is never told what a symbol means: 'zero' is the number nothing comes before, constants are
the first few numbers along the learned next-number chain."""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from worlds import N, UNDEF, NONE_CLASS


# ============================================================================ learned blocks
class NextBlock(nn.Module):
    """Embedding + MLP; logits over N numbers + 'none' with tied embeddings (~7.4K params)."""

    def __init__(self, d=32, h=64):
        super().__init__()
        self.E = nn.Embedding(N + 1, d)
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, d))

    def step(self, z):
        return self.f(z)

    def decode(self, z):
        return z @ self.E.weight.T

    def forward(self, x):
        return self.decode(self.step(self.E(x)))


def train_next_block(facts, device, steps=800, seed=0, lr=3e-3):
    torch.manual_seed(seed)
    m = NextBlock().to(device)
    x = torch.tensor([f[0] for f in facts], device=device)
    y = torch.tensor([f[1] for f in facts], device=device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def snap_next(m):
    """Learned block -> table S[x] (UNDEF where the block says 'none')."""
    dev = m.E.weight.device
    s = m(torch.arange(N, device=dev)).argmax(-1).cpu().numpy()
    s[s == NONE_CLASS] = UNDEF
    return s.astype(np.int64)


@torch.no_grad()
def latent_chain_accuracy(m, steps_list):
    """No-snap ablation: compute a + b by applying the block b times in hidden space and decoding
    only at the end (no pointer clean-up between steps). Returns {b: accuracy over valid a}."""
    dev = m.E.weight.device
    out = {}
    for b in steps_list:
        a = torch.arange(0, N - b, device=dev)
        z = m.E(a)
        for _ in range(b):
            z = m.step(z)
        pred = m.decode(z).argmax(-1)
        out[b] = float((pred == a + b).float().mean())
    return out


class FamilyBlocks(nn.Module):
    """Per-person embeddings, a parent scorer and a gender head."""

    def __init__(self, n, d=32, h=64):
        super().__init__()
        self.E = nn.Embedding(n, d)
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, d))
        self.bias = nn.Parameter(torch.zeros(n))
        self.g = nn.Linear(d, 1)

    def forward(self):
        e = self.E.weight
        return self.f(e) @ e.T + self.bias, self.g(e).squeeze(-1)


def train_family_blocks(P, G, device, steps=1500, seed=0, lr=1e-2):
    torch.manual_seed(seed)
    n = len(P)
    m = FamilyBlocks(n).to(device)
    Pt = torch.tensor(P, dtype=torch.float32, device=device)
    Gt = torch.tensor(G, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        lp, lg = m()
        loss = F.binary_cross_entropy_with_logits(lp, Pt, pos_weight=torch.tensor(10.0, device=device)) \
            + F.binary_cross_entropy_with_logits(lg, Gt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def snap_family(m, soft=False):
    lp, lg = m()
    if soft:
        return torch.sigmoid(lp), torch.sigmoid(lg)
    return (lp > 0).cpu().numpy(), (lg > 0).cpu().numpy()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ============================================================================ arithmetic search
def _take(t, idx):
    out = np.full(idx.shape, UNDEF, dtype=np.int64)
    ok = idx >= 0
    out[ok] = t[idx[ok]]
    return out


def counting_chain(S):
    """Find 'zero' (defined, nothing maps to it) and walk the learned next-number chain from it."""
    img = set(S[S >= 0].tolist())
    zeros = [x for x in range(N) if x not in img and S[x] >= 0]
    if not zeros:
        return None, np.array([], dtype=np.int64), np.full(N, UNDEF)
    order, seen = [zeros[0]], {zeros[0]}
    while S[order[-1]] >= 0 and S[order[-1]] not in seen:
        order.append(int(S[order[-1]]))
        seen.add(order[-1])
    rank = np.full(N, UNDEF)
    rank[order] = np.arange(len(order))
    return zeros[0], np.array(order), rank


def _inv_u(u):
    v = np.full(N, UNDEF, dtype=np.int64)
    ok = u >= 0
    cnt = np.bincount(u[ok], minlength=N)
    v[u[ok]] = np.arange(N)[ok]
    v[cnt != 1] = UNDEF
    return v


class ArithLibrary:
    """All distinct programs up to max_size built from the learned table S."""

    def __init__(self, S, max_size=7, n_consts=10, use_inverse=True):
        self.zero, self.order, self.rank = counting_chain(S)
        self.consts = self.order[:n_consts]
        self.U, self.B = {}, {}            # key -> (size, expr, table)
        self._add(self.U, S, 1, "S")
        for size in range(2, max_size + 1):
            Us, Bs = list(self.U.values()), list(self.B.values())
            for s, e, t in Us:
                if s == size - 1:
                    if use_inverse:
                        self._add(self.U, _inv_u(t), size, f"inv({e})")
                    self._add(self.B, self._iter(t), size, f"iter({e})")
            for s, e, t in Bs:
                if s == size - 1:
                    self._add(self.U, np.diag(t).copy(), size, f"diag({e})")
                    self._add(self.B, self._acc(t), size, f"acc({e})")
                    self._add(self.B, t.T.copy(), size, f"swap({e})")
                    if use_inverse:
                        self._add(self.B, np.stack([_inv_u(t[:, b]) for b in range(N)], 1), size, f"inv1({e})")
                if s == size - 2:
                    for c in self.consts:
                        self._add(self.U, t[:, c].copy(), size, f"bind({e},#{self.rank[c]})")
            for (s1, e1, t1), (s2, e2, t2) in itertools.product(Us, Us):
                if s1 + s2 == size - 1:
                    self._add(self.U, _take(t1, t2), size, f"{e1}∘{e2}")
        self.Ut = np.stack([v[2] for v in self.U.values()]) if self.U else np.zeros((0, N), np.int64)
        self.Bt = np.stack([v[2] for v in self.B.values()]) if self.B else np.zeros((0, N, N), np.int64)
        self.Ue = [(v[0], v[1]) for v in self.U.values()]
        self.Be = [(v[0], v[1]) for v in self.B.values()]

    @staticmethod
    def _add(lib, t, size, expr):
        if not (t >= 0).any():
            return
        k = t.tobytes()
        if k not in lib:
            lib[k] = (size, expr, t)

    def _iter(self, u):           # iter(u)(a, b) = u applied b times to a
        res = np.full((N, N), UNDEF, dtype=np.int64)
        pw = np.arange(N)
        for node in self.order:
            res[:, node] = pw
            pw = _take(u, pw)
        return res

    def _acc(self, B):            # acc(B)(a, b) = (x -> B(x, a)) applied b times to zero
        res = np.full((N, N), UNDEF, dtype=np.int64)
        if self.zero is None:
            return res
        s, a = np.full(N, self.zero), np.arange(N)
        for node in self.order:
            res[:, node] = s
            ok = s >= 0
            ns = np.full(N, UNDEF, dtype=np.int64)
            ns[ok] = B[s[ok], a[ok]]
            s = ns
        return res

    def __len__(self):
        return len(self.Ue) + len(self.Be)

    def solve(self, arity, examples):
        """Smallest program (by size, then build order) consistent with every example; None if none."""
        if arity == 1:
            a = np.array([x[0] for x, _ in examples])
            y = np.array([v for _, v in examples])
            ok = (self.Ut[:, a] == y).all(1) if len(self.Ut) else np.zeros(0, bool)
            entries, tabs = self.Ue, self.Ut
        else:
            a = np.array([x[0] for x, _ in examples])
            b = np.array([x[1] for x, _ in examples])
            y = np.array([v for _, v in examples])
            ok = (self.Bt[:, a, b] == y).all(1) if len(self.Bt) else np.zeros(0, bool)
            entries, tabs = self.Be, self.Bt
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            return None
        best = min(idx, key=lambda i: (entries[i][0], i))
        return dict(size=entries[best][0], expr=entries[best][1], table=tabs[best], arity=arity)

    @staticmethod
    def answer(prog, x):
        if prog is None:
            return UNDEF
        return int(prog["table"][x[0]] if prog["arity"] == 1 else prog["table"][x[0], x[1]])


# ============================================================================ family search
class FamilyLibrary:
    """All distinct relations up to max_size built from snapped parent / gender tables.
    Tables live on the CPU as bool arrays; the expensive part (chaining = matrix products) runs on device."""

    def __init__(self, P, G, device, max_size=10, use_inverse=True, chunk=64):
        self.dev = device
        P = np.asarray(P, bool)
        self.G = np.asarray(G, bool)
        self.n = P.shape[0]
        self.keys, self.tabs, self.entries = set(), [], []      # entries: (size, expr, tree)
        self._add(P, 1, "P", ("P",))
        for size in range(2, max_size + 1):
            cur = list(range(len(self.tabs)))
            for i in cur:
                s, e, tr = self.entries[i]
                if s != size - 1:
                    continue
                t = self.tabs[i]
                if use_inverse:
                    self._add(t.T.copy(), size, f"inv({e})", ("inv", tr))
                t2 = t.copy()
                np.fill_diagonal(t2, False)
                self._add(t2, size, f"nonself({e})", ("nonself", tr))
                self._add(t & self.G[None, :], size, f"male({e})", ("male", tr))
                self._add(t & ~self.G[None, :], size, f"female({e})", ("female", tr))
            by = {}
            for i in cur:
                by.setdefault(self.entries[i][0], []).append(i)
            for s1 in range(1, size - 1):
                s2 = size - 1 - s1
                if s1 not in by or s2 not in by:
                    continue
                Bm = torch.as_tensor(np.stack([self.tabs[j] for j in by[s2]]), device=device).half()
                for j0 in range(0, len(by[s1]), chunk):
                    I = by[s1][j0:j0 + chunk]
                    A = torch.as_tensor(np.stack([self.tabs[i] for i in I]), device=device).half()
                    Cm = (torch.einsum("aij,bjk->abik", A, Bm) > 0).cpu().numpy()
                    for ai, i in enumerate(I):
                        for bj, j in enumerate(by[s2]):
                            ei, ej = self.entries[i], self.entries[j]
                            self._add(Cm[ai, bj], size, f"({ei[1]} ; {ej[1]})", ("comp", ei[2], ej[2]))
        self.T = torch.as_tensor(np.stack(self.tabs), device=device)

    def _add(self, t, size, expr, tree):
        if not t.any():
            return
        k = np.packbits(t).tobytes()
        if k in self.keys:
            return
        self.keys.add(k)
        self.tabs.append(t)
        self.entries.append((size, expr, tree))

    def __len__(self):
        return len(self.tabs)

    def solve(self, R, people):
        """R: relation table (n x n bool) whose rows for `people` are the examples. Smallest consistent."""
        X = torch.as_tensor(np.asarray(people), device=self.dev)
        Y = torch.as_tensor(np.asarray(R)[np.asarray(people)], device=self.dev)
        ok = (self.T[:, X, :] == Y[None]).flatten(1).all(1)
        idx = torch.nonzero(ok).flatten().tolist()
        if not idx:
            return None
        best = min(idx, key=lambda i: (self.entries[i][0], i))
        return dict(size=self.entries[best][0], expr=self.entries[best][1], tree=self.entries[best][2],
                    table=self.tabs[best])


def soft_execute(tree, Ps, Gs):
    """No-snap ablation: run a family program on block probabilities (noisy-or chaining), threshold
    only at the end. Ps: (n, n) parent probabilities, Gs: (n,) male probabilities."""
    op = tree[0]
    if op == "P":
        return Ps
    if op == "inv":
        return soft_execute(tree[1], Ps, Gs).T
    if op == "nonself":
        t = soft_execute(tree[1], Ps, Gs).clone()
        t.fill_diagonal_(0)
        return t
    if op == "male":
        return soft_execute(tree[1], Ps, Gs) * Gs[None, :]
    if op == "female":
        return soft_execute(tree[1], Ps, Gs) * (1 - Gs)[None, :]
    a, b = soft_execute(tree[1], Ps, Gs), soft_execute(tree[2], Ps, Gs)
    # noisy-or: P(x ~ z) = 1 - prod_y (1 - a[x, y] b[y, z])
    return 1 - torch.exp(torch.log1p(-(a[:, :, None] * b[None, :, :]).clamp(max=1 - 1e-6)).sum(1))
