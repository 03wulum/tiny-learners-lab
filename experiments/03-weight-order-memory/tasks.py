"""Procedurally generated few-shot tasks (no downloads).

Every problem is a prompt of k worked examples followed by a query:
    [BOS, TAG] x1 -> y1 SEP x2 -> y2 SEP ... xq -> yq
The model is scored on the query answer tokens (exact match of all of them).

T1  modular rules   y = (a*x + b) mod 17          25% of (a, b) rules held out
T2  new facts       random key -> value table, given only in the prompt
T3  grid transforms 4-cell row: rotate, mirror, recolor; 25% of combos held out
T4  base addition   2-digit + 2-digit in base B; base 7 never seen in training
"""
import itertools
import numpy as np

N_SYM = 17                         # symbol tokens 0..16
ARROW, SEP, PLUS, PAD, BOS = 17, 18, 19, 20, 21
TAG = {"T1": 22, "T2": 23, "T3": 24, "T4": 25}
VOCAB = 26
TASKS = ("T1", "T2", "T3", "T4")
EX_LEN = {"T1": 4, "T2": 4, "T3": 10, "T4": 10}   # x + ARROW + y + SEP
Y_LEN = {"T1": 1, "T2": 1, "T3": 4, "T4": 3}
MAX_K = 16
MAX_LEN = 2 + (MAX_K + 1) * 10

_split_rng = np.random.default_rng(12345)          # fixed: the split never changes

T1_RULES = np.array([(a, b) for a in range(1, 17) for b in range(17)])
_p = _split_rng.permutation(len(T1_RULES))
_n_hold = len(T1_RULES) // 4
T1_SPLIT = {"val": T1_RULES[_p[:_n_hold // 2]], "test": T1_RULES[_p[_n_hold // 2:_n_hold]],
            "train": T1_RULES[_p[_n_hold:]]}

PERMS4 = np.array(list(itertools.permutations(range(4))))
T3_COMBOS = np.array([(r, m, c) for r in range(4) for m in range(2) for c in range(24)])
_p = _split_rng.permutation(len(T3_COMBOS))
_n_hold = len(T3_COMBOS) // 4
T3_SPLIT = {"val": T3_COMBOS[_p[:_n_hold // 2]], "test": T3_COMBOS[_p[_n_hold // 2:_n_hold]],
            "train": T3_COMBOS[_p[_n_hold:]]}

T4_BASES = {"train": np.array([5, 6, 8, 9, 10, 11, 12]), "val": np.array([7]), "test": np.array([7])}


def _t1(rng, n, k, split, rule_idx=None):
    rules = T1_SPLIT[split]
    if rule_idx is None:
        rule_idx = rng.integers(len(rules), size=n)
    a, b = rules[rule_idx, 0][:, None], rules[rule_idx, 1][:, None]
    x = rng.integers(N_SYM, size=(n, k + 1))
    y = (a * x + b) % N_SYM
    return x[..., None], y[..., None]


def _t2(rng, n, k, split, rule_idx=None):
    keys = np.argsort(rng.random((n, N_SYM)), axis=1)[:, :k]          # distinct keys
    vals = rng.integers(N_SYM, size=(n, k))
    q = rng.integers(k, size=n)
    x = np.concatenate([keys, keys[np.arange(n), q][:, None]], 1)
    y = np.concatenate([vals, vals[np.arange(n), q][:, None]], 1)
    return x[..., None], y[..., None]


def t3_apply(x, r, m, c):
    """x: (n, e, 4) colors; r, m, c: (n,) rotate, mirror, recolor-permutation index."""
    idx = (np.arange(4)[None, :] - r[:, None]) % 4
    out = np.take_along_axis(x, idx[:, None, :].repeat(x.shape[1], 1), axis=2)
    out = np.where(m[:, None, None] == 1, out[..., ::-1], out)
    perm = PERMS4[c]                                                  # (n, 4)
    return np.take_along_axis(perm[:, None, :].repeat(x.shape[1], 1), out, axis=2)


def _t3(rng, n, k, split, rule_idx=None):
    combos = T3_SPLIT[split]
    if rule_idx is None:
        rule_idx = rng.integers(len(combos), size=n)
    r, m, c = combos[rule_idx].T
    x = rng.integers(4, size=(n, k + 1, 4))
    return x, t3_apply(x, r, m, c)


def _t4(rng, n, k, split, rule_idx=None):
    bases = T4_BASES[split]
    if rule_idx is None:
        rule_idx = rng.integers(len(bases), size=n)
    B = bases[rule_idx][:, None]
    u = (rng.random((n, k + 1)) * B * B).astype(int)
    v = (rng.random((n, k + 1)) * B * B).astype(int)
    s = u + v
    x = np.stack([u // B, u % B, np.full_like(u, PLUS), v // B, v % B], -1)
    y = np.stack([s // (B * B), (s // B) % B, s % B], -1)
    return x, y


GEN = {"T1": _t1, "T2": _t2, "T3": _t3, "T4": _t4}


def n_rules(task, split):
    return {"T1": len(T1_SPLIT[split]), "T3": len(T3_SPLIT[split]),
            "T4": len(T4_BASES[split])}.get(task, 0)


def make_batch(rng, task, n, k, split="train", shuffle_labels=False, rule_idx=None):
    """Return dict of int64 arrays:
    full   (n, L)   support + query prompt
    ymask  (n, L)   1 where the token is a query answer token (target at that position)
    supp   (n, Ls)  support-only prompt (used for test-time adaptation)
    smask  (n, Ls)  1 at every support answer token
    """
    x, y = GEN[task](rng, n, k, split, rule_idx)
    return assemble(rng, task, x, y, shuffle_labels)


def assemble(rng, task, x, y, shuffle_labels=False):
    """x, y: (n, k+1, len) arrays; the last entry is the query."""
    n, k = x.shape[0], x.shape[1] - 1
    if shuffle_labels and k > 1:
        perm = np.argsort(rng.random((n, k)), axis=1)
        y = y.copy()
        y[:, :k] = np.take_along_axis(y[:, :k], perm[..., None].repeat(y.shape[2], 2), 1)
    ex = np.concatenate([x, np.full(x.shape[:2] + (1,), ARROW), y,
                         np.full(x.shape[:2] + (1,), SEP)], -1)               # (n, k+1, E)
    E = ex.shape[2]
    head = np.tile(np.array([BOS, TAG[task]]), (n, 1))
    full = np.concatenate([head, ex.reshape(n, -1)[:, :-1]], 1)               # drop final SEP
    supp = np.concatenate([head, ex[:, :k].reshape(n, -1)], 1)
    ylen, xlen = y.shape[2], x.shape[2]
    pos_in_ex = np.arange(E)
    is_y = (pos_in_ex >= xlen + 1) & (pos_in_ex < xlen + 1 + ylen)
    sm = np.concatenate([np.zeros(2, bool), np.tile(is_y, k)])
    fm = np.concatenate([np.zeros(2, bool), np.zeros(k * E, bool), is_y[:-1]])
    return {"full": full.astype(np.int64), "ymask": np.tile(fm, (n, 1)).astype(np.int64),
            "supp": supp.astype(np.int64), "smask": np.tile(sm, (n, 1)).astype(np.int64)}
