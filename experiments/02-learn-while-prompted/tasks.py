"""Procedurally generated learning tasks. Nothing is downloaded, and every prompt carries
a rule or a set of facts the model has never seen as a whole.

Vocabulary
    inputs  x: 0..NX-1   (rules use 0..P-1, fact keys use 0..NX-1)
    outputs y: 0..P-1

Tasks
    T1 "rules":   y = (a*x + b) mod P, a fresh (a, b) per prompt. 25% of rules are held out
                  of training (seeded split), so held-out accuracy measures generalization.
    T2 "facts":   a fresh random lookup table key -> value per prompt. Recall is only
                  possible for keys shown in the prompt; unseen keys must stay at chance.
    T4 "quad":    y = (a*x^2 + b) mod P, a rule family never used in training (exploratory).
    shuffled:     T1 inputs with random labels. Nothing to learn: the negative control.
T3 is T1/T2 with more examples (up to 64) than training ever used (16).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

P = 17          # output alphabet (prime, so affine rules are bijective)
NX = 32         # input alphabet (fact keys)
CHANCE = 1.0 / P

ALL_RULES = np.array([(a, b) for a in range(1, P) for b in range(P)])  # 272 rules


def rule_split(seed: int = 0, heldout_frac: float = 0.25):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ALL_RULES))
    n = int(len(ALL_RULES) * heldout_frac)
    return ALL_RULES[idx[n:]], ALL_RULES[idx[:n]]  # train, heldout


TRAIN_RULES, HELDOUT_RULES = rule_split()


def apply_rule(family: str, ab: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b = ab[:, :1], ab[:, 1:]
    if family == "quad":
        return (a * x * x + b) % P
    return (a * x + b) % P


@dataclass
class Batch:
    """k context pairs (px, py) and Q queries (qx -> qy) per prompt. pvalid masks pairs."""
    px: np.ndarray      # [N, k] int
    py: np.ndarray      # [N, k] int
    pvalid: np.ndarray  # [N, k] bool
    qx: np.ndarray      # [N, Q] int
    qy: np.ndarray      # [N, Q] int
    qseen: np.ndarray   # [N, Q] bool: query key appeared in the context
    ideal: np.ndarray   # [N, Q] float: Bayes-optimal probability of answering correctly


def _ideal_rules(family, px, py, pvalid, qx):
    """Brute-force ideal learner: every rule of the family consistent with the context,
    uniform prior, answer the most common prediction. Returns P(correct) per query."""
    rules = ALL_RULES
    xs = np.arange(P)
    table = apply_rule(family, rules, np.broadcast_to(xs, (len(rules), P)))  # [R, P]
    N, k = px.shape
    ok = np.ones((N, len(rules)), bool)
    for j in range(k):
        pred = table[:, np.clip(px[:, j], 0, P - 1)].T  # [N, R]
        ok &= ~pvalid[:, j:j + 1] | (pred == py[:, j:j + 1])
    out = np.zeros(qx.shape)
    for q in range(qx.shape[1]):
        pred = table[:, qx[:, q]].T  # [N, R]
        counts = np.stack([((pred == v) & ok).sum(1) for v in range(P)], 1)
        out[:, q] = counts.max(1) / np.maximum(ok.sum(1), 1)
    return out


def make_batch(task: str, n: int, k: int, q: int, rng: np.random.Generator,
               rules: np.ndarray | None = None, with_ideal: bool = True) -> Batch:
    """task in {rules, facts, quad, shuffled}. Queries for 'facts' are half shown keys,
    half unseen keys (when k > 0); for rule tasks they are random inputs."""
    kk = max(k, 1)  # k = 0 is represented as one masked-out pair
    pvalid = np.ones((n, kk), bool) if k > 0 else np.zeros((n, kk), bool)
    if task in ("rules", "quad", "shuffled"):
        if rules is None:
            rules = TRAIN_RULES if task != "quad" else ALL_RULES
        ab = rules[rng.integers(len(rules), size=n)]
        px = rng.integers(P, size=(n, kk))
        qx = rng.integers(P, size=(n, q))
        fam = "quad" if task == "quad" else "affine"
        py = apply_rule(fam, ab, px)
        qy = apply_rule(fam, ab, qx)
        if task == "shuffled":
            py = rng.integers(P, size=(n, kk))
            qy = rng.integers(P, size=(n, q))
        qseen = (qx[:, :, None] == np.where(pvalid, px, -1)[:, None, :]).any(-1)
        if task == "shuffled" or not with_ideal:
            ideal = np.full((n, q), CHANCE)
        else:
            ideal = _ideal_rules(fam, px, py, pvalid, qx)
    elif task == "facts":
        table = rng.integers(P, size=(n, NX))
        px = rng.integers(NX, size=(n, kk))
        py = np.take_along_axis(table, px, 1)
        shown = np.where(pvalid, px, -1)
        half = q // 2 if k > 0 else 0
        qa = np.take_along_axis(px, rng.integers(kk, size=(n, half)), 1)  # shown keys
        qb = rng.integers(NX, size=(n, q - half))                         # unseen keys
        for _ in range(50):  # rejection-sample keys that are not in the prompt
            bad = (qb[:, :, None] == shown[:, None, :]).any(-1)
            if not bad.any():
                break
            qb = np.where(bad, rng.integers(NX, size=qb.shape), qb)
        qx = np.concatenate([qa, qb], 1)
        qy = np.take_along_axis(table, qx, 1)
        qseen = (qx[:, :, None] == shown[:, None, :]).any(-1)
        ideal = np.where(qseen, 1.0, CHANCE)
    else:
        raise ValueError(task)
    return Batch(px, py, pvalid, qx, qy, qseen, ideal)


def make_prefix_batch(task: str, n: int, k: int, rng: np.random.Generator) -> Batch:
    """Training format: k pairs; query i asks x_i having seen pairs 0..i-1 (qx = px)."""
    b = make_batch(task, n, k, 1, rng, with_ideal=False)
    return Batch(b.px, b.py, b.pvalid, b.px.copy(), b.py.copy(), b.qseen, b.ideal)


def make_meta_batch(task: str, n: int, k: int, q: int, rng: np.random.Generator) -> Batch:
    """Meta-training format: k context pairs and q fresh queries about the same rule/table.
    For facts, 3/4 of queries ask shown keys so recall gets a strong training signal."""
    b = make_batch(task, n, k, q, rng, with_ideal=False)
    if task == "facts" and k > 0:
        # re-draw queries: 3/4 ask shown keys (answers come from the shown pairs)
        pick = rng.random((n, q)) < 0.75
        j = rng.integers(b.px.shape[1], size=(n, q))
        b.qx = np.where(pick, np.take_along_axis(b.px, j, 1), b.qx)
        b.qy = np.where(pick, np.take_along_axis(b.py, j, 1), b.qy)
        b.qseen = (b.qx[:, :, None] == b.px[:, None, :]).any(-1)
    return b
