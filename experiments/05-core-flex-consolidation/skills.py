"""Skills and the continual-learning stream (all generated, no downloads).

A skill = a cue bound to a hidden rule y = (a*x + b) mod 17, x in 0..16.
The cue is a random unit vector in 64-d ("a point on a sphere"); every time the skill shows up,
its cue arrives with fresh noise (cosine ~0.9 to the clean cue), the way a request is never
worded the same way twice.

Rules are split once, with a fixed seed: 208 rules are known before the stream (pretraining)
and 64 are new, used only by stream skills.
"""
import numpy as np

P = 17
CUE_DIM = 64
CUE_NOISE = 0.06                    # per-dim std; cos(clean, noisy) ~ 0.9
RULES = np.array([(a, b) for a in range(1, P) for b in range(P)])          # 272 rules
_perm = np.random.default_rng(2026).permutation(len(RULES))
PRE_RULES = RULES[_perm[:208]]
NEW_RULES = RULES[_perm[208:]]                                              # 64 rules

FULL = dict(n_freq=8, rep_freq=12, n_occ=24, rep_occ=4, n_one=32, burst=8, k=6, n_never=16)
QUICK = dict(n_freq=2, rep_freq=4, n_occ=3, rep_occ=2, n_one=4, burst=2, k=6, n_never=4)
GROUPS = ("freq", "occ", "one")


def table(rules):
    """(n, 2) rules -> (n, 17) answers for every x."""
    x = np.arange(P)
    return (rules[:, :1] * x[None] + rules[:, 1:2]) % P


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def random_cues(rng, n):
    return unit(rng.standard_normal((n, CUE_DIM))).astype(np.float32)


def noisy(rng, cues):
    return unit(cues + CUE_NOISE * rng.standard_normal(cues.shape)).astype(np.float32)


def make_stream(rng, cfg=FULL):
    """Return the stream for one seed.

    skills: rules (n, 2), cues (n, 64), group (n,) in GROUPS
    episodes: list of (skill id, support xs (k,)), recurring skills in random order with the
              one-off skills inserted as bursts of cfg['burst'] consecutive episodes
    burst_ends: episode indices right after each burst (where C5 is measured)
    never: rules and cues of skills that never appear (control)
    """
    n_skill = cfg["n_freq"] + cfg["n_occ"] + cfg["n_one"]
    assert n_skill <= len(NEW_RULES)
    order = rng.permutation(len(NEW_RULES))
    rules = NEW_RULES[order[:n_skill]]
    group = np.array(["freq"] * cfg["n_freq"] + ["occ"] * cfg["n_occ"] + ["one"] * cfg["n_one"])
    cues = random_cues(rng, n_skill)
    rec = ([i for i in range(cfg["n_freq"]) for _ in range(cfg["rep_freq"])] +
           [cfg["n_freq"] + i for i in range(cfg["n_occ"]) for _ in range(cfg["rep_occ"])])
    rec = list(rng.permutation(rec))
    ones = list(range(cfg["n_freq"] + cfg["n_occ"], n_skill))
    n_b = len(ones) // cfg["burst"]
    seq, burst_ends, start = [], [], 0
    for b in range(n_b):
        cut = int(round(len(rec) * (b + 1) / (n_b + 1)))
        seq += rec[start:cut]
        seq += ones[b * cfg["burst"]:(b + 1) * cfg["burst"]]
        burst_ends.append(len(seq))
        start = cut
    seq += rec[start:]
    eps = [(int(s), np.sort(rng.choice(P, cfg["k"], replace=False))) for s in seq]
    never_rules = NEW_RULES[rng.integers(len(NEW_RULES), size=cfg["n_never"])]
    return dict(rules=rules, cues=cues, group=group, episodes=eps, burst_ends=burst_ends,
                never_rules=never_rules, never_cues=random_cues(rng, cfg["n_never"]))
