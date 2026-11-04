"""Generated worlds for experiment 04 (no downloads).

Arithmetic: numbers 0..N-1 are opaque symbols. The composer only ever sees next-number facts.
Family: a generated forest of people with parent links and genders, a new forest per seed.
UNDEF (-1) marks "no answer inside this world" (e.g. 99 + 1, or 3 - 5)."""
import random
import numpy as np

N = 100
UNDEF = -1
NONE_CLASS = N            # "there is no next number" as a class for the learned block


# ----------------------------------------------------------------------------- arithmetic
def succ_facts(drop=0.0, seed=0):
    """The composer's whole training set: (x, next(x)) for x in 0..N-1 (100 facts).
    drop > 0 removes that fraction of facts (missing-facts ablation)."""
    facts = [(x, x + 1 if x + 1 < N else NONE_CLASS) for x in range(N)]
    if drop > 0:
        rng = random.Random(seed)
        keep = set(rng.sample(range(N), int(round(N * (1 - drop)))))
        facts = [f for f in facts if f[0] in keep]
    return facts


def _u(f):
    return np.array([f(a) if 0 <= f(a) < N else UNDEF for a in range(N)])


def _b(f):
    return np.array([[f(a, b) if 0 <= f(a, b) < N else UNDEF for b in range(N)] for a in range(N)])


def _half(a):
    return a // 2 if a % 2 == 0 else UNDEF


ARITH_TRAIN = {   # the baseline meta-trains on these 14 operations (the composer never sees them)
    "succ": (1, lambda a: a + 1), "pred": (1, lambda a: a - 1), "add": (2, lambda a, b: a + b),
    "double": (1, lambda a: 2 * a),
    **{f"add{c}": (1, (lambda c: lambda a: a + c)(c)) for c in (2, 4, 5, 6, 7, 8, 9)},
    **{f"minus{c}": (1, (lambda c: lambda a: a - c)(c)) for c in (2, 5, 7)},
}
ARITH_TEST = {    # held out from everyone; 4 of 5 need "going the opposite way" or repetition
    "add3": (1, lambda a: a + 3), "minus3": (1, lambda a: a - 3), "sub": (2, lambda a, b: a - b),
    "half": (1, _half), "mul": (2, lambda a, b: a * b),
}
NEEDS_UNDO_ARITH = ("minus3", "sub", "half")


def arith_table(name):
    arity, f = {**ARITH_TRAIN, **ARITH_TEST}[name]
    return arity, (_u(f) if arity == 1 else _b(f))


def random_unary_op(seed):
    """Negative control: a random permutation of 0..N-1, no structure to compose."""
    rng = np.random.default_rng(10_000 + seed)
    return 1, rng.permutation(N)


def valid_inputs(arity, table):
    if arity == 1:
        return [(a, None) for a in range(N) if table[a] != UNDEF]
    return [(a, b) for a in range(N) for b in range(N) if table[a, b] != UNDEF]


def lookup(arity, table, x):
    a, b = x
    return int(table[a] if arity == 1 else table[a, b])


def arith_episodes(arity, table, k, n_episodes, seed):
    """Each episode: k worked examples + 1 query, all distinct and defined.
    Returns list of (examples [(x, y)], query x, answer y)."""
    rng = random.Random(seed)
    pool = valid_inputs(arity, table)
    eps = []
    for _ in range(n_episodes):
        xs = rng.sample(pool, k + 1)
        ex = [(x, lookup(arity, table, x)) for x in xs[:k]]
        eps.append((ex, xs[k], lookup(arity, table, xs[k])))
    return eps


# ----------------------------------------------------------------------------- family
def family(seed, n_couples=6, gens=4, max_people=200):
    """P[x, y] = True when y is a parent of x. G[x] = True when x is male."""
    rng = random.Random(seed)
    gender, parents = [], []

    def new(g, ps=()):
        gender.append(g)
        parents.append(list(ps))
        return len(gender) - 1

    couples = [(new(1), new(0)) for _ in range(n_couples)]
    for _ in range(gens - 1):
        kids = []
        for f, m in couples:
            for _ in range(rng.randint(2, 3)):
                if len(gender) < max_people:
                    kids.append(new(rng.randint(0, 1), (f, m)))
        couples = []
        for c in kids:
            if rng.random() < 0.75 and len(gender) < max_people:
                s = new(1 - gender[c])
                couples.append((c, s) if gender[c] == 1 else (s, c))
    n = len(gender)
    P = np.zeros((n, n), bool)
    for c, ps in enumerate(parents):
        for p in ps:
            P[c, p] = True
    return P, np.array(gender, bool)


def _mm(a, b):
    return (a.astype(np.int32) @ b.astype(np.int32)) > 0


def relations(P, G):
    """Ground-truth relation tables R[x, y] = 'y is x's <relation>'."""
    C = P.T
    sib = _mm(P, C)
    np.fill_diagonal(sib, False)
    gp = _mm(P, P)
    pa_sib = _mm(P, sib)          # parents' siblings
    sib_kids = _mm(sib, C)        # siblings' children
    M, F = G[None, :], ~G[None, :]
    return {
        "parent": P, "child": C, "grandparent": gp, "sibling": sib, "brother": sib & M, "sister": sib & F,
        "grandchild": gp.T, "grandmother": gp & F, "uncle": pa_sib & M, "aunt": pa_sib & F,
        "nephew": sib_kids & M, "cousin": _mm(pa_sib, C),
    }


REL_TRAIN = ("parent", "child", "grandparent", "sibling", "brother", "sister")
REL_TEST = ("grandchild", "grandmother", "uncle", "aunt", "nephew", "cousin")
NEEDS_UNDO_REL = ("grandchild", "uncle", "aunt", "nephew", "cousin")


def random_relation(n, density, seed):
    """Negative control: a random relation with no family structure."""
    rng = np.random.default_rng(20_000 + seed)
    return rng.random((n, n)) < density


def rel_episodes(R, k, n_episodes, seed):
    """Each episode: k example people with their full answer sets, then every other person with a
    non-empty answer as test queries. Returns list of (example_people, test_people)."""
    rng = random.Random(seed)
    people = [x for x in range(len(R)) if R[x].any()]
    eps = []
    for _ in range(n_episodes):
        ex = rng.sample(people, min(k, len(people) - 1))
        eps.append((ex, [x for x in people if x not in ex]))
    return eps
