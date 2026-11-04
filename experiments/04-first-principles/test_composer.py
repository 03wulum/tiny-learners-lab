"""Correctness tests for experiment 04 (fast, CPU). These check code, not the hypothesis."""
import numpy as np
import torch

import worlds as W
import composer as C
import baselines as B

EXACT_S = np.array([x + 1 if x + 1 < W.N else W.UNDEF for x in range(W.N)])


def test_succ_facts_are_the_whole_curriculum():
    f = W.succ_facts()
    assert len(f) == 100 and f[0] == (0, 1) and f[-1] == (99, W.NONE_CLASS)
    assert len(W.succ_facts(drop=0.1)) == 90


def test_ops_tables():
    assert W.arith_table("sub")[1][7, 3] == 4 and W.arith_table("sub")[1][3, 7] == W.UNDEF
    assert W.arith_table("half")[1][10] == 5 and W.arith_table("half")[1][11] == W.UNDEF
    assert W.arith_table("mul")[1][9, 11] == 99 and W.arith_table("mul")[1][10, 10] == W.UNDEF
    assert not set(W.ARITH_TRAIN) & set(W.ARITH_TEST)


def test_counting_chain_finds_zero_without_labels():
    perm = np.random.default_rng(0).permutation(W.N)          # relabel every symbol
    S = np.full(W.N, W.UNDEF)
    for x in range(W.N - 1):
        S[perm[x]] = perm[x + 1]
    z, order, rank = C.counting_chain(S)
    assert z == perm[0] and len(order) == W.N and rank[perm[42]] == 42


def test_library_finds_every_target_on_exact_blocks():
    lib = C.ArithLibrary(EXACT_S)
    for op in W.ARITH_TEST:
        arity, T = W.arith_table(op)
        prog = lib.solve(arity, W.arith_episodes(arity, T, 8, 1, 0)[0][0])
        assert prog is not None and np.array_equal(prog["table"], T), op


def test_symbol_relabelling_does_not_matter():
    perm = np.random.default_rng(1).permutation(W.N)
    inv = np.argsort(perm)
    S = np.full(W.N, W.UNDEF)
    for x in range(W.N - 1):
        S[perm[x]] = perm[x + 1]
    lib = C.ArithLibrary(S)
    arity, T = W.arith_table("sub")
    ex = [((perm[a], perm[b]), perm[a - b]) for a, b in [(9, 4), (50, 8), (30, 30), (77, 1)]]
    prog = lib.solve(2, ex)
    assert C.ArithLibrary.answer(prog, (perm[60], perm[18])) == perm[42]


def test_no_inverse_library_cannot_subtract():
    lib = C.ArithLibrary(EXACT_S, use_inverse=False)
    arity, T = W.arith_table("sub")
    assert lib.solve(arity, W.arith_episodes(arity, T, 4, 1, 0)[0][0]) is None


def test_family_relations_consistent():
    P, G = W.family(0)
    R = W.relations(P, G)
    assert (P.sum(1) <= 2).all() and np.array_equal(R["child"], P.T)
    assert not np.diag(R["sibling"]).any() and not (R["uncle"] & ~G[None, :]).any()
    assert np.array_equal(R["grandchild"], R["grandparent"].T)


def test_family_library_finds_targets_on_exact_blocks():
    P, G = W.family(0)
    R = W.relations(P, G)
    lib = C.FamilyLibrary(P, G, "cpu")
    for rel in ("grandchild", "grandmother", "uncle", "cousin"):
        people = [x for x in range(len(P)) if R[rel][x].any()][:12]
        p = lib.solve(R[rel], people)
        assert p is not None and np.array_equal(p["table"], R[rel]), rel


def test_soft_execute_matches_hard_on_crisp_tables():
    P, G = W.family(1)
    R = W.relations(P, G)
    tree = ("male", ("comp", ("P",), ("nonself", ("comp", ("P",), ("inv", ("P",))))))
    out = C.soft_execute(tree, torch.tensor(P, dtype=torch.float32), torch.tensor(G, dtype=torch.float32)) > 0.5
    assert np.array_equal(out.numpy(), R["uncle"])


def test_blocks_and_snapping_shapes():
    torch.manual_seed(0)
    s = C.snap_next(C.NextBlock())
    assert s.shape == (W.N,) and s.min() >= W.UNDEF
    acc = C.latent_chain_accuracy(C.NextBlock(), [1, 3])
    assert set(acc) == {1, 3}
    P, G = W.family(0)
    Ps, Gs = C.snap_family(C.FamilyBlocks(len(P)))
    assert Ps.shape == P.shape and Gs.shape == G.shape


def test_icl_encoding_hides_query_answer():
    eps = W.arith_episodes(*W.arith_table("sub"), 4, 3, 0)
    seq, ans = B.encode_episodes(eps, "cpu")
    assert seq.shape == (3, 5 * 4 + 3) and int(seq[0, -1]) == B.ARROW
    assert int(ans[0]) == eps[0][2] and int(seq[0, 3]) == eps[0][0][0][1]
    assert B.ICLTransformer()(seq).shape == (3, W.N)


def test_gnn_shapes_and_fewshot():
    P, G = W.family(0)
    R = W.relations(P, G)
    m = B.PathGNN(**B.SIZES_GNN["small"])
    vec = B.gnn_fewshot(m, P, G, [R["uncle"], R["aunt"]], [[1, 2], [3, 4]], "cpu", steps=2)
    assert vec.shape == (2, B.SIZES_GNN["small"]["d"])
    assert B.gnn_predict(m, P, G, vec, "cpu").shape == (2, len(P), len(P))


def test_criteria_verdict_logic():
    import criteria
    d = lambda fp, base: dict(fp={"x": fp, "sub": fp}, fp_mean=fp, base_best_mean=base, base_best={"x": base, "sub": base},
                              fp_k1={"x": 0.0}, base_own_best=95.0, oracle_mean=95.0, controls={"c": 0.0},
                              facts=100, latent_chain={})
    m = dict(budget=dict(composer_mb=0.1, kept_mb=5, train_min=5, answer_ms=0.001, search_s=0.01, peak_ram_gb=1),
             arith=d(99.0, 20.0), family=d(95.0, 30.0))
    assert criteria.verdict(m)[1] == "PASS"
    m["family"] = d(80.0, 30.0)
    assert criteria.verdict(m)[1] == "FAIL"
    m["family"]["controls"]["c"] = 50.0
    assert criteria.verdict(m)[1] == "INCONCLUSIVE"
