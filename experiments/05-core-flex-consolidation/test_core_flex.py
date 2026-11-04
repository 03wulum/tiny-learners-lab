"""Correctness tests for tiny-learners 05 (CPU, seconds). They test the code, not the claim."""
import numpy as np
import pytest
import torch

import skills, criteria
from model import (CORE_H, N_FLEX, N_CORE, matched_plain_h, mlp_numel, slot_numel, init_stack, forward,
                   StackAdam, ce)
from memory import SlotMemory
import train


def test_rule_split_and_table():
    pre = {tuple(r) for r in skills.PRE_RULES}
    new = {tuple(r) for r in skills.NEW_RULES}
    assert len(pre) == 208 and len(new) == 64 and not pre & new
    t = skills.table(np.array([[3, 5]]))[0]
    assert all(t[x] == (3 * x + 5) % 17 for x in range(17))


def test_stream_shape():
    st = skills.make_stream(np.random.default_rng(0), skills.FULL)
    c = skills.FULL
    assert len(st["episodes"]) == c["n_freq"] * c["rep_freq"] + c["n_occ"] * c["rep_occ"] + c["n_one"]
    ids = [s for s, _ in st["episodes"]]
    for g, rep in (("freq", c["rep_freq"]), ("occ", c["rep_occ"]), ("one", 1)):
        for i in np.where(st["group"] == g)[0]:
            assert ids.count(i) == rep
    # one-offs arrive in consecutive bursts that end at burst_ends
    for e in st["burst_ends"]:
        assert all(st["group"][s] == "one" for s in ids[e - c["burst"]:e])
    assert all(len(set(xs)) == c["k"] for _, xs in st["episodes"])


def test_noisy_cue_similarity():
    rng = np.random.default_rng(0)
    c = skills.random_cues(rng, 500)
    cos = (c * skills.noisy(rng, c)).sum(1)
    assert 0.85 < cos.mean() < 0.95
    other = (c[:-1] * c[1:]).sum(1)
    assert np.abs(other).max() < 0.7          # different skills never pass the routing threshold


def test_matched_parameter_count():
    h = matched_plain_h()
    target = mlp_numel(CORE_H) + (N_FLEX + N_CORE) * slot_numel()
    assert mlp_numel(h) <= target < mlp_numel(h + 1)


def _P(R=2, h=32, seed=0):
    return init_stack(R, h, "cpu", torch.Generator().manual_seed(seed))


def test_zero_slot_is_identity():
    P = _P()
    c, x = torch.randn(2, 3, skills.CUE_DIM), torch.randint(17, (2, 3, 5))
    out = forward(P, c, x, code=torch.zeros(2, 3, 32), U=torch.zeros(2, 3, 32, 4), V=torch.randn(2, 3, 32, 4))
    assert torch.allclose(out, forward(P, c, x), atol=1e-6)


def test_stacked_runs_are_independent():
    """Training run 0 in a stack must give the same weights as training it alone."""
    P2, P1 = _P(2), _P(1)
    for k in P1:
        P1[k] = P2[k][:1].clone()
    c, x, y = torch.randn(2, 6, skills.CUE_DIM), torch.randint(17, (2, 6, 1)), torch.randint(17, (2, 6, 1))
    for P, R in ((P2, 2), (P1, 1)):
        ps = list(P.values())
        for p in ps:
            p.requires_grad_(True)
        opt = StackAdam(ps, torch.full((R,), 1e-2))
        for _ in range(5):
            ce(forward(P, c[:R], x[:R]), y[:R]).mean((1, 2)).sum().backward()
            opt.step()
    for k in P1:
        assert torch.allclose(P1[k][0], P2[k][0], atol=1e-6)


def _mem(**kw):
    return SlotMemory(32, torch.zeros(32), "cpu", torch.Generator().manual_seed(0), **kw)


def _walk(m, n_skills, order, seed=0):
    rng = np.random.default_rng(seed)
    cues = skills.random_cues(rng, n_skills)
    for t, s in enumerate(order):
        cue = skills.noisy(rng, cues[s:s + 1])[0]
        slot, _, _ = m.prepare(t, cue, s)
        slot.code = slot.code + 1.0          # stand-in for learning
        m.finish(t, slot, cue)
    return cues


def test_size_never_grows_and_nothing_is_lost():
    m = _mem(n_flex=2, n_core=3)
    _walk(m, 20, list(range(12)) + [0, 0, 0, 1, 1, 1] + list(range(12, 20)))
    assert sum(s.tier == "flex" for s in m.active) <= 2
    assert sum(s.tier == "core" for s in m.active) <= 3
    assert len({s.skill for s in m.active + m.archive}) == 20        # every skill is kept somewhere
    assert m.route_errors == 0


def test_hardening_after_repeats():
    m = _mem(n_flex=2, n_core=3, harden_at=3)
    _walk(m, 3, [0, 1, 0, 1, 0])
    tiers = {s.skill: s.tier for s in m.active}
    assert tiers[0] == "core" and tiers[1] == "flex"
    s, mult, _ = m.prepare(5, m.active[[x.skill for x in m.active].index(0)].key, 0)
    assert mult == 0.1                                                 # hardened slots learn slower


def test_restore_from_archive_is_exact():
    m = _mem(n_flex=1, n_core=0, harden_at=99)
    cues = _walk(m, 2, [0, 1])                                         # 1 pushes 0 to the archive
    assert [s.skill for s in m.archive] == [0]
    saved = m.archive[0].code.clone()
    s, where = m.lookup(cues[0])
    assert where == "archive" and torch.equal(s.code, saved)
    m2 = _mem(n_flex=1, n_core=0, harden_at=99, archive=False)
    cues = _walk(m2, 2, [0, 1])
    assert m2.archive == [] and m2.lookup(cues[0])[1] == "none"


def test_flat_lru_vs_strength_eviction():
    m = _mem(n_flex=2, n_core=1, flat="lru")                           # one ring of 3
    _walk(m, 4, [0, 0, 0, 0, 1, 2, 3])
    assert sorted(s.skill for s in m.active) == [1, 2, 3]              # 0 was strongest but oldest
    m = _mem(n_flex=2, n_core=1, flat="strength")
    _walk(m, 4, [0, 0, 0, 0, 1, 2, 3])
    assert 0 in [s.skill for s in m.active]


def test_slot_tensors_route_and_zero_the_cue():
    m = _mem()
    cues = _walk(m, 3, [0, 1])
    code, U, V, keep = m.slot_tensors(cues)
    assert keep.tolist() == [0.0, 0.0, 1.0]                            # skill 2 unseen -> cue path
    assert torch.all(code[2] == 0) and torch.all(code[0] != 0)


def test_smoke_pipeline_end_to_end(tmp_path, monkeypatch):
    """Tiny lockstep run: every arm produces a full record."""
    st = [train.prepare_stream(0, 0, skills.QUICK)]
    gen = torch.Generator().manual_seed(0)
    core = init_stack(1, CORE_H, "cpu", gen, n_codes=len(skills.PRE_RULES))
    plain = init_stack(1, matched_plain_h(), "cpu", gen)
    ps = [dict(arm=a, seed_idx=0, stream=0, lr=1e-3, lam=lam) for a, lam in (("A", 0), ("B", 0), ("C", 10.0), ("D", 0))]
    ms = [dict(arm=a, seed_idx=0, stream=0, lr=0.03, mem=kw) for a, kw in train.M_VARIANTS.items()]
    res = train.run_phase([train.PlainGroup(ps, plain, st, "cpu", 2, 0), train.MGroup(ms, core, st, "cpu", 2, 0)], st)
    assert len(res) == 4 + len(train.M_VARIANTS)
    for r in res:
        assert len(r["episodes"]) == len(st[0]["episodes"]) and "final" in r
    d = {r["arm"]: r for r in res}
    assert d["D"]["replay_bytes"] <= train.REPLAY_BYTES
    assert d["M_noarchive"]["archive_bytes"] == 0


def _m(**over):
    m = dict(budget=dict(active_mb=0.5, kept_mb=1, total_gpu_min=5, update_ms=20, route_us=5, peak_ram_gb=1),
             ret_M=90, ret_B=40, ret_D=70, ret_unlimited=95, ret_seed_std=1, first_M=80, first_B=82,
             first_heldout_M=10, recovery_M=0.97, recovery_noarch=0.1, burst_M=90, burst_flatLRU=60,
             burst_flatStrength=85, pretrain_core=99, pretrain_plain=99, ctrl_shuffled_cue=6,
             ctrl_never_seen=6, ctrl_frozen_new=6, ctrl_shuffled_labels_heldout=6, route_error_rate=0)
    m.update(over)
    return m


def test_verdict_logic():
    assert criteria.verdict(_m())[1] == "PASS"
    assert criteria.verdict(_m(ret_D=88))[1] == "FAIL"
    assert criteria.verdict(_m(ctrl_never_seen=30))[1] == "INCONCLUSIVE"
    assert criteria.verdict(_m(ret_D=85.5))[1] == "INCONCLUSIVE"            # within one seed-std
    assert criteria.verdict(_m(recovery_noarch=0.8))[1] == "FAIL"
