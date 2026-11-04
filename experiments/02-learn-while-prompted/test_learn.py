"""Code-correctness tests. They never train a model (Obi's GPU-only rule); the toy models
here only check shapes, masks, leaks and gradient wiring."""
import numpy as np
import pytest
import torch

from arms import TRAINED, load, make_model, save
from criteria import CRITERIA, spearman, verdict
from models import SIZES, Config, TinyLearner
from tasks import ALL_RULES, CHANCE, HELDOUT_RULES, NX, P, TRAIN_RULES, apply_rule, make_batch, make_meta_batch
from ttt import adapt, loo_loss


def test_rule_split_disjoint_and_complete():
    tr = {tuple(r) for r in TRAIN_RULES}
    ho = {tuple(r) for r in HELDOUT_RULES}
    assert not tr & ho and len(tr | ho) == len(ALL_RULES) == 272
    assert len(ho) == 68


def test_rules_batch_labels_follow_rule():
    rng = np.random.default_rng(0)
    b = make_batch("rules", 50, 8, 4, rng, rules=HELDOUT_RULES)
    # the ideal learner with >= 2 distinct x is certain (affine rules are pinned by 2 points)
    distinct = np.array([len(set(r)) for r in b.px])
    assert np.allclose(b.ideal[distinct >= 2], 1.0)


def test_ideal_learner_no_examples_is_chance():
    b = make_batch("rules", 20, 0, 4, np.random.default_rng(1))
    assert not b.pvalid.any()
    assert np.allclose(b.ideal, 1 / P, atol=1e-9)


def test_facts_unseen_queries_are_unseen():
    b = make_batch("facts", 100, 8, 8, np.random.default_rng(2))
    assert b.qseen[:, :4].all()            # first half asks shown keys
    assert not b.qseen[:, 4:].any()        # second half never shown
    assert np.allclose(b.ideal[b.qseen], 1.0) and np.allclose(b.ideal[~b.qseen], CHANCE)


def test_meta_batch_shown_answers_consistent():
    b = make_meta_batch("facts", 30, 6, 8, np.random.default_rng(3))
    for i in range(30):
        shown = dict(zip(b.px[i], b.py[i]))
        for x, y in zip(b.qx[i], b.qy[i]):
            if x in shown:
                assert shown[x] == y


def test_size_budget_5m_under_20mb():
    m = TinyLearner(Config(kind="attn", **SIZES["5M"]))
    assert m.n_params() * 4 / 1e6 <= 20.0
    d = TinyLearner(Config(kind="mem", **SIZES["5M"]))
    assert d.n_params() * 4 / 1e6 <= 20.0


@pytest.mark.parametrize("kind", ["attn", "mem"])
def test_prefix_queries_cannot_see_their_own_answer(kind):
    """Changing y_j must not change predictions for queries i <= j (no leak)."""
    torch.manual_seed(0)
    m = TinyLearner(Config(kind=kind, d=32, layers=2, heads=2, ff=64)).eval()
    k = 6
    px = torch.randint(0, NX, (1, k))
    py = torch.randint(0, P, (1, k))
    pv = torch.ones(1, k, dtype=torch.bool)
    base = m(px, py, pv, px, prefix=True)
    py2 = py.clone()
    py2[0, 3] = (py2[0, 3] + 1) % P
    alt = m(px, py2, pv, px, prefix=True)
    assert torch.allclose(base[:, :4], alt[:, :4], atol=1e-5)    # queries 0..3 unaffected
    assert not torch.allclose(base[:, 4:], alt[:, 4:], atol=1e-5)  # later queries do see it


@pytest.mark.parametrize("kind", ["attn", "mem"])
def test_masked_pairs_are_ignored(kind):
    torch.manual_seed(0)
    m = TinyLearner(Config(kind=kind, d=32, layers=2, heads=2, ff=64)).eval()
    px = torch.randint(0, NX, (1, 5))
    py = torch.randint(0, P, (1, 5))
    pv = torch.tensor([[True, True, False, True, True]])
    qx = torch.randint(0, NX, (1, 3))
    base = m(px, py, pv, qx)
    py2 = py.clone()
    py2[0, 2] = (py2[0, 2] + 5) % P
    px2 = px.clone()
    px2[0, 2] = (px2[0, 2] + 3) % NX
    assert torch.allclose(base, m(px2, py2, pv, qx), atol=1e-5)


def test_attention_is_order_invariant_in_all_mode():
    """No position embeddings: with full (non-causal) reads, shuffling pairs only changes the
    causal pair-to-pair paths, so check the 1-layer case where queries read pairs directly."""
    torch.manual_seed(0)
    m = TinyLearner(Config(kind="attn", d=32, layers=1, heads=2, ff=64)).eval()
    px = torch.randint(0, NX, (1, 5))
    py = torch.randint(0, P, (1, 5))
    pv = torch.ones(1, 5, dtype=torch.bool)
    qx = torch.randint(0, NX, (1, 2))
    perm = torch.randperm(5)
    assert torch.allclose(m(px, py, pv, qx), m(px[:, perm], py[:, perm], pv, qx), atol=1e-5)


def test_zero_fast_weights_equal_frozen_model():
    torch.manual_seed(0)
    m = TinyLearner(Config(d=32, layers=3, heads=2, ff=64, n_fast=2)).eval()
    px, py = torch.randint(0, P, (2, 4)), torch.randint(0, P, (2, 4))
    pv, qx = torch.ones(2, 4, dtype=torch.bool), torch.randint(0, P, (2, 3))
    assert torch.allclose(m(px, py, pv, qx), m(px, py, pv, qx, fast=m.fast_init(2)), atol=1e-6)


def test_adapt_lowers_loo_loss_and_keeps_prompts_separate():
    torch.manual_seed(0)
    m = TinyLearner(Config(d=32, layers=2, heads=2, ff=64, n_fast=2))
    px, py = torch.randint(0, P, (2, 6)), torch.randint(0, P, (2, 6))
    pv = torch.ones(2, 6, dtype=torch.bool)
    before = loo_loss(m, px, py, pv, m.fast_init(2)).item()
    fast = adapt(m, px, py, pv, steps=5, lr=1.0)
    after = loo_loss(m, px, py, pv, fast).item()
    assert after < before
    # prompt 0's fast weights depend only on prompt 0
    py_alt = py.clone()
    py_alt[1] = (py_alt[1] + 1) % P
    fast_alt = adapt(m, px, py_alt, pv, steps=2, lr=1.0)
    fast_ref = adapt(m, px, py, pv, steps=2, lr=1.0)
    assert torch.allclose(fast_alt[0][0], fast_ref[0][0], atol=1e-6)


def test_meta_gradient_reaches_slow_weights_and_lr():
    torch.manual_seed(0)
    m = TinyLearner(Config(d=32, layers=2, heads=2, ff=64, n_fast=2))
    px, py = torch.randint(0, P, (2, 5)), torch.randint(0, P, (2, 5))
    pv, qx = torch.ones(2, 5, dtype=torch.bool), torch.randint(0, P, (2, 3))
    fast = adapt(m, px, py, pv, steps=2, lr=None, create_graph=True)
    m(px, py, pv, qx, fast=fast).sum().backward()
    assert m.log_lr.grad is not None and m.log_lr.grad.abs().sum() > 0
    assert m.blocks[-1].A.grad.abs().sum() > 0


def test_checkpoint_roundtrip(tmp_path):
    m = make_model("A", smoke=True)
    save(m, "A", {"train_seconds": 1.0}, tmp_path)
    m2, meta = load("A", "cpu", tmp_path)
    assert meta["train_seconds"] == 1.0
    assert set(m.state_dict()) == set(m2.state_dict())


def test_all_arms_build():
    for arm in TRAINED:
        assert make_model(arm, smoke=True).n_params() > 0


def test_criteria_verdict_logic():
    assert len(CRITERIA) == 5
    allp = {c: True for c in ("C1", "C2", "C3", "C4", "C5")}
    assert verdict(allp, True) == "PASS"
    assert verdict(allp, False) == "INCONCLUSIVE"
    assert verdict(dict(allp, C4=False), True) == "FAIL"
    assert verdict(dict(allp, C5=False), True) == "INCONCLUSIVE"
    assert verdict(dict(allp, C1=False, C2=False), True) == "INCONCLUSIVE"
    assert spearman([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.5]) == pytest.approx(1.0)


def test_quad_family_differs_from_affine():
    x = np.arange(P)[None]
    ab = np.array([[3, 4]])
    assert not np.array_equal(apply_rule("quad", ab, x), apply_rule("affine", ab, x))
