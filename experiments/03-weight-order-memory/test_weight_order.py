import itertools
import numpy as np
import pytest
import torch

import tasks, criteria
from model import TinyLM, answer_loss
from adapt import (project_rows, project_matrix, project_neuron_wiring, random_reorder,
                   quantize_delta, log2_factorial, finetune, expand, adapt, to_t)
from run import symmetry_check, eval_seed, QUICK
from train import SMOKE_DIMS

torch.manual_seed(0)


def tiny():
    return TinyLM(tasks.VOCAB, tasks.MAX_LEN, **SMOKE_DIMS)


def test_bits():
    assert abs(log2_factorial(256) - 1684) < 1


def test_project_rows_is_pure_reordering():
    W = torch.randn(5, 7)
    S = torch.randn(3, 5, 7)
    P = project_rows(W, S)
    assert torch.equal(P.sort(-1).values, W.sort(-1).values.expand_as(P))
    assert torch.equal(project_rows(W, W.unsqueeze(0)), W.unsqueeze(0))


def test_project_rows_is_l2_closest():
    W, S = torch.randn(1, 5), torch.randn(1, 1, 5)
    best = min(((W[0, list(p)] - S[0, 0]) ** 2).sum().item() for p in itertools.permutations(range(5)))
    assert abs(((project_rows(W, S) - S) ** 2).sum().item() - best) < 1e-5


def test_matrix_and_neuron_projection_keep_values():
    W = torch.randn(6, 4)
    S = W.unsqueeze(0) + 0.5 * torch.randn(2, 6, 4)
    assert torch.equal(project_matrix(W, S).flatten(1).sort(-1).values, W.flatten().sort().values.expand(2, -1))
    N = project_neuron_wiring(W, S)
    for b in range(2):
        assert sorted(map(tuple, N[b].tolist())) == sorted(map(tuple, W.tolist()))


def test_random_reorder_keeps_rows_and_moves_some():
    W = [torch.randn(8, 50)]
    R = random_reorder(W, 0.3, 4)[0]
    assert torch.equal(R.sort(-1).values, W[0].sort(-1).values.expand_as(R))
    moved = (R != W[0]).float().mean().item()
    assert 0.1 < moved < 0.4
    assert torch.equal(random_reorder(W, 0.0, 2)[0], W[0].expand(2, -1, -1))


def test_symmetry_gate():
    sym, broken = symmetry_check(tiny(), "cpu")
    assert sym < 1e-10 and broken > 1e-8


def test_quantize_delta():
    d = torch.randn(3, 4, 64)
    assert (quantize_delta(d, 16) - d).abs().max() < 1e-3
    assert len(torch.unique(quantize_delta(d, 1)[0, 0])) <= 2


def test_batched_block_matches_shared():
    m = tiny().eval()
    tb = to_t(tasks.make_batch(np.random.default_rng(0), "T3", 3, 4, "train"), "cpu")
    a = m(tb["full"])
    b = m(tb["full"], block=expand(m.block_weights(), 3))
    assert torch.allclose(a, b, atol=1e-5)


def test_finetune_lowers_support_loss():
    m = tiny().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    tb = to_t(tasks.make_batch(np.random.default_rng(1), "T1", 4, 8, "train"), "cpu")
    W0 = expand(m.block_weights(), 4)
    l0 = answer_loss(m(tb["supp"], block=W0), tb["supp"], tb["smask"]).item()
    W1 = finetune(m, W0, tb, 1e-2, 20)
    l1 = answer_loss(m(tb["supp"], block=W1), tb["supp"], tb["smask"]).item()
    assert l1 < l0


@pytest.mark.parametrize("task", tasks.TASKS)
def test_task_answers(task):
    rng = np.random.default_rng(3)
    x, y = tasks.GEN[task](rng, 20, 4, "test")
    if task == "T1":
        rules = tasks.T1_SPLIT["test"]
        for i in range(20):
            assert any(((a * x[i] + b) % 17 == y[i]).all() for a, b in rules)
    if task == "T4":
        B = 7
        u, v = x[..., 0] * B + x[..., 1], x[..., 3] * B + x[..., 4]
        assert ((y[..., 0] * B * B + y[..., 1] * B + y[..., 2]) == u + v).all()
    if task == "T3":
        r, m, c = np.array([1]), np.array([1]), np.array([0])
        assert tasks.t3_apply(np.array([[[0, 1, 2, 3]]]), r, m, c).tolist() == [[[2, 1, 0, 3]]]
    b = tasks.make_batch(rng, task, 5, 3)
    assert b["ymask"].sum(1).tolist() == [tasks.Y_LEN[task]] * 5
    assert b["smask"].sum(1).tolist() == [3 * tasks.Y_LEN[task]] * 5


def test_splits_disjoint():
    for sp in (tasks.T1_SPLIT, tasks.T3_SPLIT):
        sets = {k: set(map(tuple, v.tolist())) for k, v in sp.items()}
        assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"])


def test_verdict_logic():
    base = dict(budget=dict(model_mb=15, kept_mb=47, base_train_min=9, total_gpu_min=18, adapt_ms=50,
                            peak_ram_gb=1.5),
                T1_A=60, T1_R=59, T1_bestP=75, avg_bestP=70, avg_Bq=71, avg_B=72, c3_seed_std=0.5,
                retention_drop_bestP=1, symmetry_max_diff=1e-12, T1_bestP_shuf=6, T1_A_shuf=6)
    assert criteria.verdict(base)[1] == "PASS"
    assert criteria.verdict(dict(base, avg_bestP=60))[1] == "FAIL"
    assert criteria.verdict(dict(base, T1_A=95, T1_bestP=99))[1] == "INCONCLUSIVE"
    assert criteria.verdict(dict(base, T1_R=70))[1] == "INCONCLUSIVE"
    assert criteria.verdict(dict(base, avg_bestP=80))[3]          # intriguing note


def test_quick_end_to_end():
    m = tiny()
    from model import Controller
    W = m.block_weights()
    r = eval_seed(0, QUICK, "cpu", m, Controller(SMOKE_DIMS["d"], W[0].shape[0], W[0].shape[1], len(W)))
    assert set(r["main"]["T1"]) >= {"A", "B", "Bq", "P1", "P2", "M", "R", "oracle"}
