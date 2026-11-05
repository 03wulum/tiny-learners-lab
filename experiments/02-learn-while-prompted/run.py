"""Score every arm on every task and print the pre-registered PASS/FAIL table.
Evaluation only: this script never trains.

    python run.py            # needs checkpoints/ (train on the GPU, or download the release)
    python run.py --quick    # CI smoke check: tiny UNTRAINED models, a few prompts
    python run.py --clean [--all]   # delete caches and quick results (--all: checkpoints too)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import criteria
from arms import CKPT_DIR, EVAL_ARMS, NAIVE_LR_GRID, TRAINED, TTT_STEPS, load, make_model
from tasks import CHANCE, HELDOUT_RULES, TRAIN_RULES, make_batch
from ttt import adapt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
K_LIST = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
TASKS = {  # name -> (generator task, rules)
    "rules_train": ("rules", TRAIN_RULES),
    "rules_heldout": ("rules", HELDOUT_RULES),
    "facts": ("facts", None),
    "quad": ("quad", None),
    "shuffled": ("shuffled", None),
}


def predict(model, ttt, lr, px, py, pv, qx):
    fast = None
    if ttt and px.shape[1] >= 2 and pv.any():
        with torch.enable_grad():
            fast = adapt(model, px, py, pv, TTT_STEPS, lr=None if ttt == "meta" else lr)
        fast = [f.detach() for f in fast]
    with torch.no_grad():
        return model(px, py, pv, qx, fast=fast).argmax(-1)


def eval_arm(model, ttt, lr, task, k, n, dev, seed=0, chunk=64):
    """Accuracy on n prompts with k examples and 8 queries each. Same prompts for every arm."""
    gen, rules = TASKS[task]
    rng = np.random.default_rng(seed * 1000 + k * 7 + list(TASKS).index(task))
    b = make_batch(gen, n, k, 8, rng, rules=rules)
    t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=dev)
    preds = []
    c = max(1, chunk // max(1, k // 8)) if ttt else chunk * 4
    for i in range(0, n, c):
        s = slice(i, i + c)
        preds.append(predict(model, ttt, lr, t(b.px[s]), t(b.py[s]), t(b.pvalid[s], torch.bool), t(b.qx[s])).cpu())
    correct = (torch.cat(preds).numpy() == b.qy)
    out = {"all": float(correct.mean()), "ideal": float(b.ideal.mean())}
    if task == "facts":
        out["shown"] = float(correct[b.qseen].mean()) if b.qseen.any() else CHANCE
        out["unseen"] = float(correct[~b.qseen].mean()) if (~b.qseen).any() else CHANCE
        out["ideal_shown"] = float(b.ideal[b.qseen].mean()) if b.qseen.any() else CHANCE
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="prompts per (task, k)")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--all", action="store_true", help="with --clean: also delete checkpoints")
    a = ap.parse_args()
    if a.clean:
        for p in [HERE / "__pycache__", HERE / ".pytest_cache", RES / "quick"] + ([CKPT_DIR] if a.all else []):
            if p.exists():
                shutil.rmtree(p)
                print(f"deleted {p.relative_to(HERE)}")
        return
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    have = all((CKPT_DIR / f"{arm}.pt").exists() for arm in TRAINED)
    if not a.quick and not have:
        sys.exit("No checkpoints in ckpt/. Train on the GPU (python train.py) or download "
                 "them from the GitHub release (see README).")
    n = a.n or (8 if a.quick else 300)
    k_list = [0, 2, 4, 8, 16, 64] if a.quick else K_LIST

    models, train_meta = {}, {}
    torch.manual_seed(0)
    for arm in TRAINED:
        if a.quick and not have:
            models[arm] = make_model(arm, smoke=True).to(dev).eval()
            train_meta[arm] = {"train_seconds": 0.0, "params": models[arm].n_params()}
        else:
            models[arm], train_meta[arm] = load(arm, dev)
    untrained = (make_model("A", smoke=True) if a.quick and not have else make_model("A")).to(dev).eval()

    # naive test-time lr for arm B: best on TRAINING rules and facts at k=8 (never held-out data)
    lr_scores = {}
    for lr in NAIVE_LR_GRID:
        lr_scores[lr] = np.mean([eval_arm(models["A"], "naive", lr, t, 8, max(8, n // 4), dev, seed=99)["all"]
                                 for t in ("rules_train", "facts")])
    naive_lr = max(lr_scores, key=lr_scores.get)
    print(f"arm B naive lr = {naive_lr} (grid scores {', '.join(f'{k}: {v:.3f}' for k, v in lr_scores.items())})")

    acc, ideal = {}, {}
    arms = dict(EVAL_ARMS, untrained=(None, None))
    for arm, (ck, ttt) in arms.items():
        model = untrained if arm == "untrained" else models[ck]
        acc[arm] = {}
        tasks = ["rules_heldout", "shuffled"] if arm == "untrained" else list(TASKS)
        for task in tasks:
            for k in k_list:
                r = eval_arm(model, ttt, naive_lr, task, k, n, dev)
                acc[arm].setdefault(task, {})[str(k)] = r["all"]
                ideal.setdefault(task, {})[str(k)] = r["ideal"]
                if task == "facts":
                    acc[arm].setdefault("facts_shown", {})[str(k)] = r["shown"]
                    acc[arm].setdefault("facts_unseen", {})[str(k)] = r["unseen"]
                    ideal.setdefault("facts_shown", {})[str(k)] = r["ideal_shown"]
        print(f"{arm:>9}: held-out rules " + " ".join(f"k{k}={acc[arm]['rules_heldout'][str(k)]:.2f}" for k in k_list),
              flush=True)

    # time to learn one 16-example prompt with arm C
    b = make_batch("rules", 1, 16, 8, np.random.default_rng(5), rules=HELDOUT_RULES)
    t = lambda x, dt=torch.long: torch.as_tensor(x, dtype=dt, device=dev)
    times = []
    for _ in range(5):
        s = time.time()
        predict(models["C"], "meta", None, t(b.px), t(b.py), t(b.pvalid, torch.bool), t(b.qx))
        if dev == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - s)

    sizes = {arm: m.n_params() * 4 / 1e6 for arm, m in models.items()}
    files = [CKPT_DIR / f"{arm}.pt" for arm in TRAINED]
    tsum = {}
    if (CKPT_DIR / "train_all.json").exists():
        tsum = json.loads((CKPT_DIR / "train_all.json").read_text())
    arm_minutes = [train_meta[arm].get("train_seconds", 0) / 60 for arm in TRAINED]
    metrics = {
        "k_list": k_list, "n_prompts": n, "device": dev, "naive_lr": naive_lr, "acc": acc, "ideal": ideal,
        "budget": {
            "model_mb_fp32_max": max(sizes.values()),
            "model_mb_fp32": sizes,
            "artifacts_mb": sum(f.stat().st_size for f in files if f.exists()) / 1e6,
            "train_minutes_arm_max": max(arm_minutes),
            "train_minutes_total": tsum.get("train_minutes", tsum.get("wall_minutes", max(arm_minutes))),
            "adapt_seconds": float(np.median(times)),
            "adapt_device": dev,
        },
        "train": {arm: {k: v for k, v in train_meta[arm].items() if k != "log"} for arm in TRAINED},
    }
    table, passed, gate = criteria.score(metrics)
    v = criteria.verdict(passed, gate)
    runtime = time.time() - t0
    try:
        import resource  # Linux/macOS only
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB
    except ImportError:  # Windows
        peak = 0.0

    print("\n" + "-" * 120)
    print(f"{'id':<4}{'criterion':<52}{'value':<58}PASS?")
    for r in table:
        print(f"{r['id']:<4}{r['name']:<52}{r['value']:<58}{'PASS' if r['pass'] else 'FAIL'}")
    print(f"base model learned (arm A >= 25% on training rules and shown facts at k=16): "
          f"{'yes' if metrics['learned'] else 'NO: this run cannot test the idea'}")
    print(f"validity gate (controls at chance, base model learned): {'OK' if gate else 'FAILED'}")
    label = " (quick: untrained toy models, not a result)" if a.quick and not have else ""
    print(f"VERDICT: {v}{label}   [{runtime:.0f}s, {peak:.2f} GB peak RAM, eval on {dev}]")

    out_dir = RES / "quick" if a.quick else RES
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"experiment": "02-learn-while-prompted", "verdict": v, "quick": a.quick, "untrained_toy": a.quick and not have,
           "criteria": table, "validity_gate": gate, "base_model_learned": metrics["learned"], "runtime_s": runtime, "peak_ram_gb": peak,
           "hypothesis": criteria.HYPOTHESIS}
    (out_dir / "verdict.json").write_text(json.dumps(out, indent=1))
    if not a.quick:
        (RES / "metrics.json").write_text(json.dumps(metrics, indent=1))
        try:
            import plots
            plots.make_all(metrics, RES)
        except ImportError as e:
            print(f"plots skipped: {e}")
    if a.quick and runtime > criteria.BUDGET["quick_seconds"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
