"""Pre-registered hypothesis, pass criteria, controls and budgets for experiment 02.

Approved by Obi on 2026-11-04 (plan: https://claude.ai/artifact/QJmX6kJcpcrz43cT4kMQkh)
before any GPU training. Do not edit thresholds after results exist; add a new
experiment instead.
"""
from __future__ import annotations

from tasks import CHANCE

HYPOTHESIS = (
    "A model of 20 MB or less can learn new rules and facts from its prompt, and with a "
    "learned update rule (meta-learned test-time training or a fast-weight memory) it keeps "
    "improving when the prompt holds more examples than it ever trained on."
)

ASSUMPTIONS = [
    "Modular-arithmetic rules and random lookup tables are fair stand-ins for 'a new rule' and "
    "'new facts': generated per prompt, so nothing can be memorized from training.",
    "5M parameters and 12 minutes on one consumer NVIDIA GPU are enough capacity and time.",
    "First-order-style meta-learning through 2 inner steps is stable at this size.",
]

BUDGET = {
    "model_mb_fp32": 20.0,       # each model's weights at fp32
    "artifacts_mb": 50.0,        # all kept checkpoints together (stored fp16)
    "train_minutes_arm": 15.0,   # each arm, on Fortran's GPU
    "train_minutes_total": 20.0,  # all arms together, wall-clock
    "adapt_seconds": 1.0,        # learning one 16-example prompt (arm C), on GPU
    "quick_seconds": 60.0,       # run.py --quick in CI
}

CRITERIA = [
    {"id": "C1", "name": "Fits the budget",
     "desc": "Each model <= 20 MB, all artifacts <= 50 MB, <= 15 min/arm and <= 20 min total GPU "
             "training, <= 1 s to learn one prompt", "metric": "budget_ok", "op": "==", "threshold": 1},
    {"id": "C2", "name": "Learns new rules (T1)",
     "desc": "Best arm on held-out rules at k=8 (ideal learner ~100%); curve rises over k=0..8 "
             "(Spearman >= 0.9)", "metric": "c2_ok", "op": "==", "threshold": 1},
    {"id": "C3", "name": "Learns new facts (T2)",
     "desc": "Best arm's recall of shown facts at k=16 >= 90%, unseen keys within 3 pts of chance",
     "metric": "c3_ok", "op": "==", "threshold": 1},
    {"id": "C4", "name": "Keeps improving past its training (T3, headline)",
     "desc": "Best of C/D at k=64 (trained to k=16) >= 80% of the ideal learner and >= A + 15 pts",
     "metric": "c4_ok", "op": "==", "threshold": 1},
    {"id": "C5", "name": "Updating must be learned",
     "desc": "Arm C beats arm B (naive test-time SGD) by >= 10 pts on held-out rules at k=4",
     "metric": "c5_ok", "op": "==", "threshold": 1},
]

THRESH = {"c2_acc": 0.80, "c2_rho": 0.90, "c3_recall": 0.90, "c3_unseen_tol": 0.03,
          "c4_of_ideal": 0.80, "c4_gain": 0.15, "c5_gain": 0.10, "gate_tol": 0.03}

CONTROLS = {
    "ceiling": "Ideal learner: brute force over every rule consistent with the prompt.",
    "negative": "Shuffled labels (every arm) and an untrained model must stay within 3 pts of "
                "chance (1/17 = 5.9%); unseen-key questions must stay at chance.",
    "baseline": "Arm A, the frozen in-context learner; arm B, the same weights plus naive SGD.",
    "ablation": "C-frozen: arm C's weights without its update; size sweep A at 0.1M / 1M / 5M.",
}


CORE = "C4"  # the headline claim

def verdict(passed: dict[str, bool], gate_ok: bool) -> str:
    """SOP v0.6 verdict words.
    PASS: all five criteria pass and the negative controls stay at chance.
    FAIL: the core criterion (C4) fails.
    INCONCLUSIVE: mixed results, or a negative control learned something (leak or bug).
    (The plan's original wording was CONFIRMED / PARTIAL / NOT CONFIRMED / INVALID; SOP v0.6
    renamed the verdicts before any GPU run. Thresholds are unchanged.)"""
    if not gate_ok:
        return "INCONCLUSIVE"
    if all(passed.values()):
        return "PASS"
    if not passed[CORE]:
        return "FAIL"
    return "INCONCLUSIVE"


def spearman(xs, ys) -> float:
    import numpy as np
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def score(m: dict) -> tuple[list[dict], dict[str, bool], bool]:
    """m = metrics from run.py. Returns table rows, pass flags, and the validity gate."""
    T = THRESH
    acc = m["acc"]  # acc[arm][task][str(k)]
    arms = [a for a in acc if a != "untrained"]
    rows, passed = [], {}

    b = m["budget"]
    c1 = (b["model_mb_fp32_max"] <= BUDGET["model_mb_fp32"] and b["artifacts_mb"] <= BUDGET["artifacts_mb"]
          and b["train_minutes_arm_max"] <= BUDGET["train_minutes_arm"]
          and b["train_minutes_total"] <= BUDGET["train_minutes_total"]
          and b["adapt_seconds"] <= BUDGET["adapt_seconds"])
    passed["C1"] = bool(c1)
    rows.append(("C1", f"{b['model_mb_fp32_max']:.1f} MB/model, {b['artifacts_mb']:.1f} MB total, "
                 f"{b['train_minutes_arm_max']:.1f}/{b['train_minutes_total']:.1f} min, "
                 f"{b['adapt_seconds']:.2f} s/prompt", "20 MB, 50 MB, 15/20 min, 1 s"))

    best2 = max(arms, key=lambda a: acc[a]["rules_heldout"]["8"])
    ks = [k for k in m["k_list"] if k <= 8]
    rho = spearman(ks, [acc[best2]["rules_heldout"][str(k)] for k in ks])
    v2 = acc[best2]["rules_heldout"]["8"]
    passed["C2"] = v2 >= T["c2_acc"] and rho >= T["c2_rho"]
    rows.append(("C2", f"{best2}: {v2:.1%}, rho {rho:.2f}", f">= {T['c2_acc']:.0%}, rho >= {T['c2_rho']}"))

    best3 = max(arms, key=lambda a: acc[a]["facts_shown"]["16"])
    v3, u3 = acc[best3]["facts_shown"]["16"], acc[best3]["facts_unseen"]["16"]
    passed["C3"] = v3 >= T["c3_recall"] and abs(u3 - CHANCE) <= T["c3_unseen_tol"]
    rows.append(("C3", f"{best3}: {v3:.1%} recall, unseen {u3:.1%}", f">= {T['c3_recall']:.0%}, unseen 5.9% +/- 3"))

    def s64(a):
        return (acc[a]["rules_heldout"]["64"] + acc[a]["facts_shown"]["64"]) / 2
    ideal64 = (m["ideal"]["rules_heldout"]["64"] + m["ideal"]["facts_shown"]["64"]) / 2
    best4 = max(["C", "D"], key=s64)
    frac, gain = s64(best4) / ideal64, s64(best4) - s64("A")
    passed["C4"] = frac >= T["c4_of_ideal"] and gain >= T["c4_gain"]
    rows.append(("C4", f"{best4}: {frac:.0%} of ideal, {gain:+.1%} vs A", f">= 80% of ideal, >= +15 pts"))

    g5 = acc["C"]["rules_heldout"]["4"] - acc["B"]["rules_heldout"]["4"]
    passed["C5"] = g5 >= T["c5_gain"]
    rows.append(("C5", f"C {acc['C']['rules_heldout']['4']:.1%} vs B {acc['B']['rules_heldout']['4']:.1%} "
                 f"({g5:+.1%})", ">= +10 pts"))

    gate = all(abs(acc[a]["shuffled"]["8"] - CHANCE) <= T["gate_tol"] for a in acc)
    gate &= abs(acc["untrained"]["rules_heldout"]["8"] - CHANCE) <= T["gate_tol"]
    table = [{"id": cid, "name": c["name"], "value": val, "threshold": th, "pass": passed[cid]}
             for (cid, val, th), c in zip(rows, CRITERIA)]
    return table, passed, bool(gate)
