"""Pre-registered criteria for tiny-learners 04, published before any training run (2026-11-04).
Plan: https://claude.ai/artifact/PCGfLn1fM2k8YpgAjzfjQ4
Accuracy = exact match on held-out questions (family: the whole answer set must match), points 0-100,
mean of 3 seeds, k = 4 worked examples per new operation. 'Best baseline' = the better of two sizes.
'Program search per new operation' = picking a program from the prebuilt library (the library is built
once per world and its build time is reported separately)."""

BUDGET = dict(composer_mb=1.0, kept_mb=50.0, train_min=10.0, answer_ms=1.0, search_s=1.0, peak_ram_gb=2.0)
C2_MIN_ACC, C2_MIN_MARGIN = 90.0, 30.0      # arithmetic, 5 held-out operations
C3_MIN_ACC, C3_MAX_FACTS = 95.0, 100        # a - b, with zero subtractions in training
C4_MIN_ACC, C4_MIN_MARGIN = 90.0, 30.0      # family, 6 held-out relations
G1_MIN_BASELINE_OWN = 80.0                  # baselines learned their own training tasks
G2_MIN_ORACLE = 70.0                        # baseline architecture can represent the targets
G3_MAX_CONTROL = 5.0                        # negative controls must fail
INTRIGUING = dict(one_shot_acc=90.0, one_shot_ops=3, latent_steps=50, latent_acc=90.0)


def verdict(m):
    """m: measured numbers from run.py. Returns (rows, verdict, gate, notes)."""
    rows = []

    def row(name, value, thr, ok):
        rows.append(dict(criterion=name, value=value, threshold=thr, result="PASS" if ok else "FAIL"))
        return ok

    b = m["budget"]
    c1 = row("C1 budget", b, BUDGET, b["composer_mb"] <= BUDGET["composer_mb"] and b["kept_mb"] <= BUDGET["kept_mb"]
             and b["train_min"] <= BUDGET["train_min"] and b["answer_ms"] <= BUDGET["answer_ms"]
             and b["search_s"] <= BUDGET["search_s"] and b["peak_ram_gb"] <= BUDGET["peak_ram_gb"])
    a = m["arith"]
    c2 = row("C2 arithmetic, 5 unseen ops (composer mean, margin over best baseline)",
             dict(composer=a["fp_mean"], best_baseline=a["base_best_mean"], margin=a["fp_mean"] - a["base_best_mean"]),
             f">= {C2_MIN_ACC} and margin >= {C2_MIN_MARGIN}",
             a["fp_mean"] >= C2_MIN_ACC and a["fp_mean"] - a["base_best_mean"] >= C2_MIN_MARGIN)
    c3 = row("C3 going the opposite way: a - b", dict(composer=a["fp"]["sub"], training_facts=a["facts"]),
             f">= {C3_MIN_ACC}, facts <= {C3_MAX_FACTS}, 0 subtractions",
             a["fp"]["sub"] >= C3_MIN_ACC and a["facts"] <= C3_MAX_FACTS)
    r = m["family"]
    c4 = row("C4 family, 6 unseen relations (composer mean, margin over best baseline)",
             dict(composer=r["fp_mean"], best_baseline=r["base_best_mean"], margin=r["fp_mean"] - r["base_best_mean"]),
             f">= {C4_MIN_ACC} and margin >= {C4_MIN_MARGIN}",
             r["fp_mean"] >= C4_MIN_ACC and r["fp_mean"] - r["base_best_mean"] >= C4_MIN_MARGIN)

    gate = {}
    for dom, d in (("arith", a), ("family", r)):
        gate[f"G1 {dom} baseline own tasks"] = d["base_own_best"] >= G1_MIN_BASELINE_OWN
        gate[f"G2 {dom} oracle"] = d["oracle_mean"] >= G2_MIN_ORACLE
        for c, v in d["controls"].items():
            gate[f"G3 {dom} {c}"] = v <= G3_MAX_CONTROL
    if not all(gate.values()):
        v = "INCONCLUSIVE"
    elif c1 and c2 and c3 and c4:
        v = "PASS"
    else:
        v = "FAIL"

    notes = []
    one = [op for op, acc in {**a["fp_k1"], **r["fp_k1"]}.items() if acc >= INTRIGUING["one_shot_acc"]]
    if len(one) >= INTRIGUING["one_shot_ops"]:
        notes.append(f"from a single example the composer got >= 90% on {len(one)} unseen operations ({', '.join(one)})")
    beaten = [op for op in a["fp"] if a["base_best"][op] > a["fp"][op]] + \
             [rel for rel in r["fp"] if r["base_best"][rel] > r["fp"][rel]]
    if beaten:
        notes.append(f"a baseline beat the composer on: {', '.join(beaten)}")
    lat = a["latent_chain"].get(INTRIGUING["latent_steps"], 0.0)
    if lat >= INTRIGUING["latent_acc"]:
        notes.append(f"chaining in hidden space without snapping stayed at {lat:.0f}% after 50 steps")
    return rows, v, gate, notes
