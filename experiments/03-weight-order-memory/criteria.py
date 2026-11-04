"""Pre-registered criteria for tiny-learners 03 (approved by Obi 2026-11-04, before any run).
Plan: https://claude.ai/artifact/5HYngMFxqr8T5yjQyDjEDh
Accuracies are exact-match on held-out problems, in points (0-100), mean over 3 seeds.
'Best P' = the better of P1 / P2, chosen on the validation split (never on test)."""

BUDGET = dict(model_mb=20.0, kept_mb=50.0, base_train_min=10.0, total_gpu_min=20.0,
              adapt_ms=100.0, peak_ram_gb=2.0)
C2_MIN_GAIN_OVER_FROZEN = 10.0     # T1, k=8
C2_MIN_GAIN_OVER_RANDOM = 10.0
C3_MAX_DEFICIT_VS_BITMATCHED = 3.0 # avg T1-T3, k=8
C4_MAX_RETENTION_DROP = 3.0        # after the 20-problem stream
GATE_SYMMETRY_MAX_DIFF = 1e-5      # neuron permutation (with outgoing weights), float64
GATE_CONTROL_MAX_GAIN = 3.0        # random reorder and shuffled labels vs frozen
# Added before any run, flagged to Obi: if frozen in-context accuracy on T1 k=8 is already
# >= 90%, C2 has no headroom, so the verdict is INCONCLUSIVE rather than FAIL.
HEADROOM_MAX_FROZEN = 90.0

INTRIGUING = dict(beat_bitmatched_by=5.0, stream_fewer_examples=0.25)


def verdict(m):
    """m: dict of measured numbers (see run.py). Returns (rows, verdict, intriguing_notes)."""
    rows = []

    def row(name, value, thr, ok):
        rows.append(dict(criterion=name, value=value, threshold=thr, result="PASS" if ok else "FAIL"))
        return ok

    b = m["budget"]
    c1 = row("C1 budget", b, BUDGET, all([
        b["model_mb"] <= BUDGET["model_mb"], b["kept_mb"] <= BUDGET["kept_mb"],
        b["base_train_min"] <= BUDGET["base_train_min"], b["total_gpu_min"] <= BUDGET["total_gpu_min"],
        b["adapt_ms"] <= BUDGET["adapt_ms"], b["peak_ram_gb"] <= BUDGET["peak_ram_gb"]]))
    g_fro, g_rnd = m["T1_bestP"] - m["T1_A"], m["T1_bestP"] - m["T1_R"]
    c2 = row("C2 reordering learns (T1 k=8)",
             dict(gain_vs_frozen=g_fro, gain_vs_random=g_rnd),
             f">= {C2_MIN_GAIN_OVER_FROZEN} and >= {C2_MIN_GAIN_OVER_RANDOM}",
             g_fro >= C2_MIN_GAIN_OVER_FROZEN and g_rnd >= C2_MIN_GAIN_OVER_RANDOM)
    d3 = m["avg_bestP"] - m["avg_Bq"]
    c3 = row("C3 bits well spent (avg T1-T3 k=8, bestP - B')", d3,
             f">= -{C3_MAX_DEFICIT_VS_BITMATCHED}", d3 >= -C3_MAX_DEFICIT_VS_BITMATCHED)
    c4 = row("C4 keeps old skills (retention drop)", m["retention_drop_bestP"],
             f"<= {C4_MAX_RETENTION_DROP}", m["retention_drop_bestP"] <= C4_MAX_RETENTION_DROP)

    gate = dict(symmetry=m["symmetry_max_diff"] < GATE_SYMMETRY_MAX_DIFF,
                random=m["T1_R"] - m["T1_A"] <= GATE_CONTROL_MAX_GAIN,
                shuffled=m["T1_bestP_shuf"] - m["T1_A_shuf"] <= GATE_CONTROL_MAX_GAIN,
                headroom=m["T1_A"] < HEADROOM_MAX_FROZEN)
    noisy = abs(d3 + C3_MAX_DEFICIT_VS_BITMATCHED) < m["c3_seed_std"]
    if not all(gate.values()) or noisy:
        v = "INCONCLUSIVE"
    elif c1 and c2 and c3 and c4:
        v = "PASS"
    elif not (c2 and c3):
        v = "FAIL"
    else:
        v = "FAIL"   # C1 or C4 failed: the claim holds only partially; details in results.md
    notes = []
    if d3 >= INTRIGUING["beat_bitmatched_by"]:
        notes.append(f"reordering beat the bit-matched fine-tune by {d3:.1f} pts")
    if m["avg_bestP"] > m["avg_B"]:
        notes.append("reordering beat the full, un-squeezed fine-tune")
    if m.get("T4_bestP_beats_all"):
        notes.append("on the unseen number base, reordering beat every other arm")
    if m.get("stream_reduction", 0) >= INTRIGUING["stream_fewer_examples"]:
        notes.append(f"later stream problems needed {100*m['stream_reduction']:.0f}% fewer examples")
    return rows, v, dict(gate=gate, c3_noisy=noisy), notes
