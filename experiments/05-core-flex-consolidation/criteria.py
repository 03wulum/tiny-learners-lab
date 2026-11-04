"""Pre-registered criteria for tiny-learners 05 "core + flexible weights" (written 2026-11-04,
committed before any GPU run). Obi's no-ask rule applies: published, then built and run.

Setting (see README / EXPLAINER): a continual-learning stream of 64 new skills. A skill is a
cue (a random point on a 64-d sphere, seen with noise) bound to a hidden rule y = (a*x + b) mod 17.
Frequent skills come back 12 times, occasional ones 4 times, one-offs once, in bursts of 8.
Every episode shows 6 worked examples of one skill; the learner gets 20 gradient steps.
"Recall" = answer all 17 inputs from the cue alone, with no examples in front of it.
All numbers are accuracy in points (0-100), mean over 3 seeds. Chance = 100/17 = 5.9.

Arms (same active parameter count): A frozen, B plain fine-tune, C online EWC,
D replay with a buffer as big (in bytes) as M's largest possible archive, M core + flex (ours).
Ablations: M without archive, flat LRU cache (one tier), flat strength cache, harden-at-once,
unlimited slots, capacity sweep. Controls: shuffled cues, never-seen skills, shuffled labels.
"""

CHANCE = 100.0 / 17

BUDGET = dict(active_mb=20.0, kept_mb=20.0, total_gpu_min=10.0, update_ms=50.0,
              route_us=100.0, peak_ram_gb=2.0)

# C2 (headline) retention at the end of the stream, cue only. Score = mean of
# (pretrained-skill recall, stream-skill recall), system as deployed (archive on for M).
C2_MIN_GAIN_OVER_FINETUNE = 30.0
C2_MIN_GAIN_OVER_REPLAY = 5.0
# C3 learning speed: recall right after a skill's FIRST episode (6 examples).
C3_MAX_DEFICIT_VS_FINETUNE = 5.0
# C4 recovery: when a skill returns after its slot went to the archive, recall before any
# new examples, as a fraction of its recall right after it was last trained.
C4_MIN_RECOVERY = 0.90
C4_MAX_RECOVERY_NOARCHIVE = 0.50     # the no-archive ablation must NOT recover (shows the cause)
# C5 hardening: frequent-skill recall from the ACTIVE weights only (archive switched off),
# measured right after each burst of one-offs, M vs a one-tier LRU cache of the same size.
C5_MIN_GAIN_OVER_FLAT_LRU = 10.0

# Validity gate (any failure -> INCONCLUSIVE)
GATE_PRETRAIN_MIN = 90.0             # both pretrained nets recall their pretrained skills
GATE_CONTROL_MAX = CHANCE + 5.0      # shuffled cues, never-seen skills, frozen on new skills,
                                     # shuffled labels (unseen inputs) stay near chance
GATE_MAX_ROUTE_ERRORS = 0.02         # fraction of episodes routed to the wrong slot

INTRIGUING = dict(heldout_x_after_first=50.0, near_unlimited=2.0)


def verdict(m):
    """m: measured numbers (see run.py aggregate). Returns rows, verdict, gate, notes."""
    rows = []

    def row(name, value, thr, ok):
        rows.append(dict(criterion=name, value=value, threshold=thr, result="PASS" if ok else "FAIL"))
        return ok

    b = m["budget"]
    c1 = row("C1 budget", b, BUDGET, all(b[k] <= BUDGET[k] for k in BUDGET))
    g_b, g_d = m["ret_M"] - m["ret_B"], m["ret_M"] - m["ret_D"]
    c2 = row("C2 retention (M - fine-tune, M - replay)", dict(vs_finetune=g_b, vs_replay=g_d),
             f">= {C2_MIN_GAIN_OVER_FINETUNE} and >= {C2_MIN_GAIN_OVER_REPLAY}",
             g_b >= C2_MIN_GAIN_OVER_FINETUNE and g_d >= C2_MIN_GAIN_OVER_REPLAY)
    d3 = m["first_M"] - m["first_B"]
    c3 = row("C3 learning speed (first episode, M - fine-tune)", d3,
             f">= -{C3_MAX_DEFICIT_VS_FINETUNE}", d3 >= -C3_MAX_DEFICIT_VS_FINETUNE)
    c4 = row("C4 recovery from archive (M, no-archive)",
             dict(M=m["recovery_M"], no_archive=m["recovery_noarch"]),
             f"M >= {C4_MIN_RECOVERY}, no-archive <= {C4_MAX_RECOVERY_NOARCHIVE}",
             m["recovery_M"] >= C4_MIN_RECOVERY and m["recovery_noarch"] <= C4_MAX_RECOVERY_NOARCHIVE)
    g5 = m["burst_M"] - m["burst_flatLRU"]
    c5 = row("C5 hardening (frequent skills after bursts, M - flat LRU)", g5,
             f">= {C5_MIN_GAIN_OVER_FLAT_LRU}", g5 >= C5_MIN_GAIN_OVER_FLAT_LRU)

    gate = dict(pretrain=min(m["pretrain_core"], m["pretrain_plain"]) >= GATE_PRETRAIN_MIN,
                shuffled_cues=m["ctrl_shuffled_cue"] <= GATE_CONTROL_MAX,
                never_seen=m["ctrl_never_seen"] <= GATE_CONTROL_MAX,
                frozen_new=m["ctrl_frozen_new"] <= GATE_CONTROL_MAX,
                shuffled_labels=m["ctrl_shuffled_labels_heldout"] <= GATE_CONTROL_MAX,
                routing=m["route_error_rate"] <= GATE_MAX_ROUTE_ERRORS)
    # headline too close to call: within one seed-std of either C2 threshold
    noisy = (abs(g_b - C2_MIN_GAIN_OVER_FINETUNE) < m["ret_seed_std"] or
             abs(g_d - C2_MIN_GAIN_OVER_REPLAY) < m["ret_seed_std"])
    if not all(gate.values()) or noisy:
        v = "INCONCLUSIVE"
    elif c1 and c2 and c3 and c4 and c5:
        v = "PASS"
    else:
        v = "FAIL"
    notes = []
    if m["first_heldout_M"] >= INTRIGUING["heldout_x_after_first"]:
        notes.append(f"after 6 examples M already answers {m['first_heldout_M']:.0f}% of the 11 "
                     "inputs it never saw (it found the rule, not just the pairs)")
    if m["ret_unlimited"] - m["ret_M"] <= INTRIGUING["near_unlimited"]:
        notes.append("the fixed-size model retained as much as one with unlimited slots")
    if m["burst_flatStrength"] >= m["burst_M"]:
        notes.append("a one-tier cache that evicts by strength matched the two tiers")
    return rows, v, dict(gate=gate, headline_noisy=noisy), notes
