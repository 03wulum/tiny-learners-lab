# Results: tiny-learners 05, core + flexible weights

**Verdict: FAIL** · ✨ INTRIGUING: the fixed-size model retained as much as one with unlimited slots; a one-tier cache that evicts by strength matched the two tiers

| Criterion | Value | Threshold | Result |
|---|---|---|---|
| C1 budget | active_mb 0.5, kept_mb 0.9, total_gpu_min 1.9, update_ms 43.3, route_us 3.3, peak_ram_gb 1.6 | active_mb ≤ 20.0, kept_mb ≤ 20.0, total_gpu_min ≤ 10.0, update_ms ≤ 50.0, route_us ≤ 100.0, peak_ram_gb ≤ 2.0 | PASS |
| C2 retention (M - fine-tune, M - replay) | vs_finetune 29.4, vs_replay 14.3 | >= 30.0 and >= 5.0 | FAIL |
| C3 learning speed (first episode, M - fine-tune) | -1.9 | >= -5.0 | PASS |
| C4 recovery from archive (M, no-archive) | M 1.0, no_archive 0.2 | M >= 0.9, no-archive <= 0.5 | PASS |
| C5 hardening (frequent skills after bursts, M - flat LRU) | 30.0 | >= 10.0 | PASS |

Validity gate: pretrain ✓, shuffled_cues ✓, never_seen ✓, frozen_new ✓, shuffled_labels ✓, routing ✓; headline too close to call: False

## Every arm (accuracy in points, mean of 3 seeds, chance 5.9)

| Arm | Retention score | Pretrained skills | Stream skills | frequent | occasional | one-off | Stream, active weights only | After 1st episode | 1st episode, unseen x | Recall on return / last | Frequent after bursts (active) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A frozen | 52.5 | 100.0 | 4.9 | 4.9 | 4.7 | 5.1 | 4.9 | 5.1 | 4.9 | 0.78 | 6.2 |
| B fine-tune | 50.1 | 73.9 | 26.3 | 67.4 | 32.4 | 11.5 | 26.3 | 37.3 | 3.6 | 0.66 | 39.9 |
| C EWC | 27.6 | 26.9 | 28.3 | 66.4 | 36.0 | 13.0 | 28.3 | 37.2 | 3.0 | 0.72 | 44.2 |
| D replay (same bytes) | 65.2 | 87.8 | 42.7 | 90.0 | 53.3 | 22.9 | 42.7 | 37.2 | 3.4 | 0.85 | 60.4 |
| M core + flex (ours) | 79.5 | 100.0 | 59.1 | 98.0 | 77.6 | 35.4 | 22.7 | 35.4 | 0.2 | 1.00 | 77.3 |
| M, no archive | 61.6 | 100.0 | 23.2 | 96.1 | 23.4 | 4.8 | 23.2 | 35.4 | 0.2 | 0.75 | 69.3 |
| one ring, LRU | 79.8 | 100.0 | 59.6 | 96.1 | 79.5 | 35.5 | 27.1 | 35.4 | 0.2 | 1.00 | 47.4 |
| one ring, strength | 79.7 | 100.0 | 59.3 | 96.1 | 78.9 | 35.5 | 27.8 | 35.4 | 0.2 | 1.00 | 80.3 |
| harden at first sight | 79.4 | 100.0 | 58.8 | 97.3 | 77.2 | 35.3 | 20.7 | 35.4 | 0.2 | 1.00 | 72.7 |
| unlimited slots | 79.7 | 100.0 | 59.5 | 95.6 | 79.6 | 35.4 | 59.5 | 35.4 | 0.2 | 1.00 | 81.4 |
| M, shuffled labels (control) | 54.6 | 100.0 | 9.2 | 13.2 | 11.1 | 6.7 | 6.4 | 6.0 | 0.5 | 1.00 | 13.3 |
| M, half the slots | 79.6 | 100.0 | 59.1 | 96.1 | 78.5 | 35.4 | 13.9 | 35.4 | 0.2 | 1.00 | 55.5 |
| M, double the slots | 79.3 | 100.0 | 58.6 | 98.0 | 76.3 | 35.4 | 38.7 | 35.4 | 0.2 | 1.00 | 78.4 |

## Run facts

- Device: NVIDIA GeForce RTX 4080 SUPER, 1 process, total 1.9 GPU min
- Active parameters: 249,233 (core 185,873 + 18 slots × 3,520); plain net width 451 = 248,518 parameters
- Archive at the end: 340 KB; replay buffer: 400 KB
- One model's update for one episode: 43.3 ms
- Pointer lookup: 3.3 µs per query (CPU); routing errors 0.0%
- Recovery events: {'M': 194, 'no_archive': 218}
- Pretrained recall: core 100.0, plain 100.0, core code path 100.0
- Tuned on the validation stream: {'B': 0.0003, 'D': 0.0003, 'M': 0.01, 'C': [0.001, 1000.0]}
