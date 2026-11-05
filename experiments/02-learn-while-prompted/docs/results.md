# Experiment 02: results in detail

Run on 2026-11-05 on Fortran (RTX 4080 SUPER, 16 GB), via pcguard. Code at a2eeb7b, results commit f2d4399. Full tables: [tables.md](tables.md).

## Verdict: FAIL (core criterion C4 failed), with valid controls

| # | Criterion | Result | Threshold | |
|---|---|---|---|---|
| C1 | Fits the budget | 19.0 MB largest model, 15.8 MB all checkpoints, 0.014 s to learn one prompt; timer read 11.5 min total | ≤ 20 MB, ≤ 50 MB, ≤ 1 s, ≤ 10 min | FAIL* |
| C2 | Learns new rules | best arm 38.7% at k=8 (A-0.1M), Spearman 1.00 | ≥ 80% | FAIL |
| C3 | Learns new facts | 100% recall at k=16, unseen keys 6.2% | ≥ 90%, unseen 5.9% ± 3 | PASS |
| C4 | Keeps improving past training | arm C 99% of ideal at k=64, but only +1.3 pts over frozen A | ≥ 80% of ideal and ≥ +15 pts | FAIL |
| C5 | Updating must be learned | C 22.6% vs B 22.5% at k=4 (+0.2) | ≥ +10 pts | FAIL |

Validity gate OK: shuffled labels and the untrained model stayed at chance (5-7%), and the base model learned (arm A: 100% on training rules and on shown facts at k=16).

\*C1: pcguard suspended the training process for 6.8 minutes (07:34:49 to 07:41:37) while one of Obi's render jobs needed RAM. train.py couldn't see the suspension and counted it as training time, almost all inside arm A-5M. Real GPU compute was about 4.7 minutes, which would pass. The verdict is FAIL either way. train.py now detects outside suspensions (a step longer than 30 s) and doesn't count them.

## What happened

**1. Every model learned to copy, none learned the rule.** On held-out rules, every arm's accuracy matches the "copy-only" curve: the chance that the question's input already appeared in the prompt, 1 − (16/17)^k. At k=8 that is 38.4%; the arms scored 37.5-38.7%. An ideal learner needs only 2 examples to pin down y = (a·x + b) mod 17 and scores 95% at k=2. The same holds for the new rule family (T4).

**2. The fully trained model memorised its 204 training rules instead.** Arm A, the one that got the most steps (4,508), scores 97% at k=8 on rules it saw in training, far above copying (38%). It identifies which of the 204 memorised rules the prompt matches. On the 68 held-out rules that falls apart into pure copying. This is the "task diversity" effect of Raventós et al. 2023: with too few distinct training tasks, a transformer learns a lookup over those tasks rather than the general algorithm. 204 rules is below that threshold.

**3. Copying generalises perfectly and cheaply.** New facts are recalled at 100% after one example, by every arm including the 0.1M-param (0.4 MB) model, and unseen keys stay at chance. It keeps working at k=64, four times the longest prompt seen in training, because the models have no position embeddings (each example is one token in a set).

**4. Test-time training changed nothing.** B matches A, and C matches C-frozen, to within 0.2 points at every k. The adapter updates are too small to move the predictions. Meta-training (C) did not learn a useful update in 1,375 steps. The naive learning rate picked on training data (0.1) barely mattered (75.3-75.8% across the grid).

**5. C4 rewarded copying.** "99% of ideal at k=64" sounds strong, but copying alone gets 98% at k=64. The criterion didn't separate rule learning from recall. Future experiments in this repo will include the copy-only baseline as a pre-registered control.

## Training

| Arm | Params | Steps | Batch | GPU seconds |
|---|---|---|---|---|
| A | 1.0M | 4,508 | 1024 | 75 |
| C | 1.0M | 1,375 | 512 | 105 |
| D | 1.0M | 485 | 1024 | 75 |
| A-0.1M | 0.1M | 1,956 | 1024 | 15 |
| A-5M | 4.8M | 427 | 1024 | ~14 real (422 counted) |

Peak GPU memory 5.3 GB. Peak system RAM 92% (Obi's render jobs plus ours). The GPU was shared with a Shorts job (about 12% of compute).

## What I'd change next

- **More rule diversity**: a larger modulus or several rule families mixed, so there are thousands of distinct training rules. That is the known way past memorisation.
- **Longer training for C and D**: they got 485-1,375 steps against A's 4,508, so the comparison between update rules is weak.
- **Copy-only control** pre-registered for any rule task.
