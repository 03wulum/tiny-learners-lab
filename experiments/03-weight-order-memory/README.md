# 03 · Weight-order memory

**Goal:** can a tiny model adapt to a new problem by *reordering* the values in one block of its weights (never changing a value), as well as fine-tuning does with the same number of bits?

**Verdict:** _pending GPU run_ · [plan and criteria](https://claude.ai/artifact/5HYngMFxqr8T5yjQyDjEDh) · details: `results/` after the run

| # | Criterion (pre-registered) | Threshold | Result |
|---|---|---|---|
| C1 | Budget | model ≤ 20 MB, kept ≤ 50 MB, training ≤ 10 min GPU, whole run ≤ 20 min, adaptation ≤ 100 ms/problem, RAM ≤ 2 GB | – |
| C2 | Reordering learns at all (unseen modular rules, 8 examples) | best reorder beats frozen **and** random reorder by ≥ 10 pts | – |
| C3 | Bits well spent (avg of 3 tasks) | best reorder ≥ bit-matched fine-tune − 3 pts | – |
| C4 | Keeps old skills after a 20-problem stream | drop ≤ 3 pts | – |

Validity gate (else INCONCLUSIVE): moving whole neurons with their outgoing weights changes outputs by < 1e-5; random reordering and shuffled labels gain ≤ 3 pts over frozen; frozen accuracy on T1 is < 90% (headroom).

## How it works

- **Model:** 7.5 M-parameter transformer (~15 MB fp16), trained ~7 min to solve few-shot problems in context. The *adaptive block* is the MLP input matrix of the last 2 layers (0.82 M weights).
- **Tasks** (generated, no downloads): T1 `y = (a·x + b) mod 17` with 25% of rules held out · T2 new key→value facts · T3 grid transforms with held-out combinations · T4 addition in an unseen number base (exploratory).
- **Arms:** A frozen · B fine-tune · B′ fine-tune squeezed to the bits a reordering carries (≈ 7 bits/weight) · **P1** sort-reorder: fine-tune a shadow copy, then re-order each row of the *original* values to follow its ranking (the closest pure reordering) · **P2** learned reorder: a 0.3 M-param controller reads the prompt and proposes the ranking · M mask ("pause" weights) · R random reorder (control) · oracle (100-step fine-tune).
- **Ablations:** reorder within rows / whole matrix / whole neurons; early vs late layers; block size (0.1 / 0.4 / 0.8 M weights → accuracy per bit); timing (adapt before vs after answering) in a same-rule stream.

**Changes from the plan, made before any run:** P2 snaps to a permutation with sort + straight-through gradients instead of Gumbel-Sinkhorn (a 1280×1280 Sinkhorn per row does not fit the budget). The fast-weight arm is left to experiment 02. A headroom check was added to the gate.

## Run

```bash
python train.py            # GPU only: 3 seeds in parallel, ~9 min, writes ckpt/
python run.py              # GPU: all arms, controls, ablations -> verdict table + results/
python run.py --quick      # < 30 s plumbing check on a tiny untrained model (CPU ok)
python run.py --clean      # caches + quick results (--all also deletes ckpt/)
python -m pytest -q        # 16 correctness tests
```
