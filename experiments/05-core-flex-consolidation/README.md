# 05 · Core + flexible weights

**Goal:** can a fixed-size tiny model keep a stable core, pick up new skills in a small flexible ring, harden the ones that keep coming back, and archive (never lose) the ones it replaces, so it remembers better than fine-tuning or replay at the same size?

**Verdict:** _pending GPU run_ · details: [EXPLAINER.md](EXPLAINER.md) · `results/results.md` after the run

| # | Criterion (pre-registered in [criteria.py](criteria.py)) | Threshold | Result |
|---|---|---|---|
| C1 | Budget | active model ≤ 20 MB, kept ≤ 20 MB, ≤ 10 GPU min (≈ 5 planned), update ≤ 50 ms, slot lookup ≤ 100 µs, RAM ≤ 2 GB | – |
| C2 | Remembers old + new skills at the end (headline) | ≥ fine-tune + 30 pts **and** ≥ replay (same bytes) + 5 pts | – |
| C3 | Learns a new skill fast (after its first 6 examples) | ≥ fine-tune − 5 pts | – |
| C4 | Recovers archived skills with no new examples | ≥ 90% of last recall; the no-archive ablation ≤ 50% | – |
| C5 | Hardening protects frequent skills through bursts of one-offs | ≥ one-ring LRU cache + 10 pts | – |

Validity gate (else INCONCLUSIVE): both pretrained nets ≥ 90% on known skills; shuffled cues, never-seen skills, the frozen net on new skills and shuffled labels (on unseen inputs) all ≤ chance + 5 (chance 5.9%); wrong-slot routing ≤ 2%.

## How it works

- **Skill:** a cue (a random point on a 64-d sphere, arriving with noise each time) bound to a hidden rule `y = (a·x + b) mod 17`. 208 rules are known before the stream, 64 are new.
- **Stream (per seed):** 224 episodes of 6 worked examples. 8 frequent skills ×12, 24 occasional ×4, 32 one-offs in 4 bursts of 8. Recall is tested from the cue alone, with no examples.
- **M, core + flex (ours):** a frozen 186K-param core plus 18 slots (each a key, a code and a rank-4 change that plug into the core): a flexible ring of 6 that learns fast, and a core ring of 12 that a skill joins after 3 visits and that learns 10× slower. Routing is a nearest-key lookup. When a ring is full, the weakest slot goes to the archive and comes back when its cue does. The active size never grows.
- **Baselines at the same active parameter count (249K):** A frozen · B fine-tune · C online EWC · D replay with as many bytes as M's largest possible archive.
- **Ablations:** no archive · one ring with LRU or strength eviction · harden at first sight · unlimited slots · half and double the slots. Hyperparameters are tuned on a separate validation stream.

## Run

```bash
python train.py            # GPU only, ~5 min: pretrain, tune, every arm on 3 seeds -> results/raw/
python run.py              # score: verdict table, results/results.md, plots (no GPU needed)
python run.py --quick      # < 30 s CPU plumbing check (tiny run, not a result)
python run.py --clean      # caches + quick results (--all also deletes ckpt/)
python -m pytest -q        # correctness tests
```
