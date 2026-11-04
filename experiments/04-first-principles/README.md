# 04 · First-principles composer

**Goal:** can a tiny model that learns only one kind of basic fact (which number comes next; who is whose parent) answer questions about operations it never saw, such as a − b, halving, uncles or cousins, by snapping its learned facts together?

**Verdict:** _pending GPU run_ · [plan and pre-registered criteria](https://claude.ai/artifact/PCGfLn1fM2k8YpgAjzfjQ4) · [plain-language explainer](EXPLAINER.md) · details: `results/results.md` after the run

| # | Criterion (pre-registered) | Threshold | Result |
|---|---|---|---|
| C1 | Budget | composer ≤ 1 MB, kept ≤ 50 MB, training ≤ 10 min GPU, answer ≤ 1 ms, program pick ≤ 1 s, RAM ≤ 2 GB | – |
| C2 | Arithmetic, 5 unseen ops (add 3, minus 3, a − b, half, a × b), 4 examples each | composer ≥ 90% and ≥ best baseline + 30 pts | – |
| C3 | Going the opposite way: a − b | ≥ 95%, trained on ≤ 100 facts, 0 subtractions | – |
| C4 | Family, 6 unseen relations (grandchild, grandmother, uncle, aunt, nephew, cousin) | composer ≥ 90% and ≥ best baseline + 30 pts | – |

Validity gate (else INCONCLUSIVE): baselines ≥ 80% on their own training tasks; baselines trained *on* the targets reach ≥ 70%; every negative control ≤ 5% (shuffled examples, a random operation, untrained blocks, the composer without its "undo" move).

## How it works

- **Learned blocks** (~7K parameters each): a next-number block trained on 100 facts (0→1 … 98→99, 99→none), or a parent + gender block per family. Each is **snapped** into a lookup table, so every later step is a constant-time pointer lookup.
- **Five moves, given not learned:** undo (`inv`), repeat (`iter`, `acc`), chain (`∘`, `;`), same input twice (`diag`), fix an input (`bind`); family adds male / female / not-yourself filters.
- **Search:** enumerate every distinct program up to a small size (merging programs with identical tables), then keep the smallest that fits the 4 worked examples, or abstain. Example: a − b = `iter(inv(S))`, "undo next, b times"; cousin = `(P ; inv((P ; nonself((P ; inv(P))))))`.
- **Baselines** (more data, same 4 examples): an MLC-style in-context transformer meta-trained on 14 arithmetic operations (including some "backwards" ones), and a Neural Bellman-Ford style path GNN meta-trained on 300 generated families. Two sizes each (25K / 0.7M and 32K / 0.9M parameters), plus oracle versions trained on the targets.
- **Ablations:** accuracy vs number of examples (1, 2, 4, 8); no snapping (chain in hidden space); 10% of first-principle facts missing; undo removed.

The composer is never told what a symbol means. It finds "zero" as the number nothing comes before and counts along its own learned chain, so relabelling every number changes nothing (tested).

## Run

```bash
python train.py            # GPU only, one process, ~5 min: composer blocks (seconds) + baselines -> ckpt/
python run.py              # evaluation: verdict table, results/verdict.json, results/results.md, plots
python run.py --quick      # < 30 s plumbing check on exact blocks and untrained baselines (CPU ok)
python run.py --clean      # caches and quick results (--all also deletes ckpt/)
python -m pytest -q        # 13 correctness tests
```
