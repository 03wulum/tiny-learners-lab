# tiny-learners-lab

Tiny models (≤ 20 MB) that **learn while they are being used**: from the prompt, from feedback, or by rewiring themselves, trained in minutes on one GPU.

## Results

| # | Experiment | Verdict | Criteria | One-line finding |
|---|---|---|---|---|
| 02 | [Learn while prompted](experiments/02-learn-while-prompted): can a ≤ 20 MB model learn new rules and facts from its prompt? | **FAIL** ✨ | 1 of 5 pass | Learns new facts perfectly from one example (even at 0.4 MB) but only copies on new rules; test-time training added nothing. [Details](experiments/02-learn-while-prompted/docs/results.md) |
| 03 | [Weight-order memory](experiments/03-weight-order-memory): adapt by *reordering* weights, not rewriting them | _pending GPU run_ | – | – |

Every experiment pre-registers PASS / FAIL criteria and negative controls before any run, and prints a verdict table (PASS / FAIL / INCONCLUSIVE, plus an optional ✨ INTRIGUING tag that never changes the verdict).

## Quick start

```bash
git clone https://github.com/03wulum/tiny-learners-lab && cd tiny-learners-lab
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # plus PyTorch, see below
python run_all.py --quick                             # plumbing check, < 30 s per experiment
```

PyTorch: on a CUDA machine install it from [pytorch.org](https://pytorch.org/get-started/locally/); CPU-only is enough for tests and `--quick`. Training runs on a GPU only.

`python run_all.py --clean` deletes caches and quick results in every experiment (`--all` also deletes checkpoints and `.venv`).

## License

MIT
