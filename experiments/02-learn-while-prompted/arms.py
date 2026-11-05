"""The arms of the experiment and how each one is trained and evaluated."""
from __future__ import annotations

from pathlib import Path

import torch

from models import SIZES, Config, TinyLearner

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "ckpt"   # git-ignored; published as a GitHub release asset

# name -> (model kind, size, training recipe)
TRAINED = {
    # main arms at 1M params so they can learn inside a 5-minute GPU budget
    "A": ("attn", "1M", "icl"),       # frozen in-context learner
    "C": ("attn", "1M", "meta"),      # meta-learned test-time training
    "D": ("mem", "1M", "icl"),        # fast-weight memory
    "A-0.1M": ("attn", "0.1M", "icl"),  # size sweep
    "A-5M": ("attn", "5M", "icl"),
}

# evaluation arms: name -> (checkpoint, test-time training mode)
#   ttt=None: frozen. ttt="naive": SGD with a fixed lr picked on training rules.
#   ttt="meta": the model's own meta-learned step sizes.
EVAL_ARMS = {
    "A": ("A", None),
    "B": ("A", "naive"),
    "C": ("C", "meta"),
    "C-frozen": ("C", None),   # ablation: C's weights without the update
    "D": ("D", None),
    "A-0.1M": ("A-0.1M", None),
    "A-5M": ("A-5M", None),
}

TTT_STEPS = 2                 # inner steps for B and C, at training and test time
NAIVE_LR_GRID = [1e-3, 1e-2, 1e-1, 1.0]
TRAIN_K = 16                  # max examples per prompt seen in training


def make_model(arm: str, smoke: bool = False) -> TinyLearner:
    kind, size, _ = TRAINED[arm]
    s = dict(d=32, layers=2, heads=2, ff=64) if smoke else SIZES[size]
    return TinyLearner(Config(kind=kind, **s))


def save(model: TinyLearner, arm: str, meta: dict, out_dir: Path = CKPT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arm}.pt"
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"config": vars(model.c), "state": state, "meta": meta}, path)
    return path


def load(arm: str, device="cpu", ckpt_dir: Path = CKPT_DIR) -> tuple[TinyLearner, dict]:
    blob = torch.load(ckpt_dir / f"{arm}.pt", map_location="cpu", weights_only=True)
    model = TinyLearner(Config(**blob["config"]))
    model.load_state_dict({k: v.float() for k, v in blob["state"].items()})
    return model.to(device).eval(), blob["meta"]
