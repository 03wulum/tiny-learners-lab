"""Train one arm on the NVIDIA GPU. Training never runs on CPU (Obi's rule); the only CPU
path is --smoke, which runs 3 steps of a toy-size model to check the code and saves
nothing unless --out is given.

    python train.py --arm A --minutes 12              # on Fortran's GPU
    python train_all.py                               # all arms at once, sharing the GPU
    python train.py --arm C --smoke                   # code check, any machine
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from arms import CKPT_DIR, TRAIN_K, TRAINED, TTT_STEPS, make_model, save
from tasks import HELDOUT_RULES, make_batch, make_meta_batch, make_prefix_batch
from ttt import adapt


def to_dev(b, dev):
    t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=dev)
    return (t(b.px), t(b.py), t(b.pvalid, torch.bool), t(b.qx), t(b.qy))


def mixed(maker, n, rng, *args):
    """Half rule prompts, half fact prompts, as one batch."""
    a, b = maker("rules", n // 2, *args, rng=rng), maker("facts", n - n // 2, *args, rng=rng)
    out = []
    for x, y in zip(to_dev(a, "cpu"), to_dev(b, "cpu")):
        out.append(torch.cat([x, y], 0))
    return out


def icl_loss(model, batch, dev):
    px, py, pv, qx, qy = (t.to(dev) for t in batch)
    logits = model(px, py, pv, qx, prefix=True)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), qy.reshape(-1))


def step_loss(model, recipe, n, rng, dev):
    pre = mixed(lambda task, n, k, rng: make_prefix_batch(task, n, k, rng), n, rng, TRAIN_K)
    loss = icl_loss(model, pre, dev)
    if recipe == "meta":
        k = int(rng.integers(2, TRAIN_K + 1))
        mb = mixed(lambda task, n, k, q, rng: make_meta_batch(task, n, k, q, rng), n // 2, rng, k, 8)
        px, py, pv, qx, qy = (t.to(dev) for t in mb)
        fast = adapt(model, px, py, pv, TTT_STEPS, lr=None, create_graph=True)
        logits = model(px, py, pv, qx, prefix=False, fast=fast)
        loss = loss + F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), qy.reshape(-1))
    return loss


@torch.no_grad()
def quick_heldout_acc(model, dev, rng, k=8, n=256):
    b = make_batch("rules", n, k, 8, rng, rules=HELDOUT_RULES)
    px, py, pv, qx, qy = to_dev(b, dev)
    return (model(px, py, pv, qx).argmax(-1) == qy).float().mean().item()


def pick_batch(model, recipe, rng, dev, opt, candidates):
    """Largest batch that fits in this process's share of GPU memory."""
    for n in candidates:
        try:
            loss = step_loss(model, recipe, n, rng, dev)
            loss.backward()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return n
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise SystemExit("no batch size fits in GPU memory")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(TRAINED))
    ap.add_argument("--minutes", type=float, default=12.0, help="hard wall-clock cap")
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--mem-frac", type=float, default=0.9, help="share of GPU memory for this process")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="3 steps, toy size, CPU allowed")
    a = ap.parse_args()

    if not a.smoke and not torch.cuda.is_available():
        sys.exit("Refusing to train on CPU: this experiment trains on an NVIDIA GPU only. "
                 "Use --smoke for a 3-step code check.")
    dev = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    _, size, recipe = TRAINED[a.arm]
    model = make_model(a.arm, smoke=a.smoke).to(dev)
    base_lr = a.lr or (1e-3 if size != "5M" else 5e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
        torch.backends.cuda.matmul.allow_tf32 = True
    if a.smoke:
        n = 8
    elif a.batch == "auto":
        n = pick_batch(model, recipe, rng, dev, opt, [2048, 1024, 512, 256, 128, 64])
    else:
        n = int(a.batch)
    amp = dev == "cuda" and torch.cuda.is_bf16_supported()
    print(f"arm {a.arm}: {model.n_params():,} params, recipe={recipe}, batch={n}, device={dev}, "
          f"cap={a.minutes} min", flush=True)

    budget = 3 if a.smoke else a.minutes * 60
    t0, step, log = time.time(), 0, []
    while True:
        el = time.time() - t0
        if (a.smoke and step >= 3) or (not a.smoke and el >= budget):
            break
        frac = min(el / budget, 1.0) if not a.smoke else 0
        warm = min(1.0, (step + 1) / 200)
        for g in opt.param_groups:
            g["lr"] = base_lr * warm * 0.5 * (1 + math.cos(math.pi * frac))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = step_loss(model, recipe, n, rng, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0 or a.smoke:
            model.eval()
            acc = quick_heldout_acc(model, dev, rng, n=32 if a.smoke else 256)
            model.train()
            log.append({"step": step, "t": round(time.time() - t0, 1), "loss": round(loss.item(), 4),
                        "heldout_acc_k8": round(acc, 4)})
            print(json.dumps(log[-1]), flush=True)

    train_s = time.time() - t0
    meta = {"arm": a.arm, "recipe": recipe, "params": model.n_params(), "steps": step,
            "batch": n, "train_seconds": round(train_s, 1), "device": dev,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "none",
            "torch": torch.__version__, "python": platform.python_version(), "seed": a.seed,
            "log": log}
    out = a.out if a.out else (None if a.smoke else CKPT_DIR)
    if out is not None:
        path = save(model.cpu(), a.arm, meta, out)
        (out / f"{a.arm}.train.json").write_text(json.dumps(meta, indent=1))
        print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"done: {step} steps in {train_s:.0f}s")


if __name__ == "__main__":
    main()
