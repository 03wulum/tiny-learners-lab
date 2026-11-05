"""Train the arms on the NVIDIA GPU, all in ONE process (one CUDA context, low system RAM).

Arms take turns step by step, each getting a fixed share of the wall-clock budget, so the
whole run finishes in --minutes (default 5; Obi's rule: about 5, never more than 10). Prompts are generated on the GPU; nothing
large lives in system RAM. If system RAM passes --ram-limit percent (default 92), training pauses
until it drops; after --max-pause minutes it stops early and saves.

Training never runs on CPU (Obi's rule). The only CPU path is --smoke: 3 steps of toy-size
models to check the code; it saves nothing unless --out is given.

    python train.py                      # all arms, 5 min total, on Fortran's GPU
    python train.py --arms A C --minutes 3
    python train.py --smoke              # code check, any machine
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from arms import CKPT_DIR, TRAIN_K, TRAINED, TTT_STEPS, make_model, save
from gpu_tasks import GpuTasks
from tasks import HELDOUT_RULES, P
from ttt import adapt

# share of the wall-clock budget per arm (meta-learning C is the most expensive per step)
SHARE = {"A": 2.5, "C": 3.5, "D": 2.5, "A-0.1M": 0.5, "A-5M": 1.0}
BATCH = {"A": 512, "C": 256, "D": 512, "A-0.1M": 512, "A-5M": 512}


def ram_percent() -> float:
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return 0.0


def step_loss(model, recipe, n, gt: GpuTasks):
    px, py, pv, qx, qy = gt.mixed(n, TRAIN_K, TRAIN_K, prefix=True)
    logits = model(px, py, pv, qx, prefix=True)
    loss = F.cross_entropy(logits.reshape(-1, P).float(), qy.reshape(-1))
    if recipe == "meta":
        k = int(torch.randint(2, TRAIN_K + 1, (1,)).item())
        px, py, pv, qx, qy = gt.mixed(n // 2, k, 8, prefix=False)
        fast = adapt(model, px, py, pv, TTT_STEPS, lr=None, create_graph=True)
        logits = model(px, py, pv, qx, prefix=False, fast=fast)
        loss = loss + F.cross_entropy(logits.reshape(-1, P).float(), qy.reshape(-1))
    return loss


@torch.no_grad()
def heldout_acc(model, dev, n=256, k=8):
    g = torch.Generator(device=dev).manual_seed(123)
    rules = torch.as_tensor(HELDOUT_RULES, device=dev)
    ab = rules[torch.randint(len(rules), (n,), device=dev, generator=g)]
    px = torch.randint(P, (n, k), device=dev, generator=g)
    qx = torch.randint(P, (n, 8), device=dev, generator=g)
    a, b = ab[:, :1], ab[:, 1:]
    pv = torch.ones_like(px, dtype=torch.bool)
    pred = model(px, (a * px + b) % P, pv, qx).argmax(-1)
    return (pred == (a * qx + b) % P).float().mean().item()


class ArmState:
    def __init__(self, arm, dev, smoke, budget_s, seed):
        torch.manual_seed(seed)
        self.arm = arm
        _, self.size, self.recipe = TRAINED[arm]
        self.model = make_model(arm, smoke=smoke).to(dev)
        self.base_lr = 5e-4 if self.size == "5M" else 1e-3
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=0.01)
        self.batch = 8 if smoke else BATCH[arm]
        self.budget = budget_s
        self.used = 0.0
        self.steps = 0
        self.log = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(TRAINED), choices=list(TRAINED))
    ap.add_argument("--minutes", type=float, default=5.0, help="total wall-clock budget, all arms (max 10)")
    ap.add_argument("--batch-scale", type=float, default=None,
                    help="multiply every arm's batch size (default: from free GPU memory)")
    ap.add_argument("--ram-limit", type=float, default=92.0,
                    help="pause training while system RAM is above this %%")
    ap.add_argument("--max-pause", type=float, default=3.0,
                    help="stop early and save if RAM stays high this many minutes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="3 steps per arm, toy size, CPU allowed")
    a = ap.parse_args()

    if not a.smoke and not torch.cuda.is_available():
        sys.exit("Refusing to train on CPU: this experiment trains on an NVIDIA GPU only. "
                 "Use --smoke for a 3-step code check.")
    if a.minutes > 10:
        sys.exit("Training budget is capped at 10 minutes (Obi's rule).")
    dev = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}, {free / 1e9:.1f} of {total / 1e9:.1f} GB free; "
              f"system RAM {ram_percent():.0f}% used", flush=True)
        if a.batch_scale is None:  # one training at a time: use the GPU's free memory
            a.batch_scale = 4.0 if free > 16e9 else 2.0 if free > 8e9 else 1.0 if free > 4e9 else 0.5
    if a.batch_scale is None:
        a.batch_scale = 1.0
    amp = dev == "cuda" and torch.cuda.is_bf16_supported()
    total_s = a.minutes * 60
    shares = {arm: SHARE[arm] for arm in a.arms}
    states = [ArmState(arm, dev, a.smoke, total_s * shares[arm] / sum(shares.values()), a.seed + i)
              for i, arm in enumerate(a.arms)]
    for s in states:
        s.batch = max(8, int(s.batch * a.batch_scale)) if not a.smoke else s.batch
        print(f"arm {s.arm}: {s.model.n_params():,} params, recipe={s.recipe}, batch={s.batch}, "
              f"budget {s.budget / 60:.1f} min", flush=True)
    gt = GpuTasks(dev, a.seed)

    t0 = time.time()
    stopped_for_ram = False
    paused_s = 0.0
    while True:
        live = [s for s in states if (s.steps < 3 if a.smoke else s.used < s.budget)]
        if not live:
            break
        s = min(live, key=lambda x: x.used / x.budget)  # the arm furthest behind its share
        ts = time.time()
        frac = 0.0 if a.smoke else min(s.used / s.budget, 1.0)
        for g in s.opt.param_groups:
            g["lr"] = s.base_lr * min(1.0, (s.steps + 1) / 200) * 0.5 * (1 + math.cos(math.pi * frac))
        s.model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = step_loss(s.model, s.recipe, s.batch, gt)
        s.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s.model.parameters(), 1.0)
        s.opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - ts
        if dt > 30 and s.steps > 10:  # the process was suspended from outside (e.g. by pcguard)
            paused_s += dt
            dt = s.used / s.steps  # count the step at its typical length
        s.used += dt
        s.steps += 1
        if s.steps % 200 == 0 or a.smoke:
            s.model.eval()
            entry = {"step": s.steps, "t": round(s.used, 1), "loss": round(loss.item(), 4),
                     "heldout_acc_k8": round(heldout_acc(s.model, dev, n=32 if a.smoke else 256), 4),
                     "ram_pct": ram_percent()}
            s.log.append(entry)
            print(s.arm, json.dumps(entry), flush=True)
        if not a.smoke and (s.steps % 25 == 0) and ram_percent() > a.ram_limit:
            # pause (not stop): wait for other jobs to free RAM; paused time doesn't count
            print(f"system RAM above {a.ram_limit}%: pausing training", flush=True)
            waited = 0.0
            while ram_percent() > a.ram_limit and waited < a.max_pause * 60:
                time.sleep(5)
                waited += 5
            paused_s += waited
            if ram_percent() > a.ram_limit:
                print(f"RAM still above {a.ram_limit}% after {a.max_pause} min: stopping early and saving")
                stopped_for_ram = True
                break
            print(f"resumed after {waited:.0f}s", flush=True)

    wall = time.time() - t0
    out = a.out if a.out else (None if a.smoke else CKPT_DIR)
    summary = {"wall_minutes": round(wall / 60, 2), "train_minutes": round((wall - paused_s) / 60, 2),
               "paused_minutes": round(paused_s / 60, 2), "stopped_for_ram": stopped_for_ram, "device": dev,
               "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "none",
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else 0,
               "torch": torch.__version__, "python": platform.python_version(), "arms": {}}
    for s in states:
        meta = {"arm": s.arm, "recipe": s.recipe, "params": s.model.n_params(), "steps": s.steps,
                "batch": s.batch, "train_seconds": round(s.used, 1), "device": dev, "seed": a.seed,
                "log": s.log}
        summary["arms"][s.arm] = {k: v for k, v in meta.items() if k != "log"}
        if out is not None:
            path = save(s.model.cpu(), s.arm, meta, out)
            (out / f"{s.arm}.train.json").write_text(json.dumps(meta, indent=1))
            print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if out is not None:
        (out / "train_all.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
