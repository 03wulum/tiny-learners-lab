"""Train the base model (in-context learner) and the P2 reorder controller for one seed.

GPU only (lab rule): refuses to run without CUDA unless --smoke (5 steps, plumbing test).
Usage:  python train.py --seeds 0 1 2          (one process per seed, in parallel)
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import torch

import tasks
from model import TinyLM, Controller, answer_loss, n_params
from adapt import expand, project_rows

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt")
TRAIN_TASKS = tasks.TASKS


def sample(rng, B, split="train", kmax=tasks.MAX_K, task=None):
    task = task or TRAIN_TASKS[rng.integers(len(TRAIN_TASKS))]
    k = int(rng.integers(1, kmax + 1))
    return tasks.make_batch(rng, task, B, k, split)


def train_seed(seed, base_min, ctrl_min, batch, smoke, dims):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    model = TinyLM(tasks.VOCAB, tasks.MAX_LEN, **dims).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.05)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    t0, step, log = time.time(), 0, []
    budget = base_min * 60
    while True:
        el = time.time() - t0
        if (smoke and step >= 5) or (not smoke and el >= budget):
            break
        frac = min(1.0, el / budget) if not smoke else 0
        lr = 1e-3 * min(1.0, (step + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        tb = {k: torch.as_tensor(v, device=dev) for k, v in sample(rng, batch).items()}
        with torch.autocast(**amp):
            loss = answer_loss(model(tb["full"]), tb["full"], _all_answers(tb))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            log.append(dict(step=step, sec=round(el, 1), loss=round(loss.item(), 4)))
            print(f"[seed {seed}] base step {step} {el:.0f}s loss {loss.item():.3f}", flush=True)
        step += 1
    base_sec = time.time() - t0
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- P2 controller: meta-trained, base frozen, reorder via straight-through sort
    W = model.block_weights()
    ctrl = Controller(dims["d"], W[0].shape[0], W[0].shape[1], len(W)).to(dev)
    copt = torch.optim.AdamW(ctrl.parameters(), lr=1e-3)
    t1, cstep = time.time(), 0
    while True:
        if (smoke and cstep >= 3) or (not smoke and time.time() - t1 >= ctrl_min * 60):
            break
        tb = {k: torch.as_tensor(v, device=dev) for k, v in sample(rng, batch // 4).items()}
        Bn = tb["supp"].shape[0]
        with torch.no_grad():
            _, hdn = model(tb["supp"], return_hidden=True)
        deltas = ctrl(hdn.mean(1).float())
        block = []
        for w, d in zip(W, deltas):
            shadow = w.unsqueeze(0) + d
            proj = project_rows(w, shadow.detach())
            block.append(shadow + (proj - shadow).detach())          # straight-through
        loss = answer_loss(model(tb["full"], block=block), tb["full"], tb["ymask"])
        copt.zero_grad(set_to_none=True)
        loss.backward()
        copt.step()
        if cstep % 100 == 0:
            print(f"[seed {seed}] ctrl step {cstep} loss {loss.item():.3f}", flush=True)
        cstep += 1
    ctrl_sec = time.time() - t1

    out = os.path.join(CKPT, f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    torch.save({"cfg": model.cfg, "state": {k: v.half() for k, v in model.state_dict().items()}},
               os.path.join(out, "base.pt"))
    torch.save({"state": {k: v.half() for k, v in ctrl.state_dict().items()}},
               os.path.join(out, "controller.pt"))
    info = dict(seed=seed, device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                params=n_params(model), ctrl_params=n_params(ctrl), base_steps=step,
                base_sec=round(base_sec, 1), ctrl_steps=cstep, ctrl_sec=round(ctrl_sec, 1),
                batch=batch, log=log)
    with open(os.path.join(out, "train_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"[seed {seed}] done: {step} base steps, {cstep} ctrl steps", flush=True)


def _all_answers(tb):
    """Train the base on every answer in the prompt (support answers + query answer)."""
    full, ym = tb["full"], tb["ymask"]
    L = tb["supp"].shape[1]
    m = torch.zeros_like(full)
    m[:, :L] = tb["smask"]
    return m + ym


DIMS = dict(d=320, n_layer=6, n_head=5, adaptive=(4, 5))
SMOKE_DIMS = dict(d=32, n_layer=2, n_head=2, adaptive=(1,))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--base-min", type=float, default=7.0)
    ap.add_argument("--ctrl-min", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="5 steps on a tiny model (tests only)")
    a = ap.parse_args()
    if not torch.cuda.is_available() and not a.smoke:
        sys.exit("No CUDA GPU found. Training runs on Fortran's GPU only (lab rule).")
    dims = SMOKE_DIMS if a.smoke else DIMS
    if len(a.seeds) == 1:
        train_seed(a.seeds[0], a.base_min, a.ctrl_min, a.batch, a.smoke, dims)
    else:   # one process per seed, all at once on the GPU
        procs = [subprocess.Popen([sys.executable, __file__, "--seeds", str(s), "--base-min", str(a.base_min),
                                   "--ctrl-min", str(a.ctrl_min), "--batch", str(a.batch)]
                                  + (["--smoke"] if a.smoke else [])) for s in a.seeds]
        sys.exit(max(p.wait() for p in procs))
