"""Train the base model (in-context learner) and the P2 reorder controller for one seed.

GPU only (lab rule): refuses to run without CUDA unless --smoke (5 steps, plumbing test).
Usage:  python train.py --seeds 0 1 2   (one process; seeds take turns on the GPU)
"""
import argparse, json, os, sys, time
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


def train_seeds(seeds, base_min, ctrl_min, batch, smoke, dims):
    """All seeds in ONE process (one CUDA context, little system RAM), taking turns step by
    step, so the GPU stays busy without three copies of PyTorch in memory."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)                     # keep CPU use low; the GPU does the work
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = TinyLM(tasks.VOCAB, tasks.MAX_LEN, **dims).to(dev)
        runs.append(dict(seed=seed, model=model, rng=np.random.default_rng(1000 + seed), step=0, log=[],
                         opt=torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98),
                                               weight_decay=0.05)))
    t0, budget = time.time(), base_min * 60
    while True:
        el = time.time() - t0
        if (smoke and runs[0]["step"] >= 5) or (not smoke and el >= budget):
            break
        frac = 0 if smoke else min(1.0, el / budget)
        for r in runs:
            lr = 1e-3 * min(1.0, (r["step"] + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * frac)))
            for g in r["opt"].param_groups:
                g["lr"] = lr
            tb = {k: torch.as_tensor(v, device=dev) for k, v in sample(r["rng"], batch).items()}
            with torch.autocast(**amp):
                loss = answer_loss(r["model"](tb["full"]), tb["full"], _all_answers(tb))
            r["opt"].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(r["model"].parameters(), 1.0)
            r["opt"].step()
            if r["step"] % 200 == 0:
                r["log"].append(dict(step=r["step"], sec=round(el, 1), loss=round(loss.item(), 4)))
                print(f"[seed {r['seed']}] base step {r['step']} {el:.0f}s loss {loss.item():.3f}", flush=True)
            r["step"] += 1
    base_sec = time.time() - t0

    # ---- P2 controllers: meta-trained, base frozen, reorder via straight-through sort
    for r in runs:
        r["model"].eval()
        for p in r["model"].parameters():
            p.requires_grad_(False)
        W = r["model"].block_weights()
        r["ctrl"] = Controller(dims["d"], W[0].shape[0], W[0].shape[1], len(W)).to(dev)
        r["copt"], r["cstep"] = torch.optim.AdamW(r["ctrl"].parameters(), lr=1e-3), 0
    t1 = time.time()
    while True:
        if (smoke and runs[0]["cstep"] >= 3) or (not smoke and time.time() - t1 >= ctrl_min * 60):
            break
        for r in runs:
            model, W = r["model"], r["model"].block_weights()
            tb = {k: torch.as_tensor(v, device=dev) for k, v in sample(r["rng"], batch // 4).items()}
            with torch.no_grad():
                _, hdn = model(tb["supp"], return_hidden=True)
            deltas = r["ctrl"](hdn.mean(1).float())
            block = []
            for w, d in zip(W, deltas):
                shadow = w.unsqueeze(0) + d
                proj = project_rows(w, shadow.detach())
                block.append(shadow + (proj - shadow).detach())          # straight-through
            loss = answer_loss(model(tb["full"], block=block), tb["full"], tb["ymask"])
            r["copt"].zero_grad(set_to_none=True)
            loss.backward()
            r["copt"].step()
            if r["cstep"] % 100 == 0:
                print(f"[seed {r['seed']}] ctrl step {r['cstep']} loss {loss.item():.3f}", flush=True)
            r["cstep"] += 1
    ctrl_sec = time.time() - t1

    for r in runs:
        out = os.path.join(CKPT, f"seed{r['seed']}")
        os.makedirs(out, exist_ok=True)
        torch.save({"cfg": r["model"].cfg, "state": {k: v.half() for k, v in r["model"].state_dict().items()}},
                   os.path.join(out, "base.pt"))
        torch.save({"state": {k: v.half() for k, v in r["ctrl"].state_dict().items()}},
                   os.path.join(out, "controller.pt"))
        info = dict(seed=r["seed"], device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                    params=n_params(r["model"]), ctrl_params=n_params(r["ctrl"]), base_steps=r["step"],
                    base_sec=round(base_sec, 1), ctrl_steps=r["cstep"], ctrl_sec=round(ctrl_sec, 1),
                    batch=batch, seeds_sharing_gpu=len(runs), peak_ram_gb=round(peak_ram_gb(), 2),
                    gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else 0,
                    log=r["log"])
        with open(os.path.join(out, "train_info.json"), "w") as f:
            json.dump(info, f, indent=1)
    print(f"done: {runs[0]['step']} base steps and {runs[0]['cstep']} ctrl steps per seed; "
          f"peak RAM {peak_ram_gb():.2f} GB", flush=True)


def peak_ram_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        if hasattr(mi, "peak_wset"):
            return mi.peak_wset / 1e9
    except ImportError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except ImportError:
        return float("nan")


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
    ap.add_argument("--base-min", type=float, default=3.75)
    ap.add_argument("--ctrl-min", type=float, default=1.25)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--smoke", action="store_true", help="5 steps on a tiny model (tests only)")
    a = ap.parse_args()
    if not torch.cuda.is_available() and not a.smoke:
        sys.exit("No CUDA GPU found. Training runs on Fortran's GPU only (lab rule).")
    dims = SMOKE_DIMS if a.smoke else DIMS
    train_seeds(a.seeds, a.base_min, a.ctrl_min, a.batch, a.smoke, dims)
