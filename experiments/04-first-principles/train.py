"""Train everything for experiment 04 on the GPU, in one process, about 5 minutes in total.

python train.py            # needs CUDA; writes ckpt/ (git-ignored) and ckpt/train_log.json
Composer blocks train in seconds. Most of the budget goes to the baselines."""
import json
import os
import sys
import time

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import worlds as W
import composer as C
import baselines as B

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt")
SEEDS = (0, 1, 2)
SECONDS = dict(icl_small=40, icl_large=65, icl_oracle=65, gnn_small=25, gnn_large=40, gnn_oracle=40)
DROP = 0.10   # missing-facts ablation


def drop_parent_facts(P, frac, seed):
    rng = np.random.default_rng(30_000 + seed)
    Pd = P.copy()
    idx = np.argwhere(P)
    gone = idx[rng.random(len(idx)) < frac]
    Pd[gone[:, 0], gone[:, 1]] = False
    return Pd


def main():
    if not torch.cuda.is_available():
        sys.exit("train.py trains on the GPU only (project rule). Use run.py --quick for a CPU plumbing check.")
    dev = torch.device("cuda")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(CKPT, exist_ok=True)
    log_lines, info = [], dict(gpu=torch.cuda.get_device_name(0), seconds={}, steps={}, params={})

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    t_all = time.time()
    # ---------------------------------------------------------------- composer blocks
    t0 = time.time()
    for s in SEEDS:
        for tag, drop in (("", 0.0), ("_missing", DROP)):
            m = C.train_next_block(W.succ_facts(drop=drop, seed=s), dev, seed=s)
            fit = float((C.snap_next(m)[[f[0] for f in W.succ_facts(drop, s)]] ==
                         np.array([f[1] if f[1] < W.N else W.UNDEF for f in W.succ_facts(drop, s)])).mean())
            torch.save(m.state_dict(), os.path.join(CKPT, f"next{tag}_s{s}.pt"))
            P, G = W.family(s)
            Pt = drop_parent_facts(P, drop, s) if drop else P
            f = C.train_family_blocks(Pt, G, dev, seed=s)
            Ps, Gs = C.snap_family(f)
            torch.save(f.state_dict(), os.path.join(CKPT, f"family{tag}_s{s}.pt"))
            log(f"composer seed {s}{tag}: next-number fit {fit:.3f}, parent fit {(Ps == Pt).mean():.4f}, "
                f"gender fit {(Gs == G).mean():.3f}")
    info["seconds"]["composer"] = time.time() - t0
    info["params"]["composer_next"] = C.n_params(C.NextBlock())
    info["params"]["composer_family"] = C.n_params(C.FamilyBlocks(len(W.family(0)[0])))
    info["facts"] = dict(composer_arith=len(W.succ_facts()), composer_family_parent_pairs=int(W.family(0)[0].sum()))

    # ---------------------------------------------------------------- arithmetic baselines
    jobs = [("icl_small", list(W.ARITH_TRAIN), "small"), ("icl_large", list(W.ARITH_TRAIN), "large"),
            ("icl_oracle", list(W.ARITH_TRAIN) + list(W.ARITH_TEST), "large")]
    for name, ops, size in jobs:
        t0 = time.time()
        m, steps = B.train_icl(ops, size, dev, SECONDS[name], log=log)
        torch.save({k: v.half() for k, v in m.state_dict().items()}, os.path.join(CKPT, f"{name}.pt"))
        info["seconds"][name], info["steps"][name] = time.time() - t0, steps
        info["params"][name] = C.n_params(m)
        log(f"{name}: {steps} steps x 1024 episodes in {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- family baselines
    jobs = [("gnn_small", W.REL_TRAIN, "small"), ("gnn_large", W.REL_TRAIN, "large"),
            ("gnn_oracle", W.REL_TRAIN + W.REL_TEST, "large")]
    for name, rels, size in jobs:
        t0 = time.time()
        m, steps = B.train_gnn(rels, size, dev, SECONDS[name], log=log)
        torch.save({k: v.half() for k, v in m.state_dict().items()}, os.path.join(CKPT, f"{name}.pt"))
        info["seconds"][name], info["steps"][name] = time.time() - t0, steps
        info["params"][name] = C.n_params(m)
        log(f"{name}: {steps} steps x 128 queries in {time.time() - t0:.0f}s")

    info["total_train_min"] = (time.time() - t_all) / 60
    info["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    info["log"] = log_lines
    with open(os.path.join(CKPT, "train_log.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    log(f"done: {info['total_train_min']:.1f} min total, peak GPU memory {info['peak_gpu_mem_gb']:.2f} GB")


if __name__ == "__main__":
    main()
