"""Train every arm at once on one NVIDIA GPU (Fortran). Each arm is its own process with
an equal share of GPU memory and picks the largest batch that fits in it, so the GPU is
used as fully as it can be. Wall-clock: about --minutes plus startup (default 12 + ~1).

    python train_all.py                 # all arms, 12 min cap each, in parallel
    python train_all.py --minutes 3     # a fast first look
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import torch

from arms import CKPT_DIR, TRAINED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--arms", nargs="*", default=list(TRAINED))
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible. This experiment trains on an NVIDIA GPU only.")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}, {free / 1e9:.1f} of {total / 1e9:.1f} GB free")
    # bigger models get a bigger share of memory
    weight = {arm: (4 if TRAINED[arm][1] == "5M" else 1) for arm in a.arms}
    usable = 0.9 * free / total
    CKPT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    procs = {}
    for arm in a.arms:
        frac = usable * weight[arm] / sum(weight.values())
        log = open(CKPT_DIR / f"{arm}.log", "w")
        procs[arm] = (subprocess.Popen([sys.executable, "train.py", "--arm", arm, "--minutes", str(a.minutes),
                                        "--mem-frac", f"{frac:.3f}"], stdout=log, stderr=subprocess.STDOUT), log)
        print(f"started {arm} (memory share {frac:.0%})", flush=True)
    codes = {}
    for arm, (p, log) in procs.items():
        codes[arm] = p.wait()
        log.close()
        print(f"{arm}: exit {codes[arm]} after {(time.time() - t0) / 60:.1f} min", flush=True)
    wall = (time.time() - t0) / 60
    summary = {"wall_minutes": round(wall, 2), "gpu": torch.cuda.get_device_name(0), "exit_codes": codes,
               "minutes_cap": a.minutes}
    (CKPT_DIR / "train_all.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    sys.exit(0 if all(c == 0 for c in codes.values()) else 1)


if __name__ == "__main__":
    main()
