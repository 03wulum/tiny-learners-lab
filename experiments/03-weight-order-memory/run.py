"""Evaluate tiny-learners 03 "weight-order memory": all arms, controls, ablations, verdict.

  python run.py            evaluate the trained checkpoints in ckpt/ (GPU; one process, seeds in turn)
  python run.py --quick    <30 s plumbing check on a tiny untrained model (CPU ok, no verdict claim)
  python run.py --clean    delete caches, quick results (add --all to delete checkpoints too)
"""
import argparse, json, os, shutil, sys, time
import numpy as np
import torch

import tasks, criteria
from model import TinyLM, Controller, answer_loss
from adapt import (adapt, expand, query_acc, to_t, moved_fraction, perm_bits_per_weight,
                   mask_bits, log2_factorial)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT, RES = os.path.join(HERE, "ckpt"), os.path.join(HERE, "results")
SEEDS = (0, 1, 2)
ARMS_MAIN = ("A", "B", "Bq", "P1", "P2", "M", "R", "oracle")
LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
MASK_GRID = (0.005, 0.02, 0.05)
FULL = dict(n_val=48, n_test=100, n_sweep=64, n_abl=48, n_ret=100, streams=8, stream_len=20,
            same_rule_streams=16, chunk=50, steps=10, oracle_steps=100)
QUICK = dict(n_val=8, n_test=8, n_sweep=8, n_abl=4, n_ret=8, streams=2, stream_len=3,
             same_rule_streams=2, chunk=8, steps=2, oracle_steps=3)


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
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # KB on Linux
    except ImportError:
        return float("nan")


class Evaluator:
    def __init__(self, model, ctrl, cfg, dev, seed):
        self.m, self.ctrl, self.cfg, self.dev, self.seed = model, ctrl, cfg, dev, seed
        self.cache = {}
        self.hp = dict(steps=cfg["steps"], oracle_steps=cfg["oracle_steps"], lr={}, mask_frac=0.02,
                       rand_frac=0.5)

    @property
    def W(self):
        return [w.detach().float() for w in self.m.block_weights()]

    def problems(self, task, k, split, n, shuffle=False):
        key = (task, k, split, n, shuffle)
        if key not in self.cache:
            rng = np.random.default_rng(abs(hash((self.seed,) + key)) % 2**32 if False else
                                        [self.seed, k, n, int(shuffle), tasks.TASKS.index(task),
                                         ("train", "val", "test").index(split)])
            self.cache[key] = tasks.make_batch(rng, task, n, k, split, shuffle_labels=shuffle)
        return self.cache[key]

    def run(self, arm, task, k, split, n, shuffle=False, hp=None, **kw):
        """Mean exact-match accuracy in points (and P1's moved fraction)."""
        hp = hp or self.hp
        pb, ch = self.problems(task, k, split, n, shuffle), self.cfg["chunk"]
        accs, moved = [], []
        W = self.W
        for i in range(0, n, ch):
            tb = to_t({key: v[i:i + ch] for key, v in pb.items()}, self.dev)
            block = adapt(self.m, arm, tb, hp, W, controller=self.ctrl, **kw)
            accs.append(query_acc(self.m, tb, block).cpu())
            if arm == "P1":
                moved.append(moved_fraction(W, block))
        acc = 100 * torch.cat(accs).mean().item()
        return (acc, float(np.mean(moved))) if arm == "P1" else acc


def symmetry_check(model, dev):
    """Moving whole neurons WITH their outgoing weights must not change anything."""
    m = TinyLM(**model.cfg).to(dev).double()
    m.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    rng = np.random.default_rng(0)
    tb = to_t(tasks.make_batch(rng, "T1", 4, 8, "test"), dev)
    with torch.no_grad():
        ref = m(tb["full"])
        blk = m.blocks[m.adaptive[-1]]
        perm = torch.randperm(blk.fc_in.weight.shape[0], device=dev)
        blk.fc_in.weight.copy_(blk.fc_in.weight[perm]); blk.fc_in.bias.copy_(blk.fc_in.bias[perm])
        saved = blk.fc_out.weight.clone()
        blk.fc_out.weight.copy_(blk.fc_out.weight[:, perm])
        sym = (m(tb["full"]) - ref).abs().max().item()
        blk.fc_out.weight.copy_(saved)                        # outgoing weights NOT moved
        broken = (m(tb["full"]) - ref).abs().max().item()
    return sym, broken


def eval_seed(seed, cfg, dev, model=None, ctrl=None):
    t0 = time.time()
    if model is None:
        ck = torch.load(os.path.join(CKPT, f"seed{seed}", "base.pt"), map_location=dev)
        model = TinyLM(**ck["cfg"]).to(dev)
        model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
        W = model.block_weights()
        ctrl = Controller(model.cfg["d"], W[0].shape[0], W[0].shape[1], len(W)).to(dev)
        ctrl.load_state_dict({k: v.float() for k, v in torch.load(
            os.path.join(CKPT, f"seed{seed}", "controller.pt"), map_location=dev)["state"].items()})
    model.eval(); ctrl.eval()
    for p in list(model.parameters()) + list(ctrl.parameters()):
        p.requires_grad_(False)
    E = Evaluator(model, ctrl, cfg, dev, seed)
    out = dict(seed=seed)
    out["symmetry_max_diff"], out["symmetry_broken_diff"] = symmetry_check(model, dev)

    # ---- tune each arm's learning rate on the validation split (T1-T3, k=8)
    val_tasks = ("T1", "T2", "T3")
    tune = {}
    for arm in ("B", "Bq", "P1"):
        scores = {}
        for lr in LR_GRID:
            hp = dict(E.hp, lr={arm: lr})
            scores[lr] = np.mean([(lambda r: r[0] if isinstance(r, tuple) else r)(
                E.run(arm, t, 8, "val", cfg["n_val"], hp=hp)) for t in val_tasks])
        E.hp["lr"][arm] = max(scores, key=scores.get)
        tune[arm] = scores
    m_scores = {}
    for lr in LR_GRID:
        for mf in MASK_GRID:
            hp = dict(E.hp, lr={"M": lr}, mask_frac=mf)
            m_scores[(lr, mf)] = np.mean([E.run("M", t, 8, "val", cfg["n_val"], hp=hp) for t in val_tasks])
    E.hp["lr"]["M"], E.hp["mask_frac"] = max(m_scores, key=m_scores.get)
    tune["M"] = {f"{k[0]}|{k[1]}": v for k, v in m_scores.items()}
    p2_val = np.mean([E.run("P2", t, 8, "val", cfg["n_val"]) for t in val_tasks])
    p1_val = tune["P1"][E.hp["lr"]["P1"]]
    best_p = "P1" if p1_val >= p2_val else "P2"
    out.update(hp=dict(lr=E.hp["lr"], mask_frac=E.hp["mask_frac"]), tune=tune,
               val=dict(P1=p1_val, P2=p2_val), best_p=best_p)

    # ---- main table: every arm, every task, k=8, test split
    main = {}
    for t in tasks.TASKS:
        main[t] = {}
        p1, frac = E.run("P1", t, 8, "test", cfg["n_test"])
        main[t]["P1"], main[t]["P1_moved_frac"] = p1, frac
        E.hp["rand_frac"] = frac
        for arm in ARMS_MAIN:
            if arm != "P1":
                main[t][arm] = E.run(arm, t, 8, "test", cfg["n_test"])
    out["main"] = main
    shuf = {arm: E.run(arm, "T1", 8, "test", cfg["n_test"], shuffle=True) for arm in ("A", "P1", "P2")}
    shuf["P1"] = shuf["P1"][0]
    out["shuffled_T1"] = shuf

    # ---- k sweep on T1
    out["k_sweep_T1"] = {arm: {k: (lambda r: r[0] if isinstance(r, tuple) else r)(
        E.run(arm, "T1", k, "test", cfg["n_sweep"])) for k in (1, 2, 4, 8, 16)}
        for arm in ("A", "B", "Bq", "P1", "P2")}

    # ---- C4: 20-problem stream of new problems, changes carried forward, then old-skill retention
    W = E.W
    ret_sets = [E.problems(t, 8, "train", cfg["n_ret"]) for t in tasks.TASKS]

    def retention(block_s=None):
        acc = []
        for pb in ret_sets:
            tb = to_t(pb, dev)
            blk = None if block_s is None else [b.unsqueeze(0).expand(pb["full"].shape[0], *b.shape) for b in block_s]
            acc.append(100 * query_acc(model, tb, blk if blk is not None else expand(W, pb["full"].shape[0])).mean().item())
        return float(np.mean(acc))

    base_ret = retention()
    stream = {"base_retention": base_ret}
    rng = np.random.default_rng([seed, 99])
    S = cfg["streams"]
    for arm in ("P1", "P2", "Bq", "B"):
        cur, s_acc = expand(W, S), []
        for step in range(cfg["stream_len"]):
            t = ("T1", "T2", "T3")[step % 3]
            tb = to_t(tasks.make_batch(rng, t, S, 8, "test"), dev)
            cur = adapt(model, arm, tb, E.hp, W, current=cur, controller=ctrl)
            s_acc.append(100 * query_acc(model, tb, cur).mean().item())
        rets = [retention([c[s] for c in cur]) for s in range(S)]
        stream[arm] = dict(stream_acc=float(np.mean(s_acc)), retention=float(np.mean(rets)),
                           retention_drop=base_ret - float(np.mean(rets)))
    out["stream_retention"] = stream

    # ---- same-rule stream: does it need fewer examples over time? + timing ablation
    out["same_rule"] = same_rule_stream(E, seed, cfg, dev)

    # ---- ablations (T1, k=8)
    n = cfg["n_abl"]
    abl = {"granularity": {g: E.run("P1", "T1", 8, "test", n, granularity=g)[0]
                           for g in ("row", "matrix", "neuron")}}
    abl["granularity"]["frozen"] = E.run("A", "T1", 8, "test", n)
    L_out = W[0].shape[0]
    subsets = {"quarter_last": [None] * (len(W) - 1) + [torch.arange(L_out // 4, device=dev)],
               "last_layer": [None] * (len(W) - 1) + [torch.arange(L_out, device=dev)],
               "all": None}
    row_len = W[0].shape[1]
    abl["block_size"] = {}
    for name, sub in subsets.items():
        n_w = (sum(len(s) for s in sub if s is not None) if sub else L_out * len(W)) * row_len
        abl["block_size"][name] = dict(
            n_weights=n_w, bits=n_w * perm_bits_per_weight(row_len),
            P1=E.run("P1", "T1", 8, "test", n, row_subset=sub)[0],
            Bq=E.run("Bq", "T1", 8, "test", n, row_subset=sub))
    saved = model.adaptive
    model.adaptive = tuple(range(len(W)))                       # earliest layers instead
    abl["location_early"] = dict(layers=model.adaptive,
                                 P1=E.run("P1", "T1", 8, "test", n)[0], Bq=E.run("Bq", "T1", 8, "test", n))
    model.adaptive = saved
    abl["location_late"] = dict(layers=saved, P1=main["T1"]["P1"], Bq=main["T1"]["Bq"])
    out["ablations"] = abl

    # ---- bits and latency
    n_w = sum(w.numel() for w in W)
    out["bits"] = dict(block_weights=n_w, perm_bits=n_w * perm_bits_per_weight(row_len),
                       perm_bits_per_weight=perm_bits_per_weight(row_len),
                       mask_bits=sum(mask_bits(w.numel(), E.hp["mask_frac"]) for w in W),
                       fp16_bits=16 * n_w)
    lat = []
    pb = E.problems("T1", 8, "test", cfg["n_test"])
    for i in range(10 if cfg is FULL else 2):
        tb = to_t({k: v[i:i + 1] for k, v in pb.items()}, dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        adapt(model, "P1", tb, E.hp, W)
        if dev == "cuda":
            torch.cuda.synchronize()
        lat.append(1000 * (time.time() - t1))
    out["adapt_ms_P1"] = float(np.median(lat[1:] if len(lat) > 1 else lat))
    out["peak_ram_gb"] = peak_ram_gb()
    out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
    out["eval_sec"] = time.time() - t0
    return out


def same_rule_stream(E, seed, cfg, dev):
    """Each stream keeps one unseen T1 rule. At every problem we count how many worked
    examples (1..8) the model needs before it answers the query right (with whatever it has
    learned so far), then let it learn from the problem ('after' timing). Timing ablation:
    k=1 problems with adaptation before answering, after answering, or both."""
    model, W, S = E.m, E.W, cfg["same_rule_streams"]
    rng = np.random.default_rng([seed, 7])
    rules = rng.integers(tasks.n_rules("T1", "test"), size=S)
    res = {}
    for arm in ("A", "P1", "Bq"):
        cur, need = expand(W, S), []
        for step in range(cfg["stream_len"]):
            x, y = tasks.GEN["T1"](rng, S, 8, "test", rules)
            got = torch.full((S,), 9.0)
            for j in range(8, 0, -1):
                xs, ys = np.concatenate([x[:, :j], x[:, -1:]], 1), np.concatenate([y[:, :j], y[:, -1:]], 1)
                tb = to_t(tasks.assemble(rng, "T1", xs, ys), dev)
                ok = query_acc(model, tb, cur).cpu() > 0
                got = torch.where(ok, torch.full_like(got, j), got)
            need.append(got.mean().item())
            if arm != "A":   # learn from all 8 examples + the answered query
                xs, ys = np.concatenate([x, x[:, :1]], 1), np.concatenate([y, y[:, :1]], 1)
                tb = to_t(tasks.assemble(rng, "T1", xs, ys), dev)
                cur = adapt(model, arm, tb, E.hp, W, current=cur)
        first, last = np.mean(need[:5]), np.mean(need[-5:])
        res[arm] = dict(examples_needed=need, first5=first, last5=last, reduction=1 - last / first)
    timing = {}
    for mode in ("before", "after", "both"):
        cur, accs = expand(W, S), []
        for step in range(cfg["stream_len"]):
            x, y = tasks.GEN["T1"](rng, S, 1, "test", rules)
            tb = to_t(tasks.assemble(rng, "T1", x, y), dev)
            if mode in ("before", "both"):
                cur = adapt(model, "P1", tb, E.hp, W, current=cur)
            accs.append(100 * query_acc(model, tb, cur).mean().item())
            if mode in ("after", "both"):
                xs, ys = np.concatenate([x, x[:, -1:]], 1), np.concatenate([y, y[:, -1:]], 1)
                cur = adapt(model, "P1", to_t(tasks.assemble(rng, "T1", xs, ys), dev), E.hp, W, current=cur)
        timing[mode] = dict(acc=accs, first5=float(np.mean(accs[:5])), last5=float(np.mean(accs[-5:])))
    res["timing_P1_k1"] = timing
    return res


def aggregate(per_seed, budget):
    def mean(f):
        return float(np.mean([f(r) for r in per_seed]))

    def bp(r, t):
        return r["main"][t][r["best_p"]]
    d3 = [np.mean([bp(r, t) for t in ("T1", "T2", "T3")]) - np.mean([r["main"][t]["Bq"] for t in ("T1", "T2", "T3")])
          for r in per_seed]
    m = dict(
        budget=budget,
        T1_A=mean(lambda r: r["main"]["T1"]["A"]), T1_R=mean(lambda r: r["main"]["T1"]["R"]),
        T1_bestP=mean(lambda r: bp(r, "T1")),
        avg_bestP=mean(lambda r: np.mean([bp(r, t) for t in ("T1", "T2", "T3")])),
        avg_Bq=mean(lambda r: np.mean([r["main"][t]["Bq"] for t in ("T1", "T2", "T3")])),
        avg_B=mean(lambda r: np.mean([r["main"][t]["B"] for t in ("T1", "T2", "T3")])),
        c3_seed_std=float(np.std(d3, ddof=1)) if len(d3) > 1 else 0.0,
        retention_drop_bestP=mean(lambda r: r["stream_retention"][r["best_p"]]["retention_drop"]),
        symmetry_max_diff=max(r["symmetry_max_diff"] for r in per_seed),
        T1_bestP_shuf=mean(lambda r: r["shuffled_T1"][r["best_p"]]),
        T1_A_shuf=mean(lambda r: r["shuffled_T1"]["A"]),
        T4_bestP_beats_all=all(bp(r, "T4") > max(v for a, v in r["main"]["T4"].items()
                                                  if a in ARMS_MAIN and a not in ("oracle", r["best_p"]))
                               for r in per_seed),
        stream_reduction=mean(lambda r: r["same_rule"]["P1"]["reduction"]),
    )
    return m


def dir_mb(path, pattern=(".pt",)):
    tot = 0
    for root, _, files in os.walk(path):
        tot += sum(os.path.getsize(os.path.join(root, f)) for f in files if f.endswith(pattern))
    return tot / 1e6


def write_report(m, rows, v, gate, notes, per_seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    js = dict(experiment="tiny-learners 03 weight-order memory", verdict=v, intriguing=notes,
              criteria=rows, gate=gate, measured=m, per_seed=per_seed)
    with open(os.path.join(out_dir, "verdict.json"), "w") as f:
        json.dump(js, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\n" + "=" * 78)
    for r in rows:
        print(f"{r['criterion']:<48} {r['result']:<5} value={r['value']}")
    print(f"gate: {gate}")
    print(f"VERDICT: {v}" + (f"   ✨ INTRIGUING: {'; '.join(notes)}" if notes else ""))
    arms = ARMS_MAIN
    lines = ["| Task | " + " | ".join(arms) + " |", "|---" * (len(arms) + 1) + "|"]
    for t in tasks.TASKS:
        lines.append(f"| {t} | " + " | ".join(f"{np.mean([r['main'][t][a] for r in per_seed]):.1f}" for a in arms) + " |")
    with open(os.path.join(out_dir, "main_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--all", action="store_true", help="with --clean: also delete checkpoints")
    a = ap.parse_args()
    if a.clean:
        freed = 0
        targets = ["__pycache__", ".pytest_cache", os.path.join("results", "quick")] + (["ckpt"] if a.all else [])
        for t in targets:
            p = os.path.join(HERE, t)
            if os.path.exists(p):
                freed += sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
                shutil.rmtree(p)
        print(f"cleaned {targets}: freed {freed / 1e6:.1f} MB")
        return
    if a.quick:
        torch.manual_seed(0)
        from train import SMOKE_DIMS
        model = TinyLM(tasks.VOCAB, tasks.MAX_LEN, **SMOKE_DIMS)
        W = model.block_weights()
        ctrl = Controller(SMOKE_DIMS["d"], W[0].shape[0], W[0].shape[1], len(W))
        r = eval_seed(0, QUICK, "cpu", model, ctrl)
        budget = dict(model_mb=0, kept_mb=0, base_train_min=0, total_gpu_min=0,
                      adapt_ms=r["adapt_ms_P1"], peak_ram_gb=r["peak_ram_gb"])
        m = aggregate([r], budget)
        rows, v, gate, notes = criteria.verdict(m)
        write_report(m, rows, v, gate, notes, [r], os.path.join(RES, "quick"))
        print("(quick mode: untrained tiny model, plumbing check only, not a result)")
        return
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. Full evaluation runs on Fortran's GPU.")
    os.makedirs(RES, exist_ok=True)
    torch.set_num_threads(2)
    t0 = time.time()
    for s in SEEDS:              # one process, one seed at a time (keeps system RAM low)
        r = eval_seed(s, FULL, "cuda")
        with open(os.path.join(RES, f"seed{s}.json"), "w") as f:
            json.dump(r, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"seed {s} evaluated in {r['eval_sec'] / 60:.1f} min, peak RAM {r['peak_ram_gb']:.2f} GB, "
              f"GPU peak {r['gpu_peak_gb']:.2f} GB", flush=True)
        torch.cuda.empty_cache()
    eval_min = (time.time() - t0) / 60
    per_seed = [json.load(open(os.path.join(RES, f"seed{s}.json"))) for s in SEEDS]
    infos = [json.load(open(os.path.join(CKPT, f"seed{s}", "train_info.json"))) for s in SEEDS]
    train_min = max((i["base_sec"] + i["ctrl_sec"]) / 60 for i in infos)
    budget = dict(model_mb=os.path.getsize(os.path.join(CKPT, "seed0", "base.pt")) / 1e6,
                  kept_mb=dir_mb(CKPT), base_train_min=train_min,
                  total_gpu_min=train_min + eval_min,
                  adapt_ms=float(np.mean([r["adapt_ms_P1"] for r in per_seed])),
                  peak_ram_gb=max([r["peak_ram_gb"] for r in per_seed] + [i.get("peak_ram_gb", 0) for i in infos]))
    m = aggregate(per_seed, budget)
    rows, v, gate, notes = criteria.verdict(m)
    write_report(m, rows, v, gate, notes, per_seed, RES)
    try:
        import plots
        plots.make(per_seed, RES)
    except Exception as e:  # plots are a nicety; never lose the verdict over them
        print("plotting failed:", e)


if __name__ == "__main__":
    main()
