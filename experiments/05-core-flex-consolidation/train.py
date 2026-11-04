"""GPU run for tiny-learners 05, in ONE process (Fortran rule: one process, data on the GPU,
stop early if system RAM passes 85%). Everything that learns happens here.

  python train.py              full run on CUDA (~3-5 min): writes ckpt/ and results/raw/
  python train.py --smoke      seconds-long plumbing run on CPU (tests and run.py --quick)

Every run (an arm on one seed's stream) is a slice of stacked tensors, and all runs step
in lockstep through the 224 episodes, so each GPU kernel serves every arm and seed at once.
  1. pretrain: 3 cores (width 384, with slot port) + 3 plain nets (wider, same active
     parameter count) on the 208 known rules, all at once.
  2. tune: a validation stream (seed-0 nets, its own skills and order): learning rate per
     arm and EWC strength, picked by mean(retention, first-episode recall).
  3. test: every arm and ablation on the 3 seeds' streams -> results/raw/seed<N>.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

import skills
from model import (CORE_H, RANK, N_FLEX, N_CORE, matched_plain_h, mlp_numel, slot_numel, init_stack,
                   forward, ce, StackAdam, pretrain, recall)
from memory import SlotMemory

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
VAL_STREAM = 100
STEPS = 20
N_REPLAY = 26
BYTES_PER_EXAMPLE = skills.CUE_DIM * 2 + 2                    # fp16 cue + x + y
REPLAY_BYTES = 64 * slot_numel() * 2                          # = M's largest possible archive (64 slots, fp16)
GRID = dict(B=[3e-4, 1e-3, 3e-3, 1e-2], D=[3e-4, 1e-3, 3e-3, 1e-2], M=[0.01, 0.03, 0.1, 0.3],
            C=[(lr, lam) for lr in (1e-3, 3e-3) for lam in (10.0, 100.0, 1000.0)])
DEFAULT_HP = dict(B=1e-3, D=1e-3, M=0.03, C=(1e-3, 100.0))
M_VARIANTS = {"M": {}, "M_noarchive": dict(archive=False), "flat_lru": dict(flat="lru"),
              "flat_strength": dict(flat="strength"), "harden_at_once": dict(harden_at=1),
              "unlimited": dict(n_flex=None, flat="strength"), "M_shuffled_labels": {},
              "cap_small": dict(n_flex=N_FLEX // 2, n_core=N_CORE // 2),
              "cap_large": dict(n_flex=N_FLEX * 2, n_core=N_CORE * 2)}
PEAK = {"rss": 0.0}
DEADLINE = {"t": float("inf")}                                # hard cap: Obi's 10 GPU-minute limit


def ram_guard():
    if time.time() > DEADLINE["t"]:
        sys.exit("STOPPED EARLY: the run passed 9.5 minutes (cap is 10). Nothing was kept from this run.")
    try:
        import psutil
    except ImportError:
        return
    PEAK["rss"] = max(PEAK["rss"], psutil.Process().memory_info().rss / 1e9)
    pct = psutil.virtual_memory().percent
    if pct > 85:
        sys.exit(f"STOPPED EARLY: system RAM at {pct:.0f}% (> 85%). Nothing was kept from this run.")


def pre_cues(seed):
    return skills.random_cues(np.random.default_rng([seed, 1]), len(skills.PRE_RULES))


def acc_rows(pred, tab):
    return 100.0 * (pred == tab).mean(-1)


def prepare_stream(stream_seed, net_seed, cfg):
    """A stream plus every noisy cue it will ever show, drawn once, so all arms see the same."""
    st = skills.make_stream(np.random.default_rng([stream_seed, 0]), cfg)
    rng = np.random.default_rng([stream_seed, 3])
    st["net_seed"] = net_seed
    st["tab"] = skills.table(st["rules"])
    st["ep_cues"] = np.stack([skills.noisy(rng, st["cues"][s:s + 1])[0] for s, _ in st["episodes"]])
    freq = np.where(st["group"] == "freq")[0]
    st["freq"] = freq
    st["burst_cues"] = [skills.noisy(rng, st["cues"][freq]) for _ in st["burst_ends"]]
    st["final_cues"] = skills.noisy(rng, st["cues"])
    st["shuf_cues"] = skills.noisy(rng, st["cues"][np.roll(np.arange(len(st["cues"])), 1)])
    st["never_noisy"] = skills.noisy(rng, st["never_cues"])
    st["pre_noisy"] = skills.noisy(rng, pre_cues(net_seed))
    st["shuffle_rng"] = np.random.default_rng([stream_seed, 4])
    return st


# ---------------------------------------------------------------------------------------
class PlainGroup:
    """Stacked baselines on the plain net: A frozen (lr 0), B fine-tune, C online EWC,
    D replay. Every run has its own copy of the weights."""

    def __init__(self, specs, pre_plain, streams, dev, steps, seed):
        self.specs, self.dev, self.steps = specs, dev, steps
        idx = torch.as_tensor([s["seed_idx"] for s in specs], device=dev)
        self.P = {k: v[idx].clone() for k, v in pre_plain.items()}
        self.R = len(specs)
        self.lr = torch.as_tensor([s["lr"] for s in specs], device=dev, dtype=torch.float32)
        self.lam = torch.as_tensor([s.get("lam", 0.0) for s in specs], device=dev, dtype=torch.float32)
        self.rng = np.random.default_rng([seed, 5])
        self.ewc = bool((self.lam > 0).any())
        pre_tab = skills.table(skills.PRE_RULES)
        pcs = [pre_cues(streams[s["stream"]]["net_seed"]) for s in specs]
        if self.ewc:                      # diagonal Fisher on pretraining data, then per episode
            self.F = {k: torch.zeros_like(v) for k, v in self.P.items()}
            for _ in range(32):
                ii = self.rng.integers(len(pre_tab), size=(self.R, 64))
                xx = self.rng.integers(skills.P, size=(self.R, 64))
                c = np.stack([skills.noisy(self.rng, pcs[r][ii[r]]) for r in range(self.R)])
                self._fisher(c, xx, pre_tab[ii, xx], 1 / 32)
            self.star = {k: v.clone() for k, v in self.P.items()}
        self.cap = int(REPLAY_BYTES // BYTES_PER_EXAMPLE)
        self.buf = {}
        for r, s in enumerate(specs):
            if s["arm"] == "D":
                half = self.cap // 2
                ii, xx = self.rng.integers(len(pre_tab), size=half), self.rng.integers(skills.P, size=half)
                self.buf[r] = dict(pc=skills.noisy(self.rng, pcs[r][ii]).astype(np.float16), px=xx, py=pre_tab[ii, xx],
                                   rc=np.zeros((self.cap - half, skills.CUE_DIM), np.float16),
                                   rx=np.zeros(self.cap - half, np.int64), ry=np.zeros(self.cap - half, np.int64),
                                   n=0, seen=0)

    def _t(self, a, dt=torch.float32):
        return torch.as_tensor(np.asarray(a), device=self.dev, dtype=dt)

    def _loss(self, c, x, y, w):
        out = forward(self.P, c, x[..., None])[:, :, 0]                       # (R, N, 17)
        l = (ce(out, y) * w).sum(1) / w.sum(1).clamp_min(1e-9)
        return l

    def _fisher(self, c, x, y, weight):
        for p in self.P.values():
            p.requires_grad_(True)
        self._loss(self._t(c), self._t(x, torch.long), self._t(y, torch.long), torch.ones(x.shape, device=self.dev)).sum().backward()
        with torch.no_grad():
            for k, p in self.P.items():
                self.F[k] += weight * p.grad ** 2
                p.grad = None
                p.requires_grad_(False)

    def where(self, r, cue):
        return "weights"

    @torch.no_grad()
    def predict(self, cues, use_archive=True):
        pred, _ = recall(self.P, self._t(cues), np.zeros(cues.shape[:2] + (skills.P,), np.int64))
        return pred.cpu().numpy()

    def episode(self, t, cues, xs, ys, sk):
        R, k = xs.shape
        S = self.steps
        ec = np.zeros((S, R, N_REPLAY, skills.CUE_DIM), np.float32)
        ex = np.zeros((S, R, N_REPLAY), np.int64)
        ey = np.zeros((S, R, N_REPLAY), np.int64)
        ew = np.zeros((R, N_REPLAY), np.float32)
        for r, b in self.buf.items():
            n_res = 0 if b["n"] == 0 else N_REPLAY // 2
            i = self.rng.integers(len(b["px"]), size=(S, N_REPLAY - n_res))
            ec[:, r, :N_REPLAY - n_res], ex[:, r, :N_REPLAY - n_res], ey[:, r, :N_REPLAY - n_res] = b["pc"][i], b["px"][i], b["py"][i]
            if n_res:
                j = self.rng.integers(b["n"], size=(S, n_res))
                ec[:, r, N_REPLAY - n_res:], ex[:, r, N_REPLAY - n_res:], ey[:, r, N_REPLAY - n_res:] = b["rc"][j], b["rx"][j], b["ry"][j]
            ew[r] = 1.0
        cur_c = self._t(np.repeat(cues[:, None], k, 1))
        cur_x, cur_y = self._t(xs, torch.long), self._t(ys, torch.long)
        ec, ex, ey = self._t(ec), self._t(ex, torch.long), self._t(ey, torch.long)
        w = torch.cat([torch.ones(R, k, device=self.dev), self._t(ew)], 1)
        params = list(self.P.values())
        for p in params:
            p.requires_grad_(True)
        opt = StackAdam(params, self.lr)
        for s in range(S):
            loss = self._loss(torch.cat([cur_c, ec[s]], 1), torch.cat([cur_x, ex[s]], 1), torch.cat([cur_y, ey[s]], 1), w)
            if self.ewc:
                pen = sum((self.F[kk] * (p - self.star[kk]) ** 2).flatten(1).sum(1) for kk, p in self.P.items())
                loss = loss + 0.5 * self.lam * pen
            loss.sum().backward()
            opt.step()
        for p in params:
            p.requires_grad_(False)
        if self.ewc:
            self._fisher(np.repeat(cues[:, None], k, 1), xs, ys, 1.0)
            self.star = {kk: v.clone() for kk, v in self.P.items()}
        for r, b in self.buf.items():                     # reservoir over stream examples
            for i in range(k):
                b["seen"] += 1
                if b["n"] < len(b["rx"]):
                    j = b["n"]
                    b["n"] += 1
                else:
                    j = self.rng.integers(b["seen"])
                    if j >= len(b["rx"]):
                        continue
                b["rc"][j], b["rx"][j], b["ry"][j] = cues[r].astype(np.float16), xs[r, i], ys[r, i]
        return ["weights"] * R

    def extras(self, r):
        if r in self.buf:
            return dict(replay_bytes=(len(self.buf[r]["px"]) + self.buf[r]["n"]) * BYTES_PER_EXAMPLE)
        return {}


class MGroup:
    """Stacked core + flex runs: each run has the frozen core of its seed and its own slot
    memory. Per episode, the one slot each run is learning is stacked and trained together."""

    def __init__(self, specs, pre_core, streams, dev, steps, seed):
        self.specs, self.dev, self.steps = specs, dev, steps
        idx = torch.as_tensor([s["seed_idx"] for s in specs], device=dev)
        self.P = {k: v[idx].clone() for k, v in pre_core.items() if k != "codes"}
        self.R, self.h = len(specs), self.P["W2"].shape[1]
        self.lr = np.array([s["lr"] for s in specs], np.float32)
        mean_codes = pre_core["codes"].mean(1)
        self.mem = [SlotMemory(self.h, mean_codes[s["seed_idx"]].clone(), dev,
                               torch.Generator().manual_seed(1000 * seed + r), **s.get("mem", {}))
                    for r, s in enumerate(specs)]

    def where(self, r, cue):
        return self.mem[r].lookup(cue)[1]

    @torch.no_grad()
    def predict(self, cues, use_archive=True):
        parts = [m.slot_tensors(cues[r], use_archive) for r, m in enumerate(self.mem)]
        code = torch.stack([p[0] for p in parts])
        U, V = torch.stack([p[1] for p in parts]), torch.stack([p[2] for p in parts])
        keep = torch.as_tensor(np.stack([p[3] for p in parts]), device=self.dev)[..., None]
        c = torch.as_tensor(cues, device=self.dev) * keep
        pred, _ = recall(self.P, c, np.zeros(cues.shape[:2] + (skills.P,), np.int64), code=code, U=U, V=V)
        return pred.cpu().numpy()

    def episode(self, t, cues, xs, ys, sk):
        R, k = xs.shape
        prep = [m.prepare(t, cues[r], int(sk[r])) for r, m in enumerate(self.mem)]
        slots = [p[0] for p in prep]
        lr = torch.as_tensor(self.lr * np.array([p[1] for p in prep], np.float32), device=self.dev)
        code = torch.stack([s.code for s in slots]).requires_grad_(True)
        U = torch.stack([s.U for s in slots]).requires_grad_(True)
        V = torch.stack([s.V for s in slots]).requires_grad_(True)
        opt = StackAdam([code, U, V], lr)
        c = torch.zeros(R, 1, skills.CUE_DIM, device=self.dev)
        x = torch.as_tensor(xs, device=self.dev)[:, None]
        y = torch.as_tensor(ys, device=self.dev)
        for _ in range(self.steps):
            out = forward(self.P, c, x, code=code[:, None], U=U[:, None], V=V[:, None])[:, 0]
            ce(out, y).mean(1).sum().backward()
            opt.step()
        for r, (m, s) in enumerate(zip(self.mem, slots)):
            s.code, s.U, s.V = code[r].detach().clone(), U[r].detach().clone(), V[r].detach().clone()
            m.finish(t, s, cues[r])
        return [p[2] for p in prep]

    def extras(self, r):
        m = self.mem[r]
        return dict(route_errors=m.route_errors, active_bytes=m.bytes("active"), archive_bytes=m.bytes("archive"),
                    n_active=len(m.active), n_archive=len(m.archive))


# ---------------------------------------------------------------------------------------
def run_phase(groups, streams):
    """Walk every run through its stream in lockstep. groups: list of Plain/MGroup."""
    T = len(streams[0]["episodes"])
    assert all(len(s["episodes"]) == T for s in streams)
    recs = {(gi, r): dict(episodes=[], bursts=[], last={}, seen=set()) for gi, g in enumerate(groups) for r in range(g.R)}
    for t in range(T):
        ram_guard()
        for gi, g in enumerate(groups):
            sts = [streams[s["stream"]] for s in g.specs]
            if t in sts[0]["burst_ends"]:
                _burst(g, gi, sts, recs, sts[0]["burst_ends"].index(t))
            sid = np.array([st["episodes"][t][0] for st in sts])
            xs = np.stack([st["episodes"][t][1] for st in sts])
            cues = np.stack([st["ep_cues"][t] for st in sts])
            tabs = np.stack([st["tab"][s] for st, s in zip(sts, sid)])
            where = [g.where(r, cues[r]) for r in range(g.R)]
            pre = acc_rows(g.predict(cues[:, None])[:, 0], tabs)
            ys = np.take_along_axis(tabs, xs, 1).copy()
            for r, s in enumerate(g.specs):
                if s.get("shuffle"):
                    ys[r] = sts[r]["shuffle_rng"].permutation(ys[r])
            g.episode(t, cues, xs, ys, sid)
            pred = g.predict(cues[:, None])[:, 0]
            post = acc_rows(pred, tabs)
            for r in range(g.R):
                rec, s = recs[(gi, r)], int(sid[r])
                held = np.setdiff1d(np.arange(skills.P), xs[r])
                rec["episodes"].append(dict(t=t, skill=s, group=str(sts[r]["group"][s]), first=s not in rec["seen"],
                                            before=where[r], pre=round(float(pre[r]), 2), post=round(float(post[r]), 2),
                                            held=round(acc_rows(pred[r][held], tabs[r][held]).item(), 2),
                                            last_post=rec["last"].get(s)))
                rec["seen"].add(s)
                rec["last"][s] = rec["episodes"][-1]["post"]
    for gi, g in enumerate(groups):
        sts = [streams[s["stream"]] for s in g.specs]
        if T in sts[0]["burst_ends"]:
            _burst(g, gi, sts, recs, sts[0]["burst_ends"].index(T))
        _final(g, gi, sts, recs)
    out = []
    for gi, g in enumerate(groups):
        for r, s in enumerate(g.specs):
            rec = recs[(gi, r)]
            rec.pop("last"), rec.pop("seen")
            rec.update(arm=s["arm"], hp=s.get("hp"), stream=s["stream"], **g.extras(r))
            out.append(rec)
    return out


def _burst(g, gi, sts, recs, b):
    cues = np.stack([st["burst_cues"][b] for st in sts])
    pred = g.predict(cues, use_archive=False)
    for r, st in enumerate(sts):
        rec = recs[(gi, r)]
        mask = np.array([f in rec["seen"] for f in st["freq"]])
        if mask.any():
            rec["bursts"].append(dict(t=st["burst_ends"][b], recall=float(acc_rows(pred[r][mask], st["tab"][st["freq"]][mask]).mean())))


def _final(g, gi, sts, recs):
    pre_tab = skills.table(skills.PRE_RULES)
    p_pre = g.predict(np.stack([st["pre_noisy"] for st in sts]))
    p_sys = g.predict(np.stack([st["final_cues"] for st in sts]), use_archive=True)
    p_act = g.predict(np.stack([st["final_cues"] for st in sts]), use_archive=False)
    p_shuf = g.predict(np.stack([st["shuf_cues"] for st in sts]))
    p_nev = g.predict(np.stack([st["never_noisy"] for st in sts]))
    for r, st in enumerate(sts):
        f = dict(pretrained=float(acc_rows(p_pre[r], pre_tab).mean()))
        for mode, p in (("system", p_sys), ("active", p_act)):
            per = acc_rows(p[r], st["tab"])
            f[mode] = {gname: float(per[st["group"] == gname].mean()) for gname in skills.GROUPS}
            f[mode]["all"] = float(per.mean())
        f["shuffled_cue"] = float(acc_rows(p_shuf[r], st["tab"]).mean())
        f["never_seen"] = float(acc_rows(p_nev[r], skills.table(st["never_rules"])).mean())
        recs[(gi, r)]["final"] = f


def score(res):
    f = res["final"]
    ret = 0.5 * (f["pretrained"] + f["system"]["all"])
    first = np.mean([e["post"] for e in res["episodes"] if e["first"]])
    return 0.5 * (ret + first)


def bench_update_ms(pre_core, dev, steps, reps=10):
    """Latency of ONE model's update for one episode (one slot, `steps` Adam steps)."""
    g = MGroup([dict(arm="M", seed_idx=0, stream=0, lr=0.03)], pre_core, None, dev, steps, 0)
    rng = np.random.default_rng(9)
    cues = skills.random_cues(rng, reps)
    xs = np.sort(rng.choice(skills.P, 6, replace=False))[None]
    ts = []
    for i in range(reps + 2):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.episode(i, cues[i % reps][None], xs, xs % skills.P, np.array([i]))
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    return float(np.median(ts[2:]))


def bench_route_us(n_active=N_FLEX + N_CORE, n_archive=64, reps=20000):
    """One pointer lookup (active ring, then archive) on the CPU, in microseconds."""
    rng = np.random.default_rng(0)
    m = SlotMemory.__new__(SlotMemory)
    m.tau, m.archive_on = 0.7, True
    mk = lambda keys: [type("S", (), {"key": k})() for k in keys]
    m.active, m.archive = mk(skills.random_cues(rng, n_active)), mk(skills.random_cues(rng, n_archive))
    q = skills.random_cues(rng, 64)
    t0 = time.perf_counter_ns()
    for i in range(reps):
        m.lookup(q[i % 64])
    return (time.perf_counter_ns() - t0) / reps / 1e3


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU plumbing run (not a result)")
    ap.add_argument("--pretrain-sec", type=float, default=60.0)
    a = ap.parse_args()
    if not a.smoke and not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. Training runs on Fortran's GPU only (lab rule).")
    dev = "cpu" if a.smoke else "cuda"
    torch.set_num_threads(2)                                   # Fortran rule: light on the shared CPU
    if not a.smoke:
        torch.backends.cuda.matmul.allow_tf32 = True
    ck = os.path.join(HERE, "results", "quick", "ckpt") if a.smoke else os.path.join(HERE, "ckpt")
    out = os.path.join(HERE, "results", "quick" if a.smoke else "raw")
    os.makedirs(out, exist_ok=True)
    os.makedirs(ck, exist_ok=True)
    seeds = (0,) if a.smoke else SEEDS
    cfg = skills.QUICK if a.smoke else skills.FULL
    steps = 3 if a.smoke else STEPS
    t_start = time.time()
    DEADLINE["t"] = t_start + 9.5 * 60
    gen = torch.Generator().manual_seed(2026)

    # 1. pretrain every net at once
    R = len(seeds)
    core = init_stack(R, CORE_H, dev, gen, n_codes=len(skills.PRE_RULES))
    plain = init_stack(R, matched_plain_h(), dev, gen)
    cues = torch.as_tensor(np.stack([pre_cues(s) for s in seeds]), device=dev)
    info_pre = pretrain([(core, True), (plain, False)], skills.PRE_RULES, cues, a.pretrain_sec, dev, gen,
                        max_steps=40 if a.smoke else None, ram_guard=ram_guard)
    pre_tab = np.repeat(skills.table(skills.PRE_RULES)[None], R, 0)
    noisy = torch.as_tensor(np.stack([skills.noisy(np.random.default_rng([s, 2]), pre_cues(s)) for s in seeds]), device=dev)
    rc_core = recall({k: v for k, v in core.items() if k != "codes"}, noisy, pre_tab)[1].mean(1).tolist()
    rc_plain = recall(plain, noisy, pre_tab)[1].mean(1).tolist()
    x = torch.arange(skills.P, device=dev).expand(R, len(skills.PRE_RULES), skills.P)
    rc_code = (forward(core, torch.zeros_like(noisy), x, code=core["codes"]).argmax(-1).cpu().numpy() == pre_tab).mean((1, 2)) * 100
    print(f"[pretrain] {info_pre['steps']} steps in {info_pre['sec']}s; recall core {rc_core}, plain {rc_plain}, "
          f"code path {rc_code.tolist()}", flush=True)
    torch.save({k: v.half().cpu() for k, v in core.items()}, os.path.join(ck, "pretrained_core.pt"))
    torch.save({k: v.half().cpu() for k, v in plain.items()}, os.path.join(ck, "pretrained_plain.pt"))

    # 2. tune on the validation stream (seed-0 nets)
    hp, tuning = dict(DEFAULT_HP), []
    if not a.smoke:
        t0 = time.time()
        vs = [prepare_stream(VAL_STREAM, seeds[0], cfg)]
        pspecs = ([dict(arm="B", seed_idx=0, stream=0, lr=v, hp=v, key="B") for v in GRID["B"]] +
                  [dict(arm="D", seed_idx=0, stream=0, lr=v, hp=v, key="D") for v in GRID["D"]] +
                  [dict(arm="C", seed_idx=0, stream=0, lr=v[0], lam=v[1], hp=v, key="C") for v in GRID["C"]])
        mspecs = [dict(arm="M", seed_idx=0, stream=0, lr=v, hp=v, key="M") for v in GRID["M"]]
        res = run_phase([PlainGroup(pspecs, plain, vs, dev, steps, 7), MGroup(mspecs, core, vs, dev, steps, 7)], vs)
        for key in GRID:
            cands = [r for r in res if r["arm"] == key]
            best = max(cands, key=score)
            hp[key] = best["hp"]
            tuning += [dict(arm=key, hp=r["hp"], score=round(float(score(r)), 2)) for r in cands]
            print(f"[tune] {key}: picked {hp[key]} (" + ", ".join(f"{r['hp']}={score(r):.1f}" for r in cands) + ")", flush=True)
        print(f"[tune] {time.time() - t0:.0f}s", flush=True)

    # 3. test: every arm on every seed's stream
    t0 = time.time()
    ts = [prepare_stream(s, s, cfg) for s in seeds]
    pspecs = [dict(arm=arm, seed_idx=i, stream=i, lr=lr, lam=lam, hp=h)
              for i in range(R) for arm, lr, lam, h in (("A", 0.0, 0.0, None), ("B", hp["B"], 0.0, hp["B"]),
                                                        ("C", hp["C"][0], hp["C"][1], hp["C"]), ("D", hp["D"], 0.0, hp["D"]))]
    mspecs = [dict(arm=arm, seed_idx=i, stream=i, lr=hp["M"], hp=hp["M"], mem=kw, shuffle=arm == "M_shuffled_labels")
              for i in range(R) for arm, kw in M_VARIANTS.items()]
    mg = MGroup(mspecs, core, ts, dev, steps, 11)
    res = run_phase([PlainGroup(pspecs, plain, ts, dev, steps, 11), mg], ts)
    print(f"[test] {len(pspecs) + len(mspecs)} runs x {len(ts[0]['episodes'])} episodes in {time.time() - t0:.0f}s", flush=True)
    state_mb = {}
    for r, s in enumerate(mspecs):
        if s["arm"] == "M":
            seed = seeds[s["seed_idx"]]
            p = os.path.join(ck, f"core_flex_final_seed{seed}.pt")
            torch.save(dict(core={k: v[s["seed_idx"]].half().cpu() for k, v in core.items() if k != "codes"},
                            **mg.mem[r].state()), p)
            state_mb[seed] = os.path.getsize(p) / 1e6
    total_min = (time.time() - t_start) / 60
    update_ms = bench_update_ms(core, dev, steps)
    for i, s in enumerate(seeds):
        d = {r["arm"]: r for r in res if r["stream"] == i}
        d["M"]["state_mb"] = state_mb[s]
        with open(os.path.join(out, f"seed{s}.json"), "w") as f:
            json.dump(d, f, separators=(",", ":"))
    ram_guard()
    info = dict(total_gpu_min=round(total_min, 2), device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                processes=1, hp=hp, tuning=tuning, smoke=a.smoke,
                pretrain=[dict(seed=s, which=w, steps=info_pre["steps"], sec=info_pre["sec"], recall_cue=float(rc[i]),
                               **({"recall_code": float(rc_code[i])} if w == "core" else {}))
                          for i, s in enumerate(seeds) for w, rc in (("core", rc_core), ("plain", rc_plain))],
                pretrain_log=info_pre["log"], update_ms=update_ms, route_us=bench_route_us(),
                core_params=mlp_numel(CORE_H), slot_params=slot_numel(), plain_h=matched_plain_h(),
                plain_params=mlp_numel(matched_plain_h()),
                active_params=mlp_numel(CORE_H) + (N_FLEX + N_CORE) * slot_numel(),
                replay_bytes=REPLAY_BYTES, peak_ram_gb=round(PEAK["rss"], 2))
    with open(os.path.join(out, "train_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"done in {total_min:.1f} min; update {update_ms:.1f} ms, lookup {info['route_us']:.1f} us, "
          f"peak RAM {PEAK['rss']:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
