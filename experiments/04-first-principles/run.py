"""Evaluate experiment 04: composer vs baselines, controls, ablations -> verdict table + results/.

python run.py            # after train.py (GPU recommended; evaluation only, no training)
python run.py --quick    # < 30 s plumbing check: exact blocks + untrained baselines, CPU ok
python run.py --clean    # delete caches and quick results (--all also deletes ckpt/)"""
import argparse
import collections
import json
import os
import shutil
import sys
import time

import numpy as np
import psutil
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import worlds as W
import composer as C
import baselines as B
import criteria

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt")
KS = (1, 2, 4, 8)


def pct(x):
    return 100.0 * float(np.mean(x)) if len(x) else 0.0


def dir_mb(path):
    tot = 0
    for root, _, files in os.walk(path):
        tot += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return tot / 1e6


def peak_ram_gb():
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)
    if peak is None:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except ImportError:
            peak = mi.rss
    return peak / 1e9


def load(cls, kwargs, path, dev):
    m = cls(**kwargs)
    m.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
    return m.to(dev).eval()


def shuffle_arith(eps):
    """Negative control: rotate the answers among the k examples (the examples now carry no signal)."""
    out = []
    for ex, q, y in eps:
        ys = [v for _, v in ex]
        ys = ys[1:] + ys[:1]
        out.append(([(x, v) for (x, _), v in zip(ex, ys)], q, y))
    return out


def fp_arith_acc(lib, arity, eps, times=None):
    hits = []
    for ex, q, y in eps:
        t0 = time.perf_counter()
        prog = lib.solve(arity, ex)
        if times is not None:
            times.append(time.perf_counter() - t0)
        hits.append(C.ArithLibrary.answer(prog, q) == y)
    return pct(hits)


def fp_family_acc(lib, R, eps, R_examples=None, times=None, progs=None):
    accs = []
    for ex, test in eps:
        t0 = time.perf_counter()
        p = lib.solve(R if R_examples is None else R_examples, ex)
        if times is not None:
            times.append(time.perf_counter() - t0)
        if progs is not None:
            progs.append(p)
        accs.append(0.0 if p is None else np.mean([np.array_equal(p["table"][x], R[x]) for x in test]))
    return pct(accs)


def table_acc(pred, R, eps):
    """pred: (n, n) bool predicted relation; eps: [(examples, test people)]."""
    return pct([np.mean([np.array_equal(pred[x], R[x]) for x in test]) for _, test in eps])


# =============================================================================== arithmetic
def eval_arith(seed, quick, dev, icl, n_ep, log):
    out = dict(fp={}, fp_k={}, base={s: {} for s in icl if s != "oracle"}, base_k={}, oracle={}, programs={})
    if quick:
        S = np.array([x + 1 if x + 1 < W.N else W.UNDEF for x in range(W.N)])
        S_miss = S.copy()
        S_miss[np.random.default_rng(seed).choice(W.N, 10, replace=False)] = W.UNDEF
        block = None
    else:
        block = load(C.NextBlock, {}, os.path.join(CKPT, f"next_s{seed}.pt"), dev)
        S = C.snap_next(block)
        S_miss = C.snap_next(load(C.NextBlock, {}, os.path.join(CKPT, f"next_missing_s{seed}.pt"), dev))
    torch.manual_seed(500 + seed)
    S_untr = C.snap_next(C.NextBlock().to(dev))
    t0 = time.perf_counter()
    lib = C.ArithLibrary(S)
    build_s = time.perf_counter() - t0
    libs = dict(noinv=C.ArithLibrary(S, use_inverse=False), missing=C.ArithLibrary(S_miss),
                untrained=C.ArithLibrary(S_untr))
    out["library_size"], out["library_build_s"] = len(lib), build_s
    out["block_fit"] = pct(S == np.array([x + 1 if x + 1 < W.N else W.UNDEF for x in range(W.N)]))
    times = []
    ctrl = collections.defaultdict(list)
    for i, op in enumerate(W.ARITH_TEST):
        arity, T = W.arith_table(op)
        for k in KS:
            eps = W.arith_episodes(arity, T, k, n_ep, seed=seed * 1000 + i * 10 + k)
            acc = fp_arith_acc(lib, arity, eps, times if k == 4 else None)
            out["fp_k"].setdefault(op, {})[k] = acc
            for s, m in icl.items():
                a = pct(B.icl_answer(m, eps, dev) == np.array([y for _, _, y in eps]))
                out["base_k"].setdefault(s, {}).setdefault(op, {})[k] = a
            if k == 4:
                out["fp"][op] = acc
                for s in icl:
                    (out["oracle"] if s == "oracle" else out["base"][s])[op] = out["base_k"][s][op][4]
                prog = lib.solve(arity, eps[0][0])
                out["programs"][op] = prog["expr"] if prog else None
                ctrl["shuffled"].append(fp_arith_acc(lib, arity, shuffle_arith(eps)))
                ctrl["untrained"].append(fp_arith_acc(libs["untrained"], arity, eps))
                if op in W.NEEDS_UNDO_ARITH:
                    ctrl["no_undo"].append(fp_arith_acc(libs["noinv"], arity, eps))
                out.setdefault("missing", {})[op] = fp_arith_acc(libs["missing"], arity, eps)
    arity, T = W.random_unary_op(seed)
    eps = W.arith_episodes(arity, T, 4, n_ep, seed=seed * 1000 + 999)
    ctrl["random_op"].append(fp_arith_acc(lib, arity, eps))
    out["random_op_baseline"] = {s: pct(B.icl_answer(m, eps, dev) == np.array([y for _, _, y in eps]))
                                 for s, m in icl.items() if s != "oracle"}
    out["controls"] = {c: float(np.mean(v)) for c, v in ctrl.items()}
    own = {}
    for s, m in icl.items():
        if s == "oracle":
            continue
        accs = []
        for i, op in enumerate(W.ARITH_TRAIN):
            arity, T = W.arith_table(op)
            eps = W.arith_episodes(arity, T, 4, max(20, n_ep // 2), seed=seed * 1000 + 500 + i)
            accs.append(pct(B.icl_answer(m, eps, dev) == np.array([y for _, _, y in eps])))
        own[s] = float(np.mean(accs))
    out["base_own"] = own
    out["search_s"] = float(np.max(times)) if times else 0.0
    out["latent_chain"] = C.latent_chain_accuracy(block, [1, 2, 5, 10, 20, 50]) if block is not None else {}
    out["latent_chain"] = {b: 100 * v for b, v in out["latent_chain"].items()}
    prog = lib.solve(2, [((9, 4), 5), ((50, 8), 42), ((30, 30), 0), ((77, 1), 76)])
    lat = []
    for a in range(200):
        x = (a % 90 + 9, a % 9)
        t0 = time.perf_counter_ns()
        C.ArithLibrary.answer(prog, x)
        lat.append(time.perf_counter_ns() - t0)
    out["answer_us"] = float(np.median(lat)) / 1000
    log(f"  arith seed {seed}: library {len(lib)} programs in {build_s:.2f}s, composer {out['fp']}, "
        f"baselines {out['base']}, controls {out['controls']}")
    return out


# =============================================================================== family
def eval_family(seed, quick, dev, gnn, n_ep, fs_steps, log):
    out = dict(fp={}, fp_k={}, base={s: {} for s in gnn if s != "oracle"}, oracle={}, programs={}, missing={}, soft={})
    P, G = W.family(seed)
    R = W.relations(P, G)
    n = len(P)
    if quick:
        Ps, Gs, Ps_soft, Gs_soft = P, G, torch.tensor(P, dtype=torch.float32), torch.tensor(G, dtype=torch.float32)
        Pm = P & (np.random.default_rng(seed).random(P.shape) > 0.1)
        Gm = G
    else:
        blk = load(C.FamilyBlocks, dict(n=n), os.path.join(CKPT, f"family_s{seed}.pt"), dev)
        Ps, Gs = C.snap_family(blk)
        Ps_soft, Gs_soft = C.snap_family(blk, soft=True)
        Pm, Gm = C.snap_family(load(C.FamilyBlocks, dict(n=n), os.path.join(CKPT, f"family_missing_s{seed}.pt"), dev))
    torch.manual_seed(600 + seed)
    Pu, Gu = C.snap_family(C.FamilyBlocks(n).to(dev))
    t0 = time.perf_counter()
    lib = C.FamilyLibrary(Ps, Gs, dev)
    build_s = time.perf_counter() - t0
    libs = dict(noinv=C.FamilyLibrary(Ps, Gs, dev, use_inverse=False), missing=C.FamilyLibrary(Pm, Gm, dev),
                untrained=C.FamilyLibrary(Pu, Gu, dev))
    out["library_size"], out["library_build_s"] = len(lib), build_s
    out["block_fit"] = pct(np.concatenate([(Ps == P).ravel(), Gs == G]))
    times, ctrl, fs_jobs = [], collections.defaultdict(list), []
    for i, rel in enumerate(W.REL_TEST):
        for k in KS:
            eps = W.rel_episodes(R[rel], k, n_ep, seed=seed * 1000 + i * 10 + k)
            progs = [] if k == 4 else None
            acc = fp_family_acc(lib, R[rel], eps, times=times if k == 4 else None, progs=progs)
            out["fp_k"].setdefault(rel, {})[k] = acc
            if k != 4:
                continue
            out["fp"][rel] = acc
            out["programs"][rel] = progs[0]["expr"] if progs[0] else None
            fs_jobs.append((rel, eps))
            # negative controls and ablations
            sh = []
            for ex, test in eps:
                Rf = R[rel].copy()
                Rf[np.array(ex)] = R[rel][np.roll(np.array(ex), 1)]
                sh.append(fp_family_acc(lib, R[rel], [(ex, test)], R_examples=Rf))
            ctrl["shuffled"].append(float(np.mean(sh)))
            ctrl["untrained"].append(fp_family_acc(libs["untrained"], R[rel], eps))
            if rel in W.NEEDS_UNDO_REL:
                ctrl["no_undo"].append(fp_family_acc(libs["noinv"], R[rel], eps))
            out["missing"][rel] = fp_family_acc(libs["missing"], R[rel], eps)
            cache, soft = {}, []
            for (ex, test), p in zip(eps, progs):
                if p is None:
                    soft.append(0.0)
                    continue
                if p["expr"] not in cache:
                    cache[p["expr"]] = (C.soft_execute(p["tree"], Ps_soft.to(dev), Gs_soft.to(dev)) > 0.5).cpu().numpy()
                soft.append(np.mean([np.array_equal(cache[p["expr"]][x], R[rel][x]) for x in test]))
            out["soft"][rel] = pct(soft)
    Rr = W.random_relation(n, float(R["uncle"].mean()), seed)
    eps_r = W.rel_episodes(Rr, 4, n_ep, seed=seed * 1000 + 999)
    ctrl["random_relation"].append(fp_family_acc(lib, Rr, eps_r))
    out["controls"] = {c: float(np.mean(v)) for c, v in ctrl.items()}
    # baselines: few-shot relation vectors fitted on the same k = 4 examples
    jobs = fs_jobs + [("random_relation", eps_r)]
    tables = {**R, "random_relation": Rr}
    R_list = [tables[rel] for rel, eps in jobs for _ in eps]
    people = [ex for rel, eps in jobs for ex, _ in eps]
    out["random_relation_baseline"], out["base_own"] = {}, {}
    for s, m in gnn.items():
        if s == "oracle":
            vec = m.rel.weight[[B.REL_ALL.index(r) for r in W.REL_TEST]].detach()
            pred = B.gnn_predict(m, P, G, vec, dev)
            for j, (rel, eps) in enumerate(fs_jobs):
                out["oracle"][rel] = table_acc(pred[j], R[rel], eps)
            continue
        vec = B.gnn_fewshot(m, P, G, R_list, people, dev, steps=fs_steps)
        pred = B.gnn_predict(m, P, G, vec, dev)
        e = 0
        for rel, eps in jobs:
            accs = [table_acc(pred[e + j], tables[rel], [eps[j]]) for j in range(len(eps))]
            e += len(eps)
            if rel == "random_relation":
                out["random_relation_baseline"][s] = float(np.mean(accs))
            else:
                out["base"][s][rel] = float(np.mean(accs))
        own_vec = m.rel.weight[[B.REL_ALL.index(r) for r in W.REL_TRAIN]].detach()
        own_pred = B.gnn_predict(m, P, G, own_vec, dev)
        own = []
        for j, rel in enumerate(W.REL_TRAIN):
            people_ne = [x for x in range(n) if R[rel][x].any()]
            own.append(pct([np.array_equal(own_pred[j][x], R[rel][x]) for x in people_ne]))
        out["base_own"][s] = float(np.mean(own))
    out["search_s"] = float(np.max(times)) if times else 0.0
    log(f"  family seed {seed}: {n} people, library {len(lib)} relations in {build_s:.2f}s, composer {out['fp']}, "
        f"baselines {out['base']}, controls {out['controls']}")
    return out


# =============================================================================== aggregate
def mean_over(dicts, key=None):
    """Average a list of {name: value} (or {name: {k: value}}) over seeds."""
    keys = dicts[0].keys()
    if isinstance(next(iter(dicts[0].values()), 0), dict):
        return {k: mean_over([d[k] for d in dicts]) for k in keys}
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def summarize(per_seed, base_sizes):
    s = dict(fp=mean_over([p["fp"] for p in per_seed]), fp_k=mean_over([p["fp_k"] for p in per_seed]),
             base={b: mean_over([p["base"][b] for p in per_seed]) for b in base_sizes},
             oracle=mean_over([p["oracle"] for p in per_seed]), controls=mean_over([p["controls"] for p in per_seed]),
             missing=mean_over([p["missing"] for p in per_seed]), base_own=mean_over([p["base_own"] for p in per_seed]),
             programs=[p["programs"] for p in per_seed],
             library_size=[p["library_size"] for p in per_seed],
             library_build_s=float(np.max([p["library_build_s"] for p in per_seed])),
             search_s=float(np.max([p["search_s"] for p in per_seed])),
             block_fit=float(np.mean([p["block_fit"] for p in per_seed])))
    s["fp_mean"] = float(np.mean(list(s["fp"].values())))
    s["base_means"] = {b: float(np.mean(list(v.values()))) for b, v in s["base"].items()}
    best = max(s["base_means"], key=s["base_means"].get)
    s["base_best_size"], s["base_best_mean"], s["base_best"] = best, s["base_means"][best], s["base"][best]
    s["oracle_mean"] = float(np.mean(list(s["oracle"].values())))
    s["base_own_best"] = max(s["base_own"].values())
    s["fp_k1"] = {k: v[1] for k, v in s["fp_k"].items()}
    s["fp_seed_means"] = [float(np.mean(list(p["fp"].values()))) for p in per_seed]
    return s


def write_md(path, m, rows, v, gate, notes):
    a, r, b = m["arith"], m["family"], m["budget"]
    L = [f"# Results · experiment 04\n", f"**Verdict: {v}**" + (f" · ✨ INTRIGUING: {'; '.join(notes)}" if notes else ""), "",
         "| Criterion | Value | Threshold | Result |", "|---|---|---|---|"]
    for row in rows:
        val = row["value"]
        val = ", ".join(f"{k} {x:.2f}" if isinstance(x, float) else f"{k} {x}" for k, x in val.items()) if isinstance(val, dict) else val
        thr = row["threshold"] if not isinstance(row["threshold"], dict) else "see criteria.py"
        L.append(f"| {row['criterion']} | {val} | {thr} | {row['result']} |")
    L += ["", "Validity gate: " + ", ".join(f"{k} {'ok' if ok else 'FAILED'}" for k, ok in gate.items()), ""]
    for name, d, unit in (("Arithmetic (unseen operations)", a, "op"), ("Family (unseen relations)", r, "relation")):
        sizes = list(d["base"])
        L += [f"## {name}, k = 4, mean of seeds", "",
              f"| {unit} | composer | " + " | ".join(f"baseline {s}" for s in sizes) + " | oracle baseline | composer, 10% facts missing | program found (seed 0) |",
              "|---|---|" + "---|" * len(sizes) + "---|---|---|"]
        for op in d["fp"]:
            L.append(f"| {op} | {d['fp'][op]:.1f} | " + " | ".join(f"{d['base'][s][op]:.1f}" for s in sizes) +
                     f" | {d['oracle'][op]:.1f} | {d['missing'][op]:.1f} | `{d['programs'][0][op]}` |")
        L.append(f"| **mean** | **{d['fp_mean']:.1f}** | " + " | ".join(f"{d['base_means'][s]:.1f}" for s in sizes) +
                 f" | {d['oracle_mean']:.1f} | {np.mean(list(d['missing'].values())):.1f} | |")
        L += ["", "Composer accuracy vs number of examples k: " +
              "; ".join(f"{op} " + "/".join(f"{d['fp_k'][op][k]:.0f}" for k in KS) for op in d["fp_k"]) + " (k = 1/2/4/8)", "",
              "Negative controls (composer, should be ~0): " + ", ".join(f"{c} {x:.1f}" for c, x in d["controls"].items()),
              f"Baselines on their own training tasks: " + ", ".join(f"{s} {x:.1f}" for s, x in d["base_own"].items()),
              f"Library: {d['library_size']} distinct programs per seed, built in ≤ {d['library_build_s']:.2f} s; "
              f"slowest pick of a program for a new operation {1000 * d['search_s']:.1f} ms; learned block fit {d['block_fit']:.1f}%.", ""]
    if a.get("latent_chain"):
        L.append("No-snap ablation, arithmetic (a + b by chaining the next-number block b times in hidden space): " +
                 ", ".join(f"{k} steps {x:.0f}%" for k, x in a["latent_chain"].items()))
    if r.get("soft"):
        L.append("No-snap ablation, family (programs run on soft probabilities): " + ", ".join(f"{k} {x:.1f}" for k, x in r["soft"].items()))
    L += ["", f"Budget: composer files {b['composer_mb']:.3f} MB, kept {b['kept_mb']:.1f} MB, training {b['train_min']:.2f} min, "
          f"answer {1000 * b['answer_ms']:.2f} µs median, program pick ≤ {b['search_s'] * 1000:.1f} ms, peak RAM {b['peak_ram_gb']:.2f} GB."]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.clean:
        for p in ("__pycache__", ".pytest_cache", os.path.join("results", "quick")) + (("ckpt",) if args.all else ()):
            full = os.path.join(HERE, p)
            if os.path.isdir(full):
                mb = dir_mb(full)
                shutil.rmtree(full)
                print(f"deleted {p} ({mb:.1f} MB)")
        return
    t_start = time.time()
    torch.set_num_threads(2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = os.path.join(HERE, "results", "quick") if args.quick else os.path.join(HERE, "results")
    os.makedirs(res, exist_ok=True)
    seeds = (0,) if args.quick else (0, 1, 2)
    n_ep_a, n_ep_r, fs_steps = (20, 2, 3) if args.quick else (200, 20, 200)
    if args.quick:
        torch.manual_seed(0)
        icl = {s: B.ICLTransformer(**B.SIZES_ICL["small"]).to(dev).eval() for s in ("small", "large", "oracle")}
        gnn = {s: B.PathGNN(**B.SIZES_GNN["small"]).to(dev).eval() for s in ("small", "large", "oracle")}
        train = dict(total_train_min=0.0)
    else:
        if not os.path.isfile(os.path.join(CKPT, "train_log.json")):
            sys.exit("no checkpoints: run train.py on the GPU first (or run.py --quick)")
        icl = {s: load(B.ICLTransformer, B.SIZES_ICL["large" if s == "oracle" else s], os.path.join(CKPT, f"icl_{s}.pt"), dev)
               for s in ("small", "large", "oracle")}
        gnn = {s: load(B.PathGNN, B.SIZES_GNN["large" if s == "oracle" else s], os.path.join(CKPT, f"gnn_{s}.pt"), dev)
               for s in ("small", "large", "oracle")}
        train = json.load(open(os.path.join(CKPT, "train_log.json")))
    log = lambda s: print(s, flush=True)
    log(f"device {dev}; seeds {seeds}")
    pa = [eval_arith(s, args.quick, dev, icl, n_ep_a, log) for s in seeds]
    pf = [eval_family(s, args.quick, dev, gnn, n_ep_r, fs_steps, log) for s in seeds]
    A, Fm = summarize(pa, ["small", "large"]), summarize(pf, ["small", "large"])
    A["facts"] = len(W.succ_facts())
    A["latent_chain"] = mean_over([p["latent_chain"] for p in pa]) if pa[0]["latent_chain"] else {}
    Fm["soft"] = mean_over([p["soft"] for p in pf])
    A["random_op_baseline"] = mean_over([p["random_op_baseline"] for p in pa])
    Fm["random_relation_baseline"] = mean_over([p["random_relation_baseline"] for p in pf])
    comp_mb = 0.0 if args.quick else (os.path.getsize(os.path.join(CKPT, "next_s0.pt")) +
                                      os.path.getsize(os.path.join(CKPT, "family_s0.pt"))) / 1e6
    budget = dict(composer_mb=comp_mb, kept_mb=(0.0 if args.quick else dir_mb(CKPT)) + dir_mb(res),
                  train_min=train["total_train_min"], answer_ms=float(np.median([p["answer_us"] for p in pa])) / 1000,
                  search_s=max(A["search_s"], Fm["search_s"]), peak_ram_gb=peak_ram_gb())
    m = dict(budget=budget, arith=A, family=Fm, train=train, eval_min=(time.time() - t_start) / 60, quick=args.quick)
    rows, v, gate, notes = criteria.verdict(m)
    print("\n%-72s %-32s %s" % ("criterion", "threshold", "result"))
    for row in rows:
        print("%-72s %-32s %s" % (row["criterion"][:72], str(row["threshold"])[:32], row["result"]))
    print("gate:", {k: ("ok" if ok else "FAILED") for k, ok in gate.items()})
    print(f"VERDICT: {v}" + (f"  ✨ INTRIGUING: {'; '.join(notes)}" if notes else "") + ("  (quick plumbing run)" if args.quick else ""))
    with open(os.path.join(res, "verdict.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(verdict=v, rows=rows, gate=gate, intriguing=notes, metrics=m), fh, indent=1, default=str)
    write_md(os.path.join(res, "results.md"), m, rows, v, gate, notes)
    try:
        import plots
        plots.make(m, res)
    except Exception as e:      # plots never break a run
        print("plots skipped:", e)
    print(f"wrote {res} in {(time.time() - t_start):.0f}s")


if __name__ == "__main__":
    main()
