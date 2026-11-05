"""Score tiny-learners 05 "core + flexible weights": criteria table, verdict, results.

  python run.py            score the GPU run in results/raw/ (written by train.py on Fortran)
  python run.py --quick    < 30 s plumbing check: tiny CPU run of train.py --smoke, then scoring
  python run.py --clean    delete caches and quick results (add --all to delete ckpt/ too)

No learning happens here, so it needs no GPU.
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np

import criteria

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ARM_LABEL = {"A": "A frozen", "B": "B fine-tune", "C": "C EWC", "D": "D replay (same bytes)",
             "M": "M core + flex (ours)", "M_noarchive": "M, no archive", "flat_lru": "one ring, LRU",
             "flat_strength": "one ring, strength", "harden_at_once": "harden at first sight",
             "unlimited": "unlimited slots", "M_shuffled_labels": "M, shuffled labels (control)",
             "cap_small": "M, half the slots", "cap_large": "M, double the slots"}


def per_arm(r):
    f, eps = r["final"], r["episodes"]
    first = [e for e in eps if e["first"]]
    ret_ev = [e for e in eps if not e["first"] and e["last_post"]]
    return dict(
        retention=0.5 * (f["pretrained"] + f["system"]["all"]), pretrained=f["pretrained"],
        stream=f["system"]["all"], stream_active=f["active"]["all"],
        **{f"stream_{g}": f["system"][g] for g in ("freq", "occ", "one")},
        first=float(np.mean([e["post"] for e in first])), first_held=float(np.mean([e["held"] for e in first])),
        return_ratio=sum(e["pre"] for e in ret_ev) / max(1e-9, sum(e["last_post"] for e in ret_ev)),
        burst=float(np.mean([b["recall"] for b in r["bursts"]])) if r["bursts"] else float("nan"),
        shuffled_cue=f["shuffled_cue"], never_seen=f["never_seen"])


def recovery(r, before):
    ev = [e for e in r["episodes"] if not e["first"] and e["before"] == before and e["last_post"]]
    if not ev:
        return float("nan"), 0
    return sum(e["pre"] for e in ev) / sum(e["last_post"] for e in ev), len(ev)


def aggregate(per_seed, info):
    arms = list(per_seed[0].keys())
    tab = {a: {k: float(np.mean([per_arm(s[a])[k] for s in per_seed])) for k in per_arm(per_seed[0][a])}
           for a in arms}
    mean = lambda f: float(np.mean([f(s) for s in per_seed]))
    rec_m = [recovery(s["M"], "archive") for s in per_seed]
    rec_n = [recovery(s["M_noarchive"], "none") for s in per_seed]
    n_eps = sum(len(s["M"]["episodes"]) for s in per_seed)
    budget = dict(active_mb=info["active_params"] * 2 / 1e6,
                  kept_mb=max(s["M"].get("state_mb", 0.0) for s in per_seed),
                  total_gpu_min=info["total_gpu_min"], update_ms=info["update_ms"],
                  route_us=info["route_us"], peak_ram_gb=info["peak_ram_gb"])
    pre = info["pretrain"]
    m = dict(budget=budget,
             ret_M=tab["M"]["retention"], ret_B=tab["B"]["retention"], ret_D=tab["D"]["retention"],
             ret_unlimited=tab["unlimited"]["retention"],
             ret_seed_std=float(np.std([per_arm(s["M"])["retention"] for s in per_seed])),
             first_M=tab["M"]["first"], first_B=tab["B"]["first"], first_heldout_M=tab["M"]["first_held"],
             recovery_M=float(np.nanmean([x for x, _ in rec_m])) if any(n for _, n in rec_m) else float("nan"),
             recovery_noarch=float(np.nanmean([x for x, _ in rec_n])) if any(n for _, n in rec_n) else float("nan"),
             recovery_events=dict(M=sum(n for _, n in rec_m), no_archive=sum(n for _, n in rec_n)),
             burst_M=tab["M"]["burst"], burst_flatLRU=tab["flat_lru"]["burst"],
             burst_flatStrength=tab["flat_strength"]["burst"],
             pretrain_core=min(p["recall_cue"] for p in pre if p["which"] == "core"),
             pretrain_plain=min(p["recall_cue"] for p in pre if p["which"] == "plain"),
             pretrain_code_path=min(p.get("recall_code", 100) for p in pre if p["which"] == "core"),
             ctrl_shuffled_cue=tab["M"]["shuffled_cue"], ctrl_never_seen=tab["M"]["never_seen"],
             ctrl_frozen_new=tab["A"]["first"], ctrl_shuffled_labels_heldout=tab["M_shuffled_labels"]["first_held"],
             route_error_rate=sum(s["M"]["route_errors"] for s in per_seed) / max(1, n_eps),
             archive_kb=mean(lambda s: s["M"]["archive_bytes"]) / 1e3,
             replay_kb=mean(lambda s: s["D"]["replay_bytes"]) / 1e3)
    return m, tab


def nan_to_none(o):
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [nan_to_none(v) for v in o]
    return o


def write_report(m, tab, info, out):
    for k in ("recovery_M", "recovery_noarch", "burst_M", "burst_flatLRU"):
        if m[k] != m[k]:          # nan: no events, the criterion cannot pass
            m[k] = -1.0 if k in ("recovery_M", "burst_M") else 1e9
    rows, v, gate, notes = criteria.verdict(m)
    os.makedirs(out, exist_ok=True)
    js = dict(experiment="tiny-learners 05 core + flexible weights", verdict=v, intriguing=notes,
              criteria=rows, gate=gate, measured=m, arms=tab, hp=info["hp"], smoke=info.get("smoke", False))
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(nan_to_none(js), f, indent=1)

    fmt = lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)
    print("\n" + "=" * 86)
    for r in rows:
        val = r["value"]
        val = ", ".join(f"{k}={fmt(x)}" for k, x in val.items()) if isinstance(val, dict) else fmt(val)
        print(f"{r['criterion']:<58} {r['result']:<5} {val}")
    print(f"gate: {gate}")
    print(f"VERDICT: {v}" + (f"   ✨ INTRIGUING: {'; '.join(notes)}" if notes else ""))

    cols = [("retention", "Retention score"), ("pretrained", "Pretrained skills"), ("stream", "Stream skills"),
            ("stream_freq", "frequent"), ("stream_occ", "occasional"), ("stream_one", "one-off"),
            ("stream_active", "Stream, active weights only"), ("first", "After 1st episode"),
            ("first_held", "1st episode, unseen x"), ("return_ratio", "Recall on return / last"),
            ("burst", "Frequent after bursts (active)")]
    lines = ["| Arm | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for a in tab:
        lines.append(f"| {ARM_LABEL.get(a, a)} | " + " | ".join(
            (f"{tab[a][k]:.2f}" if k == "return_ratio" else f"{tab[a][k]:.1f}") for k, _ in cols) + " |")
    md = ["# Results: tiny-learners 05, core + flexible weights", "",
          f"**Verdict: {v}**" + (f" · ✨ INTRIGUING: {'; '.join(notes)}" if notes else ""), "",
          "| Criterion | Value | Threshold | Result |", "|---|---|---|---|"]
    for r in rows:
        val = r["value"]
        val = ", ".join(f"{k} {fmt(x)}" for k, x in val.items()) if isinstance(val, dict) else fmt(val)
        thr = r["threshold"] if not isinstance(r["threshold"], dict) else ", ".join(f"{k} ≤ {x}" for k, x in r["threshold"].items())
        md.append(f"| {r['criterion']} | {val} | {thr} | {r['result']} |")
    md += ["", f"Validity gate: {', '.join(f'{k} {chr(10003) if ok else chr(10007)}' for k, ok in gate['gate'].items())}"
           f"; headline too close to call: {gate['headline_noisy']}", "",
           "## Every arm (accuracy in points, mean of 3 seeds, chance 5.9)", "", *lines, "",
           "## Run facts", "",
           f"- Device: {info['device']}, {info['processes']} process, total {info['total_gpu_min']:.1f} GPU min",
           f"- Active parameters: {info['active_params']:,} (core {info['core_params']:,} + 18 slots × {info['slot_params']:,}); "
           f"plain net width {info['plain_h']} = {info['plain_params']:,} parameters",
           f"- Archive at the end: {m['archive_kb']:.0f} KB; replay buffer: {m['replay_kb']:.0f} KB",
           f"- One model's update for one episode: {info['update_ms']:.1f} ms",
           f"- Pointer lookup: {info['route_us']:.1f} µs per query (CPU); routing errors {100 * m['route_error_rate']:.1f}%",
           f"- Recovery events: {m['recovery_events']}",
           f"- Pretrained recall: core {m['pretrain_core']:.1f}, plain {m['pretrain_plain']:.1f}, core code path {m['pretrain_code_path']:.1f}",
           f"- Tuned on the validation stream: {info['hp']}"]
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(lines))
    return v


def score(raw, out):
    info = json.load(open(os.path.join(raw, "train_info.json")))
    seeds = sorted(f for f in os.listdir(raw) if f.startswith("seed") and f.endswith(".json"))
    per_seed = [json.load(open(os.path.join(raw, f))) for f in seeds]
    m, tab = aggregate(per_seed, info)
    v = write_report(m, tab, info, out)
    try:
        import plots
        plots.make(per_seed, tab, out)
    except Exception as e:   # plots are a nicety; never lose the verdict over them
        print("plotting failed:", e)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--all", action="store_true", help="with --clean: also delete ckpt/")
    a = ap.parse_args()
    if a.clean:
        freed, targets = 0, ["__pycache__", ".pytest_cache", os.path.join("results", "quick")] + (["ckpt"] if a.all else [])
        for t in targets:
            p = os.path.join(HERE, t)
            if os.path.exists(p):
                freed += sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
                shutil.rmtree(p)
        print(f"cleaned {targets}: freed {freed / 1e6:.1f} MB")
        return
    if a.quick:
        q = os.path.join(RES, "quick")
        rc = subprocess.call([sys.executable, os.path.join(HERE, "train.py"), "--smoke"], cwd=HERE)
        if rc:
            sys.exit(rc)
        score(q, q)
        print("(quick mode: tiny CPU plumbing run, not a result)")
        return
    raw = os.path.join(RES, "raw")
    if not os.path.exists(os.path.join(raw, "train_info.json")):
        sys.exit("No GPU run found in results/raw/. Run train.py on Fortran's GPU first.")
    score(raw, RES)


if __name__ == "__main__":
    main()
