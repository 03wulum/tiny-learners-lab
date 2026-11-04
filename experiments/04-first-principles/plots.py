"""Figures for experiment 04 (called by run.py; never breaks a run)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
COL = dict(composer="#2563eb", small="#a3a3a3", large="#525252", oracle="#d97706")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def make(m, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[5, 6]))
    for ax, key, title in ((axes[0], "arith", "Arithmetic: unseen operations"), (axes[1], "family", "Family: unseen relations")):
        d = m[key]
        ops = list(d["fp"])
        x = np.arange(len(ops))
        series = [("composer", d["fp"], "composer (learned 1 fact type)"), ("small", d["base"]["small"], "baseline, small"),
                  ("large", d["base"]["large"], "baseline, large"), ("oracle", d["oracle"], "baseline trained on the targets")]
        w = 0.2
        for i, (c, vals, lab) in enumerate(series):
            ax.bar(x + (i - 1.5) * w, [vals[o] for o in ops], w, color=COL[c], label=lab)
        ax.set_xticks(x, ops, color=INK)
        ax.set_ylim(0, 105)
        ax.set_ylabel("exact-match accuracy, k = 4 (%)", color=MUTED)
        ax.set_title(title, color=INK, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "unseen_ops.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ks = [1, 2, 4, 8]
    ax = axes[0]
    for key in ("arith", "family"):
        fpk = m[key]["fp_k"]
        mean = [np.mean([fpk[o][k] for o in fpk]) for k in ks]
        ax.plot(ks, mean, marker="o", color=COL["composer"] if key == "arith" else "#7c3aed", label=f"composer, {key}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, ks)
    ax.set_xlabel("worked examples shown (k)", color=MUTED)
    ax.set_ylabel("mean accuracy on unseen ops (%)", color=MUTED)
    ax.set_ylim(0, 105)
    ax.set_title("How many examples does it need?", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    ax = axes[1]
    lc = m["arith"].get("latent_chain") or {}
    if lc:
        steps = sorted(int(k) for k in lc)
        ax.plot(steps, [lc[s] if s in lc else lc[str(s)] for s in steps], marker="o", color="#dc2626", label="no snapping (hidden-space chain)")
        ax.plot(steps, [100] * len(steps), ls="--", color=COL["composer"], label="snap to lookup table each step")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("steps chained (a + b with b steps)", color=MUTED)
    ax.set_ylabel("accuracy (%)", color=MUTED)
    ax.set_ylim(0, 105)
    ax.set_title("Why snap to pointers: errors compound", color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "examples_and_snapping.png"), dpi=150)
    plt.close(fig)
