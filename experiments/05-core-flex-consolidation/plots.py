"""Phone-readable plots for tiny-learners 05. Ours in blue, everything else gray/orange."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OURS, OTHER, GRAY, INK = "#2a78d6", "#eb6834", "#9a9a94", "#3a3a36"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
CHANCE = 100 / 17


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)


def make(per_seed, tab, out):
    from run import ARM_LABEL
    # 1. headline: retention per arm
    arms = ["A", "B", "C", "D", "M"]
    vals = [tab[a]["retention"] for a in arms]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.barh([ARM_LABEL[a] for a in arms], vals, color=[OURS if a == "M" else GRAY for a in arms], height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}", va="center", color=INK)
    ax.axvline(CHANCE, color=GRAY, ls=":", lw=1)
    ax.set_xlim(0, 108)
    ax.invert_yaxis()
    ax.set_xlabel("recall from the cue alone at the end (%)")
    ax.set_title("Who still remembers? (old + new skills, same active size)", loc="left", fontsize=12)
    _save(fig, out, "retention.png")

    # 2. recall when a skill comes back, along the stream
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for a, col in (("B", GRAY), ("D", OTHER), ("M_noarchive", "#c3c2b7"), ("M", OURS)):
        ev = [(e["t"], e["pre"]) for s in per_seed for e in s[a]["episodes"] if not e["first"]]
        if not ev:
            continue
        t, p = np.array(ev).T
        edges = np.linspace(0, t.max() + 1, 12)
        mid = 0.5 * (edges[1:] + edges[:-1])
        y = [p[(t >= lo) & (t < hi)].mean() if ((t >= lo) & (t < hi)).any() else np.nan
             for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(mid, y, color=col, lw=2, label=ARM_LABEL[a])
    ax.axhline(CHANCE, color=GRAY, ls=":", lw=1)
    ax.set_ylim(0, 105)
    ax.set_xlabel("episode in the stream")
    ax.set_ylabel("recall when a skill returns (%)")
    ax.set_title("Coming back to a skill: recall before any new examples", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    _save(fig, out, "recall_on_return.png")

    # 3. frequent skills right after each burst of one-offs (active weights only)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for a, col in (("flat_lru", GRAY), ("flat_strength", OTHER), ("M", OURS)):
        rec = np.array([[b["recall"] for b in s[a]["bursts"]] for s in per_seed])
        if rec.size == 0:
            continue
        ax.plot(np.arange(1, rec.shape[1] + 1), rec.mean(0), "-o", color=col, lw=2, ms=8, label=ARM_LABEL[a])
    ax.set_ylim(0, 105)
    ax.set_xlabel("burst of one-off skills")
    ax.set_ylabel("frequent-skill recall (%)")
    ax.set_title("Do bursts of novelty push out the important skills?", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    _save(fig, out, "bursts.png")

    # 4. capacity: active-only vs with archive
    arms = ["cap_small", "M", "cap_large", "unlimited"]
    lab = ["9 slots", "18 slots (M)", "36 slots", "unlimited"]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.18, [tab[a]["stream_active"] for a in arms], 0.34, color=GRAY, label="active weights only")
    ax.bar(x + 0.18, [tab[a]["stream"] for a in arms], 0.34, color=OURS, label="with archive")
    ax.set_xticks(x, lab)
    ax.set_ylim(0, 105)
    ax.set_ylabel("stream-skill recall at the end (%)")
    ax.set_title("Fixed size + archive vs more slots", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    _save(fig, out, "capacity.png")
