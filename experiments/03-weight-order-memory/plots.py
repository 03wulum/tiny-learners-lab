"""Static result plots (PNG) for the README. Colors follow the entity, fixed per arm."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = {"A": "#8a8a85", "P1": "#2a78d6", "P2": "#1baf7a", "Bq": "#eb6834", "B": "#eda100"}
NAME = {"A": "Frozen (in-context only)", "P1": "P1 sort-reorder", "P2": "P2 learned reorder",
        "Bq": "B′ bit-matched fine-tune", "B": "B full fine-tune"}


def _style(ax, title, xl, yl):
    ax.set_title(title, loc="left", fontsize=11, color="#222")
    ax.set_xlabel(xl, color="#555"); ax.set_ylabel(yl, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors="#555")


def make(per_seed, out):
    os.makedirs(out, exist_ok=True)
    # 1) accuracy vs number of examples (T1)
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=150)
    ks = [1, 2, 4, 8, 16]
    for arm in ("A", "Bq", "P1", "P2"):
        y = np.array([[r["k_sweep_T1"][arm][str(k)] if str(k) in r["k_sweep_T1"][arm] else r["k_sweep_T1"][arm][k]
                       for k in ks] for r in per_seed])
        ax.errorbar(ks, y.mean(0), yerr=y.std(0), color=COLOR[arm], lw=2, marker="o", ms=5, capsize=3,
                    label=NAME[arm])
    ax.set_xscale("log", base=2); ax.set_xticks(ks, [str(k) for k in ks]); ax.set_ylim(0, 100)
    _style(ax, "Unseen modular rules: accuracy vs worked examples", "examples in the prompt (k)", "exact match (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "k_sweep.png")); plt.close(fig)

    # 2) accuracy vs bits written (block-size ablation)
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=150)
    names = ("quarter_last", "last_layer", "all")
    for arm in ("P1", "Bq"):
        bits = [per_seed[0]["ablations"]["block_size"][n]["bits"] / 8e3 for n in names]
        y = np.array([[r["ablations"]["block_size"][n][arm] for n in names] for r in per_seed])
        ax.errorbar(bits, y.mean(0), yerr=y.std(0), color=COLOR[arm], lw=2, marker="o", ms=5, capsize=3,
                    label=NAME[arm])
    fro = np.mean([r["main"]["T1"]["A"] for r in per_seed])
    ax.axhline(fro, color=COLOR["A"], lw=1.5, ls="--", label=NAME["A"])
    ax.set_xscale("log"); ax.set_ylim(0, 100)
    _style(ax, "Same bit budget: reorder vs fine-tune (T1, k=8)", "information written into the block (KB)",
           "exact match (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "bits_curve.png")); plt.close(fig)

    # 3) same-rule stream: examples needed per problem
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=150)
    for arm in ("A", "Bq", "P1"):
        y = np.array([r["same_rule"][arm]["examples_needed"] for r in per_seed])
        ax.plot(np.arange(1, y.shape[1] + 1), y.mean(0), color=COLOR[arm], lw=2, marker="o", ms=4, label=NAME[arm])
    ax.set_ylim(0, 9.5)
    _style(ax, "Learning across a stream of problems with one unseen rule", "problem number in the stream",
           "examples needed to answer (9 = never)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "stream.png")); plt.close(fig)
