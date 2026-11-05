"""Learning-curve plots from results/metrics.json."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"A": ("#7f7f7f", "-"), "B": ("#d62728", "--"), "C": ("#1f77b4", "-"), "C-frozen": ("#1f77b4", ":"),
         "D": ("#2ca02c", "-"), "A-0.1M": ("#dddddd", "-"), "A-5M": ("#bbbbbb", "-")}
PANELS = [("rules_train", "Rules seen in training"), ("rules_heldout", "T1 new rules (held out)"),
          ("quad", "T4 new rule family"), ("facts_shown", "T2 new facts (shown keys)")]
P_OUT = 17


def copy_only(k):
    """Accuracy of a learner that only repeats answers it was shown: P(query x was in the prompt)."""
    return 1 - (1 - 1 / P_OUT) ** k


def make_all(m: dict, out_dir):
    ks = m["k_list"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, (task, title) in zip(axes, PANELS):
        ax.plot(ks, [m["ideal"][task][str(k)] for k in ks], color="black", lw=1, label="ideal learner")
        ax.axhline(1 / 17, color="black", ls=":", lw=0.8)
        if task != "facts_shown":
            ax.plot(ks, [copy_only(k) for k in ks], color="#ff7f0e", ls="--", lw=1.5, label="copy-only (no rule)")
        for arm, (col, ls) in STYLE.items():
            if arm in m["acc"] and task in m["acc"][arm]:
                ax.plot(ks, [m["acc"][arm][task][str(k)] for k in ks], color=col, ls=ls, marker="o", ms=3, label=arm)
        ax.axvline(16, color="#999", lw=0.8)
        ax.text(16.5, 0.02, "trained up to here", fontsize=7, color="#666")
        ax.set_xscale("symlog", linthresh=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("examples in the prompt (k)")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=130)
    plt.close(fig)
