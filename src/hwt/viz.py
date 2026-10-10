"""Figures from the saved result tables. Computes nothing; reads results/*.csv only."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config

COL = {"rule": "#9aa0a6", "hgb": "#1f77b4", "cnn": "#d95f02"}
LABEL = {"rule": "lateral-speed rule", "hgb": "gradient boosting", "cnn": "BEV CNN"}


def _short(rung: str) -> str:
    """'R3 highD+5% exiD' -> 'R3 +5%'; other rungs keep their code."""
    code = rung.split(" ")[0]
    return f"{code} {rung.split('highD')[-1].split(' ')[0]}" if code == "R3" else code


def ladder(cfg: Config) -> None:
    d = pd.read_csv(cfg.paths.results / "matched.csv")
    d = d[d.H == cfg.eval.primary_horizon_s]
    models = [m for m in COL if f"ap_{m}" in d]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, sub, title in ((axes[0], "all", "All anchors"), (axes[1], "pre_drift", "Pre-drift anchors (not yet moving sideways)")):
        t = d[d.subset == sub]
        keys = list(dict.fromkeys(zip(t.rung, t.test)))
        x = np.arange(len(keys))
        for i, m in enumerate(models):
            ys, lo, hi = [], [], []
            for r, tb in keys:
                row = t[(t.rung == r) & (t.test == tb)].iloc[0]
                ys.append(row[f"ap_{m}"])
                lo.append(row[f"ap_{m}"] - row[f"ap_{m}_lo"])
                hi.append(row[f"ap_{m}_hi"] - row[f"ap_{m}"])
            ax.bar(x + (i - 1) * 0.27, ys, 0.27, yerr=[lo, hi], color=COL[m], label=LABEL[m], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{_short(r)}" + chr(10) + tb.replace("_", " ") for r, tb in keys], rotation=60, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("PR-AUC, lane change within 2 s (95% CI)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Shift ladder on matched samples: R0 highD, R1 new highD location, R2 highD to exiD, R3 + k% exiD, R4 exiD only")
    fig.tight_layout()
    (cfg.paths.results / "figures").mkdir(exist_ok=True)
    fig.savefig(cfg.paths.results / "figures" / "ladder.png", dpi=150)
    plt.close(fig)


def coverage(cfg: Config) -> None:
    d = pd.read_csv(cfg.paths.results / "conformal.csv")
    d = d[d.subset == "all"]
    keys = list(dict.fromkeys(zip(d.scenario, d.test)))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (col, name, c) in enumerate((("cov", "all anchors", "#1f77b4"), ("cov_pos", "actual lane changes", "#d95f02"))):
        ys = [d[(d.scenario == s) & (d.test == t)][col].iloc[0] for s, t in keys]
        lo = [y - d[(d.scenario == s) & (d.test == t)][col + "_lo"].iloc[0] for y, (s, t) in zip(ys, keys)]
        hi = [d[(d.scenario == s) & (d.test == t)][col + "_hi"].iloc[0] - y for y, (s, t) in zip(ys, keys)]
        ax.bar(np.arange(len(keys)) + (i - 0.5) * 0.38, ys, 0.38, yerr=[lo, hi], color=c, label=name, capsize=2)
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.text(len(keys) - 0.5, 0.91, "target 0.90", ha="right", fontsize=8)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([f"{s.split(':')[0]}: {s.split(': ')[-1]}\n{t.replace('_', ' ')}" for s, t in keys], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("empirical coverage (95% CI)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Conformal prediction sets (alpha = 0.1, 2 s horizon): coverage breaks under shift, recalibration repairs it", pad=14)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.paths.results / "figures" / "coverage.png", dpi=150)
    plt.close(fig)
