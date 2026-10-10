"""Figures for the README: label timing audit, leave-one-location-out, and example bird's-eye-view rasters."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config

BLUE, ORANGE, GREY = "#1f77b4", "#d95f02", "#9aa0a6"


def _events(cfg: Config, ds: str, n_rec: int = 30):
    """Per native flag on cars: lateral-position profile around the flag and whether it is a physical lane-width move."""
    fr = cfg.data.frame_rate
    w, k = round(cfg.data.event_window_s * fr), round(cfg.data.axis_sample_s * fr)
    man = pd.read_csv(cfg.paths.canonical / "manifest.csv")
    recs = man[man.dataset == ds].recording_id.to_numpy()
    recs = recs[np.linspace(0, len(recs) - 1, min(n_rec, len(recs))).astype(int)]
    offs = np.arange(-w, w + 1, 5)
    profiles, physical = [], []
    for rid in recs:
        df = pd.read_parquet(cfg.paths.canonical / f"{ds}_{rid:02d}.parquet", columns=["track_id", "lat_pos", "lc_flag", "cls"])
        df = df[df.cls.astype(str) == "car"]
        for _, g in df.groupby("track_id"):
            lat, flags = g.lat_pos.to_numpy(), np.flatnonzero(g.lc_flag.to_numpy())
            for e in flags:
                if e - w < 0 or e + w >= len(lat):
                    continue
                seg = lat[e - w:e + w + 1]
                l0, net = seg[:k].mean(), seg[-k:].mean() - seg[:k].mean()
                ok = abs(net) >= cfg.data.min_lateral_disp_m
                physical.append(ok)
                if ok:
                    profiles.append((lat[e + offs] - l0) / net)
    return offs / fr, np.array(profiles), float(np.mean(physical)), len(physical)


def timing(cfg: Config) -> None:
    """Native lane-change flags on the lane-relative coordinate: both datasets mark nearly the same moment."""
    import json
    audit = json.loads((cfg.paths.results / "label_audit.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for ds, col, name in (("highd", BLUE, "highD"), ("exid", ORANGE, "exiD")):
        t, P, _, _ = _events(cfg, ds)
        q1, med, q3 = np.quantile(P, [0.25, 0.5, 0.75], axis=0)
        ax[0].fill_between(t, q1, q3, color=col, alpha=0.18, lw=0)
        ax[0].plot(t, med, color=col, lw=2.2, label=f"{name}: median and interquartile band ({len(P)} lane changes)")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].axhline(0.5, color=GREY, ls="--", lw=1)
    ax[0].text(0.05, 0.03, "flag", fontsize=9)
    ax[0].set_xlabel("seconds relative to the dataset's own lane-change flag")
    ax[0].set_ylabel("fraction of the sideways move completed")
    ax[0].set_title("Both datasets flag a lane change at nearly the same moment")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].grid(alpha=0.3)
    names = ["highD", "exiD"]
    keys = ["highd", "exid"]
    vals = [100 * (1 - audit[k]["share_of_measurable_below_threshold"]) for k in keys]
    ax[1].bar(names, vals, color=[BLUE, ORANGE], width=0.55)
    for i, k in enumerate(keys):
        a_ = audit[k]
        ax[1].text(i, vals[i] + 1.2, f"{vals[i]:.1f}%\n(n={a_['measurable']:,}; flag is {a_['flag_minus_midpoint_s_median']:+.2f} s\nafter the midpoint)",
                   ha="center", fontsize=8.5)
    ax[1].set_ylim(0, 118)
    ax[1].set_ylabel("measurable flags that are a real lane-width move (%)")
    ax[1].set_title(f"Flags with a lateral move of at least {cfg.data.min_lateral_disp_m} m")
    ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    (cfg.paths.results / "figures").mkdir(exist_ok=True)
    fig.savefig(cfg.paths.results / "figures" / "label_timing.png", dpi=150)
    plt.close(fig)


def lolo(cfg: Config) -> None:
    d = pd.read_csv(cfg.paths.results / "lolo.csv")
    d = d[d.subset == "pre_drift"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, ds, title in ((axes[0], "highd", "highD (6 locations)"), (axes[1], "exid", "exiD (7 locations)")):
        x = d[d.dataset == ds].sort_values("held_out_location")
        pos = np.arange(len(x))
        for off, m, col, lab in ((-0.12, "rule", GREY, "lateral-speed rule"), (0.12, "hgb", BLUE, "boosting trained on the other locations")):
            y = x[f"ap_{m}"].to_numpy()
            ax.errorbar(pos + off, y, yerr=[y - x[f"ap_{m}_lo"], x[f"ap_{m}_hi"] - y], fmt="o", color=col, capsize=3, label=lab)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"loc {int(v)}\n({int(p)} pos.)" for v, p in zip(x.held_out_location, x.positives)], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("PR-AUC on cars not yet moving sideways (95% CI)")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Early-intent skill on a road the model has never seen, one location held out at a time")
    fig.tight_layout()
    fig.savefig(cfg.paths.results / "figures" / "lolo.png", dpi=150)
    plt.close(fig)


def bev_examples(cfg: Config) -> None:
    from . import cnn
    b = cfg.bev
    pre_thr = cfg.eval.pre_drift_max_disp_1s_m
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    for r, (ds, name) in enumerate((("highd", "highD"), ("exid", "exiD"))):
        z = np.load(cfg.paths.interim.parent / "bev" / f"{ds}.npz", allow_pickle=False)
        names = list(z["names"])
        test = np.flatnonzero(z["split"] == "test")
        y = cnn.horizon_labels(z["ev"][test], z["ttl"][test])[:, 1]
        d1, n_own = z["S"][test, names.index("lat_disp_m1s")], z["S"][test, names.index("n_left")] + z["S"][test, names.index("n_right")]
        rng = np.random.default_rng(3)
        picks = [
            ("keeps its lane", test[(y == 0) & (n_own >= 3)]),
            ("lane change within 2 s, not yet moving sideways", test[(y == 1) & (d1 < pre_thr) & (n_own >= 2)]),
            ("lane change within 2 s, already drifting", test[(y == 1) & (d1 >= 0.6) & (n_own >= 2)])]
        for c, (title, pool) in enumerate(picks):
            i = int(rng.choice(pool))
            X = z["X"][i]
            occ = X[0] > 0
            rel = (X[1].astype(float) - 128) / 127 * b.speed_clip_ms
            np.where(occ, rel, np.nan)
            ax = axes[r, c]
            ax.set_facecolor("#ececec")
            ys, xs = np.nonzero(occ)
            ax.scatter((xs + 0.5) * b.cell_lat_m - b.half_width_m, (ys + 0.5) * b.cell_long_m - b.behind_m, c=rel[ys, xs], cmap="coolwarm",
                       vmin=-15, vmax=15, marker="s", s=22)
            ax.scatter([0], [0], marker="^", color="k", s=70, zorder=5)
            ax.set_xlim(-b.half_width_m, b.half_width_m)
            ax.set_ylim(-b.behind_m, b.ahead_m)
            ax.set_title(f"{name}: {title}", fontsize=8.5)
            ax.set_xlabel("metres to the left (+)" if r == 1 else "")
            ax.set_ylabel("metres ahead (+)" if c == 0 else "")
            ax.grid(alpha=0.25)
    fig.suptitle("What the CNN sees: surrounding vehicles around the car (black triangle); colour = speed relative to it (red faster, blue slower)")
    fig.tight_layout()
    fig.savefig(cfg.paths.results / "figures" / "bev_examples.png", dpi=150)
    plt.close(fig)


def run(cfg: Config) -> None:
    timing(cfg)
    lolo(cfg)
    bev_examples(cfg)
