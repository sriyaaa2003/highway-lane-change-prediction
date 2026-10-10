"""Audit of the native lane-change flags on the lane-relative coordinate, over every recording.

For each flag with a measurable +-event_window_s window: is it a real lane-width move, and when is its crossing midpoint relative
to the flag? Replaces the earlier chord-axis audit, which is unreliable on curved roads.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config


def run(cfg: Config) -> dict:
    fr = cfg.data.frame_rate
    w, k = round(cfg.data.event_window_s * fr), round(cfg.data.axis_sample_s * fr)
    man = pd.read_csv(cfg.paths.canonical / "manifest.csv")
    out: dict = {}
    for ds in ("highd", "exid"):
        n_flags = n_edge = n_short = 0
        mid_offsets = []
        for rid in man[man.dataset == ds].recording_id:
            df = pd.read_parquet(cfg.paths.canonical / f"{ds}_{rid:02d}.parquet", columns=["track_id", "lat_pos", "lc_flag", "cls"])
            df = df[df.cls.astype(str) == "car"]
            for _, g in df.groupby("track_id"):
                lat = g.lat_pos.to_numpy()
                for e in np.flatnonzero(g.lc_flag.to_numpy()):
                    n_flags += 1
                    a, b = max(e - w, 0), min(e + w, len(lat) - 1)
                    if (e - a) < w // 2 or (b - e) < w // 2 or (b - a) < 4 * k:
                        n_edge += 1
                        continue
                    seg = lat[a:b + 1]
                    l0 = seg[:k].mean()
                    net = seg[-k:].mean() - l0
                    if abs(net) < cfg.data.min_lateral_disp_m:
                        n_short += 1
                        continue
                    mid = int(np.argmax(np.abs(seg - l0) >= 0.5 * abs(net)))
                    mid_offsets.append((e - a - mid) / fr)       # seconds the flag comes AFTER the crossing midpoint
        m = np.array(mid_offsets)
        measurable = n_flags - n_edge
        out[ds] = {"cars_flags": n_flags, "too_close_to_track_edge": n_edge, "measurable": measurable,
                   "share_of_measurable_below_threshold": round(n_short / max(measurable, 1), 4),
                   "flag_minus_midpoint_s_median": round(float(np.median(m)), 3),
                   "flag_minus_midpoint_s_iqr": [round(float(np.quantile(m, .25)), 3), round(float(np.quantile(m, .75)), 3)]}
        print(ds, out[ds], flush=True)
    (cfg.paths.results / "label_audit.json").write_text(json.dumps(out, indent=1))
    return out
