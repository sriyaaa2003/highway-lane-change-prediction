"""Lane-change event labels and the sample index.

An anchor is a (track, frame) with enough history and enough future. For each anchor we store the first
lane-change event inside the longest horizon (direction and time-to-event); a label for any shorter horizon H
is derived later as `ev_dir if ttl <= H else 0`, so all horizons share exactly the same anchors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

KEEP, CHANGE = 0, 1


def physical_events(lat_pos, lc_idx, n, w, min_disp, k):
    """Harmonise native lane-change flags into physical events (blueprint ADR-4, revised after the P0 audit).

    Lateral motion is measured on `lat_pos`, each dataset's own lane-relative coordinate, so it is independent of
    heading and of road curvature. A native flag is a *physical* event only if the vehicle's lateral position
    changes by >= min_disp metres across +-w frames (means over the first and last k frames). Its time is set to
    the frame where half of that change is covered (the lane-crossing midpoint), so highD (flag at the crossing)
    and exiD (flag about 0.7 s later) share one definition. Direction is not used: the sign of the exiD offset is
    not established, so the task is binary.
    Returns (event_times, ambiguous_native_indices).
    """
    times, amb = [], []
    for ei in lc_idx:
        a, b = max(ei - w, 0), min(ei + w, n - 1)
        if (ei - a) < w // 2 or (b - ei) < w // 2 or (b - a) < 4 * k:
            amb.append(ei)                     # too close to the track edge to measure
            continue
        lat = lat_pos[a:b + 1]
        l0 = lat[:k].mean()
        net = lat[-k:].mean() - l0
        if abs(net) < min_disp:
            amb.append(ei)                     # flag without a lane-width lateral move
            continue
        times.append(a + int(np.argmax(np.abs(lat - l0) >= 0.5 * abs(net))))
    return np.unique(np.array(times, dtype=int)), np.array(amb, dtype=int)


def build_index(df: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """df: one recording, canonical schema, sorted by (track_id, frame) with a RangeIndex."""
    fr = cfg.data.frame_rate
    hist_n = round(cfg.data.history_s * fr)
    hmax_n = cfg.hmax_frames
    excl_n = round(cfg.data.exclude_after_event_s * fr)
    w = round(cfg.data.event_window_s * fr)
    k_axis = round(cfg.data.axis_sample_s * fr)
    stride = cfg.data.sample_stride_frames

    lat_pos = df.lat_pos.to_numpy()
    lc = df.lc_flag.to_numpy()
    tid = df.track_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    ends = np.r_[starts[1:], len(df)]

    rows, ev, ttl = [], [], []
    for s, e in zip(starts, ends):
        n = e - s
        if n < hist_n + hmax_n + 1:
            continue                       # too short: no sample (censoring, see blueprint ADR-3)
        et, amb = physical_events(lat_pos[s:e], np.flatnonzero(lc[s:e]), n, w, cfg.data.min_lateral_disp_m, k_axis)
        i = np.arange(hist_n, n - hmax_n, stride)   # anchors; future of length hmax_n always exists
        if len(amb):                       # an unresolved flag nearby: neither "keep" nor an event
            lo = np.searchsorted(amb, i - excl_n, side="left")
            hi = np.searchsorted(amb, i + hmax_n, side="right")
            i = i[hi == lo]
        if len(et):
            pos_prev = np.searchsorted(et, i, side="right") - 1
            prev = np.where(pos_prev >= 0, et[np.maximum(pos_prev, 0)], -10**9)
            i = i[(i - prev) > excl_n]
        if len(i) == 0:
            continue
        d = np.zeros(len(i), dtype=np.int8)
        t = np.full(len(i), np.nan, dtype=np.float32)
        if len(et):
            nxt = np.searchsorted(et, i, side="right")          # first event strictly after the anchor
            kk = np.clip(nxt, 0, len(et) - 1)
            within = (nxt < len(et)) & ((et[kk] - i) <= hmax_n)
            d[within] = 1
            t[within] = (et[kk][within] - i[within]) / fr
        rows.append(s + i)
        ev.append(d)
        ttl.append(t)

    if not rows:
        return pd.DataFrame(columns=["row", "ev", "ttl"])
    idx = pd.DataFrame({"row": np.concatenate(rows), "ev": np.concatenate(ev), "ttl": np.concatenate(ttl)})
    if len(idx) > cfg.data.max_samples_per_recording:      # uniform subsample keeps prevalence unbiased
        idx = idx.iloc[np.sort(rng.choice(len(idx), cfg.data.max_samples_per_recording, replace=False))]
    return idx.reset_index(drop=True)


def labels_for_horizon(ev: np.ndarray, ttl: np.ndarray, horizon_s: float) -> np.ndarray:
    """1 if a lane change (crossing midpoint) occurs within horizon_s after the anchor, else 0."""
    return ((np.nan_to_num(ttl, nan=np.inf) <= horizon_s + 1e-9) & (ev > 0)).astype(np.int8)


def run(cfg: Config) -> pd.DataFrame:
    from .features import build as build_features
    cfg.paths.interim.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(cfg.paths.canonical / "manifest.csv")
    parts = []
    for r in man.itertuples():
        out = cfg.paths.interim / f"samples_{r.dataset}_{r.recording_id:02d}.parquet"
        if not out.exists():
            df = pd.read_parquet(cfg.paths.canonical / f"{r.dataset}_{r.recording_id:02d}.parquet")
            rng = np.random.default_rng([cfg.splits.seed, r.recording_id, 0 if r.dataset == "highd" else 1])
            idx = build_index(df, cfg, rng)
            feats = build_features(df, idx, cfg)
            s = pd.concat([idx.drop(columns="row"), feats], axis=1)
            meta = df.iloc[idx.row.to_numpy()][["dataset", "recording_id", "location_id", "track_id", "frame"]]
            s = pd.concat([meta.reset_index(drop=True), s], axis=1)
            s.to_parquet(out, index=False)
        else:
            s = pd.read_parquet(out)
        pos = int((s.ev > 0).sum())
        print(f"{r.dataset} {r.recording_id:02d} samples={len(s)} events_in_window={pos}", flush=True)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)
