"""Scalar features at an anchor, defined identically for both datasets (ego frame, SI units).

Only quantities available in both highD and exiD are used. Lane-centre offset is deliberately absent: exiD
provides it, highD does not (blueprint section 11.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

CLASS_CODE = {"car": 0, "van": 1, "truck": 2, "motorcycle": 3}


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * a))


def ref_axis(df: pd.DataFrame, r: np.ndarray, cfg: Config):
    """Forward (c, s) of the vehicle's own recent path: chord from t-ref_from_s to t-ref_to_s (causal)."""
    fr = cfg.data.frame_rate
    k0, k1 = round(cfg.features.ref_from_s * fr), round(cfg.features.ref_to_s * fr)
    x, y = df.x.to_numpy(), df.y.to_numpy()
    ang = np.arctan2(y[r - k1] - y[r - k0], x[r - k1] - x[r - k0])
    return np.cos(ang), np.sin(ang)


def build(df: pd.DataFrame, idx: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    fr = cfg.data.frame_rate
    r = idx.row.to_numpy()
    x, y, lat = df.x.to_numpy(), df.y.to_numpy(), df.lat_pos.to_numpy()
    vx, vy, ax, ay = df.vx.to_numpy(), df.vy.to_numpy(), df.ax.to_numpy(), df.ay.to_numpy()
    speed = np.hypot(vx, vy)
    va = np.arctan2(vy, vx)                     # direction of travel (defined the same way in both datasets)
    kv = max(1, round(cfg.features.lat_vel_window_s * fr))

    def lat_speed(rr):                          # |d lat_pos / dt| over the window ending at rr
        return np.abs(lat[rr] - lat[rr - kv]) / (kv / fr)

    # Rotation for the neighbour bands: direction of the vehicle's own recent path (a past chord, causal).
    k0, k1 = round(cfg.features.ref_from_s * fr), round(cfg.features.ref_to_s * fr)
    ang = np.arctan2(y[r - k1] - y[r - k0], x[r - k1] - x[r - k0])
    c, s = np.cos(ang), np.sin(ang)             # forward (c, s) and left (-s, c)
    v_lon = speed[r]
    out = {
        "v_lon": v_lon,
        "a_lon": (ax[r] * vx[r] + ay[r] * vy[r]) / np.maximum(speed[r], 0.1),
        "v_lat": lat_speed(r),
        "a_lat": (lat_speed(r) - lat_speed(r - kv)) / (kv / fr),     # rate of change of |lateral speed|
        "length": df.length.to_numpy()[r],
        "width": df.width.to_numpy()[r],
        "cls": df.cls.astype(str).map(CLASS_CODE).fillna(4).to_numpy()[r],
        "thw": df.thw.to_numpy()[r],
        "ttc": df.ttc.to_numpy()[r],
    }
    for dt in cfg.features.hist_steps_s:        # history is always available (anchors need history_s)
        k = round(dt * fr)
        rp = r - k
        tag = f"{dt:g}s"
        out[f"v_lon_m{tag}"] = speed[rp]
        out[f"v_lat_m{tag}"] = lat_speed(rp)
        out[f"lat_disp_m{tag}"] = np.abs(lat[r] - lat[rp])           # lateral move over the window (metres)
        out[f"dvel_angle_m{tag}"] = np.abs(_wrap(va[r] - va[rp]))
    feat = pd.DataFrame(out)
    feat.attrs["axis"] = (c, s)
    feat = pd.concat([feat, _neighbors(df, idx, cfg, c, s, v_lon)], axis=1)
    return feat.astype("float32")


def neighbor_rows(df: pd.DataFrame, idx: pd.DataFrame, cfg: Config, c, s, v_lon, max_dy: float | None = None):
    """Every other vehicle in the anchor's frame, expressed in the anchor's path-aligned frame.

    Columns: aid (anchor number), dx (ahead +), dy (left +), dv (longitudinal speed minus the ego speed),
    dvl / dvt (relative velocity along / across the reference axis), length, width.
    """
    lw, rng_m = cfg.features.lane_width_m, cfg.features.neighbor_range_m
    max_dy = 2.5 * lw if max_dy is None else max_dy
    r = idx.row.to_numpy()
    a = pd.DataFrame({"aid": np.arange(len(idx)), "frame": df.frame.to_numpy()[r],
                      "tid": df.track_id.to_numpy()[r], "ex": df.x.to_numpy()[r], "ey": df.y.to_numpy()[r],
                      "evx": df.vx.to_numpy()[r], "evy": df.vy.to_numpy()[r], "c": c, "s": s, "ev": v_lon})
    nb = df[["frame", "track_id", "x", "y", "vx", "vy", "length", "width"]]
    m = a.merge(nb, on="frame")
    m = m[m.track_id != m.tid]
    dxw, dyw = m.x.to_numpy() - m.ex.to_numpy(), m.y.to_numpy() - m.ey.to_numpy()
    m = m.assign(dx=dxw * m.c + dyw * m.s, dy=-dxw * m.s + dyw * m.c)
    dvx, dvy = m.vx.to_numpy() - m.evx.to_numpy(), m.vy.to_numpy() - m.evy.to_numpy()
    m = m.assign(dv=(m.vx * m.c + m.vy * m.s) - m.ev, dvl=dvx * m.c + dvy * m.s, dvt=-dvx * m.s + dvy * m.c)
    return m[(m.dx.abs() <= rng_m) & (m.dy.abs() <= max_dy)]


def _neighbors(df: pd.DataFrame, idx: pd.DataFrame, cfg: Config, c, s, v_lon) -> pd.DataFrame:
    """Nearest vehicle gaps in the own, left and right lane bands (hand-built BEV summary)."""
    lw = cfg.features.lane_width_m
    m = neighbor_rows(df, idx, cfg, c, s, v_lon)
    out = pd.DataFrame(index=np.arange(len(idx)))
    zones = {"own": (-lw / 2, lw / 2), "left": (lw / 2, 2.5 * lw), "right": (-2.5 * lw, -lw / 2)}
    for name, (lo, hi) in zones.items():
        z = m[(m.dy > lo) & (m.dy <= hi)]
        out[f"n_{name}"] = z.groupby("aid").size().reindex(out.index).fillna(0)
        for side, sub in (("front", z[z.dx > 0]), ("rear", z[z.dx <= 0])):
            g = sub.assign(adx=sub.dx.abs()).sort_values("adx").drop_duplicates("aid").set_index("aid")
            out[f"{name}_{side}_gap"] = g.adx.reindex(out.index)
            out[f"{name}_{side}_dv"] = g.dv.reindex(out.index)
    return out
