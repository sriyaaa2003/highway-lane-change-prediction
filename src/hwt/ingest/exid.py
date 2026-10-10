"""exiD -> canonical. Only column mapping lives here."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import SchemaError, finalize


def _heading_rad(h: np.ndarray, vx: np.ndarray, vy: np.ndarray, unit: str) -> np.ndarray:
    if unit == "auto":
        # decide by agreement with the velocity direction on moving rows
        mv = np.hypot(vx, vy) > 5.0
        ang = np.arctan2(vy[mv], vx[mv])
        err = {}
        for u, f in (("rad", 1.0), ("deg", np.pi / 180.0)):
            d = np.angle(np.exp(1j * (h[mv] * f - ang)))
            err[u] = np.median(np.abs(d))
        unit = min(err, key=err.get)
        if err[unit] > 0.3:
            raise SchemaError(f"exiD heading agrees with velocity in neither rad nor deg: {err}")
    out = h * (np.pi / 180.0 if unit == "deg" else 1.0)
    return np.angle(np.exp(1j * out))


def unwrap_lat_offset(track_id: pd.Series, off: pd.Series, lane_width: pd.Series) -> pd.Series:
    """Continuous lateral coordinate from latLaneCenterOffset.

    The offset is measured to the centre of the *current* lane, so it jumps by about one lane width whenever the
    vehicle is assigned to a neighbouring lane. A step larger than half a lane width in one frame (0.04 s) is not
    physical, so it is replaced by the previous step before integrating. Sign convention of the offset is unknown.
    """
    g = off.groupby(track_id)
    d = g.diff()
    half = 0.5 * lane_width.fillna(lane_width.median())
    jump = d.abs() > half
    prev = d.groupby(track_id).shift(1).fillna(0.0)
    d_corr = np.where(jump, prev, d.fillna(0.0))
    return pd.Series(d_corr, index=off.index).groupby(track_id).cumsum() + g.transform("first")


def read_recording(root: Path, rid: int, speed_max: float, heading_unit: str) -> pd.DataFrame:
    p = root / f"{rid:02d}"
    tm = pd.read_csv(f"{p}_tracksMeta.csv")[["trackId", "class"]]
    rm = pd.read_csv(f"{p}_recordingMeta.csv").iloc[0]
    t = pd.read_csv(f"{p}_tracks.csv", low_memory=False).merge(tm, on="trackId", how="left")
    t = t.sort_values(["trackId", "frame"], kind="stable").reset_index(drop=True)

    out = pd.DataFrame({
        "dataset": "exid",
        "recording_id": rid,
        "location_id": int(rm["locationId"]),
        "track_id": t.trackId,
        "frame": t.frame,
        "x": t.xCenter,
        "y": t.yCenter,
        "heading": _heading_rad(t.heading.to_numpy(), t.xVelocity.to_numpy(), t.yVelocity.to_numpy(),
                                heading_unit),
        "vx": t.xVelocity,
        "vy": t.yVelocity,
        "ax": t.xAcceleration,
        "ay": t.yAcceleration,
        "length": t.length,
        "width": t.width,
        "cls": t["class"].str.lower(),
        "lat_pos": unwrap_lat_offset(t.trackId, t.latLaneCenterOffset, t.laneWidth),
        "lane_key": t.laneletId.fillna(-1),   # -1: off-lanelet rows; lane_key is not used for exiD events
        "lc_flag": t.laneChange.fillna(0).astype(int).eq(1),
        "ttc": t.leadTTC.where(np.isfinite(t.leadTTC) & (t.leadTTC > 0)),
        "thw": t.leadTHW.where(np.isfinite(t.leadTHW) & (t.leadTHW > 0)),
        "lead_id": t.leadId.fillna(-1),
    })
    return finalize(out, speed_max)
