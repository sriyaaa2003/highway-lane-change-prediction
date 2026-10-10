"""Canonical scene table. Units: metres, seconds, radians. World frame is y-up (x east, y north).

highD stores y downward; the highD mapper flips y, vy and ay.

`lat_pos` is a continuous lateral coordinate along the road in metres, taken from each dataset's own lane
reference (highD: y, the roads are straight; exiD: latLaneCenterOffset with the lane-switch jumps removed).
Its SIGN IS ARBITRARY in exiD, so only magnitudes (|lat_pos differences|) may be used. `heading` must not be used
for lateral geometry: it follows the velocity in exiD but is a fixed road axis in highD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCHEMA_VERSION = 2

COLUMNS = [
    "dataset", "recording_id", "location_id", "track_id", "frame",
    "x", "y", "heading", "vx", "vy", "ax", "ay", "length", "width", "cls", "lat_pos",
    "lane_key", "lc_flag", "ttc", "thw", "lead_id",
]
KEY = ["dataset", "recording_id", "track_id", "frame"]


class SchemaError(ValueError):
    pass


def finalize(df: pd.DataFrame, speed_max: float) -> pd.DataFrame:
    """Order, type and validate. Fails loudly; never repairs."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns {missing}")
    df = df[COLUMNS].sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)
    if df.duplicated(KEY).any():
        bad = df.loc[df.duplicated(KEY), "track_id"].unique()[:5]
        raise SchemaError(f"duplicate (track, frame) keys, e.g. tracks {bad}")
    gap = df.groupby("track_id").frame.diff().fillna(1)
    if (gap != 1).any():
        bad = df.loc[gap != 1, "track_id"].unique()[:5]
        raise SchemaError(f"non-contiguous frames, e.g. tracks {bad}")
    if not np.isfinite(df[["x", "y", "heading", "vx", "vy", "lat_pos"]].to_numpy()).all():
        raise SchemaError("non-finite position/heading/velocity")
    if (np.hypot(df.vx, df.vy) > speed_max).any():
        raise SchemaError(f"speed above sanity bound {speed_max} m/s")
    if (df.heading.abs() > np.pi + 1e-6).any():
        raise SchemaError("heading outside [-pi, pi]")
    f32 = ["x", "y", "heading", "vx", "vy", "ax", "ay", "length", "width", "ttc", "thw", "lat_pos"]
    df[f32] = df[f32].astype("float32")
    for c in ["recording_id", "location_id", "track_id", "frame", "lane_key", "lead_id"]:
        df[c] = df[c].astype("int32")
    df["lc_flag"] = df["lc_flag"].astype(bool)
    df["dataset"] = df["dataset"].astype("category")
    df["cls"] = df["cls"].astype("category")
    return df
