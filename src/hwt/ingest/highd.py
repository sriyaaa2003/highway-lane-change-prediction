"""highD -> canonical. Only column mapping lives here."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import finalize


def read_recording(root: Path, rid: int, speed_max: float) -> pd.DataFrame:
    p = root / f"{rid:02d}"
    tm = pd.read_csv(f"{p}_tracksMeta.csv")[["id", "drivingDirection", "class"]]
    rm = pd.read_csv(f"{p}_recordingMeta.csv").iloc[0]
    t = pd.read_csv(f"{p}_tracks.csv", low_memory=False).merge(tm, on="id", how="left")
    t = t.sort_values(["id", "frame"], kind="stable").reset_index(drop=True)

    # drivingDirection 1: upper carriageway, moves toward -x; 2: lower, toward +x (checked in the P0 audit).
    heading = np.where(t.drivingDirection == 1, np.pi, 0.0)
    lane_changed = t.groupby("id").laneId.diff().fillna(0).ne(0)

    out = pd.DataFrame({
        "dataset": "highd",
        "recording_id": rid,
        "location_id": int(rm["locationId"]),
        "track_id": t.id,
        "frame": t.frame,
        # highD x,y is the top-left corner and y points down: centre the box, then flip y to y-up.
        "x": t.x + t.width / 2.0,
        "y": -(t.y + t.height / 2.0),
        "heading": heading,
        "vx": t.xVelocity,
        "vy": -t.yVelocity,
        "ax": t.xAcceleration,
        "ay": -t.yAcceleration,
        "length": t.width,   # highD 'width' is the box extent along x, i.e. vehicle length
        "width": t.height,
        "cls": t["class"].str.lower(),
        "lat_pos": -(t.y + t.height / 2.0),   # straight road: y is the lateral coordinate
        "lane_key": t.laneId,
        "lc_flag": lane_changed,
        "ttc": t.ttc.where(t.ttc > 0),   # 0 means "no leader" in highD
        "thw": t.thw.where(t.thw > 0),
        "lead_id": t.precedingId.replace(0, -1),
    })
    return finalize(out, speed_max)
