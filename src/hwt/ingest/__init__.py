from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config
from . import exid, highd
from .schema import SCHEMA_VERSION


def recording_ids(root: Path, suffix: str) -> list[int]:
    return sorted(int(p.name[:2]) for p in root.glob(f"[0-9][0-9]_{suffix}"))


def run(cfg: Config, datasets: tuple[str, ...] = ("highd", "exid")) -> pd.DataFrame:
    """Write one parquet per recording and return the recording manifest."""
    cfg.paths.canonical.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        root = cfg.paths.raw / ds
        ids = recording_ids(root, "tracks.csv")
        for rid in ids:
            out = cfg.paths.canonical / f"{ds}_{rid:02d}.parquet"
            if out.exists():
                df = pd.read_parquet(out, columns=["location_id", "track_id", "lc_flag"])
            else:
                df = (highd.read_recording(root, rid, cfg.data.speed_sanity_max_ms) if ds == "highd" else
                      exid.read_recording(root, rid, cfg.data.speed_sanity_max_ms, cfg.data.exid_heading_unit))
                df.attrs["schema_version"] = SCHEMA_VERSION
                df.to_parquet(out, index=False)
            rows.append({"dataset": ds, "recording_id": rid, "location_id": int(df.location_id.iloc[0]),
                             "n_tracks": int(df.track_id.nunique()), "n_events": int(df.lc_flag.sum())})
            print(f"{ds} {rid:02d} ok  tracks={rows[-1]['n_tracks']} events={rows[-1]['n_events']}", flush=True)
    man = pd.DataFrame(rows)
    man.to_csv(cfg.paths.canonical / "manifest.csv", index=False)
    (cfg.paths.canonical / "schema.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION}))
    return man
