"""Deterministic group-wise splits. The unit is the recording (and location), never the frame."""
from __future__ import annotations

import hashlib

import pandas as pd

from .config import Config

HOLDOUT = "holdout_loc"


def _u(dataset: str, rid: int, seed: int) -> float:
    h = hashlib.sha1(f"{seed}:{dataset}:{rid}".encode()).hexdigest()
    return int(h[:12], 16) / 16**12


def assign(man: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """man: manifest with dataset, recording_id, location_id. Returns it with a `split` column."""
    f = cfg.splits.frac
    edges = [("train", f.train), ("val", f.val), ("cal", f.cal), ("test", f.test)]
    tot = sum(v for _, v in edges)
    hold = {"highd": set(cfg.splits.highd_holdout_locations), "exid": set(cfg.splits.exid_holdout_locations)}
    out = []
    for r in man.itertuples():
        if r.location_id in hold[r.dataset]:
            out.append(HOLDOUT)
            continue
        u, acc = _u(r.dataset, r.recording_id, cfg.splits.seed) * tot, 0.0
        for name, v in edges:
            acc += v
            if u < acc:
                out.append(name)
                break
    return man.assign(split=out)


def check_integrity(man: pd.DataFrame) -> None:
    """A recording appears in exactly one split; holdout locations never appear in train/val/cal/test."""
    if man.duplicated(["dataset", "recording_id"]).any():
        raise AssertionError("recording listed twice")
    hl = man[man.split == HOLDOUT].groupby("dataset").location_id.apply(set)
    for ds, locs in hl.items():
        leak = man[(man.dataset == ds) & (man.split != HOLDOUT) & (man.location_id.isin(locs))]
        if len(leak):
            raise AssertionError(f"held-out location leaks into {leak.split.unique()}")
