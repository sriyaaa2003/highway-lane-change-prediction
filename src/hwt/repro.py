"""Reproducibility check: re-derive saved artefacts from the raw files and compare.

Checks (written to results/repro.json): ingest of sample recordings equals the stored parquet; labels and features of the
same recordings equal the stored interim tables; BEV rasters equal the stored array; boosting and CPU CNN training with a
fixed seed reproduce themselves. GPU runs are not bit-reproducible, so their seed-to-seed spread is reported in cnn.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import cnn
from .bev import build_recording
from .config import Config
from .features import build as build_features
from .ingest import exid, highd
from .labels import build_index
from .matched import _fit


def run(cfg: Config) -> dict:
    out: dict = {}
    for ds, rid in (("highd", 7), ("highd", 41), ("exid", 12), ("exid", 70)):
        root = cfg.paths.raw / ds
        new = (highd.read_recording(root, rid, cfg.data.speed_sanity_max_ms) if ds == "highd" else
               exid.read_recording(root, rid, cfg.data.speed_sanity_max_ms, cfg.data.exid_heading_unit))
        old = pd.read_parquet(cfg.paths.canonical / f"{ds}_{rid:02d}.parquet")
        key = f"{ds}_{rid}"
        out[key] = {"ingest_equal": bool(new.reset_index(drop=True).equals(old.reset_index(drop=True)))}
        rng = np.random.default_rng([cfg.splits.seed, rid, 0 if ds == "highd" else 1])
        idx = build_index(old, cfg, rng)
        feats = build_features(old, idx, cfg)
        stored = pd.read_parquet(cfg.paths.interim / f"samples_{ds}_{rid:02d}.parquet")
        cols = [c for c in feats.columns if c in stored.columns]
        out[key]["labels_equal"] = bool(np.array_equal(idx.ev.to_numpy(), stored.ev.to_numpy())
                                        and np.allclose(np.nan_to_num(idx.ttl.to_numpy()), np.nan_to_num(stored.ttl.to_numpy())))
        out[key]["features_max_abs_diff"] = float(np.nanmax(np.abs(feats[cols].to_numpy() - stored[cols].to_numpy())))
        smp = stored.reset_index(drop=True)
        X = build_recording(old, smp, cfg)
        z = np.load(cfg.paths.interim.parent / "bev" / f"{ds}.npz", allow_pickle=False)
        m = np.flatnonzero(z["rec"] == rid)
        out[key]["bev_equal"] = bool(np.array_equal(X, z["X"][m]))
        print("repro", key, out[key], flush=True)

    # model determinism on a fixed subset
    z = np.load(cfg.paths.interim.parent / "bev" / "highd.npz", allow_pickle=False)
    tr = np.flatnonzero(z["split"] == "train")[:20000]
    y = cnn.horizon_labels(z["ev"][tr], z["ttl"][tr])[:, 1]
    p = json.loads((cfg.paths.results / "hgb_selected.json").read_text())
    a, b = _fit(z["S"][tr], y, p).predict_proba(z["S"][:5000]), _fit(z["S"][tr], y, p).predict_proba(z["S"][:5000])
    out["hgb_deterministic_max_abs_diff"] = float(np.abs(a - b).max())
    cc = {**cfg.cnn.model_dump(), "epochs": 2}
    runs = []
    for _ in range(2):
        mdl, _ = cnn.train(z["X"][tr[:4000]], ((z["S"][tr[:4000]] - z["S"][tr].mean(0)) / (z["S"][tr].std(0) + 1e-6)).astype(np.float32),
                           cnn.horizon_labels(z["ev"][tr[:4000]], z["ttl"][tr[:4000]]),
                           z["X"][tr[4000:6000]], ((z["S"][tr[4000:6000]] - z["S"][tr].mean(0)) / (z["S"][tr].std(0) + 1e-6)).astype(np.float32),
                           cnn.horizon_labels(z["ev"][tr[4000:6000]], z["ttl"][tr[4000:6000]]), cc, 0, "cpu", log=lambda *_: None)
        runs.append(cnn.predict(mdl, z["X"][tr[6000:7000]], ((z["S"][tr[6000:7000]] - z["S"][tr].mean(0)) / (z["S"][tr].std(0) + 1e-6)).astype(np.float32), "cpu")[0])
    out["cnn_cpu_deterministic_max_abs_diff"] = float(np.abs(runs[0] - runs[1]).max())
    (cfg.paths.results / "repro.json").write_text(json.dumps(out, indent=1))
    return out


