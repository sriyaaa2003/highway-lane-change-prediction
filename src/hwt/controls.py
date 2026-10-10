"""Null distribution of the CNN's score: networks trained on shuffled labels, several seeds, evaluated on real labels.

A model trained on labels that carry no information should score about the prevalence on average; the spread across seeds
shows how far an untrained network can drift from it by chance alone. A mean well above the prevalence would mean leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import cnn
from .config import Config
from .evalcnn import TESTS, load_bev


def run(cfg: Config) -> pd.DataFrame:
    out_dir = cfg.paths.interim.parent / "out"
    bev = load_bev(cfg)
    thr, hi = cfg.eval.pre_drift_max_disp_1s_m, list(cnn.HORIZONS).index(cfg.eval.primary_horizon_s)
    rows = []
    for tag, tests in (("shuf", ["highd_test", "highd_new_loc", "exid_seen_loc", "exid_new_loc"]),
                       ("shufx", ["exid_seen_loc", "exid_new_loc"])):
        files = sorted(out_dir.glob(f"{tag}_s*.npz"))
        for tname in tests:
            ds, split = TESTS[tname]
            b = bev[ds]
            names = list(b["names"])
            for subset in ("all", "pre_drift"):
                aps, prev = [], None
                for f in files:
                    z = np.load(f)
                    idx = z[f"{ds}_idx"]
                    keep = b["split"][idx] == split
                    rows_i = idx[keep]
                    y = cnn.horizon_labels(b["ev"][rows_i], b["ttl"][rows_i])[:, hi]
                    sm = np.ones(len(rows_i), bool) if subset == "all" else b["S"][rows_i, names.index("lat_disp_m1s")] < thr
                    prev = float(y[sm].mean())
                    aps.append(average_precision_score(y[sm], z[f"{ds}_P"][keep][sm][:, hi]) if y[sm].any() else np.nan)
                rows.append({"trained_on": {"shuf": "highD", "shufx": "exiD"}[tag], "test": tname, "subset": subset, "seeds": len(files),
                                 "prevalence": prev, "ap_mean": float(np.nanmean(aps)), "ap_min": float(np.nanmin(aps)),
                                 "ap_max": float(np.nanmax(aps)), "per_seed": [round(a, 3) for a in aps]})
    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "controls.csv", index=False)
    return res
