"""Leave-one-location-out: how much does a model lose on a road it has never seen, for every location?

Within one dataset, train on all other locations (uniformly capped), evaluate on the held-out location at the primary
horizon, overall and on pre-drift anchors, with recording-cluster CIs. Uses the tuned boosting configuration.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import cnn
from .config import Config
from .evalcnn import load_bev
from .matched import _fit
from .stats import Cluster, ci

TRAIN_CAP = 200_000


def run(cfg: Config) -> pd.DataFrame:
    bev = load_bev(cfg)
    params = json.loads((cfg.paths.results / "hgb_selected.json").read_text())
    hi = list(cnn.HORIZONS).index(cfg.eval.primary_horizon_s)
    thr, B = cfg.eval.pre_drift_max_disp_1s_m, cfg.eval.bootstrap_resamples
    rng = np.random.default_rng(cfg.eval.bootstrap_seed)
    rows = []
    for ds in ("highd", "exid"):
        d = bev[ds]
        y = cnn.horizon_labels(d["ev"], d["ttl"])[:, hi]
        names = list(d["names"])
        pre = d["S"][:, names.index("lat_disp_m1s")] < thr
        for loc in np.unique(d["loc"]):
            te = d["loc"] == loc
            tr = np.flatnonzero(~te)
            if len(tr) > TRAIN_CAP:
                tr = np.sort(rng.choice(tr, TRAIN_CAP, replace=False))
            m = _fit(d["S"][tr], y[tr], params)
            p = m.predict_proba(d["S"][te])[:, 1]
            for subset, sm in (("all", np.ones(te.sum(), bool)), ("pre_drift", pre[te])):
                cl = Cluster(y[te][sm], d["rec"][te][sm])
                counts = cl.draws(B, cfg.eval.bootstrap_seed)
                row = {"dataset": ds, "held_out_location": int(loc), "subset": subset, "n_test": int(sm.sum()),
                           "positives": int(y[te][sm].sum()), "recordings": int(cl.n_rec), "prevalence": float(y[te][sm].mean())}
                for name, score in (("hgb", p[sm]), ("rule", d["S"][te][sm][:, names.index("v_lat")])):
                    lo, hi_ = ci(cl.replicates(score, counts))
                    row.update({f"ap_{name}": cl.point(score), f"ap_{name}_lo": lo, f"ap_{name}_hi": hi_})
                rows.append(row)
            print("lolo", ds, loc, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "lolo.csv", index=False)
    return res
