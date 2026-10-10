"""How much do the conclusions depend on the label definition and the pre-drift cutoff?

The lane-change threshold (min_lateral_disp_m), the measuring window (event_window_s) and the pre-drift cutoff were
set after inspecting the data, so each is varied here. Samples are rebuilt per variant (same recordings, same caps, same
splits) and boosting with the tuned configuration is re-evaluated on three rungs at the primary horizon.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import labels
from .baselines import load_samples
from .bev import fill_scalars
from .config import Config
from .labels import labels_for_horizon
from .matched import _fit
from .stats import Cluster, ci

VARIANTS = {
    "base": {},
    "disp 1.2 m": {"min_lateral_disp_m": 1.2},
    "disp 2.4 m": {"min_lateral_disp_m": 2.4},
    "window 3 s": {"event_window_s": 3.0},
    "window 5 s": {"event_window_s": 5.0},
}
CUTOFFS = [0.15, 0.3, 0.6]       # pre-drift cutoffs, evaluated on the base samples (evaluation-only parameter)


def _variant_cfg(cfg: Config, name: str, over: dict) -> Config:
    interim = cfg.paths.interim if name == "base" else cfg.paths.interim.parent / f"interim_sens_{name.replace(' ', '_')}"
    return cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"interim": interim}),
                                  "data": cfg.data.model_copy(update=over)})


def _evaluate(s: pd.DataFrame, cfg: Config, params: dict, cutoff: float, tag: dict) -> list[dict]:
    H, B = cfg.eval.primary_horizon_s, cfg.eval.bootstrap_resamples
    cols = [c for c in s.columns if c not in {"dataset", "recording_id", "location_id", "track_id", "frame", "ev", "ttl",
                                              "split", "rec", "trk"}]
    from .splits import HOLDOUT
    s = pd.concat([s.drop(columns=cols), fill_scalars(s[cols], cfg)], axis=1)     # same missing-value encoding as the CNN data
    hi, ex = s[s.dataset == "highd"], s[s.dataset == "exid"]
    models = {"src": _fit(hi[hi.split == "train"][cols], labels_for_horizon(hi[hi.split == "train"].ev.to_numpy(),
              hi[hi.split == "train"].ttl.to_numpy(), H), params),
              "exid": _fit(ex[ex.split == "train"][cols], labels_for_horizon(ex[ex.split == "train"].ev.to_numpy(),
              ex[ex.split == "train"].ttl.to_numpy(), H), params)}
    tests = [("R0 highD->highD (test)", "src", hi[hi.split == "test"]),
             ("R2 highD->exiD seen loc", "src", ex[ex.split == "test"]),
             ("R2 highD->exiD new loc", "src", ex[ex.split == HOLDOUT]),
             ("R4 exiD->exiD seen loc", "exid", ex[ex.split == "test"]),
             ("R4 exiD->exiD new loc", "exid", ex[ex.split == HOLDOUT])]
    rows = []
    for name, mk, te in tests:
        p = models[mk].predict_proba(te[cols])[:, 1]
        y = labels_for_horizon(te.ev.to_numpy(), te.ttl.to_numpy(), H)
        pre = te.lat_disp_m1s.to_numpy() < cutoff
        for subset, sm in (("all", np.ones(len(te), bool)), ("pre_drift", pre)):
            cl = Cluster(y[sm], te.rec.to_numpy()[sm])
            counts = cl.draws(B, cfg.eval.bootstrap_seed)
            row = dict(**tag, rung=name, subset=subset, n_test=int(sm.sum()), positives=int(y[sm].sum()))
            for mname, score in (("hgb", p[sm]), ("rule", te.v_lat.to_numpy()[sm])):
                lo, hi_ = ci(cl.replicates(score, counts))
                row.update({f"ap_{mname}": cl.point(score), f"ap_{mname}_lo": lo, f"ap_{mname}_hi": hi_})
            rows.append(row)
    return rows


def run(cfg: Config) -> pd.DataFrame:
    params = json.loads((cfg.paths.results / "hgb_selected.json").read_text())
    rows = []
    for name, over in VARIANTS.items():
        vc = _variant_cfg(cfg, name, over)
        if name != "base" and not (vc.paths.interim / "samples_highd_01.parquet").exists():
            labels.run(vc)
        s = load_samples(vc)
        rows += _evaluate(s, vc, params, cfg.eval.pre_drift_max_disp_1s_m, {"variant": name, "pre_drift_cutoff": cfg.eval.pre_drift_max_disp_1s_m})
        print("sensitivity", name, "done", flush=True)
        if name == "base":
            for cutoff in CUTOFFS:
                if cutoff != cfg.eval.pre_drift_max_disp_1s_m:
                    rows += [r for r in _evaluate(s, vc, params, cutoff, {"variant": f"cutoff {cutoff} m", "pre_drift_cutoff": cutoff})
                             if r["subset"] == "pre_drift"]
    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "sensitivity.csv", index=False)
    return res
