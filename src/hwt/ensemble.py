"""Hierarchy and stacking on top of the rule, boosting and CNN.

Variants (all at the primary horizon):
  gen    boosting trained on every anchor (the baseline to beat)
  hier   two stages: cars already drifting (lat_disp_1s >= cutoff) are scored by the generalist, cars not yet moving
         sideways by a specialist trained only on such anchors; each stage is calibrated on a calibration split so the
         two stages' probabilities can be ranked together
  stack3 regime-aware logistic stack of {lateral-speed rule, boosting, CNN}
  stack4 the same plus the specialist
Everything is fitted on train / val / cal splits; test splits are only scored. The calibration split is recording-
disjoint from the test split. "source-calibrated" uses the calibration split of the training domain, "target-calibrated"
the one of the test domain (the same idea as the conformal repair experiment).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from . import cnn
from .config import Config
from .evalcnn import TESTS, Run, load_bev
from .matched import _fit, _masks, _stack, _val_masks
from .stats import Cluster, ci, holm, paired_p

SPEC_GRID = {"max_leaf_nodes": [4, 8, 16], "learning_rate": [0.03, 0.06, 0.1]}
VARIANTS = ["gen", "hier", "stack3", "stack4"]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def regime_design(feats: list[np.ndarray], pre: np.ndarray) -> np.ndarray:
    """Features plus their interaction with the pre-drift regime, so each regime gets its own blend weights."""
    F = np.column_stack(feats)
    r = pre.astype(float)[:, None]
    return np.hstack([F, F * r, r])


def hierarchy_scores(p_drift: np.ndarray, p_pre: np.ndarray, pre: np.ndarray, cal: dict) -> np.ndarray:
    """Calibrate each stage on the calibration split (isotonic) and route by regime."""
    iso_d = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal["p_drift"][~cal["pre"]], cal["y"][~cal["pre"]])
    iso_p = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal["p_pre"][cal["pre"]], cal["y"][cal["pre"]])
    return np.where(pre, iso_p.predict(p_pre), iso_d.predict(p_drift))


def tune_specialist(S, y, Sv, yv, sel: dict) -> dict:
    best, best_ap = None, -1.0
    for leaf in SPEC_GRID["max_leaf_nodes"]:
        for lr in SPEC_GRID["learning_rate"]:
            p = {**sel, "max_leaf_nodes": leaf, "learning_rate": lr}
            ap = average_precision_score(yv, _fit(S, y, p).predict_proba(Sv)[:, 1])
            if ap > best_ap:
                best, best_ap = p, ap
    return {**best, "val_ap": best_ap}


def run(cfg: Config) -> pd.DataFrame:
    bev = load_bev(cfg)
    sel = json.loads((cfg.paths.results / "hgb_selected.json").read_text())
    H0 = cfg.eval.primary_horizon_s
    hi, thr, B = list(cnn.HORIZONS).index(H0), cfg.eval.pre_drift_max_disp_1s_m, cfg.eval.paired_resamples
    out_dir = cfg.paths.interim.parent / "out"
    names = {ds: list(bev[ds]["names"]) for ds in bev}
    col = lambda ds, n: names[ds].index(n)

    configs = [("R0/R1 highD (source-calibrated)", "src", "highd", ["highd_test", "highd_new_loc"]),
               ("R2 highD->exiD, source-calibrated", "src", "highd", ["exid_seen_loc", "exid_new_loc"]),
               ("R2 highD->exiD, target-calibrated", "src", "exid", ["exid_seen_loc", "exid_new_loc"]),
               ("R3 +5% exiD, target-calibrated", f"k{cfg.eval.k_fractions[1]}", "exid", ["exid_seen_loc", "exid_new_loc"]),
               ("R4 exiD only, target-calibrated", "exid", "exid", ["exid_seen_loc", "exid_new_loc"])]
    models: dict = {}
    meta = []
    for prefix in sorted({c[1] for c in configs}):
        S, ev, ttl = _stack(bev, _masks(bev, prefix, cfg))
        y = cnn.horizon_labels(ev, ttl)[:, hi]
        # the specialist's domain is the generalist's pre-drift regime; the cutoff is the same as the evaluation subset
        dn = [d for d, _ in _masks(bev, prefix, cfg)]
        pre_tr = np.concatenate([bev[d]["S"][m][:, col(d, "lat_disp_m1s")] < thr for d, m in _masks(bev, prefix, cfg)])
        Sv, evv, ttlv = _stack(bev, _val_masks(prefix, bev))
        yv = cnn.horizon_labels(evv, ttlv)[:, hi]
        pre_v = np.concatenate([bev[d]["S"][m][:, col(d, "lat_disp_m1s")] < thr for d, m in _val_masks(prefix, bev)])
        gen = _fit(S, y, sel)
        spec_p = tune_specialist(S[pre_tr], y[pre_tr], Sv[pre_v], yv[pre_v], sel)
        spec = _fit(S[pre_tr], y[pre_tr], spec_p)
        models[prefix] = (gen, spec)
        meta.append({"training set": prefix, "specialist leaves": spec_p["max_leaf_nodes"], "specialist lr": spec_p["learning_rate"],
                     "specialist val AP (pre-drift)": round(spec_p["val_ap"], 4), "specialist train positives": int(y[pre_tr].sum()),
                     "train domains": dn})
        print("fitted", prefix, meta[-1], flush=True)
    pd.DataFrame(meta).to_csv(cfg.paths.results / "ensemble_models.csv", index=False)

    def scores(prefix, ds, split):
        gen, spec = models[prefix]
        mk = bev[ds]["split"] == split
        Sx = bev[ds]["S"][mk]
        crun = Run(cfg, prefix, bev, out_dir)
        cidx, cP = crun.test_arrays(ds, split)
        assert np.array_equal(cidx, np.flatnonzero(mk))
        y = cnn.horizon_labels(bev[ds]["ev"][mk], bev[ds]["ttl"][mk])[:, hi]
        return {"y": y, "rec": bev[ds]["rec"][mk], "rule": Sx[:, col(ds, "v_lat")], "p_gen": gen.predict_proba(Sx)[:, 1],
                    "p_spec": spec.predict_proba(Sx)[:, 1], "p_cnn": cP[:, :, hi].mean(0),
                    "pre": Sx[:, col(ds, "lat_disp_m1s")] < thr}

    rows = []
    for label, prefix, cal_ds, tests in configs:
        cal = scores(prefix, cal_ds, "cal")
        cal_pack = {"p_drift": cal["p_gen"], "p_pre": cal["p_spec"], "pre": cal["pre"], "y": cal["y"]}
        Xc3 = regime_design([logit(cal["p_gen"]), logit(cal["p_cnn"]), np.log1p(10 * cal["rule"])], cal["pre"])
        Xc4 = regime_design([logit(cal["p_gen"]), logit(cal["p_cnn"]), np.log1p(10 * cal["rule"]), logit(cal["p_spec"])], cal["pre"])
        mu3, sd3, mu4, sd4 = Xc3.mean(0), Xc3.std(0) + 1e-6, Xc4.mean(0), Xc4.std(0) + 1e-6
        st3 = LogisticRegression(C=0.3, max_iter=2000).fit((Xc3 - mu3) / sd3, cal["y"])
        st4 = LogisticRegression(C=0.3, max_iter=2000).fit((Xc4 - mu4) / sd4, cal["y"])
        for tname in tests:
            ds, split = TESTS[tname]
            t = scores(prefix, ds, split)
            X3 = regime_design([logit(t["p_gen"]), logit(t["p_cnn"]), np.log1p(10 * t["rule"])], t["pre"])
            X4 = regime_design([logit(t["p_gen"]), logit(t["p_cnn"]), np.log1p(10 * t["rule"]), logit(t["p_spec"])], t["pre"])
            sc = {"gen": t["p_gen"], "rule": t["rule"], "cnn": t["p_cnn"],
                  "hier": hierarchy_scores(t["p_gen"], t["p_spec"], t["pre"], cal_pack),
                  "stack3": st3.decision_function((X3 - mu3) / sd3), "stack4": st4.decision_function((X4 - mu4) / sd4)}
            for subset, sm in (("all", np.ones(len(t["y"]), bool)), ("pre_drift", t["pre"])):
                cl = Cluster(t["y"][sm], t["rec"][sm])
                counts = cl.draws(B, cfg.eval.bootstrap_seed)
                reps = {k: cl.replicates(v[sm], counts) for k, v in sc.items()}
                point = {k: cl.point(v[sm]) for k, v in sc.items()}
                row = {"config": label, "test": tname, "subset": subset, "positives": int(t["y"][sm].sum()),
                       "cal_positives": int(cal["y"].sum()), "cal_pre_drift_positives": int(cal["y"][cal["pre"]].sum())}
                for k in sc:
                    lo, hi_ = ci(reps[k])
                    row.update({f"ap_{k}": point[k], f"ap_{k}_lo": lo, f"ap_{k}_hi": hi_})
                for k in ("hier", "stack3", "stack4"):
                    d = reps[k] - reps["gen"]
                    lo, hi_ = ci(d)
                    row.update({f"d_{k}_gen": point[k] - point["gen"], f"d_{k}_gen_lo": lo, f"d_{k}_gen_hi": hi_, f"p_{k}_gen": paired_p(d)})
                rows.append(row)
            print(label, tname, "evaluated", flush=True)
    res = pd.DataFrame(rows)
    pcols = [f"p_{k}_gen" for k in ("hier", "stack3", "stack4")]
    flat = res[pcols].to_numpy().ravel().tolist()
    adj = np.array(holm(flat)).reshape(len(res), len(pcols))
    for i, c in enumerate(pcols):
        res[c + "_holm"] = adj[:, i]
    res.to_csv(cfg.paths.results / "ensemble.csv", index=False)
    return res
