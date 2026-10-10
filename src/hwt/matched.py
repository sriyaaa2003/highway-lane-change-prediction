"""Matched comparison on the same samples the CNN uses: lateral-speed rule, gradient boosting (tuned on the
validation split only), shuffled-label controls, and the CNN, with paired recording-cluster bootstraps.

Primary family for multiple-comparison control: every (rung, test, subset) cell at the primary horizon.
Holm adjustment is applied separately to CNN minus boosting and to boosting minus the rule.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from . import cnn
from .config import Config
from .evalcnn import TESTS, Run, load_bev, rung_table
from .stats import Cluster, ci, holm, paired_p


def _masks(bev: dict, prefix: str, cfg: Config) -> list[tuple[str, np.ndarray]]:
    hi, ex = bev["highd"], bev["exid"]
    if prefix in ("src", "shuf"):
        return [("highd", hi["split"] == "train")]
    if prefix in ("exid", "shufx"):
        return [("exid", ex["split"] == "train")]
    k = prefix[1:]
    return [("highd", hi["split"] == "train"), ("exid", ex[f"kmask_{k}"])]


def _val_masks(prefix: str, bev: dict) -> list[tuple[str, np.ndarray]]:
    ds = ["highd"] if prefix in ("src", "shuf") else ["exid"] if prefix in ("exid", "shufx") else ["highd", "exid"]
    return [(d, bev[d]["split"] == "val") for d in ds]


def _stack(bev: dict, parts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    S = np.concatenate([bev[d]["S"][m] for d, m in parts])
    ev = np.concatenate([bev[d]["ev"][m] for d, m in parts])
    ttl = np.concatenate([bev[d]["ttl"][m] for d, m in parts])
    return S, ev, ttl


def _fit(S, y, params: dict, seed: int = 0):
    m = HistGradientBoostingClassifier(max_iter=params["max_iter"], learning_rate=params["learning_rate"],
                                       max_leaf_nodes=params["max_leaf_nodes"],
                                       l2_regularization=params["l2_regularization"],
                                       class_weight=params["class_weight"], early_stopping=False, random_state=seed)
    return m.fit(S, y)


def tune_hgb(cfg: Config, bev: dict) -> dict:
    """Pick (max_leaf_nodes, learning_rate) by mean validation AP at the primary horizon over the highD-only and
    exiD-only training sets. Test splits are never read."""
    H = cfg.eval.primary_horizon_s
    base = cfg.model.hgb.model_dump()
    rows = []
    for leaf in cfg.model.hgb_grid["max_leaf_nodes"]:
        for lr in cfg.model.hgb_grid["learning_rate"]:
            p = {**base, "max_leaf_nodes": leaf, "learning_rate": lr}
            vals = []
            for prefix in ("src", "exid"):
                S, ev, ttl = _stack(bev, _masks(bev, prefix, cfg))
                Sv, evv, ttlv = _stack(bev, _val_masks(prefix, bev))
                m = _fit(S, cnn.horizon_labels(ev, ttl)[:, list(cnn.HORIZONS).index(H)], p)
                yv = cnn.horizon_labels(evv, ttlv)[:, list(cnn.HORIZONS).index(H)]
                vals.append(average_precision_score(yv, m.predict_proba(Sv)[:, 1]))
            rows.append({"max_leaf_nodes": leaf, "learning_rate": lr, "val_ap_src": vals[0], "val_ap_exid": vals[1],
                         "val_ap_mean": float(np.mean(vals))})
            print("tune hgb", rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(cfg.paths.results / "hgb_tuning.csv", index=False)
    best = t.sort_values("val_ap_mean", ascending=False).iloc[0]
    sel = {**base, "max_leaf_nodes": int(best.max_leaf_nodes), "learning_rate": float(best.learning_rate)}
    (cfg.paths.results / "hgb_selected.json").write_text(json.dumps(sel, indent=1))
    return sel


def run(cfg: Config) -> pd.DataFrame:
    bev = load_bev(cfg)
    sel_path = cfg.paths.results / "hgb_selected.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else tune_hgb(cfg, bev)
    H0, thr, B = cfg.eval.primary_horizon_s, cfg.eval.pre_drift_max_disp_1s_m, cfg.eval.bootstrap_resamples
    out_dir = cfg.paths.interim.parent / "out"

    # ---- fit boosting per training set and horizon, predict every evaluation split (cached on disk)
    cdir = cfg.paths.interim.parent / "out_hgb"
    cdir.mkdir(exist_ok=True)
    cache: dict = {}
    for prefix in ("src", *[f"k{k}" for k in cfg.eval.k_fractions], "exid", "shuf", "shufx"):
        S, ev, ttl = _stack(bev, _masks(bev, prefix, cfg))
        Y = cnn.horizon_labels(ev, ttl)
        shuf_rng = np.random.default_rng(cfg.eval.bootstrap_seed + (1 if prefix == "shuf" else 2))
        for hi, H in enumerate(cnn.HORIZONS):
            if prefix.startswith("shuf") and H != H0:
                continue
            fn = cdir / f"{prefix}_H{H:g}.npz"
            if fn.exists():
                z = np.load(fn)
                for ds in ("highd", "exid"):
                    for split in ("test", "holdout_loc"):
                        cache[(prefix, H, ds, split)] = z[f"{ds}_{split}"]
                continue
            y = shuf_rng.permutation(Y[:, hi]) if prefix.startswith("shuf") else Y[:, hi]
            m = _fit(S, y, sel)
            for ds in ("highd", "exid"):
                for split in ("test", "holdout_loc"):
                    cache[(prefix, H, ds, split)] = m.predict_proba(bev[ds]["S"][bev[ds]["split"] == split])[:, 1]
            np.savez_compressed(fn, **{f"{ds}_{split}": cache[(prefix, H, ds, split)]
                                       for ds in ("highd", "exid") for split in ("test", "holdout_loc")})
            print("fit", prefix, H, flush=True)

    # ---- evaluation cells
    rows = []
    for rung, prefix, tests in rung_table(cfg):
        crun = Run(cfg, prefix, bev, out_dir)
        for tname in tests:
            ds, split = TESTS[tname]
            mk = bev[ds]["split"] == split
            idx = np.flatnonzero(mk)
            Y = cnn.horizon_labels(bev[ds]["ev"][idx], bev[ds]["ttl"][idx])
            rec = bev[ds]["rec"][idx]
            names = list(bev[ds]["names"])
            vlat = bev[ds]["S"][idx, names.index("v_lat")]
            pre = bev[ds]["S"][idx, names.index("lat_disp_m1s")] < thr
            if crun.seeds:
                cidx, cP = crun.test_arrays(ds, split)
                assert np.array_equal(cidx, idx)
            shuf_run = Run(cfg, {"src": "shuf", "exid": "shufx"}.get(prefix, "none"), bev, out_dir)
            if shuf_run.seeds:
                sidx, sP = shuf_run.test_arrays(ds, split)
                assert np.array_equal(sidx, idx)
            for subset, sm in (("all", np.ones(len(idx), bool)), ("pre_drift", pre)):
                for hi, H in enumerate(cnn.HORIZONS):
                    y = Y[sm, hi]
                    cl = Cluster(y, rec[sm])
                    counts = cl.draws(cfg.eval.paired_resamples if H == H0 else B, cfg.eval.bootstrap_seed)
                    scores = {"rule": [vlat[sm]], "hgb": [cache[(prefix, H, ds, split)][sm]]}
                    if crun.seeds:
                        scores["cnn"] = [cP[s, sm, hi] for s in range(len(crun.seeds))]
                    if shuf_run.seeds and H == H0:
                        scores["cnn_shuf"] = [sP[s, sm, hi] for s in range(len(shuf_run.seeds))]
                    for ctl, label in (("shuf", "hgb_shuf"), ("shufx", "hgb_shuf")):
                        if prefix == {"shuf": "src", "shufx": "exid"}[ctl] and H == H0:
                            scores[label] = [cache[(ctl, H, ds, split)][sm]]
                    reps = {k: np.mean([cl.replicates(s, counts) for s in v], axis=0) for k, v in scores.items()}
                    point = {k: float(np.mean([cl.point(s) for s in v])) for k, v in scores.items()}
                    row = {"rung": rung, "test": tname, "subset": subset, "H": H, "n_test": int(sm.sum()), "positives": int(y.sum()),
                               "prevalence": float(y.mean()), "n_recordings": int(cl.n_rec)}
                    for k in scores:
                        lo, hi_ = ci(reps[k])
                        row.update({f"ap_{k}": point[k], f"ap_{k}_lo": lo, f"ap_{k}_hi": hi_})
                    for a, b in (("cnn", "hgb"), ("hgb", "rule")):
                        if a in reps and b in reps:
                            d = reps[a] - reps[b]
                            lo, hi_ = ci(d)
                            row.update({f"d_{a}_{b}": point[a] - point[b], f"d_{a}_{b}_lo": lo, f"d_{a}_{b}_hi": hi_,
                                        f"p_{a}_{b}": paired_p(d)})
                    rows.append(row)
            print(rung, tname, "evaluated", flush=True)
    res = pd.DataFrame(rows)
    for col in ("p_cnn_hgb", "p_hgb_rule"):               # Holm over the primary-horizon cells
        m = (res.H == H0) & res[col].notna()
        res.loc[m, col + "_holm"] = holm(res.loc[m, col].tolist())
    res.to_csv(cfg.paths.results / "matched.csv", index=False)
    return res
