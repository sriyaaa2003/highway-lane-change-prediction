"""A broader family of ensembles and hierarchies, with selection on the calibration split only.

Members (all trained on the training split of the rung): boosting on every anchor (gen), specialists trained only on
pre-drift anchors at three cutoffs, ExtraTrees, logistic regression on the scalars, a residual specialist whose input
includes the generalist's out-of-fold score, and the CNN (3-seed mean) and the one-feature rule.
Combiners are fitted on the calibration split. For each configuration the variant with the best cross-validated
calibration AP (recording-grouped folds, mean of all-anchor and pre-drift AP) is selected; only that variant is then
tested against plain boosting (paired bootstrap, Holm over cells). All variants' test scores are reported descriptively.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from . import cnn
from .config import Config
from .ensemble import hierarchy_scores, logit, regime_design
from .evalcnn import TESTS, Run, load_bev
from .matched import _fit, _masks, _val_masks
from .stats import Cluster, ci, holm, paired_p

CUTS = [0.15, 0.3, 0.6]
SPEC2 = {"max_leaf_nodes": [8, 16, 32], "learning_rate": [0.03, 0.06, 0.1]}


def _rows(bev, parts):
    """Concatenate (S, labels, group key) over (dataset, mask) parts; group = recording across datasets."""
    S = np.concatenate([bev[d]["S"][m] for d, m in parts])
    ev = np.concatenate([bev[d]["ev"][m] for d, m in parts])
    ttl = np.concatenate([bev[d]["ttl"][m] for d, m in parts])
    g = np.concatenate([(0 if d == "highd" else 1000) + bev[d]["rec"][m] for d, m in parts])
    return S, ev, ttl, g


def fit_members(cfg: Config, bev: dict, prefix: str, sel: dict) -> dict:
    """Predict cal/test/holdout of both datasets with every trained member; cached on disk."""
    fn = cfg.paths.interim.parent / "out_ens" / f"{prefix}.npz"
    fn.parent.mkdir(exist_ok=True)
    if fn.exists():
        z = np.load(fn)
        return {k: z[k] for k in z.files}
    hi = list(cnn.HORIZONS).index(cfg.eval.primary_horizon_s)
    names = list(bev["highd"]["names"])
    d1 = names.index("lat_disp_m1s")
    S, ev, ttl, g = _rows(bev, _masks(bev, prefix, cfg))
    y = cnn.horizon_labels(ev, ttl)[:, hi]
    Sv, evv, ttlv, _ = _rows(bev, _val_masks(prefix, bev))
    yv = cnn.horizon_labels(evv, ttlv)[:, hi]
    gen = _fit(S, y, sel)
    members = {"gen": gen}
    # specialists at three cutoffs (small validation grid each)
    for c in CUTS:
        m, mv = S[:, d1] < c, Sv[:, d1] < c
        best, best_ap = None, -1.0
        for leaf in SPEC2["max_leaf_nodes"]:
            for lr in SPEC2["learning_rate"]:
                p = {**sel, "max_leaf_nodes": leaf, "learning_rate": lr}
                ap = average_precision_score(yv[mv], _fit(S[m], y[m], p).predict_proba(Sv[mv])[:, 1])
                if ap > best_ap:
                    best, best_ap = p, ap
        members[f"spec{c}"] = _fit(S[m], y[m], best)
        print(prefix, "specialist", c, "val AP", round(best_ap, 3), "positives", int(y[m].sum()), flush=True)
    et = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample",
                              n_jobs=-1, random_state=0).fit(S, y)
    mu, sd = S.mean(0), S.std(0) + 1e-6
    lr = LogisticRegression(C=1.0, max_iter=2000).fit((S - mu) / sd, y)
    # residual specialist: pre-drift anchors, input = scalars + the generalist's OUT-OF-FOLD logit
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(S, y, g):
        oof[te] = logit(_fit(S[tr], y[tr], sel).predict_proba(S[te])[:, 1])
    m = S[:, d1] < 0.3
    res = _fit(np.column_stack([S[m], oof[m]]), y[m], {**sel, "max_leaf_nodes": 16, "learning_rate": 0.06})
    out = {}
    for ds in ("highd", "exid"):
        for split in ("cal", "test", "holdout_loc"):
            mk = bev[ds]["split"] == split
            X = bev[ds]["S"][mk]
            pg = gen.predict_proba(X)[:, 1]
            out[f"{ds}_{split}_gen"] = pg
            for c in CUTS:
                out[f"{ds}_{split}_spec{c}"] = members[f"spec{c}"].predict_proba(X)[:, 1]
            out[f"{ds}_{split}_et"] = et.predict_proba(X)[:, 1]
            out[f"{ds}_{split}_lr"] = lr.predict_proba((X - mu) / sd)[:, 1]
            out[f"{ds}_{split}_res"] = res.predict_proba(np.column_stack([X, logit(pg)]))[:, 1]
    np.savez_compressed(fn, **out)
    return out


# ---------------------------------------------------------------- combiners: fit(C) -> predict(T)
def _rank(v):
    return np.argsort(np.argsort(v, kind="stable"), kind="stable") / max(len(v) - 1, 1)


def _flat(D):
    return np.column_stack([logit(D["gen"]), logit(D["cnn"]), np.log1p(10 * D["rule"])])


def _wide(D):
    return [logit(D["gen"]), logit(D["cnn"]), np.log1p(10 * D["rule"]), logit(D["spec0.3"]), logit(D["et"]), logit(D["lr"]), logit(D["res"])]


def _lr(X, y, c=0.3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=c, max_iter=3000).fit((X - mu) / sd, y)
    return lambda X2: m.decision_function((X2 - mu) / sd)


def v_gen(C):
    return lambda T: T["gen"]


def v_avg(C):
    return lambda T: _rank(T["gen"]) + _rank(T["cnn"]) + _rank(T["rule"])


def v_wavg(C):
    best, bw = -1.0, (1, 1, 1)
    r = [_rank(C["gen"]), _rank(C["cnn"]), _rank(C["rule"])]
    for a in np.arange(0, 1.01, 0.1):
        for b in np.arange(0, 1.01 - a, 0.1):
            ap = average_precision_score(C["y"], a * r[0] + b * r[1] + (1 - a - b) * r[2])
            if ap > best:
                best, bw = ap, (a, b, 1 - a - b)
    return lambda T: bw[0] * _rank(T["gen"]) + bw[1] * _rank(T["cnn"]) + bw[2] * _rank(T["rule"])


def v_lr_flat(C):
    f = _lr(_flat(C), C["y"])
    return lambda T: f(_flat(T))


def v_lr_regime(C):
    f = _lr(regime_design([logit(C["gen"]), logit(C["cnn"]), np.log1p(10 * C["rule"])], C["pre"]), C["y"])
    return lambda T: f(regime_design([logit(T["gen"]), logit(T["cnn"]), np.log1p(10 * T["rule"])], T["pre"]))


def v_lr_split(C):
    fp = _lr(_flat(C)[C["pre"]], C["y"][C["pre"]])
    fd = _lr(_flat(C)[~C["pre"]], C["y"][~C["pre"]])
    return lambda T: np.where(T["pre"], fp(_flat(T)), fd(_flat(T)))


def v_gbm(C):
    X = np.column_stack([_flat(C), C["pre"]])
    m = HistGradientBoostingClassifier(max_leaf_nodes=4, max_iter=100, learning_rate=0.05, random_state=0).fit(X, C["y"])
    return lambda T: m.predict_proba(np.column_stack([_flat(T), T["pre"]]))[:, 1]


def v_lr_wide(C):
    f = _lr(regime_design(_wide(C), C["pre"]), C["y"], 0.1)
    return lambda T: f(regime_design(_wide(T), T["pre"]))


def v_hier(key, cut):
    def make(C):
        pre_c = C["d1"] < cut
        cal = {"p_drift": C["gen"], "p_pre": C[key], "pre": pre_c, "y": C["y"]}
        return lambda T: hierarchy_scores(T["gen"], T[key], T["d1"] < cut, cal)
    return make


def v_soft(C):
    iso_g = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(C["gen"], C["y"])
    pre = C["d1"] < 0.3
    iso_s = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(C["spec0.3"][pre], C["y"][pre])
    best, bp = -1.0, (0.3, 20.0)
    for c in (0.15, 0.3, 0.6):
        for beta in (5.0, 10.0, 20.0, 40.0):
            g = 1 / (1 + np.exp(-beta * (c - C["d1"])))
            ap = average_precision_score(C["y"], (1 - g) * iso_g.predict(C["gen"]) + g * iso_s.predict(C["spec0.3"]))
            if ap > best:
                best, bp = ap, (c, beta)
    c, beta = bp

    def predict(T):
        g = 1 / (1 + np.exp(-beta * (c - T["d1"])))
        return (1 - g) * iso_g.predict(T["gen"]) + g * iso_s.predict(T["spec0.3"])
    return predict


VARIANTS = {"gen": v_gen, "avg": v_avg, "wavg": v_wavg, "lr_flat": v_lr_flat, "lr_regime": v_lr_regime, "lr_split": v_lr_split,
            "gbm_stack": v_gbm, "lr_wide": v_lr_wide, "soft_gate": v_soft,
            "hier_0.15": v_hier("spec0.15", 0.15), "hier_0.3": v_hier("spec0.3", 0.3), "hier_0.6": v_hier("spec0.6", 0.6),
            "cascade_res": v_hier("res", 0.3)}


def _sub(D, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in D.items()}


def cal_cv_score(make, C, thr: float) -> float:
    """Mean of all-anchor and pre-drift AP of out-of-fold calibration scores (recording-grouped folds)."""
    g = pd.factorize(C["rec"])[0]
    oof = np.zeros(len(C["y"]))
    for tr, te in GroupKFold(min(5, len(np.unique(g)))).split(C["y"], C["y"], g):
        oof[te] = make(_sub(C, tr))(_sub(C, te))
    pre = C["d1"] < thr
    a = average_precision_score(C["y"], oof)
    b = average_precision_score(C["y"][pre], oof[pre]) if C["y"][pre].any() else a
    return float((a + b) / 2)


def run(cfg: Config) -> pd.DataFrame:
    bev = load_bev(cfg)
    sel = json.loads((cfg.paths.results / "hgb_selected.json").read_text())
    H0, thr, B = cfg.eval.primary_horizon_s, cfg.eval.pre_drift_max_disp_1s_m, cfg.eval.paired_resamples
    hi = list(cnn.HORIZONS).index(H0)
    names = {ds: list(bev[ds]["names"]) for ds in bev}
    out_dir = cfg.paths.interim.parent / "out"
    configs = [("R0/R1 highD (source-cal)", "src", "highd", ["highd_test", "highd_new_loc"]),
               ("R2 highD->exiD (source-cal)", "src", "highd", ["exid_seen_loc", "exid_new_loc"]),
               ("R2 highD->exiD (target-cal)", "src", "exid", ["exid_seen_loc", "exid_new_loc"]),
               ("R3 +5% exiD (target-cal)", f"k{cfg.eval.k_fractions[1]}", "exid", ["exid_seen_loc", "exid_new_loc"]),
               ("R4 exiD only (target-cal)", "exid", "exid", ["exid_seen_loc", "exid_new_loc"])]
    cache = {p: fit_members(cfg, bev, p, sel) for p in sorted({c[1] for c in configs})}
    print("members fitted", flush=True)

    def block(prefix, ds, split):
        mk = bev[ds]["split"] == split
        Sx = bev[ds]["S"][mk]
        crun = Run(cfg, prefix, bev, out_dir)
        cidx, cP = crun.test_arrays(ds, split)
        assert np.array_equal(cidx, np.flatnonzero(mk))
        D = {"y": cnn.horizon_labels(bev[ds]["ev"][mk], bev[ds]["ttl"][mk])[:, hi], "rec": bev[ds]["rec"][mk],
             "rule": Sx[:, names[ds].index("v_lat")], "d1": Sx[:, names[ds].index("lat_disp_m1s")],
             "cnn": cP[:, :, hi].mean(0)}
        for k in ("gen", "spec0.15", "spec0.3", "spec0.6", "et", "lr", "res"):
            D[k] = cache[prefix][f"{ds}_{split}_{k}"]
        D["pre"] = D["d1"] < thr
        return D

    sel_rows, test_rows = [], []
    for label, prefix, cal_ds, tests in configs:
        C = block(prefix, cal_ds, "cal")
        cv = {n: cal_cv_score(mk, C, thr) for n, mk in VARIANTS.items()}
        best = max(cv, key=cv.get)
        sel_rows.append({"config": label, "selected": best, "cal_positives": int(C["y"].sum()), **{f"cv_{k}": v for k, v in cv.items()}})
        print(label, "selected", best, {k: round(v, 3) for k, v in cv.items()}, flush=True)
        fitted = {n: mk(C) for n, mk in VARIANTS.items()}
        for tname in tests:
            ds, split = TESTS[tname]
            T = block(prefix, ds, split)
            sc = {n: f(T) for n, f in fitted.items()}
            for subset, sm in (("all", np.ones(len(T["y"]), bool)), ("pre_drift", T["pre"])):
                cl = Cluster(T["y"][sm], T["rec"][sm])
                row = {"config": label, "test": tname, "subset": subset, "positives": int(T["y"][sm].sum()), "selected": best}
                row.update({f"ap_{n}": cl.point(v[sm]) for n, v in sc.items()})
                counts = cl.draws(B, cfg.eval.bootstrap_seed)
                rg, rs = cl.replicates(sc["gen"][sm], counts), cl.replicates(sc[best][sm], counts)
                d = rs - rg
                lo, hi_ = ci(d)
                row.update({"d_sel_gen": row[f"ap_{best}"] - row["ap_gen"], "d_lo": lo, "d_hi": hi_, "p_raw": paired_p(d)})
                test_rows.append(row)
            print(label, tname, "tested", flush=True)
    res = pd.DataFrame(test_rows)
    res["p_holm"] = holm(res.p_raw.tolist())
    pd.DataFrame(sel_rows).to_csv(cfg.paths.results / "ensemble2_selection.csv", index=False)
    res.to_csv(cfg.paths.results / "ensemble2.csv", index=False)
    return res
