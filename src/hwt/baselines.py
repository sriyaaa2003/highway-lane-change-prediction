"""Class prior, lateral-velocity rule and gradient boosting across the shift ladder (CPU only)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from .config import Config
from .labels import labels_for_horizon
from .splits import HOLDOUT, assign, check_integrity

META = ["dataset", "recording_id", "location_id", "track_id", "frame", "ev", "ttl"]


def load_samples(cfg: Config) -> pd.DataFrame:
    man = assign(pd.read_csv(cfg.paths.canonical / "manifest.csv"), cfg)
    check_integrity(man)
    parts = [pd.read_parquet(cfg.paths.interim / f"samples_{r.dataset}_{r.recording_id:02d}.parquet")
             for r in man.itertuples()]
    s = pd.concat(parts, ignore_index=True)
    s = s.merge(man[["dataset", "recording_id", "split"]], on=["dataset", "recording_id"])
    s["rec"] = s.dataset.astype(str) + "_" + s.recording_id.astype(str)
    s["trk"] = s.rec + "_" + s.track_id.astype(str)
    return s


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision of 'lane change within the horizon'; NaN if the test set has no positives."""
    return float(average_precision_score(y, p[:, 1])) if y.any() else float("nan")


def cluster_bootstrap(y, p, rec, B, seed) -> tuple[float, float, float]:
    """95% CI resampling whole recordings (frames within a recording are not independent)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(rec)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([order[bounds[k]:bounds[k + 1]] for k in pick])
        vals.append(pr_auc(y[ii], p[ii]))
    vals = np.array(vals)
    return pr_auc(y, p), float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def feature_cols(s: pd.DataFrame) -> list[str]:
    drop = set(META) | {"split", "rec", "trk"}
    return [c for c in s.columns if c not in drop]


def fit(kind: str, tr: pd.DataFrame, H: float, cfg: Config):
    """Return a predict(te) -> (N,3) probability/score callable."""
    ytr = labels_for_horizon(tr.ev.to_numpy(), tr.ttl.to_numpy(), H)
    if kind == "prior":
        pr = np.bincount(ytr, minlength=2) / len(ytr)
        return lambda te: np.tile(pr, (len(te), 1))
    if kind == "rule":                       # rank by lateral speed only
        return lambda te: np.c_[np.zeros(len(te)), te.v_lat.to_numpy()]
    h = cfg.model.hgb
    m = HistGradientBoostingClassifier(max_iter=h.max_iter, learning_rate=h.learning_rate,
                                       max_leaf_nodes=h.max_leaf_nodes, l2_regularization=h.l2_regularization,
                                       class_weight=h.class_weight, early_stopping=False, random_state=0)
    cols = feature_cols(tr)
    m.fit(tr[cols], ytr)

    def predict(te: pd.DataFrame) -> np.ndarray:
        P = np.zeros((len(te), 2))
        P[:, m.classes_] = m.predict_proba(te[cols])
        return P
    return predict


def rungs(s: pd.DataFrame, cfg: Config) -> dict[str, tuple[pd.DataFrame, dict[str, pd.DataFrame]]]:
    hi, ex = s[s.dataset == "highd"], s[s.dataset == "exid"]
    hi_tr, ex_tr = hi[hi.split == "train"], ex[ex.split == "train"]
    tgt = {"exid_seen_loc": ex[ex.split == "test"], "exid_new_loc": ex[ex.split == HOLDOUT]}
    out = {
        "R0 highD->highD": (hi_tr, {"highd_test": hi[hi.split == "test"]}),
        "R1 highD->highD new location": (hi_tr, {"highd_new_loc": hi[hi.split == HOLDOUT]}),
        "R2 highD->exiD": (hi_tr, tgt),
    }
    trk = ex_tr.trk.drop_duplicates().sample(frac=1.0, random_state=cfg.eval.bootstrap_seed).to_numpy()
    for k in cfg.eval.k_fractions:
        keep = set(trk[: max(1, round(k * len(trk)))])
        out[f"R3 highD+{k:.0%} exiD"] = (pd.concat([hi_tr, ex_tr[ex_tr.trk.isin(keep)]]), tgt)
    out["R4 exiD->exiD"] = (ex_tr, tgt)
    return {k: v for k, v in out.items() if all(len(t) for t in v[1].values()) and len(v[0])}


def run(cfg: Config) -> None:
    """Resumable: each result row is appended to results/baselines.csv as soon as it exists."""
    cfg.paths.results.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.results / "baselines.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    have = set(zip(done.rung, done.H, done.model, done.test)) if len(done) else set()
    s = load_samples(cfg)
    t0 = time.time()
    for rung, (tr, tests) in rungs(s, cfg).items():
        thr = cfg.eval.pre_drift_max_disp_1s_m      # "pre-drift": no sideways move in the last second yet
        tests = {**tests, **{f"{k}|pre_drift": v[v["lat_disp_m1s"] < thr] for k, v in tests.items()}}
        for H in cfg.data.horizons_s:
            for kind in ("prior", "rule", "hgb"):
                if all((rung, H, kind, t) in have for t in tests):
                    continue
                predict = fit(kind, tr, H, cfg)
                for tname, te in tests.items():
                    y = labels_for_horizon(te.ev.to_numpy(), te.ttl.to_numpy(), H)
                    pr, lo, hi = cluster_bootstrap(y, predict(te), te.rec.to_numpy(),
                                                   cfg.eval.bootstrap_resamples, cfg.eval.bootstrap_seed)
                    row = pd.DataFrame([{"rung": rung, "H": H, "model": kind, "test": tname, "n_train": len(tr),
                                             "n_test": len(te), "n_test_recordings": int(te.rec.nunique()),
                                             "positives": int(y.sum()),
                                             "pr_auc": pr, "ci_lo": lo, "ci_hi": hi, "prevalence": float(y.mean())}])
                    row.to_csv(path, mode="a", header=not path.exists(), index=False)
                    print(f"{rung} H={H} {kind} {tname}: PR-AUC {pr:.3f} [{lo:.3f},{hi:.3f}] "
                          f"({time.time() - t0:.0f}s)", flush=True)
                del predict
    (cfg.paths.results / "baselines_meta.json").write_text(json.dumps(
        {"wall_s": time.time() - t0, "horizons_s": cfg.data.horizons_s, "hgb": cfg.model.hgb.model_dump(),
         "bootstrap_resamples": cfg.eval.bootstrap_resamples}))
