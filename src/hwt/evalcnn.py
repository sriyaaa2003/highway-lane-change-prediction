"""CNN results on the shift ladder, with the same test sets, horizons and pre-drift subset as the baselines.

Statistic: PR-AUC averaged over training seeds; CI by a recording-cluster bootstrap in which each resample is applied
to every seed jointly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import cnn
from .config import Config

TESTS = {"highd_test": ("highd", "test"), "highd_new_loc": ("highd", "holdout_loc"),
         "exid_seen_loc": ("exid", "test"), "exid_new_loc": ("exid", "holdout_loc")}


def rung_table(cfg: Config) -> list[tuple[str, str, list[str]]]:
    rows = [("R0 highD->highD", "src", ["highd_test"]),
            ("R1 highD->highD new location", "src", ["highd_new_loc"]),
            ("R2 highD->exiD", "src", ["exid_new_loc", "exid_seen_loc"])]
    rows += [(f"R3 highD+{k:.0%} exiD", f"k{k}", ["exid_new_loc", "exid_seen_loc"]) for k in cfg.eval.k_fractions]
    rows.append(("R4 exiD->exiD", "exid", ["exid_new_loc", "exid_seen_loc"]))
    return rows


class Run:
    """All seeds of one training setup, aligned to the BEV npz order."""

    def __init__(self, cfg: Config, prefix: str, bev: dict, out_dir):
        self.prefix, self.bev = prefix, bev
        self.seeds = [s for s in cfg.cnn.seeds if (out_dir / f"{prefix}_s{s}.npz").exists()]
        self.z = {s: np.load(out_dir / f"{prefix}_s{s}.npz") for s in self.seeds}

    def test_arrays(self, ds: str, split: str):
        """idx (rows into the BEV npz), probs (seeds, n, 3)."""
        b = self.bev[ds]
        idx0 = self.z[self.seeds[0]][f"{ds}_idx"]
        keep = b["split"][idx0] == split
        P = np.stack([self.z[s][f"{ds}_P"][keep].astype(np.float32) for s in self.seeds])
        return idx0[keep], P


def cluster_ci(y, P, rec, B, seed):
    """y (n,), P (seeds,n); statistic = mean over seeds of average precision (exact weighted bootstrap)."""
    from .stats import Cluster, ci
    cl = Cluster(y, rec)
    reps = np.mean([cl.replicates(p, cl.draws(B, seed)) for p in P], axis=0)
    point = float(np.mean([cl.point(p) for p in P]))
    lo, hi = ci(reps)
    return point, lo, hi


def load_bev(cfg: Config) -> dict:
    d = {}
    for ds in ("highd", "exid"):
        z = np.load(cfg.paths.interim.parent / "bev" / f"{ds}.npz", allow_pickle=False)
        d[ds] = {k: z[k] for k in z.files if k != "X"}
    return d


def run(cfg: Config) -> pd.DataFrame:
    out_dir = cfg.paths.interim.parent / "out"
    bev = load_bev(cfg)
    rows = []
    for rung, prefix, tests in rung_table(cfg):
        r = Run(cfg, prefix, bev, out_dir)
        if not r.seeds:
            print("missing", prefix)
            continue
        for tname in tests:
            ds, split = TESTS[tname]
            idx, P = r.test_arrays(ds, split)
            b = bev[ds]
            Y = cnn.horizon_labels(b["ev"][idx], b["ttl"][idx])
            rec = b["rec"][idx]
            pre = b["S"][idx, list(b["names"]).index("lat_disp_m1s")] < cfg.eval.pre_drift_max_disp_1s_m
            for subset, mask in (("", np.ones(len(idx), bool)), ("|pre_drift", pre)):
                for hi, H in enumerate(cnn.HORIZONS):
                    y = Y[mask, hi]
                    pr, lo, hi_ = cluster_ci(y, P[:, mask, hi], rec[mask], cfg.eval.bootstrap_resamples,
                                             cfg.eval.bootstrap_seed)
                    per_seed = [average_precision_score(y, p) for p in P[:, mask, hi]] if y.any() else [np.nan]
                    rows.append({"rung": rung, "H": H, "model": "cnn", "test": tname + subset, "n_test": int(mask.sum()),
                                     "positives": int(y.sum()), "pr_auc": pr, "ci_lo": lo, "ci_hi": hi_,
                                     "seed_min": float(np.min(per_seed)), "seed_max": float(np.max(per_seed)),
                                     "n_seeds": len(r.seeds)})
            print(rung, tname, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "cnn.csv", index=False)
    base = pd.read_csv(cfg.paths.results / "baselines.csv")
    both = pd.concat([base[["rung", "H", "model", "test", "n_test", "positives", "pr_auc", "ci_lo", "ci_hi"]],
                      res[["rung", "H", "model", "test", "n_test", "positives", "pr_auc", "ci_lo", "ci_hi"]]])
    both.to_csv(cfg.paths.results / "ladder.csv", index=False)
    return both
