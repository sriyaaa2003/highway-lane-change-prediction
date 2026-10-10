"""Split conformal prediction with class-conditional (Mondrian) thresholds, evaluated under dataset shift.

Score of a calibration sample = 1 - p(true class). For each true class c the threshold q_c is the
ceil((n_c+1)(1-alpha))/n_c empirical quantile of that class's calibration scores. A prediction set contains every
class whose score would be <= its threshold. Calibration and test sets are recording-disjoint. Calibration and test
recordings are both resampled in the bootstrap, so the interval includes calibration-set noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cnn
from .config import Config
from .evalcnn import TESTS, Run, load_bev

HEAD = 1  # the 2 s horizon (cfg.eval.primary_horizon_s)


def thresholds(p1: np.ndarray, y: np.ndarray, alpha: float, min_events: int):
    """Returns (q0, q1); a threshold is +inf (always include) if the class has too few calibration samples."""
    q = []
    for c in (0, 1):
        s = 1.0 - (p1[y == 1] if c == 1 else 1.0 - p1[y == 0])
        n = len(s)
        if n < min_events:
            q.append(np.inf)
            continue
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q.append(float(np.quantile(s, level, method="higher")))
    return q[0], q[1]


def evaluate(q0, q1, p1, y):
    """Class 0 is in the set if 1 - p(0) <= q0, i.e. p1 <= q0; class 1 if 1 - p1 <= q1."""
    in0, in1 = p1 <= q0, (1.0 - p1) <= q1
    covered = np.where(y == 1, in1, in0)
    size = in0.astype(int) + in1.astype(int)
    return {"cov": covered.mean(), "cov_pos": covered[y == 1].mean() if (y == 1).any() else np.nan,
                "cov_neg": covered[y == 0].mean(), "size": size.mean(), "empty": (size == 0).mean()}


def _boot_units(rec, rng):
    codes, uniq = pd.factorize(rec)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([order[bounds[k]:bounds[k + 1]] for k in pick])


def run(cfg: Config, alpha: float = 0.1) -> pd.DataFrame:
    out_dir = cfg.paths.interim.parent / "out"
    bev = load_bev(cfg)
    thr = cfg.eval.pre_drift_max_disp_1s_m
    scen = [  # (scenario label, model prefix, calibration (ds, split), test names)
        ("exchangeable: highD cal -> highD", "src", ("highd", "cal"), ["highd_test", "highd_new_loc"]),
        ("shift: highD cal -> exiD", "src", ("highd", "cal"), ["exid_seen_loc", "exid_new_loc"]),
        ("repair: exiD cal, highD-trained model", "src", ("exid", "cal"), ["exid_seen_loc", "exid_new_loc"]),
        *[(f"repair: exiD cal, highD+{k:.0%} exiD model", f"k{k}", ("exid", "cal"), ["exid_seen_loc", "exid_new_loc"])
          for k in cfg.eval.k_fractions],
        ("exiD-only model, exiD cal", "exid", ("exid", "cal"), ["exid_seen_loc", "exid_new_loc"]),
    ]
    rng_master = np.random.default_rng(cfg.eval.bootstrap_seed)
    rows = []
    for label, prefix, (cds, csplit), tests in scen:
        run_ = Run(cfg, prefix, bev, out_dir)
        if not run_.seeds:
            continue
        cidx, cP = run_.test_arrays(cds, csplit)
        cb = bev[cds]
        cy = cnn.horizon_labels(cb["ev"][cidx], cb["ttl"][cidx])[:, HEAD]
        crec = cb["rec"][cidx]
        for tname in tests:
            tds, tsplit = TESTS[tname]
            tidx, tP = run_.test_arrays(tds, tsplit)
            tb = bev[tds]
            ty = cnn.horizon_labels(tb["ev"][tidx], tb["ttl"][tidx])[:, HEAD]
            trec = tb["rec"][tidx]
            pre = tb["S"][tidx, list(tb["names"]).index("lat_disp_m1s")] < thr
            for subset, mask in (("all", np.ones(len(tidx), bool)), ("pre_drift", pre)):
                n_seeds = len(run_.seeds)

                def one(ci, ti, mask=mask, cP=cP, cy=cy, tP=tP, ty=ty, n_seeds=n_seeds):
                    ms = []
                    for si in range(n_seeds):
                        q0, q1 = thresholds(cP[si, ci, HEAD], cy[ci], alpha, cfg.eval.min_events_per_class)
                        tt = ti[mask[ti]]
                        ms.append(evaluate(q0, q1, tP[si, tt, HEAD], ty[tt]))
                    return {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
                point = one(np.arange(len(cy)), np.arange(len(ty)))
                reps = [one(_boot_units(crec, rng_master), _boot_units(trec, rng_master))
                        for _ in range(cfg.eval.bootstrap_resamples // 3)]
                row = {"scenario": label, "model": prefix, "cal": f"{cds}:{csplit}", "test": tname, "subset": subset,
                           "alpha": alpha, "cal_pos": int(cy.sum()), "test_n": int(mask.sum()), "test_pos": int(ty[mask].sum()),
                           "n_seeds": len(run_.seeds)}
                for k, v in point.items():
                    arr = np.array([r[k] for r in reps])
                    row.update({k: v, f"{k}_lo": float(np.nanquantile(arr, .025)),
                                f"{k}_hi": float(np.nanquantile(arr, .975))})
                rows.append(row)
            print(label, tname, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "conformal.csv", index=False)
    return res
