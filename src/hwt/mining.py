"""Scenario mining: query-by-example retrieval of behaviour-defined scenes from learned embeddings.

Tags are rules on the data itself, defined from what happens AFTER the anchor (so they are not inputs of any model):
  brake_next     minimum longitudinal acceleration in the next brake_window_s below brake_decel_ms2
  ttc_crit_next  minimum time-to-collision in the next ttc_window_s below ttc_crit_s
  lane_change    a physical lane change within lane_change_horizon_s (the CNN's own training label)
Protocol: draw n_query positive scenes from the calibration split as queries, average their normalised embeddings
into a prototype, rank every scene of the pool (test split of a recording-disjoint set) by cosine similarity, and
score precision@k. Random retrieval scores the tag prevalence. All thresholds are in the config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cnn
from .config import Config
from .evalcnn import Run, load_bev


def build_tags(cfg: Config) -> dict[str, dict[str, np.ndarray]]:
    """Per dataset, tag arrays aligned with the BEV npz order."""
    t, fr = cfg.tags, cfg.data.frame_rate
    bev = load_bev(cfg)
    out = {}
    for ds in ("highd", "exid"):
        b = bev[ds]
        n = len(b["rec"])
        brake, ttc = np.zeros(n, bool), np.zeros(n, bool)
        for rid in np.unique(b["rec"]):
            df = pd.read_parquet(cfg.paths.canonical / f"{ds}_{rid:02d}.parquet",
                                 columns=["track_id", "frame", "vx", "vy", "ax", "ay", "ttc"])
            spd = np.hypot(df.vx, df.vy).clip(lower=0.1)
            df["a_lon"] = (df.ax * df.vx + df.ay * df.vy) / spd
            df["ttc"] = df.ttc.fillna(np.inf)
            for col, win, name in (("a_lon", t.brake_window_s, "brake"), ("ttc", t.ttc_window_s, "ttc")):
                w = round(win * fr)
                # forward-looking minimum over the next w frames within each track
                fwd = df.groupby("track_id")[col].transform(lambda s, w=w: s[::-1].rolling(w, min_periods=1).min()[::-1])
                df[f"min_{name}"] = fwd
            key = pd.MultiIndex.from_arrays([df.track_id.to_numpy(), df.frame.to_numpy()])
            m = np.flatnonzero(b["rec"] == rid)
            pos = key.get_indexer(pd.MultiIndex.from_arrays([b["trk"][m], b["frame"][m]]))
            brake[m] = df.min_brake.to_numpy()[pos] < t.brake_decel_ms2
            ttc[m] = df.min_ttc.to_numpy()[pos] < t.ttc_crit_s
        lc = cnn.horizon_labels(b["ev"], b["ttl"])[:, list(cnn.HORIZONS).index(t.lane_change_horizon_s)] > 0
        out[ds] = {"brake_next": brake, "ttc_crit_next": ttc, "lane_change": lc}
    return out


def _embeddings(cfg, bev, prefix, ds, split, out_dir):
    r = Run(cfg, prefix, bev, out_dir)
    idx0 = r.z[r.seeds[0]][f"{ds}_idx"]
    keep = bev[ds]["split"][idx0] == split
    return idx0[keep], [r.z[s][f"{ds}_E"][keep].astype(np.float32) for s in r.seeds]


def _unit(e):
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)


def retrieve(Eq, Ep, tq, tp, n_query, k, n_draws, rng):
    """Mean/percentile precision@k over random query draws; NaN if too few positives."""
    pos = np.flatnonzero(tq)
    if len(pos) < n_query or tp.sum() < k:
        return None
    Eq, Ep = _unit(Eq), _unit(Ep)
    vals = []
    for _ in range(n_draws):
        proto = Eq[rng.choice(pos, n_query, replace=False)].mean(0)
        top = np.argpartition(-(Ep @ proto), k)[:k]
        vals.append(tp[top].mean())
    v = np.array(vals)
    return float(v.mean()), float(np.quantile(v, .025)), float(np.quantile(v, .975))


def run(cfg: Config) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    out_dir = cfg.paths.interim.parent / "out"
    bev = load_bev(cfg)
    tags = build_tags(cfg)
    m = cfg.mining
    rng = np.random.default_rng(cfg.eval.bootstrap_seed)
    scen = [("highD -> highD", "src", "highd", "highd"), ("exiD -> exiD (exiD-trained)", "exid", "exid", "exid"),
            ("highD -> exiD (highD-trained, zero-shot)", "src", "highd", "exid")]
    raw = {ds: np.load(cfg.paths.interim.parent / "bev" / f"{ds}.npz", allow_pickle=False)["X"] for ds in bev}
    rows = []
    for label, prefix, qds, pds in scen:
        qi, qE = _embeddings(cfg, bev, prefix, qds, "cal", out_dir)
        pi, pE = _embeddings(cfg, bev, prefix, pds, "holdout_loc" if pds == "exid" and qds == "exid" else "test", out_dir)
        sources = {"cnn": (qE, pE)}
        _ri, rE = _embeddings(cfg, bev, "rand", qds, "cal", out_dir)
        _rpi, rpE = _embeddings(cfg, bev, "rand", pds, "holdout_loc" if pds == "exid" and qds == "exid" else "test", out_dir)
        sources["random-init CNN (control)"] = (rE, rpE)
        # hand-built baselines, fitted on the query dataset's calibration split only
        mu, sd = bev[qds]["S"][qi].mean(0), bev[qds]["S"][qi].std(0) + 1e-6
        sources["scalar features"] = ([(bev[qds]["S"][qi] - mu) / sd], [(bev[pds]["S"][pi] - mu) / sd])
        sub = rng.choice(len(qi), min(20000, len(qi)), replace=False)
        pca = PCA(64, random_state=0).fit(raw[qds][qi[sub]].reshape(len(sub), -1).astype(np.float32) / 255.0)
        def tf(ds, ii, pca=pca):
            return [pca.transform(raw[ds][ii].reshape(len(ii), -1).astype(np.float32) / 255.0)]

        sources["raw BEV (PCA-64)"] = (tf(qds, qi), tf(pds, pi))
        for tag in ("brake_next", "ttc_crit_next", "lane_change"):
            tq, tp = tags[qds][tag][qi], tags[pds][tag][pi]
            for name, (Eq_list, Ep_list) in sources.items():
                res = [retrieve(a, b, tq, tp, m.n_query, m.k, m.n_draws, rng) for a, b in zip(Eq_list, Ep_list)]
                if any(r is None for r in res):
                    continue
                mean = float(np.mean([r[0] for r in res]))
                rows.append({"scenario": label, "tag": tag, "embedding": name, "precision_at_k": mean,
                                 "lo": float(np.mean([r[1] for r in res])), "hi": float(np.mean([r[2] for r in res])),
                                 "prevalence": float(tp.mean()), "enrichment": mean / max(float(tp.mean()), 1e-9),
                                 "pool_n": len(pi), "pool_pos": int(tp.sum()), "query_pos": int(tq.sum()),
                                 "k": m.k, "n_query": m.n_query})
            print(label, tag, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "mining.csv", index=False)
    return res
