"""Modelled battery energy of passenger-car trajectories and its relation to driving style.

This is a physical model, NOT measured consumption: neither dataset contains energy data, the road is assumed flat,
and the vehicle profile (configs/energy.yaml) is an assumed typical value set. Results are modelled Wh/km.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict
from scipy.signal import savgol_filter

from .config import Config
from .labels import physical_events


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mass_kg: float
    cda_m2: float
    crr: float
    rho_air: float
    eta_drive: float
    eta_regen: float
    regen_cap_w: float
    aux_w: float
    g: float


class Study(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    classes: list[str]
    min_track_km: float
    sg_window_s: float
    sg_polyorder: int
    k_range: list[int]
    stability_resamples: int
    min_stability_ari: float
    sensitivity_fraction: float
    bootstrap_resamples: int
    seed: int


def load(path: Path) -> tuple[Profile, Study]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Profile(**raw["profile"]), Study(**raw["study"])


def battery_power_w(v, a, p: Profile):
    """Battery power in W (positive = drawing from the battery) on a flat road."""
    force = p.mass_kg * a + 0.5 * p.rho_air * p.cda_m2 * v**2 + p.crr * p.mass_kg * p.g * (v > 0.5)
    wheel = force * v
    batt = np.where(wheel >= 0, wheel / p.eta_drive, np.maximum(wheel, -p.regen_cap_w) * p.eta_regen)
    return batt + p.aux_w


def track_aggregate(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, starts)


def smoothed_accel(speed: np.ndarray, starts: np.ndarray, ends: np.ndarray, dt: float, win_s: float, order: int):
    a = np.empty_like(speed)
    w = round(win_s / dt) | 1
    for s, e in zip(starts, ends):
        a[s:e] = savgol_filter(speed[s:e], w, order, deriv=1, delta=dt) if e - s >= w else np.gradient(speed[s:e], dt)
    return a


def run(cfg: Config) -> pd.DataFrame:
    prof, st = load(cfg.paths.results.parent / "configs" / "energy.yaml")
    dt = 1.0 / cfg.data.frame_rate
    frac = st.sensitivity_fraction
    variants = {"base": prof}
    for f in prof.model_fields:
        if f == "g":
            continue
        for sign, tag in ((1 - frac, "-"), (1 + frac, "+")):
            variants[f"{f}{tag}{frac:.0%}"] = prof.model_copy(update={f: getattr(prof, f) * sign})
    rows = []
    for r in pd.read_csv(cfg.paths.canonical / "manifest.csv").itertuples():
        df = pd.read_parquet(cfg.paths.canonical / f"{r.dataset}_{r.recording_id:02d}.parquet",
                             columns=["track_id", "vx", "vy", "ax", "ay", "thw", "cls", "lat_pos", "lc_flag"])
        df = df[df.cls.astype(str).isin(st.classes)].reset_index(drop=True)
        if df.empty:
            continue
        tid = df.track_id.to_numpy()
        starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
        ends = np.r_[starts[1:], len(df)]
        v = np.hypot(df.vx.to_numpy(), df.vy.to_numpy()).astype(np.float64)
        a_given = (df.ax.to_numpy() * df.vx.to_numpy() + df.ay.to_numpy() * df.vy.to_numpy()) / np.maximum(v, 0.1)
        a_smooth = smoothed_accel(v, starts, ends, dt, st.sg_window_s, st.sg_polyorder)
        km = track_aggregate(v * dt, starts) / 1000.0
        keep = km >= st.min_track_km
        rec = {"dataset": r.dataset, "recording_id": r.recording_id, "location_id": r.location_id,
               "track_id": tid[starts], "km": km}
        for name, p in variants.items():
            rec[f"wh_km_{name}"] = track_aggregate(battery_power_w(v, a_smooth, p) * dt / 3600.0, starts) / np.maximum(km, 1e-9)
        rec["wh_km_given_accel"] = track_aggregate(battery_power_w(v, a_given, prof) * dt / 3600.0, starts) / np.maximum(km, 1e-9)
        jerk = np.gradient(a_smooth, dt)
        rec["mean_speed"] = track_aggregate(v, starts) / (ends - starts)
        rec["speed_std"] = np.sqrt(np.maximum(track_aggregate(v**2, starts) / (ends - starts) - rec["mean_speed"] ** 2, 0))
        rec["jerk_rms"] = np.sqrt(track_aggregate(jerk**2, starts) / (ends - starts))
        rec["p95_abs_acc"] = np.array([np.quantile(np.abs(a_smooth[s:e]), 0.95) for s, e in zip(starts, ends)])
        thw = df.thw.to_numpy()
        rec["mean_thw"] = np.array([np.nanmean(thw[s:e]) if np.isfinite(thw[s:e]).any() else np.nan
                                    for s, e in zip(starts, ends)])
        lat, lc = df.lat_pos.to_numpy(), df.lc_flag.to_numpy()
        w, k = round(cfg.data.event_window_s * cfg.data.frame_rate), round(cfg.data.axis_sample_s * cfg.data.frame_rate)
        rec["lane_changes"] = np.array([len(physical_events(lat[s:e], np.flatnonzero(lc[s:e]), e - s, w,
                                                            cfg.data.min_lateral_disp_m, k)[0])
                                        for s, e in zip(starts, ends)])
        rows.append(pd.DataFrame(rec)[keep])
        print(r.dataset, r.recording_id, int(keep.sum()), "tracks", flush=True)
    tracks = pd.concat(rows, ignore_index=True)
    tracks.to_parquet(cfg.paths.results / "energy_tracks.parquet", index=False)
    return tracks


STYLE = ["mean_speed", "speed_std", "p95_abs_acc", "jerk_rms", "lane_changes", "mean_thw"]


def _ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return beta, 1.0 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))


def analyze(cfg: Config) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    _prof, st = load(cfg.paths.results.parent / "configs" / "energy.yaml")
    t = pd.read_parquet(cfg.paths.results / "energy_tracks.parquet")
    t = t[np.isfinite(t.wh_km_base)].reset_index(drop=True)
    rng = np.random.default_rng(st.seed)
    out: dict = {"n_tracks": len(t), "by_dataset": t.groupby("dataset").wh_km_base.agg(["mean", "std", "count"]).round(2).to_dict("index")}

    # 1. accel source and sensitivity (these are the honest uncertainty on the headline number)
    base = t.wh_km_base.mean()
    out["mean_wh_km_base"] = round(float(base), 2)
    out["accel_source"] = {"smoothed": round(float(base), 2), "dataset_provided": round(float(t.wh_km_given_accel.mean()), 2),
                           "track_corr": round(float(np.corrcoef(t.wh_km_base, t.wh_km_given_accel)[0, 1]), 3)}
    sens = [(c[len("wh_km_"):], float(t[c].mean())) for c in t.columns if c.startswith("wh_km_") and c[6:].count("%")]
    sdf = pd.DataFrame(sens, columns=["variant", "mean_wh_km"]).assign(change_pct=lambda d: 100 * (d.mean_wh_km / base - 1)).round(2)
    sdf.to_csv(cfg.paths.results / "energy_sensitivity.csv", index=False)

    # 2. style clusters with a stability check
    Z = t[STYLE].copy()
    Z["mean_thw"] = Z.mean_thw.fillna(Z.mean_thw.median())
    Z = ((Z - Z.mean()) / Z.std().replace(0, 1)).to_numpy()
    sub = rng.choice(len(Z), min(20000, len(Z)), replace=False)
    ks = {}
    for k in st.k_range:
        ref = KMeans(k, n_init=3, random_state=st.seed).fit(Z[sub])
        ref_lab = ref.predict(Z)
        sil = float(silhouette_score(Z[sub], ref.labels_, sample_size=10000, random_state=st.seed))
        aris = []
        for b in range(st.stability_resamples):
            bs = rng.choice(len(Z), min(20000, len(Z)), replace=True)
            aris.append(adjusted_rand_score(ref_lab, KMeans(k, n_init=3, random_state=b).fit(Z[bs]).predict(Z)))
        ks[k] = {"silhouette": sil, "ari_mean": float(np.mean(aris)), "ari_median": float(np.median(aris)),
                 "ari_p10": float(np.quantile(aris, 0.1)), "ari_min": float(np.min(aris))}
    stable = [k for k, m in ks.items() if m["ari_p10"] >= st.min_stability_ari]    # 90% of resamples must agree
    k_star = max(stable, key=lambda k: ks[k]["silhouette"]) if stable else max(ks, key=lambda k: ks[k]["silhouette"])
    out["clusters"] = {"per_k": ks, "chosen_k": int(k_star), "stable": bool(stable)}
    lab = KMeans(k_star, n_init=5, random_state=st.seed).fit(Z[sub]).predict(Z)
    t["cluster"] = lab

    # 3. does style matter beyond speed? (OLS, recording-cluster bootstrap)
    v = t.mean_speed.to_numpy()
    X1 = np.c_[np.ones(len(t)), v, v**2]
    D = np.eye(k_star)[lab][:, 1:]
    X2 = np.c_[X1, D]
    y = t.wh_km_base.to_numpy()
    _, r1 = _ols(X1, y)
    b2, r2 = _ols(X2, y)
    rec = (t.dataset + "_" + t.recording_id.astype(str)).to_numpy()
    codes, uniq = pd.factorize(rec)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    # robustness: replace the quadratic speed control by 20 speed quantile bins plus a dataset indicator
    bins = np.digitize(v, np.quantile(v, np.linspace(0, 1, 21)[1:-1]))
    Xb1 = np.c_[np.eye(20)[bins], (t.dataset == "exid").to_numpy()]
    _, rb1 = _ols(Xb1, y)
    bb2, rb2 = _ols(np.c_[Xb1, D], y)
    boots = []
    for _ in range(st.bootstrap_resamples):
        ii = np.concatenate([order[bounds[k]:bounds[k + 1]] for k in rng.integers(0, len(uniq), len(uniq))])
        bb, rr = _ols(X2[ii], y[ii])
        boots.append(np.r_[bb[3:], rr - _ols(X1[ii], y[ii])[1]])
    boots = np.array(boots)
    out["style_beyond_speed"] = {
        "r2_speed_only": round(float(r1), 4), "r2_speed_plus_cluster": round(float(r2), 4),
        "delta_r2": round(float(r2 - r1), 4),
        "binned_speed_control": {"r2_without_cluster": round(float(rb1), 4), "r2_with_cluster": round(float(rb2), 4),
                                 "delta_r2": round(float(rb2 - rb1), 4),
                                 "cluster_effect_wh_km": {int(i + 1): round(float(bb2[21 + i]), 2) for i in range(k_star - 1)}},
        "delta_r2_ci": [round(float(np.quantile(boots[:, -1], .025)), 4), round(float(np.quantile(boots[:, -1], .975)), 4)],
        "cluster_effect_wh_km_vs_cluster0": {int(i + 1): [round(float(b2[3 + i]), 2),
                                                           round(float(np.quantile(boots[:, i], .025)), 2),
                                                           round(float(np.quantile(boots[:, i], .975)), 2)]
                                              for i in range(k_star - 1)}}
    prof_tab = t.groupby("cluster").agg(n=("km", "size"), wh_km=("wh_km_base", "mean"), **{c: (c, "mean") for c in STYLE}).round(2)
    prof_tab.to_csv(cfg.paths.results / "energy_clusters.csv")
    out["cluster_profile"] = prof_tab.to_dict("index")
    import json
    (cfg.paths.results / "energy_summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out
