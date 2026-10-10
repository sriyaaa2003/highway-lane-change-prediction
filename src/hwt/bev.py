"""Bird's-eye-view rasters built from trajectories (not from camera video).

Frame: ego at the origin, +dx ahead, +dy left, aligned to the vehicle's own recent path (features.ref_axis), so the
frame is defined identically for highD and exiD. Channels (uint8): 0 occupancy (255 = vehicle footprint),
1 relative longitudinal speed, 2 relative lateral speed (128 = same speed, scaled by bev.speed_clip_ms).
The ego vehicle is not drawn; its own kinematics are in the scalar features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .features import neighbor_rows, ref_axis
from .splits import assign, check_integrity


def grid_shape(cfg: Config) -> tuple[int, int]:
    b = cfg.bev
    return round((b.ahead_m + b.behind_m) / b.cell_long_m), round(2 * b.half_width_m / b.cell_lat_m)


def rasterize(aid, dx, dy, dvl, dvt, length, width, n: int, cfg: Config) -> np.ndarray:
    b = cfg.bev
    H, W = grid_shape(cfg)
    out = np.full((n, 3, H, W), 128, dtype=np.uint8)
    out[:, 0] = 0
    i0 = np.floor((dx - length / 2 + b.behind_m) / b.cell_long_m).astype(int)
    i1 = np.floor((dx + length / 2 + b.behind_m) / b.cell_long_m).astype(int)
    j0 = np.floor((dy - width / 2 + b.half_width_m) / b.cell_lat_m).astype(int)
    j1 = np.floor((dy + width / 2 + b.half_width_m) / b.cell_lat_m).astype(int)
    q = lambda v: (np.clip(v / b.speed_clip_ms, -1, 1) * 127 + 128).astype(np.uint8)
    ch1, ch2 = q(dvl), q(dvt)
    ok = (i1 >= 0) & (i0 < H) & (j1 >= 0) & (j0 < W)
    for k in np.flatnonzero(ok):
        a, bb = max(i0[k], 0), min(i1[k], H - 1)
        c, d = max(j0[k], 0), min(j1[k], W - 1)
        out[aid[k], 0, a:bb + 1, c:d + 1] = 255
        out[aid[k], 1, a:bb + 1, c:d + 1] = ch1[k]
        out[aid[k], 2, a:bb + 1, c:d + 1] = ch2[k]
    return out


def fill_scalars(S: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    f = cfg.bev
    S = S.copy()
    for c in S.columns:
        if c.endswith("_gap"):
            S[c] = S[c].fillna(f.fill_gap_m)
        elif c == "thw":
            S[c] = S[c].fillna(f.fill_thw_s)
        elif c == "ttc":
            S[c] = S[c].fillna(f.fill_ttc_s)
        else:
            S[c] = S[c].fillna(0.0)
    return S


def build_recording(df: pd.DataFrame, samples: pd.DataFrame, cfg: Config):
    """samples: rows of the interim table for this recording (already capped). Returns (X, S)."""
    key = pd.MultiIndex.from_arrays([df.track_id.to_numpy(), df.frame.to_numpy()])
    row = key.get_indexer(pd.MultiIndex.from_arrays([samples.track_id.to_numpy(), samples.frame.to_numpy()]))
    if (row < 0).any():
        raise AssertionError("sample not found in canonical table")
    idx = pd.DataFrame({"row": row})
    c, s = ref_axis(df, row, cfg)
    m = neighbor_rows(df, idx, cfg, c, s, samples.v_lon.to_numpy(), max_dy=cfg.bev.half_width_m + 1.0)
    X = rasterize(m.aid.to_numpy(), m.dx.to_numpy(), m.dy.to_numpy(), m.dvl.to_numpy(), m.dvt.to_numpy(),
                  m.length.to_numpy(), m.width.to_numpy(), len(samples), cfg)
    return X


META = ["dataset", "recording_id", "location_id", "track_id", "frame", "ev", "ttl"]


def run(cfg: Config) -> None:
    out_dir = cfg.paths.interim.parent / "bev"
    out_dir.mkdir(parents=True, exist_ok=True)
    man = assign(pd.read_csv(cfg.paths.canonical / "manifest.csv"), cfg)
    check_integrity(man)
    for ds in ("highd", "exid"):
        Xs, Ss, Ms = [], [], []
        for r in man[man.dataset == ds].itertuples():
            smp = pd.read_parquet(cfg.paths.interim / f"samples_{ds}_{r.recording_id:02d}.parquet")
            rng = np.random.default_rng([cfg.splits.seed, r.recording_id, 0 if ds == "highd" else 1, 99])
            if len(smp) > cfg.bev.per_recording_cap:
                smp = smp.iloc[np.sort(rng.choice(len(smp), cfg.bev.per_recording_cap, replace=False))]
            smp = smp.reset_index(drop=True)
            df = pd.read_parquet(cfg.paths.canonical / f"{ds}_{r.recording_id:02d}.parquet")
            Xs.append(build_recording(df, smp, cfg))
            Ss.append(smp.drop(columns=META))
            Ms.append(smp[META].assign(split=r.split))
            print(f"{ds} {r.recording_id:02d} n={len(smp)}", flush=True)
        S = fill_scalars(pd.concat(Ss, ignore_index=True), cfg)
        M = pd.concat(Ms, ignore_index=True)
        arrays = {"X": np.concatenate(Xs), "S": S.to_numpy(np.float32), "names": np.array(list(S.columns), dtype="U32"),
                      "rec": M.recording_id.to_numpy(np.int32), "loc": M.location_id.to_numpy(np.int32),
                      "trk": M.track_id.to_numpy(np.int32), "frame": M.frame.to_numpy(np.int32),
                      "split": M.split.to_numpy().astype("U16"), "ev": M.ev.to_numpy(np.int8),
                      "ttl": M.ttl.to_numpy(np.float32)}
        if ds == "exid":                       # nested track subsets for the "highD + k% exiD" rung
            tr = M[M.split == "train"]
            tracks = tr[["recording_id", "track_id"]].drop_duplicates().to_numpy()
            order = np.random.default_rng(cfg.eval.bootstrap_seed).permutation(len(tracks))
            for k in cfg.eval.k_fractions:
                chosen = {tuple(t) for t in tracks[order[: max(1, round(k * len(tracks)))]]}
                arrays[f"kmask_{k}"] = np.array([(a, b) in chosen for a, b in zip(M.recording_id, M.track_id)]) \
                    & (M.split == "train").to_numpy()
        np.savez_compressed(out_dir / f"{ds}.npz", **arrays)
        print(ds, "saved", arrays["X"].shape, flush=True)
