"""End-to-end test on synthetic raw files in the real highD and exiD formats: ingest -> labels -> features -> BEV."""
import numpy as np
import pandas as pd
import pytest

from hwt import bev, ingest, labels
from hwt.config import Paths, load_config

FR, N = 25, 420


def smoothstep(n, start, dur):
    r = np.clip((np.arange(n) - start) / dur, 0, 1)
    return 3 * r**2 - 2 * r**3


def write_highd(root, rid=1):
    rows, meta = [], []
    for vid in range(1, 9):
        lane = 2 + vid % 2
        y_c = 20.0 + 3.7 * (vid % 2)                     # lane centre, y pointing down
        ramp = 3.7 * smoothstep(N, 150, 100) if vid == 1 else np.zeros(N)
        yc = y_c + ramp
        lane_id = np.where(ramp > 1.85, lane + 1, lane) if vid == 1 else np.full(N, lane)
        f = np.arange(N)
        rows.append(pd.DataFrame({
            "frame": f, "id": vid, "x": 50.0 + vid * 25 + f * 30 / FR, "y": yc - 1.0, "width": 4.5, "height": 2.0,
            "xVelocity": 30.0, "yVelocity": np.gradient(yc, 1 / FR), "xAcceleration": 0.0, "yAcceleration": 0.0,
            "thw": 0.0, "ttc": 0.0, "precedingId": 0, "laneId": lane_id}))
        meta.append({"id": vid, "drivingDirection": 2, "class": "Car" if vid % 4 else "Truck"})
    root.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(root / f"{rid:02d}_tracks.csv", index=False)
    pd.DataFrame(meta).to_csv(root / f"{rid:02d}_tracksMeta.csv", index=False)
    pd.DataFrame([{"locationId": 2}]).to_csv(root / f"{rid:02d}_recordingMeta.csv", index=False)


def write_exid(root, rid=0, degrees=True):
    rows, meta = [], []
    for vid in range(1, 9):
        f = np.arange(N)
        lat = 3.7 * smoothstep(N, 150, 100) if vid == 1 else np.zeros(N)
        y = 500.0 + 3.7 * (vid % 2) + lat
        offset = lat.copy()
        flag = np.zeros(N, int)
        if vid == 1:                                     # offset is relative to the current lane: jumps by one lane width
            sw = 215
            offset[sw:] -= 3.7
            flag[sw] = 1
        vy = np.gradient(y, 1 / FR)
        head = np.arctan2(vy, 30.0)                      # exiD heading follows the velocity
        rows.append(pd.DataFrame({
            "trackId": vid, "frame": f, "xCenter": 100.0 + vid * 25 + f * 30 / FR, "yCenter": y,
            "heading": np.degrees(head) if degrees else head, "xVelocity": 30.0, "yVelocity": vy,
            "xAcceleration": 0.0, "yAcceleration": 0.0, "length": 4.5, "width": 1.9, "laneletId": 10 + vid % 2,
            "laneChange": flag, "leadTTC": np.nan, "leadTHW": np.nan, "leadId": np.nan,
            "latLaneCenterOffset": offset, "laneWidth": 3.7}))
        meta.append({"trackId": vid, "class": "car"})
    root.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(root / f"{rid:02d}_tracks.csv", index=False)
    pd.DataFrame(meta).to_csv(root / f"{rid:02d}_tracksMeta.csv", index=False)
    pd.DataFrame([{"locationId": 1}]).to_csv(root / f"{rid:02d}_recordingMeta.csv", index=False)


@pytest.fixture()
def cfg(tmp_path):
    base = load_config("configs/base.yaml")
    paths = Paths(raw=tmp_path / "raw", canonical=tmp_path / "canon", interim=tmp_path / "interim", results=tmp_path / "res")
    split = base.splits.model_copy(update={"highd_holdout_locations": [], "exid_holdout_locations": []})
    data = base.data.model_copy(update={"sample_stride_frames": 5})
    write_highd(paths.raw / "highd")
    write_exid(paths.raw / "exid")
    return base.model_copy(update={"paths": paths, "splits": split, "data": data})


def test_ingest_maps_both_formats_to_the_canonical_schema(cfg):
    ingest.run(cfg)
    h = pd.read_parquet(cfg.paths.canonical / "highd_01.parquet")
    e = pd.read_parquet(cfg.paths.canonical / "exid_00.parquet")
    one = h[h.track_id == 1].reset_index(drop=True)
    assert one.x.iloc[0] == pytest.approx(75.0 + 2.25) and one.y.iloc[0] == pytest.approx(-23.7)   # centred, y flipped
    assert one.lc_flag.sum() == 1 and h.cls.nunique() == 2 and h.ttc.isna().all()
    # exiD: heading given in degrees is detected; offset jump is removed so lat_pos is continuous
    ev = e[e.track_id == 1].reset_index(drop=True)
    assert np.abs(np.diff(ev.lat_pos.to_numpy())).max() < 0.2 and np.abs(ev.heading).max() < np.pi
    assert ev.lc_flag.sum() == 1


def test_full_pipeline_produces_labels_features_and_rasters(cfg):
    ingest.run(cfg)
    s = labels.run(cfg)
    assert (s.ev == 1).any() and (s.ev == 0).any()
    assert set(s.dataset.unique()) == {"highd", "exid"}
    bev.run(cfg)
    z = np.load(cfg.paths.interim.parent / "bev" / "highd.npz", allow_pickle=False)
    assert z["X"].shape[1:] == (3, 90, 32) and len(z["X"]) == len(z["S"]) == len(z["ev"])
    assert np.isfinite(z["S"]).all() and (z["X"][:, 0] > 0).any()


def test_pipeline_is_deterministic(cfg):
    ingest.run(cfg)
    a = labels.run(cfg).reset_index(drop=True)
    for f in cfg.paths.interim.glob("samples_*.parquet"):
        f.unlink()
    b = labels.run(cfg).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_lane_change_event_found_in_both_datasets_with_the_same_definition(cfg):
    ingest.run(cfg)
    s = labels.run(cfg)
    pos = s[s.ev == 1].groupby("dataset").track_id.unique()
    assert list(pos["highd"]) == [1] and list(pos["exid"]) == [1]
