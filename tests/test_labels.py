import numpy as np
import pandas as pd
import pytest

from hwt.config import load_config
from hwt.labels import KEEP, build_index, labels_for_horizon

CFG = load_config("configs/base.yaml")
FR = 25


def track(n, event_at=None, direction=+1, tid=1):
    """Straight track along +x at 30 m/s; a lane change at `event_at` shifts y by direction*3.7 m."""
    f = np.arange(n)
    y = np.zeros(n)
    lc = np.zeros(n, bool)
    if event_at is not None:
        ramp = np.clip((f - (event_at - 12)) / 24.0, 0, 1)          # 1 s lane change centred on the event
        y = direction * 3.7 * ramp
        lc[event_at] = True
    return pd.DataFrame({"track_id": tid, "frame": f, "x": f * 30 / FR, "y": y, "lat_pos": y, "heading": 0.0,
                         "lc_flag": lc})


def idx(df, **over):
    cfg = CFG.model_copy(update={"data": CFG.data.model_copy(update={
        "sample_stride_frames": 1, "max_samples_per_recording": 10**9, **over})})
    return build_index(df, cfg, np.random.default_rng(0))


def test_left_event_labelled_with_time_to_event():
    ix = idx(track(400, event_at=200, direction=+1))
    r = ix[ix.row == 200 - 50]
    assert r.ev.iloc[0] == 1 and r.ttl.iloc[0] == pytest.approx(2.0, abs=0.1)
    assert labels_for_horizon(r.ev.to_numpy(), r.ttl.to_numpy(), 2.2)[0] == 1
    assert labels_for_horizon(r.ev.to_numpy(), r.ttl.to_numpy(), 1.0)[0] == KEEP


def test_event_is_detected_for_either_lateral_direction():
    left = idx(track(400, event_at=200, direction=+1))
    right = idx(track(400, event_at=200, direction=-1))
    assert left.ev.sum() == right.ev.sum() > 0


def test_lateral_geometry_ignores_heading():
    df = track(400, event_at=200)
    df["heading"] = 0.3 * np.sin(df.frame / 7.0)     # garbage heading must not change labels
    assert idx(df).equals(idx(track(400, event_at=200)))


def test_event_beyond_longest_horizon_is_keep():
    ix = idx(track(400, event_at=200))
    r = ix[ix.row == 200 - 100]          # 4 s before the event, hmax is 3 s
    assert r.ev.iloc[0] == KEEP and np.isnan(r.ttl.iloc[0])


def test_anchors_right_after_event_are_dropped():
    ix = idx(track(400, event_at=200))
    assert not ix.row.between(205, 200 + 55).any()      # excluded for exclude_after_event_s (2.5 s) after the midpoint


def test_short_future_is_censored_not_keep():
    ix = idx(track(400))
    assert ix.row.max() <= 400 - 75 - 1          # no anchor whose full horizon is outside the track


def test_track_too_short_gives_no_samples():
    assert len(idx(track(100))) == 0


def test_labels_do_not_depend_on_row_order_within_cap():
    df = track(400, event_at=200)
    a = idx(df, max_samples_per_recording=50)
    b = idx(df, max_samples_per_recording=50)
    assert a.equals(b)                              # seeded subsample is deterministic


def ramped(n, start, dur, direction=+1, flag_at=None):
    """Smooth lane change: y moves 3.7 m over `dur` frames from `start`; flag placed at `flag_at`."""
    df = track(n)
    ramp = np.clip((np.arange(n) - start) / dur, 0, 1)
    df["lat_pos"] = direction * 3.7 * (3 * ramp**2 - 2 * ramp**3)
    df["lc_flag"] = False
    df.loc[flag_at if flag_at is not None else start, "lc_flag"] = True
    return df


def test_flag_at_maneuver_start_is_retimed_to_crossing_midpoint():
    # exiD-like: flag when the move starts (frame 150); half the displacement is covered at frame 150+50
    ix = idx(ramped(500, start=150, dur=100, flag_at=150))
    r = ix[ix.row == 200 - 25]                      # 1.0 s before the midpoint (frame 200)
    assert r.ev.iloc[0] == 1 and r.ttl.iloc[0] == pytest.approx(1.0, abs=0.05)


def test_flag_without_lateral_motion_is_ambiguous_and_removes_nearby_anchors():
    df = track(500)
    df.loc[250, "lc_flag"] = True                   # flag, but the car never moves sideways
    ix = idx(df)
    assert not ix.row.between(250 - 75, 250 + 62).any()      # nothing labelled keep near the unresolved flag
    assert (ix.ev == KEEP).all() and len(ix) > 0


def test_physical_event_threshold_is_configurable():
    df = ramped(500, start=150, dur=20)
    df["lat_pos"] = df["lat_pos"] * (1.5 / 3.7)                 # only 1.5 m of movement
    assert (idx(df).ev == KEEP).all()
    assert (idx(df, min_lateral_disp_m=1.0).ev > 0).any()
