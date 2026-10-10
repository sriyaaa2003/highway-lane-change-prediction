import pandas as pd
import pytest

from hwt.config import load_config
from hwt.splits import HOLDOUT, assign, check_integrity

CFG = load_config("configs/base.yaml")


def manifest():
    rows = [("highd", r, 1 + r % 6) for r in range(1, 61)] + [("exid", r, r % 7) for r in range(93)]
    return pd.DataFrame(rows, columns=["dataset", "recording_id", "location_id"])


def with_holdout(h, e):
    s = CFG.splits.model_copy(update={"highd_holdout_locations": h, "exid_holdout_locations": e})
    return CFG.model_copy(update={"splits": s})


def test_deterministic():
    a, b = assign(manifest(), CFG), assign(manifest(), CFG)
    assert a.split.tolist() == b.split.tolist()


def test_holdout_location_is_fully_held_out_and_integrity_passes():
    cfg = with_holdout([3], [2, 5])
    m = assign(manifest(), cfg)
    check_integrity(m)
    assert set(m[m.split == HOLDOUT].location_id[m.dataset == "highd"]) == {3}
    assert not ((m.split != HOLDOUT) & (m.dataset == "exid") & m.location_id.isin([2, 5])).any()


def test_all_splits_populated():
    assert set(assign(manifest(), with_holdout([], [])).split) == {"train", "val", "cal", "test"}


def test_integrity_detects_leak():
    m = assign(manifest(), with_holdout([3], []))
    m.loc[(m.dataset == "highd") & (m.location_id == 3), "split"] = "train"
    m.loc[m.index[0], "split"] = HOLDOUT
    m.loc[m.index[0], "location_id"] = 3
    m.loc[m.index[0], "dataset"] = "highd"
    with pytest.raises(AssertionError):
        check_integrity(m)
