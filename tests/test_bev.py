import numpy as np

from hwt.bev import grid_shape, rasterize
from hwt.config import load_config

CFG = load_config("configs/base.yaml")


def one(dx, dy, dvl=0.0, dvt=0.0, length=4.0, width=1.8):
    a = lambda v: np.array([v], dtype=float)
    return rasterize(np.array([0]), a(dx), a(dy), a(dvl), a(dvt), a(length), a(width), 1, CFG)[0]


def test_grid_shape_from_config():
    assert grid_shape(CFG) == (90, 32)


def test_car_ten_metres_ahead_in_own_lane_lands_in_expected_cells():
    img = one(10.0, 0.0)
    rows, cols = np.nonzero(img[0])
    assert rows.min() == 30 + 10 - 2 and rows.max() == 30 + 10 + 2       # behind_m=30, 1 m cells, 4 m long
    assert cols.min() <= 16 <= cols.max()                                 # centred laterally


def test_left_neighbour_is_on_the_left_half_and_mirror_flips_columns():
    left, right = one(5.0, 3.7), one(5.0, -3.7)
    assert np.nonzero(left[0])[1].min() > 16 and np.nonzero(right[0])[1].max() < 16
    assert np.array_equal(left[0][:, ::-1], right[0])


def test_relative_speed_channels_encode_sign():
    img = one(10.0, 0.0, dvl=10.0, dvt=-10.0)
    r, c = np.nonzero(img[0])
    assert img[1][r[0], c[0]] > 128 and img[2][r[0], c[0]] < 128


def test_vehicle_outside_window_is_ignored():
    assert one(100.0, 0.0)[0].sum() == 0 and one(-50.0, 0.0)[0].sum() == 0
