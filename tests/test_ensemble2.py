import numpy as np
import pytest

from hwt.ensemble2 import VARIANTS, cal_cv_score


def synth(n, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    noise = lambda s: np.clip(0.04 + 0.5 * y * rng.random(n) + s * rng.random(n), 1e-3, 1 - 1e-3)
    d1 = np.where(rng.random(n) < 0.5, rng.random(n) * 0.3, 0.3 + rng.random(n))
    D = {"y": y, "rec": rng.integers(0, 12, n), "rule": 0.1 + y * rng.random(n), "d1": d1, "pre": d1 < 0.3, "cnn": noise(0.3)}
    for k in ("gen", "spec0.15", "spec0.3", "spec0.6", "et", "lr", "res"):
        D[k] = noise(0.25)
    return D


@pytest.mark.parametrize("name", list(VARIANTS))
def test_every_variant_fits_and_predicts_finite_scores(name):
    C, T = synth(3000, 0), synth(1500, 1)
    s = VARIANTS[name](C)(T)
    assert s.shape == (1500,) and np.isfinite(s).all()


def test_informative_members_beat_chance_in_cal_cv():
    C = synth(4000, 2)
    assert cal_cv_score(VARIANTS["lr_flat"], C, 0.3) > 0.08 * 2
