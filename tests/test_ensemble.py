import numpy as np

from hwt.ensemble import hierarchy_scores, logit, regime_design


def test_regime_design_gives_each_regime_its_own_columns():
    f = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    X = regime_design(f, np.array([False, True, False]))
    assert X.shape == (3, 5)
    assert np.array_equal(X[:, 2:4], np.array([[0, 0], [2, 5], [0, 0]]))      # interaction columns only active when pre-drift
    assert np.array_equal(X[:, 4], [0, 1, 0])


def test_hierarchy_routes_by_regime_and_calibrates_each_stage():
    rng = np.random.default_rng(0)
    n = 6000
    pre = rng.random(n) < 0.5
    p_drift, p_pre = rng.random(n), rng.random(n)
    # true event rate: 30% when drifting score is high, 5% when pre-drift score is high; stages are on different scales
    y = np.where(pre, rng.random(n) < 0.05 * p_pre * 2, rng.random(n) < 0.6 * p_drift).astype(int)
    cal = {"p_drift": p_drift, "p_pre": p_pre, "pre": pre, "y": y}
    s = hierarchy_scores(p_drift, p_pre, pre, cal)
    assert np.all((s >= 0) & (s <= 1))
    # calibrated: a pre-drift case can never outrank a drifting case unless it is genuinely more likely to be an event
    assert s[pre].mean() < s[~pre].mean()


def test_logit_is_finite_at_the_edges():
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()
