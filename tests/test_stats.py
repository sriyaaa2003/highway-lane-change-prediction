import numpy as np
from sklearn.metrics import average_precision_score

from hwt.stats import Cluster, holm, paired_p, weighted_ap


def data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    s = rng.normal(0, 1, n) + 1.5 * y
    rec = rng.integers(0, 20, n)
    return y, s, rec


def test_weighted_ap_matches_sklearn_with_unit_weights():
    y, s, _ = data()
    o = np.argsort(-s, kind="stable")
    assert abs(weighted_ap(y[o].astype(float), np.ones(len(y))) - average_precision_score(y, s)) < 1e-9


def test_integer_weights_equal_duplicated_rows():
    y, s, _ = data(500)
    w = np.random.default_rng(1).integers(0, 4, len(y))
    o = np.argsort(-s, kind="stable")
    got = weighted_ap(y[o].astype(float), w[o].astype(float))
    rep = np.repeat(np.arange(len(y)), w)
    assert abs(got - average_precision_score(y[rep], s[rep])) < 1e-9


def test_cluster_bootstrap_ci_contains_point_and_is_narrower_with_more_recordings():
    y, s, rec = data()
    c = Cluster(y, rec)
    reps = c.replicates(s, c.draws(300, 0))
    assert np.quantile(reps, 0.025) < c.point(s) < np.quantile(reps, 0.975)
    y2, s2, rec2 = data(40000)
    c2 = Cluster(y2, rec2 * 10 + np.arange(len(rec2)) % 10)       # 200 recordings
    r2 = c2.replicates(s2, c2.draws(300, 0))
    assert np.subtract(*np.quantile(r2, [0.975, 0.025])) < np.subtract(*np.quantile(reps, [0.975, 0.025]))


def test_paired_p_and_holm():
    assert paired_p(np.full(1000, 0.05)) < 0.01
    assert paired_p(np.random.default_rng(0).normal(0, 1, 1000)) > 0.2
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and adj[2] == 0.06 and adj[1] == 0.06 and all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03]))
