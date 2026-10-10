import numpy as np

from hwt.conformal import evaluate, thresholds


def synth(n, rng, shift=0.0):
    y = (rng.random(n) < 0.1).astype(int)
    logit = np.where(y == 1, 1.5, -1.5) + rng.normal(0, 1.2, n) + shift
    return 1 / (1 + np.exp(-logit)), y


def test_class_conditional_coverage_is_near_nominal_when_exchangeable():
    rng = np.random.default_rng(0)
    pc, yc = synth(40000, rng)
    pt, yt = synth(40000, rng)
    q0, q1 = thresholds(pc, yc, 0.1, 30)
    m = evaluate(q0, q1, pt, yt)
    assert abs(m["cov_pos"] - 0.9) < 0.03 and abs(m["cov_neg"] - 0.9) < 0.01


def test_coverage_drops_under_shift_and_recalibration_restores_it():
    rng = np.random.default_rng(1)
    pc, yc = synth(40000, rng)
    q0, q1 = thresholds(pc, yc, 0.1, 30)
    ps, ys = synth(40000, rng, shift=-1.5)          # test scores shifted: guarantee no longer applies
    assert evaluate(q0, q1, ps, ys)["cov_pos"] < 0.8
    pr, yr = synth(40000, rng, shift=-1.5)           # recalibrate on shifted data
    r0, r1 = thresholds(pr, yr, 0.1, 30)
    assert abs(evaluate(r0, r1, ps, ys)["cov_pos"] - 0.9) < 0.03


def test_too_few_events_gives_inclusive_threshold_not_a_guess():
    _q0, q1 = thresholds(np.array([0.1, 0.2, 0.9]), np.array([0, 0, 1]), 0.1, 30)
    assert q1 == np.inf
