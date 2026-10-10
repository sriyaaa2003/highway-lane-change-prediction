from pathlib import Path

import numpy as np
import pytest

from hwt.energy import battery_power_w, load

P, ST = load(Path("configs/energy.yaml"))


def test_constant_speed_is_drag_plus_rolling_over_efficiency_plus_aux():
    v = np.array([30.0])
    wheel = 0.5 * P.rho_air * P.cda_m2 * 30.0**3 + P.crr * P.mass_kg * P.g * 30.0
    assert battery_power_w(v, np.array([0.0]), P)[0] == pytest.approx(wheel / P.eta_drive + P.aux_w)


def test_standstill_draws_only_the_auxiliary_load():
    assert battery_power_w(np.array([0.0]), np.array([0.0]), P)[0] == pytest.approx(P.aux_w)


def test_hard_braking_recovery_is_capped():
    p = battery_power_w(np.array([30.0]), np.array([-8.0]), P)[0]
    assert p == pytest.approx(-P.regen_cap_w * P.eta_regen + P.aux_w)


def test_gentle_braking_recovers_energy_but_less_than_it_cost_to_accelerate():
    up, down = battery_power_w(np.array([20.0]), np.array([1.0]), P)[0], battery_power_w(np.array([20.0]), np.array([-1.0]), P)[0]
    assert down < battery_power_w(np.array([20.0]), np.array([0.0]), P)[0] < up
    assert (up - P.aux_w) > abs(down - P.aux_w)


def test_energy_rises_with_speed_at_constant_speed():
    e = battery_power_w(np.array([20.0, 30.0, 40.0]), np.zeros(3), P) / np.array([20.0, 30.0, 40.0])
    assert np.all(np.diff(e) > 0)
