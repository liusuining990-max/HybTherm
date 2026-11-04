import numpy as np
import pytest

from heatblock.analysis import apply_offsets, calibration_offsets, fit_first_order, uniformity_metrics
from heatblock.synthetic import SensorModel, calibration_bath_log


def test_first_order_fit_recovers_tau():
    rng = np.random.default_rng(0)
    t = np.arange(0, 3600, 1.0)
    true = 22.0 + (65.0 - 22.0) * np.exp(-t / 2100.0)
    y = true + rng.normal(0, 0.03, t.size)
    y = np.round(y / 0.0625) * 0.0625
    fit = fit_first_order(t, y)
    assert fit.tau == pytest.approx(2100.0, rel=0.02)
    assert fit.t_inf == pytest.approx(22.0, abs=0.5)
    assert fit.r2 > 0.999


def test_calibration_recovers_relative_offsets():
    sm = SensorModel(offsets={"p1": 0.3, "p2": -0.2, "p3": 0.1, "p4": -0.4}, lag_tau=10.0)
    labels = ["p1", "p2", "p3", "p4"]
    t, x, _ = calibration_bath_log(labels, sm, duration=900)
    cal = calibration_offsets(x, labels, settle_samples=60)
    true = np.array([0.3, -0.2, 0.1, -0.4])
    np.testing.assert_allclose(cal.offsets, true - true.mean(), atol=0.02)
    assert abs(cal.offsets.sum()) < 1e-9  # probe-mean reference


def test_calibration_with_reference_probe():
    sm = SensorModel(offsets={"ref": 0.0, "p1": 0.25, "p2": -0.15}, noise_std=0.01, quantise=False)
    labels = ["ref", "p1", "p2"]
    _, x, _ = calibration_bath_log(labels, sm)
    cal = calibration_offsets(x, labels, reference="ref", settle_samples=60)
    np.testing.assert_allclose(cal.offsets, [0.0, 0.25, -0.15], atol=0.01)
    corrected = apply_offsets(x, labels, cal)
    assert np.abs(corrected[60:, 1] - corrected[60:, 0]).mean() < 0.02


def test_uniformity_metrics_on_constructed_run():
    t = np.arange(0, 1000, 1.0)
    rise = 1 - np.exp(-t / 60.0)
    a = 22 + 43.0 * rise  # ends at 65.0
    b = 22 + 42.6 * rise  # ends at 64.6
    x = np.column_stack([a, b])
    m = uniformity_metrics(t, x, ["hot", "cold"], setpoint=65.0, t_end_hold=1000, window=200)
    assert m.steady_spread_degC == pytest.approx(0.4, abs=1e-3)
    assert m.coldest == "cold" and m.hottest == "hot"
    assert m.steady_bias_degC == pytest.approx(-0.2, abs=1e-3)
    # mean reaches 64.5 when 22 + 42.8(1 - e^{-t/60}) = 64.5  ->  t = 297.8 s
    assert m.t_first_reach_s == pytest.approx(298.0)
    # coldest probe enters the +/-1 band (>= 64.0) at t = -60 ln(1 - 42/42.6) = 255.8 s
    assert m.t_ready_s == pytest.approx(256.0)
    # spread never exceeds 0.4, so the block is "uniform" from the start
    assert m.t_uniform_s == 0.0


def test_smoothing_stops_noise_deciding_band_times():
    rng = np.random.default_rng(1)
    t = np.arange(0, 1200, 1.0)
    a = np.full(t.size, 65.0) + rng.normal(0, 0.05, t.size)
    b = np.full(t.size, 64.55) + rng.normal(0, 0.05, t.size)  # true spread 0.45 < 0.5
    x = np.column_stack([a, b])
    raw = uniformity_metrics(t, x, ["a", "b"], 65.0)
    smooth = uniformity_metrics(t, x, ["a", "b"], 65.0, smooth_samples=31)
    assert raw.t_uniform_s is None or raw.t_uniform_s > 600  # noise pushes spread over 0.5
    assert smooth.t_uniform_s is not None and smooth.t_uniform_s < 60
