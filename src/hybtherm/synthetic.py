"""Synthetic probe logs: what a DS18B20 logger would record from a simulated block.

Purpose: exercise the whole analysis pipeline end to end before any hardware
exists, and check that the pipeline recovers known truths (probe offsets,
time constants, spread).

Sensor model, per probe:
  1. first-order thermal lag (probe housing + contact), time constant tau_s
  2. fixed offset (the probe's calibration error)
  3. white noise
  4. quantisation to the 12-bit DS18B20 resolution of 0.0625 degC

Synthetic data is labelled as such in every file it writes. It is NOT a
measurement and must never be reported as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import SimResult

DS18B20_LSB = 0.0625  # degC at 12-bit resolution


@dataclass
class SensorModel:
    offsets: dict[str, float] = field(
        default_factory=lambda: {"A1": +0.31, "D6": -0.22, "A6": +0.12, "H12": -0.40, "ambient": +0.19}
    )
    noise_std: float = 0.03  # degC
    lag_tau: float = 10.0  # s
    quantise: bool = True
    seed: int = 7


def _lag(t: np.ndarray, x: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        return x.copy()
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        a = 1.0 - np.exp(-(t[i] - t[i - 1]) / tau)
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def _read(true: np.ndarray, t: np.ndarray, offset: float, sm: SensorModel, rng) -> np.ndarray:
    y = _lag(t, true, sm.lag_tau) + offset + rng.normal(0.0, sm.noise_std, size=true.shape)
    if sm.quantise:
        y = np.round(y / DS18B20_LSB) * DS18B20_LSB
    return y


def probe_log_from_sim(result: SimResult, probe_wells: list[str], sm: SensorModel | None = None):
    """Return (t, temps, labels) for the chosen wells plus an 'ambient' probe."""
    sm = sm or SensorModel()
    rng = np.random.default_rng(sm.seed)
    t = result.t
    cols, labels = [], []
    for w in probe_wells:
        true = result.wells[:, result.well_names.index(w)]
        cols.append(_read(true, t, sm.offsets.get(w, 0.0), sm, rng))
        labels.append(w)
    amb_true = np.full_like(t, result.config.boundaries.t_ambient)
    cols.append(_read(amb_true, t, sm.offsets.get("ambient", 0.0), sm, rng))
    labels.append("ambient")
    return t, np.column_stack(cols), labels


def calibration_bath_log(
    labels: list[str],
    sm: SensorModel | None = None,
    duration: float = 900.0,
    t_start: float = 64.8,
    drift_per_s: float = -0.002,
    dt: float = 1.0,
):
    """All probes in one well-stirred insulated bath that cools slowly."""
    sm = sm or SensorModel()
    rng = np.random.default_rng(sm.seed + 1)
    t = np.arange(0.0, duration + dt / 2, dt)
    true = t_start + drift_per_s * t
    cols = [_read(true, t, sm.offsets.get(lab, 0.0), sm, rng) for lab in labels]
    return t, np.column_stack(cols), true
