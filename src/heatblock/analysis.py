"""Analysis of multi-probe temperature logs.

Works the same on simulated, synthetic and real logs. A log is a time vector
``t`` (s) and a 2-D array ``temps`` (n_samples, n_probes) in degC, plus the
probe labels.

Functions
---------
calibration_offsets   relative calibration from a run where all probes sit
                      in the same isothermal bath
apply_offsets         subtract the offsets from a log
fit_first_order       fit T(t) = T_inf + (T0 - T_inf) exp(-(t - t0)/tau)
uniformity_metrics    figures of merit for a heat-up/hold run
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    labels: list[str]
    offsets: np.ndarray  # degC, reading - reference
    offset_sem: np.ndarray  # degC, standard error of each offset
    residual_std: np.ndarray  # degC, scatter of corrected readings about the reference
    reference: str  # "probe-mean" or the label of the reference probe

    def as_dict(self) -> dict:
        return {
            "reference": self.reference,
            "offsets_degC": dict(zip(self.labels, np.round(self.offsets, 4).tolist())),
            "offset_sem_degC": dict(zip(self.labels, np.round(self.offset_sem, 4).tolist())),
            "residual_std_degC": dict(zip(self.labels, np.round(self.residual_std, 4).tolist())),
        }


def calibration_offsets(
    temps: np.ndarray,
    labels: list[str],
    reference: str | None = None,
    settle_samples: int = 0,
) -> Calibration:
    """Relative calibration from an isothermal-bath run.

    Every probe sees the same true temperature at each instant, so the
    difference between a probe and the reference is its offset. The reference
    is either one named probe (e.g. a calibrated thermometer logged as a
    column) or, by default, the mean of all probes. With the probe mean as
    reference the offsets sum to zero: this removes probe-to-probe differences
    (what matters for uniformity) but not the common absolute error.
    """
    x = np.asarray(temps, dtype=float)[settle_samples:]
    if x.ndim != 2 or x.shape[1] != len(labels):
        raise ValueError("temps must be (n_samples, n_probes) matching labels")
    if reference is None:
        ref = x.mean(axis=1)
        ref_name = "probe-mean"
    else:
        ref = x[:, labels.index(reference)]
        ref_name = reference
    diff = x - ref[:, None]
    offsets = diff.mean(axis=0)
    n = diff.shape[0]
    resid = diff - offsets
    std = resid.std(axis=0, ddof=1) if n > 1 else np.zeros(len(labels))
    # Samples are autocorrelated (slow drift, quantisation), so the naive
    # std/sqrt(n) underestimates the error. Use an effective sample size based
    # on the lag-1 autocorrelation as a pragmatic correction.
    sem = np.empty(len(labels))
    for p in range(len(labels)):
        r = resid[:, p]
        if n > 2 and r.std() > 0:
            rho = np.corrcoef(r[:-1], r[1:])[0, 1]
            rho = float(np.clip(rho, 0.0, 0.99))
        else:
            rho = 0.0
        n_eff = max(1.0, n * (1 - rho) / (1 + rho))
        sem[p] = std[p] / np.sqrt(n_eff)
    return Calibration(list(labels), offsets, sem, std, ref_name)


def apply_offsets(temps: np.ndarray, labels: list[str], cal: Calibration) -> np.ndarray:
    """Return temps with each probe's offset subtracted (probes matched by label)."""
    out = np.array(temps, dtype=float, copy=True)
    for p, lab in enumerate(labels):
        if lab in cal.labels:
            out[:, p] -= cal.offsets[cal.labels.index(lab)]
    return out


# ---------------------------------------------------------------------------
# First-order fit
# ---------------------------------------------------------------------------


@dataclass
class FirstOrderFit:
    t_inf: float
    t0_value: float
    tau: float
    tau_stderr: float
    r2: float
    rmse: float


def _first_order(t, t_inf, t0_value, tau):
    return t_inf + (t0_value - t_inf) * np.exp(-t / tau)


def fit_first_order(t: np.ndarray, temp: np.ndarray, tau_guess: float | None = None) -> FirstOrderFit:
    """Fit T(t) = T_inf + (T0 - T_inf) exp(-(t - t[0]) / tau).

    This is the solution of the lumped model C dT/dt = UA (T_inf - T), i.e. the
    first-order linear ODE dT/dt + T/tau = T_inf/tau with tau = C/UA.
    Use it on a cool-down (heater off) or an open-loop constant-power step,
    not on a controlled heat-up, where the heater power is not constant.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(temp, dtype=float)
    ts = t - t[0]
    span = ts[-1] if ts[-1] > 0 else 1.0
    if tau_guess is None:
        tau_guess = span / 2.0
    p0 = (y[-1] - 0.1 * (y[0] - y[-1]), y[0], tau_guess)
    popt, pcov = curve_fit(_first_order, ts, y, p0=p0, maxfev=20000)
    pred = _first_order(ts, *popt)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    perr = np.sqrt(np.diag(pcov))
    return FirstOrderFit(
        t_inf=float(popt[0]),
        t0_value=float(popt[1]),
        tau=float(popt[2]),
        tau_stderr=float(perr[2]),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        rmse=float(np.sqrt(ss_res / len(y))),
    )


# ---------------------------------------------------------------------------
# Uniformity metrics for a heat-up / hold run
# ---------------------------------------------------------------------------


@dataclass
class UniformityMetrics:
    setpoint: float
    t_first_reach_s: float | None  # first time the probe mean reaches setpoint - band
    t_ready_s: float | None  # from then on every probe stays within +/- ready_band of setpoint
    t_uniform_s: float | None  # from then on max-min across probes stays <= uniform_band
    overshoot_degC: float  # max probe-mean temperature above setpoint during the run
    max_transient_spread_degC: float
    steady_spread_degC: float  # max - min of per-probe means over the steady window
    steady_bias_degC: float  # mean of per-probe means minus setpoint
    steady_between_std_degC: float
    steady_max_fluct_degC: float  # largest per-probe temporal std in the steady window
    coldest: str
    hottest: str
    window_s: tuple[float, float]

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def _stays_true_from(t: np.ndarray, ok: np.ndarray) -> float | None:
    """Earliest time after which ``ok`` is True for every remaining sample."""
    if not ok[-1]:
        return None
    bad = np.where(~ok)[0]
    if len(bad) == 0:
        return float(t[0])
    return float(t[bad[-1] + 1])


def uniformity_metrics(
    t: np.ndarray,
    temps: np.ndarray,
    labels: list[str],
    setpoint: float,
    t_end_hold: float | None = None,
    window: float = 300.0,
    ready_band: float = 1.0,
    uniform_band: float = 0.5,
    reach_band: float = 0.5,
    smooth_samples: int = 1,
) -> UniformityMetrics:
    """Figures of merit for a controlled heat-up and hold.

    ``temps`` holds only the probes that sit in the block (not the ambient
    probe). Only samples with t <= t_end_hold are used. The steady window is the
    last ``window`` seconds before t_end_hold. For measured data, pass
    ``smooth_samples`` (e.g. 30 at 1 Hz) so that noise does not decide the
    band-crossing times; the temporal fluctuation metric is then also smoothed.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(temps, dtype=float)
    if t_end_hold is None:
        t_end_hold = float(t[-1])
    sel = t <= t_end_hold + 1e-9
    t, x = t[sel], x[sel]
    if smooth_samples > 1:
        # centred moving average per probe, so sensor noise and quantisation
        # steps do not trigger the band tests; edges use a shrinking window
        k = np.ones(smooth_samples)
        num = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x)
        den = np.convolve(np.ones(len(t)), k, mode="same")[:, None]
        x = num / den

    mean = x.mean(axis=1)
    spread = x.max(axis=1) - x.min(axis=1)

    reach = np.where(mean >= setpoint - reach_band)[0]
    t_reach = float(t[reach[0]]) if len(reach) else None

    in_band = np.all(np.abs(x - setpoint) <= ready_band, axis=1)
    t_ready = _stays_true_from(t, in_band)
    t_uniform = _stays_true_from(t, spread <= uniform_band)

    w = t >= t_end_hold - window
    per_probe = x[w].mean(axis=0)
    return UniformityMetrics(
        setpoint=setpoint,
        t_first_reach_s=t_reach,
        t_ready_s=t_ready,
        t_uniform_s=t_uniform,
        overshoot_degC=float(max(0.0, mean.max() - setpoint)),
        max_transient_spread_degC=float(spread.max()),
        steady_spread_degC=float(per_probe.max() - per_probe.min()),
        steady_bias_degC=float(per_probe.mean() - setpoint),
        steady_between_std_degC=float(per_probe.std(ddof=0)),
        steady_max_fluct_degC=float(x[w].std(axis=0).max()),
        coldest=labels[int(np.argmin(per_probe))],
        hottest=labels[int(np.argmax(per_probe))],
        window_s=(float(t_end_hold - window), float(t_end_hold)),
    )
