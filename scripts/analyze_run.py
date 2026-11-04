"""Analyse a probe log: calibration, uniformity metrics, cool-down time constants.

    python scripts/analyze_run.py \
        --run data/synthetic/baseline_run.csv \
        --cal data/synthetic/calibration_bath.csv \
        --block-probes A1,A6,D6,H12 --setpoint 65 --t-off 1800 \
        --out results/analysis_synthetic

The same command works on a real log recorded with ``heatblock-log``.
Outputs: report.json and two figures in the --out folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatblock.analysis import apply_offsets, calibration_offsets, fit_first_order, uniformity_metrics  # noqa: E402
from heatblock.io import read_log, select  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--cal", type=Path, default=None, help="isothermal-bath log for relative calibration")
    ap.add_argument("--cal-settle", type=int, default=60, help="samples to skip at the start of the bath log")
    ap.add_argument("--block-probes", required=True, help="labels of probes in the block, comma-separated")
    ap.add_argument("--setpoint", type=float, required=True)
    ap.add_argument("--t-off", type=float, required=True, help="time the heater was switched off, s")
    ap.add_argument("--smooth", type=int, default=30, help="moving-average window in samples for band-crossing times")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    t, x, labels = read_log(args.run)
    block = [s.strip() for s in args.block_probes.split(",")]
    report = {"run": str(args.run), "block_probes": block}

    if args.cal:
        _, xc, lc = read_log(args.cal)
        cal = calibration_offsets(xc, lc, settle_samples=args.cal_settle)
        report["calibration"] = cal.as_dict()
        x_corr = apply_offsets(x, labels, cal)
    else:
        report["calibration"] = None
        x_corr = x

    xb_raw = select(x, labels, block)
    xb = select(x_corr, labels, block)
    m_raw = uniformity_metrics(t, xb_raw, block, args.setpoint, t_end_hold=args.t_off, smooth_samples=args.smooth)
    m = uniformity_metrics(t, xb, block, args.setpoint, t_end_hold=args.t_off, smooth_samples=args.smooth)
    report["uniformity_uncorrected"] = m_raw.as_dict()
    report["uniformity_corrected"] = m.as_dict()

    cool = t >= args.t_off
    fits = {}
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150)
    for p, lab in enumerate(block):
        f = fit_first_order(t[cool], xb[cool, p], tau_guess=2000.0)
        fits[lab] = {"tau_s": round(f.tau, 1), "tau_stderr_s": round(f.tau_stderr, 1), "T_inf_degC": round(f.t_inf, 2), "r2": round(f.r2, 6)}
        tt = t[cool] - t[cool][0]
        ax.plot(t[cool], xb[cool, p], lw=0.8, label=f"{lab}: tau = {f.tau:.0f} s")
        ax.plot(t[cool], f.t_inf + (f.t0_value - f.t_inf) * np.exp(-tt / f.tau), "k:", lw=0.8)
    report["cooldown_first_order_fit"] = fits
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Corrected temperature, degC")
    ax.set_title("Cool-down with first-order fits (dotted)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "cooldown_fit.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150)
    hold = t <= args.t_off
    for p, lab in enumerate(block):
        ax.plot(t[hold], xb[hold, p], lw=0.8, label=lab)
    ax.axhline(args.setpoint, color="grey", lw=0.6)
    ax.set_ylim(args.setpoint - 2, args.setpoint + 1)
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Corrected temperature, degC")
    ax.set_title("Hold phase after calibration correction", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "hold_corrected.png")
    plt.close(fig)

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
