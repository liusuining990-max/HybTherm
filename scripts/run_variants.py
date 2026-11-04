"""Run the baseline block and the design variants, save results and figures.

    python scripts/run_variants.py            # full run, dx = 2.5 mm (~5 min)
    python scripts/run_variants.py --quick    # dx = 5 mm, for a fast look

Outputs
  results/variants.json           metrics for every variant
  results/<variant>.npz           time series (all 96 wells, control, power)
  docs/figures/*.png              figures used in docs/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatblock.analysis import fit_first_order, uniformity_metrics  # noqa: E402
from heatblock.geometry import N_COLS, N_ROWS, ROWS  # noqa: E402
from heatblock.model import Boundaries, Heater, SimConfig, lumped_parameters, simulate  # noqa: E402

PROBES = ["A1", "A6", "D6", "H12"]
FIG = ROOT / "docs" / "figures"
RES = ROOT / "results"

BASE_B = Boundaries()
BASE_H = Heater()

VARIANTS = {
    "baseline": dict(desc="Centre pad 60x40 mm, open top (h=10), bare sides (h=10)"),
    "full_pad": dict(desc="Heater pad enlarged to 110x70 mm", heater=Heater(pad_x=0.110, pad_y=0.070)),
    "insulated_sides": dict(desc="10 mm foam on the four sides (h_sides 10 -> 2.3)", boundaries=Boundaries(h_sides=2.3)),
    "lid": dict(desc="Insulated lid on top (h_top 10 -> 3)", boundaries=Boundaries(h_top=3.0)),
    "draft": dict(desc="Air draft from a fan or hood (h_top 25, h_sides 15)", boundaries=Boundaries(h_top=25.0, h_sides=15.0)),
    "combined": dict(
        desc="Full pad + insulated sides + lid",
        heater=Heater(pad_x=0.110, pad_y=0.070),
        boundaries=Boundaries(h_top=3.0, h_sides=2.3),
    ),
}


def config_for(name: str, dx: float) -> SimConfig:
    v = VARIANTS[name]
    return SimConfig(
        dx=dx,
        heater=v.get("heater", BASE_H),
        boundaries=v.get("boundaries", BASE_B),
    )


def analyse(res) -> dict:
    cfg = res.config
    names = res.well_names
    m_all = uniformity_metrics(res.t, res.wells, names, cfg.controller.setpoint, t_end_hold=cfg.t_heat)
    idx = [names.index(p) for p in PROBES]
    m_probe = uniformity_metrics(res.t, res.wells[:, idx], PROBES, cfg.controller.setpoint, t_end_hold=cfg.t_heat)
    cool = res.t >= cfg.t_heat
    taus = {}
    for p in PROBES:
        f = fit_first_order(res.t[cool], res.wells[cool, names.index(p)], tau_guess=2000.0)
        taus[p] = round(f.tau, 1)
    lp = lumped_parameters(cfg)
    heat = res.t <= cfg.t_heat
    w = res.t >= cfg.t_heat - 300
    return {
        "all_96_wells": m_all.as_dict(),
        "four_probes": m_probe.as_dict(),
        "cooldown_tau_s": taus,
        "lumped": {k: round(v, 6) for k, v in lp.items()},
        "control_overshoot_degC": round(float(res.control[heat].max() - cfg.controller.setpoint), 4),
        "steady_power_W": round(float(res.power[heat & w].mean()), 3),
    }


def well_map(res, window=300.0):
    cfg = res.config
    w = (res.t >= cfg.t_heat - window) & (res.t <= cfg.t_heat)
    return res.wells[w].mean(axis=0).reshape(N_ROWS, N_COLS)


def plot_well_map(res, title, path, vmin=None, vmax=None):
    m = well_map(res)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    im = ax.imshow(m, cmap="coolwarm", origin="upper", vmin=vmin, vmax=vmax)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            ax.text(c, r, f"{m[r, c]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_xticks(range(N_COLS), [str(c + 1) for c in range(N_COLS)])
    ax.set_yticks(range(N_ROWS), list(ROWS))
    for p in PROBES:
        r, c = ROWS.index(p[0]), int(p[1:]) - 1
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=1.6, ec="black"))
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("degC (mean of last 300 s of hold)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_timeseries(res, path):
    cfg = res.config
    names = res.well_names
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5.6), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    sel = res.t <= 900
    styles = {"A1": "-", "A6": "-", "D6": "-", "H12": ":"}
    for p in PROBES:
        lab = f"well {p}" + (" (= A1 by symmetry)" if p == "H12" else "")
        ax.plot(res.t[sel], res.wells[sel, names.index(p)], styles[p], label=lab, lw=1.6 if p == "H12" else 1.2)
    ax.plot(res.t[sel], res.control[sel], "k--", lw=1.0, label="control sensor")
    ax.axhline(cfg.controller.setpoint, color="grey", lw=0.6)
    ax.set_ylabel("Temperature, degC")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Baseline heat-up: four probe wells and the control sensor (model)", fontsize=10)
    spread = res.wells.max(axis=1) - res.wells.min(axis=1)
    ax2.plot(res.t[sel], spread[sel], color="tab:red", lw=1.2, label="spread, all 96 wells")
    ax2b = ax2.twinx()
    ax2b.plot(res.t[sel], res.power[sel], color="tab:grey", lw=0.8)
    ax2b.set_ylabel("Heater W", color="tab:grey", fontsize=8)
    ax2.set_ylabel("max-min, K")
    ax2.set_xlabel("Time, s")
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_section(res, path):
    cfg = res.config
    f = res.snapshot  # (nz, ny, nx)
    j = f.shape[1] // 2
    sec = f[:, j, :]
    L, H = cfg.geometry.length_x * 1e3, cfg.geometry.height_z * 1e3
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=150)
    im = ax.imshow(sec, origin="lower", extent=[0, L, 0, H], cmap="inferno", aspect="auto")
    pad = cfg.heater.pad_x * 1e3
    ax.plot([(L - pad) / 2, (L + pad) / 2], [0.3, 0.3], color="cyan", lw=3, label="heater pad")
    ax.set_xlabel("x, mm")
    ax.set_ylabel("z, mm")
    ax.legend(fontsize=7, loc="upper right")
    fig.colorbar(im, ax=ax, label="degC")
    ax.set_title("Vertical section through the block centre at the end of the hold (model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_variants(summary, path):
    names = list(summary)
    ss = [summary[n]["all_96_wells"]["steady_spread_degC"] for n in names]
    tr = [summary[n]["all_96_wells"]["max_transient_spread_degC"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=150)
    ax[0].bar(x, ss, color="tab:blue")
    ax[0].axhline(0.5, color="tab:red", ls="--", lw=0.8, label="DS18B20 stated accuracy, +/-0.5 degC")
    ax[0].set_xticks(x, names, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("K")
    ax[0].set_title("Steady spread, 96 wells", fontsize=10)
    ax[0].legend(fontsize=7)
    for xi, v in zip(x, ss):
        ax[0].text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax[1].bar(x, tr, color="tab:orange")
    ax[1].set_xticks(x, names, rotation=30, ha="right", fontsize=8)
    ax[1].set_title("Largest spread during heat-up", fontsize=10)
    for xi, v in zip(x, tr):
        ax[1].text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Design variants (model predictions)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def grid_study(dxs, t_heat=1200.0) -> list[dict]:
    rows = []
    for dx in dxs:
        cfg = SimConfig(dx=dx, t_heat=t_heat, t_end=t_heat, record_every=2.0)
        t0 = time.time()
        res = simulate(cfg)
        m = uniformity_metrics(res.t, res.wells, res.well_names, 65.0, t_end_hold=t_heat)
        rows.append(
            {
                "dx_mm": round(dx * 1e3, 3),
                "cells": int(res.snapshot.size),
                "steady_spread_degC": round(m.steady_spread_degC, 4),
                "max_transient_spread_degC": round(m.max_transient_spread_degC, 4),
                "coldest": m.coldest,
                "runtime_s": round(time.time() - t0, 1),
            }
        )
        print("grid", rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-grid", action="store_true")
    args = ap.parse_args()
    dx = 5e-3 if args.quick else 2.5e-3
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    summary = {}
    results = {}
    for name in VARIANTS:
        t0 = time.time()
        res = simulate(config_for(name, dx))
        results[name] = res
        summary[name] = {"description": VARIANTS[name]["desc"], **analyse(res)}
        summary[name]["runtime_s"] = round(time.time() - t0, 1)
        np.savez_compressed(
            RES / f"{name}.npz", t=res.t, wells=res.wells, control=res.control, power=res.power, snapshot=res.snapshot
        )
        print(name, json.dumps(summary[name]["all_96_wells"]), flush=True)

    out = {"dx_m": dx, "probe_wells": PROBES, "variants": summary}
    if not args.skip_grid and not args.quick:
        out["grid_study"] = grid_study([5e-3, 2.5e-3, 0.02 / 12])
    (RES / "variants.json").write_text(json.dumps(out, indent=2))

    base = results["baseline"]
    plot_well_map(base, "Baseline: well temperatures at steady hold, setpoint 65 degC (model; boxed = probe wells)", FIG / "wellmap_baseline.png")
    plot_well_map(results["draft"], "Air draft variant: well temperatures at steady hold (model)", FIG / "wellmap_draft.png")
    plot_well_map(results["combined"], "Combined improvements: well temperatures at steady hold (model)", FIG / "wellmap_combined.png")
    plot_timeseries(base, FIG / "timeseries_baseline.png")
    plot_section(base, FIG / "section_baseline.png")
    plot_variants(summary, FIG / "variants_spread.png")
    print("done")


if __name__ == "__main__":
    main()
