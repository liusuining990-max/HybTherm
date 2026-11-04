"""Transient 3-D heat conduction model of a heating block.

Method
------
Finite-volume discretisation on a uniform Cartesian grid of cubic cells, with
backward-Euler (implicit) time stepping. Implicit stepping is unconditionally
stable, so the time step is chosen for accuracy, not stability, and the system
matrix is factorised once and reused for every step.

Per cell *i* with heat capacity C_i = rho * c * V:

    C_i (T_i^{n+1} - T_i^n) / dt
        = sum_j G_ij (T_j^{n+1} - T_i^{n+1})        (conduction to neighbours)
        + G_b,i (T_amb - T_i^{n+1})                  (convection at outer faces)
        + Q_i^n                                      (heater power)

G_ij = k A / dx between neighbouring cells. For a cell face on the outside,
G_b = 1 / (1/(h A) + (dx/2)/(k A)), i.e. convection in series with conduction
across half a cell, so the surface temperature is not assumed equal to the
cell-centre temperature.

The heater is a flat pad on the bottom face. Its power is set every step by a
PI controller that reads one embedded control sensor.

Assumptions (see docs/REPORT.md for discussion):
* the block is solid; well holes and the plastic tubes/liquid in them are not
  resolved, and "well temperature" means block temperature at the well's
  read-out point;
* material properties are constant; radiation is folded into the effective
  heat-transfer coefficients h;
* contact resistance between heater pad and block is neglected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import BlockGeometry


@dataclass(frozen=True)
class Material:
    name: str = "Al 6061-T6"
    k: float = 167.0  # W/(m K)
    rho: float = 2700.0  # kg/m^3
    cp: float = 896.0  # J/(kg K)

    @property
    def alpha(self) -> float:
        """Thermal diffusivity, m^2/s."""
        return self.k / (self.rho * self.cp)


@dataclass(frozen=True)
class Boundaries:
    """Effective heat-transfer coefficients h, W/(m^2 K), including radiation."""

    h_top: float = 10.0
    h_sides: float = 10.0
    h_bottom: float = 5.0  # bottom face outside the heater pad
    h_under_pad: float = 2.0  # bottom face under the (insulated) heater pad
    t_ambient: float = 22.0  # degC


@dataclass(frozen=True)
class Heater:
    """Rectangular heater pad centred on the bottom face."""

    pad_x: float = 0.060  # m
    pad_y: float = 0.040  # m
    max_power: float = 80.0  # W


@dataclass(frozen=True)
class Controller:
    """PI controller on a single embedded sensor. Output is a duty fraction 0..1."""

    setpoint: float = 65.0  # degC
    kp: float = 0.25  # 1/K
    ki: float = 0.0015  # 1/(K s)
    sensor_xyz: tuple[float, float, float] | None = None  # m; None = block centre, 5 mm up


@dataclass(frozen=True)
class SimConfig:
    geometry: BlockGeometry = field(default_factory=BlockGeometry)
    material: Material = field(default_factory=Material)
    boundaries: Boundaries = field(default_factory=Boundaries)
    heater: Heater = field(default_factory=Heater)
    controller: Controller = field(default_factory=Controller)
    dx: float = 2.5e-3  # m, cell size
    dt: float = 0.5  # s
    t_heat: float = 1800.0  # s, controller active from 0 to t_heat
    t_end: float = 5400.0  # s, heater off from t_heat to t_end (cool-down)
    record_every: float = 1.0  # s
    t_initial: float | None = None  # degC; None = ambient

    def with_(self, **changes) -> "SimConfig":
        return replace(self, **changes)


@dataclass
class SimResult:
    config: SimConfig
    t: np.ndarray  # (n,) s
    wells: np.ndarray  # (n, 96) degC
    control: np.ndarray  # (n,) degC
    power: np.ndarray  # (n,) W
    mean: np.ndarray  # (n,) degC, volume-mean block temperature
    snapshot: np.ndarray  # (nz, ny, nx) field at end of heating phase
    well_names: list[str]


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    nz: int
    dx: float

    @property
    def n(self) -> int:
        return self.nx * self.ny * self.nz

    def idx(self, i, j, k):
        return i + self.nx * (j + self.ny * k)

    def centres(self):
        xs = (np.arange(self.nx) + 0.5) * self.dx
        ys = (np.arange(self.ny) + 0.5) * self.dx
        zs = (np.arange(self.nz) + 0.5) * self.dx
        return xs, ys, zs


def make_grid(geom: BlockGeometry, dx: float) -> Grid:
    nx = int(round(geom.length_x / dx))
    ny = int(round(geom.width_y / dx))
    nz = int(round(geom.height_z / dx))
    for n, L, name in ((nx, geom.length_x, "x"), (ny, geom.width_y, "y"), (nz, geom.height_z, "z")):
        if abs(n * dx - L) > 1e-9:
            raise ValueError(f"cell size {dx} does not divide block {name}-dimension {L}")
    return Grid(nx, ny, nz, dx)


def interpolation_matrix(grid: Grid, points: np.ndarray) -> sp.csr_matrix:
    """Sparse (m, n) matrix W so that W @ T gives trilinear values at ``points``.

    Points outside the cell-centre lattice are clamped to the nearest centre
    plane, which is equivalent to zero-gradient extrapolation over half a cell.
    """
    xs, ys, zs = grid.centres()
    rows, cols, vals = [], [], []
    for p_i, (x, y, z) in enumerate(points):
        fi = np.clip((x - xs[0]) / grid.dx, 0, grid.nx - 1 - 1e-12)
        fj = np.clip((y - ys[0]) / grid.dx, 0, grid.ny - 1 - 1e-12)
        fk = np.clip((z - zs[0]) / grid.dx, 0, grid.nz - 1 - 1e-12)
        i0, j0, k0 = int(fi), int(fj), int(fk)
        wx, wy, wz = fi - i0, fj - j0, fk - k0
        for di, wxi in ((0, 1 - wx), (1, wx)):
            for dj, wyj in ((0, 1 - wy), (1, wy)):
                for dk, wzk in ((0, 1 - wz), (1, wz)):
                    w = wxi * wyj * wzk
                    if w == 0.0:
                        continue
                    ii = min(i0 + di, grid.nx - 1)
                    jj = min(j0 + dj, grid.ny - 1)
                    kk = min(k0 + dk, grid.nz - 1)
                    rows.append(p_i)
                    cols.append(grid.idx(ii, jj, kk))
                    vals.append(w)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(points), grid.n))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass
class Assembled:
    grid: Grid
    capacity: np.ndarray  # (n,) J/K
    conduction: sp.csr_matrix  # (n, n) Laplacian-type conductance matrix incl. boundary terms
    g_boundary: np.ndarray  # (n,) W/K to ambient
    heater_share: np.ndarray  # (n,) fraction of heater power into each cell, sums to 1


def _boundary_conductance(h: float, area: float, k: float, half: float) -> float:
    if h <= 0.0:
        return 0.0
    return 1.0 / (1.0 / (h * area) + half / (k * area))


def assemble(cfg: SimConfig) -> Assembled:
    g = make_grid(cfg.geometry, cfg.dx)
    m, b, h = cfg.material, cfg.boundaries, cfg.heater
    dx = g.dx
    area = dx * dx
    g_cond = m.k * area / dx
    n = g.n

    i, j, k = np.meshgrid(np.arange(g.nx), np.arange(g.ny), np.arange(g.nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    me = g.idx(i, j, k)

    rows, cols, vals = [], [], []
    diag = np.zeros(n)

    def link(mask, other):
        a = me[mask]
        o = other[mask]
        rows.extend(a.tolist())
        cols.extend(o.tolist())
        vals.extend([-g_cond] * len(a))
        np.add.at(diag, a, g_cond)

    link(i < g.nx - 1, g.idx(i + 1, j, k))
    link(i > 0, g.idx(i - 1, j, k))
    link(j < g.ny - 1, g.idx(i, j + 1, k))
    link(j > 0, g.idx(i, j - 1, k))
    link(k < g.nz - 1, g.idx(i, j, k + 1))
    link(k > 0, g.idx(i, j, k - 1))

    # Boundary conductances to ambient
    half = dx / 2.0
    g_b = np.zeros(n)
    g_side = _boundary_conductance(b.h_sides, area, m.k, half)
    g_top = _boundary_conductance(b.h_top, area, m.k, half)
    g_bot = _boundary_conductance(b.h_bottom, area, m.k, half)
    g_pad = _boundary_conductance(b.h_under_pad, area, m.k, half)

    xs, ys, _ = g.centres()
    cx, cy = cfg.geometry.length_x / 2.0, cfg.geometry.width_y / 2.0
    in_pad = (np.abs(xs[i] - cx) <= h.pad_x / 2.0 + 1e-12) & (np.abs(ys[j] - cy) <= h.pad_y / 2.0 + 1e-12)

    for mask in (i == 0, i == g.nx - 1, j == 0, j == g.ny - 1):
        np.add.at(g_b, me[mask], g_side)
    np.add.at(g_b, me[k == g.nz - 1], g_top)
    bottom = k == 0
    np.add.at(g_b, me[bottom & ~in_pad], g_bot)
    np.add.at(g_b, me[bottom & in_pad], g_pad)

    heater_share = np.zeros(n)
    pad_cells = me[bottom & in_pad]
    if len(pad_cells) == 0:
        raise ValueError("heater pad does not cover any bottom cell; enlarge the pad or refine dx")
    heater_share[pad_cells] = 1.0 / len(pad_cells)

    diag_total = diag + g_b
    rows.extend(range(n))
    cols.extend(range(n))
    vals.extend(diag_total.tolist())
    K = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    capacity = np.full(n, m.rho * m.cp * dx**3)
    return Assembled(g, capacity, K, g_b, heater_share)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def steady_state(cfg: SimConfig, power: float) -> np.ndarray:
    """Steady temperature field (flat array) for a constant heater power in W."""
    a = assemble(cfg)
    rhs = a.g_boundary * cfg.boundaries.t_ambient + a.heater_share * power
    return spla.spsolve(a.conduction.tocsc(), rhs)


def control_sensor_xyz(cfg: SimConfig) -> tuple[float, float, float]:
    if cfg.controller.sensor_xyz is not None:
        return cfg.controller.sensor_xyz
    g = cfg.geometry
    return (g.length_x / 2.0, g.width_y / 2.0, 0.005)


def simulate(cfg: SimConfig, constant_power: float | None = None) -> SimResult:
    """Run heat-up/hold (0..t_heat) then cool-down (t_heat..t_end).

    If ``constant_power`` is given, the controller is bypassed and that power is
    applied during the heating phase (open-loop step test).
    """
    a = assemble(cfg)
    g = a.grid
    dt = cfg.dt
    t_amb = cfg.boundaries.t_ambient

    A = (sp.diags(a.capacity / dt) + a.conduction).tocsc()
    lu = spla.splu(A)

    geom = cfg.geometry
    W = interpolation_matrix(g, geom.well_xyz())
    w_ctrl = interpolation_matrix(g, np.array([control_sensor_xyz(cfg)]))

    T = np.full(g.n, t_amb if cfg.t_initial is None else cfg.t_initial, dtype=float)
    n_steps = int(round(cfg.t_end / dt))
    rec_stride = max(1, int(round(cfg.record_every / dt)))
    n_rec = n_steps // rec_stride + 1

    t_out = np.empty(n_rec)
    wells = np.empty((n_rec, W.shape[0]))
    ctrl = np.empty(n_rec)
    pw = np.empty(n_rec)
    mean = np.empty(n_rec)

    c = cfg.controller
    integral = 0.0
    snapshot = None
    base_rhs = a.g_boundary * t_amb
    power = 0.0
    r = 0

    def record(step, power_now):
        nonlocal r
        t_out[r] = step * dt
        wells[r] = W @ T
        ctrl[r] = float((w_ctrl @ T)[0])
        pw[r] = power_now
        mean[r] = T.mean()
        r += 1

    record(0, 0.0)
    for step in range(1, n_steps + 1):
        t_now = (step - 1) * dt
        if t_now < cfg.t_heat:
            if constant_power is not None:
                power = constant_power
            else:
                t_meas = float((w_ctrl @ T)[0])
                err = c.setpoint - t_meas
                u_unsat = c.kp * err + c.ki * integral
                u = min(max(u_unsat, 0.0), 1.0)
                # conditional integration (anti-windup)
                if (u_unsat == u) or (u_unsat > 1.0 and err < 0) or (u_unsat < 0.0 and err > 0):
                    integral += err * dt
                power = u * cfg.heater.max_power
        else:
            power = 0.0
        rhs = a.capacity / dt * T + base_rhs + a.heater_share * power
        T = lu.solve(rhs)
        if snapshot is None and step * dt >= cfg.t_heat - 1e-9:
            snapshot = T.reshape(g.nz, g.ny, g.nx).copy()
        if step % rec_stride == 0:
            record(step, power)

    if snapshot is None:
        snapshot = T.reshape(g.nz, g.ny, g.nx).copy()

    return SimResult(
        config=cfg,
        t=t_out[:r],
        wells=wells[:r],
        control=ctrl[:r],
        power=pw[:r],
        mean=mean[:r],
        snapshot=snapshot,
        well_names=geom.well_names(),
    )


def lumped_parameters(cfg: SimConfig) -> dict:
    """Lumped-capacitance estimates: total capacity, total UA, time constant, Biot number."""
    geom, m, b, h = cfg.geometry, cfg.material, cfg.boundaries, cfg.heater
    C = m.rho * m.cp * geom.volume
    pad_area = h.pad_x * h.pad_y
    ua = (
        b.h_top * geom.area_top
        + b.h_sides * geom.area_sides
        + b.h_bottom * (geom.area_top - pad_area)
        + b.h_under_pad * pad_area
    )
    area_total = 2 * geom.area_top + geom.area_sides
    h_eff = ua / area_total
    lc = geom.volume / area_total
    return {
        "capacity_J_per_K": C,
        "UA_W_per_K": ua,
        "tau_s": C / ua,
        "biot": h_eff * lc / m.k,
        "char_length_m": lc,
    }
