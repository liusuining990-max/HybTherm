"""Verification of the heat model against cases with known answers.

These are the model's verification tests in the design-control sense: each
checks the code against an independent analytic result, not against itself.
"""

import numpy as np
import pytest

from hybtherm.geometry import BlockGeometry
from hybtherm.model import (
    Boundaries,
    Heater,
    SimConfig,
    assemble,
    lumped_parameters,
    simulate,
    steady_state,
)

COARSE = dict(dx=5e-3)


def test_steady_1d_slab_matches_analytic():
    """Uniform bottom flux, insulated sides and bottom, convective top.

    Exact 1-D solution: T(z) = T_amb + q/h + q (H - z)/k. The finite-volume
    scheme is exact for a linear profile, so agreement should be to round-off.
    """
    geom = BlockGeometry()
    b = Boundaries(h_top=12.0, h_sides=0.0, h_bottom=0.0, h_under_pad=0.0, t_ambient=20.0)
    heater = Heater(pad_x=geom.length_x, pad_y=geom.width_y, max_power=30.0)
    cfg = SimConfig(boundaries=b, heater=heater, **COARSE)
    T = steady_state(cfg, power=30.0)
    a = assemble(cfg)
    g = a.grid
    field = T.reshape(g.nz, g.ny, g.nx)
    q = 30.0 / (geom.length_x * geom.width_y)
    k = cfg.material.k
    zc = (np.arange(g.nz) + 0.5) * g.dx
    expected = 20.0 + q / 12.0 + q * (geom.height_z - zc) / k
    np.testing.assert_allclose(field.mean(axis=(1, 2)), expected, rtol=0, atol=1e-8)
    # and the field must be uniform in x, y
    assert np.ptp(field, axis=(1, 2)).max() < 1e-8


def test_transient_matches_lumped_solution():
    """Uniform h everywhere and a whole-bottom heater: Biot ~ 3e-4, so the block
    mean must follow T = T_amb + (P/UA)(1 - exp(-t/tau)), tau = C/UA."""
    geom = BlockGeometry()
    h = 10.0
    b = Boundaries(h_top=h, h_sides=h, h_bottom=h, h_under_pad=h, t_ambient=22.0)
    heater = Heater(pad_x=geom.length_x, pad_y=geom.width_y, max_power=20.0)
    cfg = SimConfig(boundaries=b, heater=heater, dt=2.0, t_heat=3000.0, t_end=3000.0, record_every=10.0, **COARSE)
    lp = lumped_parameters(cfg)
    assert lp["biot"] < 0.01
    res = simulate(cfg, constant_power=20.0)
    t = res.t
    expected = 22.0 + 20.0 / lp["UA_W_per_K"] * (1 - np.exp(-t / lp["tau_s"]))
    rise = expected - 22.0
    err = np.abs(res.mean - expected)
    # within 0.5 % of the temperature rise (backward Euler + half-cell surface resistance)
    assert np.all(err <= 0.005 * rise.max() + 1e-6)


def test_energy_balance_closed():
    """Each implicit step conserves energy: heat stored = heater input - loss to ambient.

    Internal conduction only moves heat between cells, so it must cancel in the
    sum. This checks that the assembled conduction matrix is conservative.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    cfg = SimConfig(dt=1.0, **COARSE)
    a = assemble(cfg)
    t_amb = cfg.boundaries.t_ambient
    A = (sp.diags(a.capacity / cfg.dt) + a.conduction).tocsc()
    lu = spla.splu(A)
    T = np.full(a.grid.n, t_amb)
    P = 40.0
    for _ in range(50):
        T_new = lu.solve(a.capacity / cfg.dt * T + a.g_boundary * t_amb + a.heater_share * P)
        stored = np.sum(a.capacity * (T_new - T)) / cfg.dt
        lost = np.sum(a.g_boundary * (T_new - t_amb))
        assert stored == pytest.approx(P - lost, rel=1e-9, abs=1e-9)
        T = T_new
    # internal conduction rows sum to the boundary conductance
    row_sums = np.asarray(a.conduction.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, a.g_boundary, atol=1e-12)


def test_symmetry_of_corner_wells():
    """Centred pad and centred control sensor: the four corner wells must agree."""
    cfg = SimConfig(t_heat=600.0, t_end=600.0, record_every=5.0, **COARSE)
    res = simulate(cfg)
    names = res.well_names
    corners = [names.index(w) for w in ("A1", "A12", "H1", "H12")]
    final = res.wells[-1, corners]
    assert np.ptp(final) < 1e-6


def test_controller_reaches_setpoint():
    cfg = SimConfig(t_heat=900.0, t_end=900.0, record_every=5.0, **COARSE)
    res = simulate(cfg)
    assert abs(res.control[-1] - cfg.controller.setpoint) < 0.05
    assert res.control.max() - cfg.controller.setpoint < 1.0  # modest overshoot


def test_bad_cell_size_rejected():
    with pytest.raises(ValueError):
        assemble(SimConfig(dx=3e-3))
