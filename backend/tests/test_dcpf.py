import numpy as np

from cascast.grid.cascade import settle
from cascast.grid.dcpf import dc_power_flow, islands, n1_flows


def test_kirchhoff_current_law(grid):
    """Net injection at every bus equals the sum of flows leaving it."""
    alive = np.ones(grid.n_branch, dtype=bool)
    res = dc_power_flow(grid, alive, grid.load)
    injection = np.bincount(grid.gen_bus, weights=res.pg, minlength=grid.n_bus) - res.served
    np.testing.assert_allclose(grid.incidence @ res.flow, injection, atol=1e-6)


def test_balanced_and_secure_at_nominal_load(grid):
    alive = np.ones(grid.n_branch, dtype=bool)
    res = dc_power_flow(grid, alive, grid.load)
    assert res.n_islands == 1
    assert np.isclose(res.pg.sum(), grid.load.sum())
    assert np.isclose(res.served.sum(), grid.load.sum())
    assert (np.abs(res.flow) <= grid.rating).all()


def test_out_of_service_branches_carry_no_flow(grid, rng):
    alive = np.ones(grid.n_branch, dtype=bool)
    alive[rng.choice(grid.n_branch, 5, replace=False)] = False
    res = dc_power_flow(grid, alive, grid.load)
    assert (res.flow[~alive] == 0).all()


def test_isolated_load_bus_is_shed(grid):
    """Disconnect every branch at a load-only bus: that load is lost, the rest is served."""
    cap = grid.gen_capacity_at_bus
    bus = int(np.flatnonzero((grid.load > 0) & (cap == 0))[0])
    alive = (grid.f != bus) & (grid.t != bus)
    n, label = islands(grid, alive)
    assert n >= 2
    res = dc_power_flow(grid, alive, grid.load)
    assert res.served[bus] == 0
    assert np.isclose(res.served.sum(), grid.load.sum() - grid.load[bus])


def test_lodf_n1_flows_match_full_power_flow(grid, rng):
    """The fast N-1 screen (LODF + analytic islanding) is exact under the DC model."""
    for scale, n_pre in [(1.0, 0), (1.25, 2), (0.8, 3)]:
        alive = np.ones(grid.n_branch, dtype=bool)
        alive[rng.choice(grid.n_branch, n_pre, replace=False)] = False
        load = grid.load * scale * rng.lognormal(0, 0.08, grid.n_bus)
        alive, pre = settle(grid, alive, load)
        cont, flow, served = n1_flows(grid, alive, load, pre)
        assert (cont == np.flatnonzero(alive)).all()
        for i, c in enumerate(cont):
            a = alive.copy()
            a[c] = False
            ref = dc_power_flow(grid, a, load)
            np.testing.assert_allclose(flow[i], ref.flow, atol=1e-6)
            np.testing.assert_allclose(served[i], ref.served, atol=1e-6)
