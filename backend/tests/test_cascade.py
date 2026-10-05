import numpy as np

from cascadia.grid.cascade import settle, simulate_cascade
from cascadia.ml.datagen import scenario


def test_cascade_invariants(grid, rng):
    for _ in range(40):
        load = grid.load * rng.uniform(0.9, 1.4)
        alive, pre = settle(grid, np.ones(grid.n_branch, dtype=bool), load)
        trip = rng.choice(np.flatnonzero(alive), 2, replace=False)
        res = simulate_cascade(grid, load, trip, alive=alive, pre=pre, record=True)
        assert 0.0 <= res.load_shed_frac <= 1.0
        assert (res.stage[trip] == 0).all()
        assert not res.alive[trip].any()
        # the final state is stable: no in-service branch above its rating
        assert (np.abs(res.final.flow[res.alive]) <= grid.rating[res.alive] + 1e-9).all()
        # every branch that failed in the cascade was in service before
        assert alive[res.failed].all()
        assert len(res.stages) == res.n_stages + 1


def test_nominal_grid_is_n_minus_1_secure(grid):
    """Ratings are derived so that no single outage cascades at nominal load."""
    for e in range(grid.n_branch):
        res = simulate_cascade(grid, grid.load, [e])
        assert not res.failed.any(), f"branch {e} cascades at nominal load"


def test_stress_causes_cascades(grid):
    load = grid.load * 1.35
    alive, pre = settle(grid, np.ones(grid.n_branch, dtype=bool), load)
    worst = max(simulate_cascade(grid, load, [e], alive=alive, pre=pre).failed.sum()
                for e in np.flatnonzero(alive))
    assert worst > 0


def test_datagen_scenario_shapes(grid, rng):
    s = scenario(rng)
    assert s["load"].shape == (grid.n_bus,)
    assert s["flow"].shape == s["trip"].shape == s["failed"].shape == (grid.n_branch,)
    assert s["trip"].sum() >= 1
    assert not (s["failed"] & s["trip"]).any()
