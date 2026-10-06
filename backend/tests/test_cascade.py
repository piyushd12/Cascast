import numpy as np

from cascast.grid.cascade import Protection, monte_carlo, settle, simulate_cascade
from cascast.ml.datagen import scenario


def _stressed_state(grid, rng, scale=1.25):
    load = grid.load * scale
    alive, pre = settle(grid, np.ones(grid.n_branch, dtype=bool), load)
    return load, alive, pre


def test_deterministic_cascade_invariants(grid, rng):
    for _ in range(30):
        load = grid.load * rng.uniform(0.9, 1.4)
        alive, pre = settle(grid, np.ones(grid.n_branch, dtype=bool), load)
        trip = rng.choice(np.flatnonzero(alive), 2, replace=False)
        res = simulate_cascade(grid, load, trip, alive=alive, pre=pre, record=True)
        assert 0.0 <= res.load_shed_frac <= 1.0
        assert (res.stage[trip] == 0).all() and not res.alive[trip].any()
        # stable end state: no in-service branch above its rating
        assert (np.abs(res.final.flow[res.alive]) <= grid.rating[res.alive] + 1e-9).all()
        assert alive[res.failed].all()
        assert not res.hidden.any()
        assert len(res.stages) == res.n_stages + 1


def test_stochastic_cascade_invariants(grid, rng):
    load, alive, pre = _stressed_state(grid, rng)
    for _ in range(50):
        trip = rng.choice(np.flatnonzero(alive), 1)
        res = simulate_cascade(grid, load, trip, alive=alive, pre=pre, rng=rng, record=True)
        assert 0.0 <= res.load_shed_frac <= 1.0
        assert not (res.hidden & ~res.failed).any()  # misoperations are failures
        assert alive[res.failed].all()


def test_nominal_grid_is_n_minus_1_secure(grid):
    """Ratings are derived so that no single outage overloads anything at nominal load."""
    for e in range(grid.n_branch):
        res = simulate_cascade(grid, grid.load, [e])
        assert not res.failed.any(), f"branch {e} cascades at nominal load"


def test_stress_causes_cascades(grid, rng):
    load, alive, pre = _stressed_state(grid, rng, 1.35)
    worst = max(simulate_cascade(grid, load, [e], alive=alive, pre=pre).failed.sum()
                for e in np.flatnonzero(alive))
    assert worst > 0


def test_monte_carlo_reduces_to_deterministic(grid, rng):
    load, alive, pre = _stressed_state(grid, rng)
    trip = [int(np.flatnonzero(alive)[3])]
    det = simulate_cascade(grid, load, trip, alive=alive, pre=pre)
    mc = monte_carlo(grid, load, trip, alive, pre, 5, rng, protection=Protection(0.0, 0.0))
    np.testing.assert_array_equal(mc.fail_prob, det.failed.astype(float))
    assert np.allclose(mc.shed, det.load_shed_frac)


def test_monte_carlo_probabilities(grid, rng):
    load, alive, pre = _stressed_state(grid, rng)
    mc = monte_carlo(grid, load, [int(np.flatnonzero(alive)[0])], alive, pre, 30, rng)
    assert mc.runs == 30 and mc.shed.shape == (30,)
    assert ((mc.fail_prob >= 0) & (mc.fail_prob <= 1)).all()


def test_datagen_scenario_shapes(grid, rng):
    s = scenario(rng)
    assert s["load"].shape == s["served1"].shape == (grid.n_bus,)
    assert s["flow"].shape == s["flow1"].shape == s["trip"].shape == (grid.n_branch,)
    assert s["trip"].sum() >= 1
    assert not (s["failed"] & s["trip"]).any()
