from cascast.services.simulator import GridTwin, daily_profile


def test_daily_profile_peaks_in_evening():
    assert daily_profile(19 * 60) > daily_profile(3 * 60)


def test_twin_runs_a_simulated_day(grid):
    twin = GridTwin(grid, seed=1)
    for tick in range(360):  # 12 h at 2 sim-min per tick
        twin.step(2.0)
        snap = twin.snapshot(tick)
        assert snap["served_mw"] <= snap["demand_mw"] + 1e-6
    assert len(snap["alive"]) == grid.n_branch


def test_operator_trip_and_restore(grid):
    twin = GridTwin(grid, seed=2)
    twin.apply({"cmd": "trip", "lines": [10]})
    assert not twin.alive[10]
    assert twin.events[-1]["lines"] == [10]
    twin.apply({"cmd": "restore_all"})
    twin.step(1.0)
    assert twin.alive[10]


def test_twin_recovers_from_a_major_disturbance(grid):
    """Restoration must not deadlock on island re-synchronisation or re-close into overloads."""
    twin = GridTwin(grid, seed=3, minute=2 * 60.0)  # night: low demand
    twin.apply({"cmd": "trip", "lines": list(range(0, grid.n_branch, 4))})
    assert (~twin.alive).sum() > 40
    for _ in range(4 * 30):  # 4 h
        twin.step(2.0)
    assert (~twin.alive).sum() <= 10
    assert not any(e.get("cause") == "Overload" for e in twin.events)  # no re-close churn
