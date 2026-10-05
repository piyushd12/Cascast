from cascadia.services.simulator import GridTwin, daily_profile


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
