import numpy as np

from cascadia.grid.dcpf import dc_power_flow
from cascadia.ml.features import EDGE_FEATURES, NODE_FEATURES, build_features
from cascadia.ml.inference import get_model


def test_feature_shapes(grid):
    alive = np.ones(grid.n_branch, dtype=bool)
    pre = dc_power_flow(grid, alive, grid.load)
    trip = np.eye(grid.n_branch, dtype=bool)[:7]
    xv, xe = build_features(grid, grid.load, pre.served, pre.flow, alive, trip)
    assert xv.shape == (7, grid.n_bus, len(NODE_FEATURES))
    assert xe.shape == (7, grid.n_branch, len(EDGE_FEATURES))
    assert xv.dtype == xe.dtype == np.float32
    assert (xe[np.arange(7), np.arange(7), EDGE_FEATURES.index("contingency")] == 1).all()


def test_screen_outputs(grid):
    model = get_model()
    alive = np.ones(grid.n_branch, dtype=bool)
    alive[3] = False
    pre = dc_power_flow(grid, alive, grid.load)
    sc = model.screen(grid.load, alive, pre)
    assert sc.prob.shape == (grid.n_branch - 1, grid.n_branch)
    assert ((sc.prob >= 0) & (sc.prob <= 1)).all()
    assert ((sc.shed >= 0) & (sc.shed <= 1)).all()
    # a branch can't "fail" if it is already out or is the contingency itself
    assert (sc.prob[:, 3] == 0).all()
    assert (sc.prob[np.arange(sc.contingency.size), sc.contingency] == 0).all()


def test_batch_matches_single(grid):
    model = get_model()
    alive = np.ones(grid.n_branch, dtype=bool)
    pre = dc_power_flow(grid, alive, grid.load * 1.2)
    trip = np.eye(grid.n_branch, dtype=bool)[[5, 40, 100]]
    p_batch, s_batch = model.predict(grid.load * 1.2, pre.served, pre.flow, alive, trip)
    for i in range(3):
        p, s = model.predict(grid.load * 1.2, pre.served, pre.flow, alive, trip[i:i + 1])
        np.testing.assert_allclose(p[0], p_batch[i], atol=1e-5)
        np.testing.assert_allclose(s[0], s_batch[i], atol=1e-5)
