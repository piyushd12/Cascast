import cascast  # noqa: F401, I001  (first: pins BLAS threads before numpy loads)
import numpy as np
import pytest

from cascast.grid.model import load_grid


@pytest.fixture(scope="session")
def grid():
    return load_grid()


@pytest.fixture
def rng():
    return np.random.default_rng(42)
