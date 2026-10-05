import numpy as np
import pytest

from cascadia.grid.model import load_grid


@pytest.fixture(scope="session")
def grid():
    return load_grid()


@pytest.fixture
def rng():
    return np.random.default_rng(42)
