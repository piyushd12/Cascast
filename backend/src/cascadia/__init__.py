"""Cascadia: real-time cascading-failure risk for power grids."""

import os

# The grid matrices are small (118 buses): multithreaded BLAS only adds spin-wait overhead, and
# with several services per host it oversubscribes every core. Must run before numpy import.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
