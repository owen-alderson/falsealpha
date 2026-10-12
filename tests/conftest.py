import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from falsealpha import Panel  # noqa: E402


@pytest.fixture
def random_panel():
    rng = np.random.default_rng(0)
    T, N = 300, 25
    fields = {
        "close": 1e4 + 50 * rng.normal(size=(T, N)),  # big level: stresses running sums
        "volume": rng.lognormal(10, 1, (T, N)),
        "returns": rng.normal(0, 0.02, (T, N)),
    }
    for v in fields.values():
        v[rng.random((T, N)) < 0.02] = np.nan
    fields["close"][:, 3] = 7.0  # a constant column: zero-variance paths
    universe = rng.random((T, N)) < 0.8
    return Panel(np.datetime64("2020-01-01") + np.arange(T), [f"A{i}" for i in range(N)], fields, universe)
