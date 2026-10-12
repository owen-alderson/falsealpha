"""Size of the Reality Check and SPA tests where the firewall uses them.

1,000 simulations each of K zero-mean strategies (no edge at all) over T days,
1,000 stationary-bootstrap draws with mean block 10, as in the firewall. A
test's size is how often it rejects at p < 0.05; it should be about 0.05.
Usage: python scripts/snoop_size.py
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import falsealpha as fa


def one(seed: int, K: int, T: int, rho: float) -> tuple[bool, bool]:
    rng = np.random.default_rng(seed)
    x = np.sqrt(rho) * rng.normal(size=T) + np.sqrt(1 - rho) * rng.normal(size=(K, T))
    r = fa.snoop_tests(0.01 * x, 1000, 10.0, seed + 1)
    return r["reality_check_p"] < 0.05, r["spa_p"] < 0.05


for K, T, rho in [(50, 490, 0.0), (50, 490, 0.5), (20, 1000, 0.0)]:
    with ThreadPoolExecutor() as pool:
        hits = np.array(list(pool.map(lambda s: one(s, K, T, rho), range(1000))))
    print(f"K={K:>3} T={T:>5} corr={rho}: Reality Check size {hits[:, 0].mean():.3f}   SPA size {hits[:, 1].mean():.3f}")
