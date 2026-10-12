"""How often does the firewall pass something on pure noise?

Mines 30 independent noise panels (1,500 days x 60 assets, population 200 x 6
generations, 10 bps costs) and counts survivors and SPA / Reality Check
rejections. Usage: python scripts/noise_calibration.py [workdir]
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

from falsealpha import run, synthetic

work = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
spa, rc, pbo, survivors = [], [], [], 0
for s in range(30):
    synthetic.noise(T=1500, N=60, seed=100 + s).save(work / f"noise{s}.npz")
    v = run.mine(work / f"noise{s}.npz", work / f"run{s}", run.MineConfig(population=200, generations=6, seed=s))
    spa.append(v["snoop"]["spa_p"])
    rc.append(v["snoop"]["reality_check_p"])
    pbo.append(v["pbo"]["pbo"])
    survivors += len(v["survivors"])
print(f"survivors: {survivors}")
print(f"SPA p < 0.05: {sum(p < 0.05 for p in spa)}/30   Reality Check p < 0.05: {sum(p < 0.05 for p in rc)}/30")
print(f"mean PBO: {np.mean(pbo):.2f}")
