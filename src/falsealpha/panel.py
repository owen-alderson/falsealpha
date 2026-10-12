"""A panel: named (days x assets) float fields plus a point-in-time universe mask."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _core


@dataclass
class Panel:
    dates: np.ndarray  # datetime64[D], length T
    assets: list[str]  # length N
    fields: dict[str, np.ndarray]  # each T x N float64, NaN = no data
    universe: np.ndarray  # T x N bool: tradable member on that day
    source: str = ""
    periods_per_year: float = 365.0
    _data: _core.Data | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.dates = np.asarray(self.dates, dtype="datetime64[D]")
        self.universe = np.asarray(self.universe, dtype=bool)
        T, N = len(self.dates), len(self.assets)
        if self.universe.shape != (T, N):
            raise ValueError(f"universe is {self.universe.shape}, expected {(T, N)}")
        for name, arr in self.fields.items():
            if not _core.valid_field_name(name):
                raise ValueError(f"field name {name!r} must be an identifier that is not an operator name "
                                 "(formulas refer to fields by name)")
            arr = np.ascontiguousarray(arr, dtype=np.float64)
            if arr.shape != (T, N):
                raise ValueError(f"field {name!r} is {arr.shape}, expected {(T, N)}")
            self.fields[name] = arr
        if "returns" not in self.fields:
            raise ValueError("a panel needs a 'returns' field (close-to-close simple returns)")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.dates), len(self.assets)

    def core(self) -> _core.Data:
        """The C++ view of this panel (built once, then reused)."""
        if self._data is None:
            self._data = _core.Data(self.fields, self.universe)
        return self._data

    def head(self, days: int) -> Panel:
        """The first `days` days only. Used to keep the holdout out of reach."""
        return Panel(self.dates[:days], list(self.assets), {k: v[:days] for k, v in self.fields.items()},
                     self.universe[:days], self.source, self.periods_per_year)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.dates.astype("int64").tobytes())
        h.update("\0".join(self.assets).encode())
        for name in sorted(self.fields):
            h.update(name.encode())
            h.update(self.fields[name].tobytes())
        h.update(self.universe.tobytes())
        return h.hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, dates=self.dates, assets=np.array(self.assets), universe=self.universe,
            meta=np.array([self.source, str(self.periods_per_year)]),
            **{f"field_{k}": v for k, v in self.fields.items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> Panel:
        with np.load(path, allow_pickle=False) as z:
            fields = {k[len("field_"):]: z[k] for k in z.files if k.startswith("field_")}
            source, ppy = (str(x) for x in z["meta"])
            return cls(z["dates"], [str(a) for a in z["assets"]], fields, z["universe"], source, float(ppy))
