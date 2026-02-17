from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


A_WEIGHTING = {
    63: -26.2,
    125: -16.1,
    250: -8.6,
    500: -3.2,
    1000: 0.0,
    2000: 1.2,
    4000: 1.0,
    8000: -1.1,
}
BAND_ORDER = [63, 125, 250, 500, 1000, 2000, 4000, 8000]


@dataclass
class Source:
    source_id: int
    x: float
    y: float
    z: float
    bands: dict[int, float] = field(default_factory=dict)
    lwa_total: float | None = None

    def estimated_lwa_from_bands(self) -> float | None:
        if not self.bands:
            return None
        adjusted = []
        for f in BAND_ORDER:
            if f in self.bands:
                adjusted.append(self.bands[f] + A_WEIGHTING[f])
        if not adjusted:
            return None
        return energy_sum(adjusted)

    def validate_lwa_consistency(self, tolerance_db: float = 1.0) -> tuple[bool, str | None]:
        estimated = self.estimated_lwa_from_bands()
        if estimated is None or self.lwa_total is None:
            return True, None
        diff = abs(estimated - self.lwa_total)
        if diff > tolerance_db:
            return (
                False,
                f"ID {self.source_id}: differenza LwA totale vs bande = {diff:.2f} dB (> {tolerance_db:.1f} dB).",
            )
        return True, None


@dataclass
class Barrier:
    barrier_id: int
    points: list[tuple[float, float]]
    height: float = 2.0


@dataclass
class GridSettings:
    cell_size: float = 10.0
    buffer: float = 100.0
    z_receiver: float = 4.0
    nx: int = 0
    ny: int = 0
    xmin: float = 0.0
    ymin: float = 0.0
    xmax: float = 0.0
    ymax: float = 0.0

    def update_extent(self, sources: list[Source]) -> None:
        if not sources:
            self.nx = self.ny = 0
            return
        xs = [s.x for s in sources]
        ys = [s.y for s in sources]
        self.xmin = min(xs) - self.buffer
        self.xmax = max(xs) + self.buffer
        self.ymin = min(ys) - self.buffer
        self.ymax = max(ys) + self.buffer
        width = max(self.xmax - self.xmin, self.cell_size)
        height = max(self.ymax - self.ymin, self.cell_size)
        self.nx = int((width / self.cell_size) + 0.999999)
        self.ny = int((height / self.cell_size) + 0.999999)


@dataclass
class ProjectData:
    epsg: str
    grid: GridSettings
    sources: list[Source]
    barriers: list[Barrier]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsg": self.epsg,
            "grid": asdict(self.grid),
            "sources": [asdict(s) for s in self.sources],
            "barriers": [asdict(b) for b in self.barriers],
        }


def energy_sum(levels_db: list[float]) -> float:
    if not levels_db:
        return float("-inf")
    lmax = max(levels_db)
    return lmax + 10.0 * __import__("math").log10(sum(10 ** ((l - lmax) / 10.0) for l in levels_db))
