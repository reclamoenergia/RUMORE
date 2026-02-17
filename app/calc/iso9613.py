from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.model.entities import Barrier, GridSettings, Source, energy_sum


@dataclass
class CalcResult:
    x_coords: np.ndarray
    y_coords: np.ndarray
    levels_db: np.ndarray
    contours: list[tuple[float, np.ndarray]]


def _segment_intersects(p1: tuple[float, float], p2: tuple[float, float], q1: tuple[float, float], q2: tuple[float, float]) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _barrier_attenuation(source: Source, rx_x: float, rx_y: float, z_receiver: float, barriers: list[Barrier]) -> float:
    max_att = 0.0
    for barrier in barriers:
        intersects = False
        for i in range(len(barrier.points) - 1):
            if _segment_intersects((source.x, source.y), (rx_x, rx_y), barrier.points[i], barrier.points[i + 1]):
                intersects = True
                break
        if not intersects:
            continue
        mid_z_los = (source.z + z_receiver) / 2.0
        if barrier.height > mid_z_los:
            max_att = max(max_att, min(20.0, 5.0 + 3.0 * (barrier.height - mid_z_los)))
    return max_att


def _source_level_at_receiver(source: Source, rx_x: float, rx_y: float, z_receiver: float, alpha_air: float, barriers: list[Barrier]) -> float:
    dx = rx_x - source.x
    dy = rx_y - source.y
    dz = z_receiver - source.z
    r = max(1.0, math.sqrt(dx * dx + dy * dy + dz * dz))

    a_div = 20.0 * math.log10(r) + 11.0
    a_atm = alpha_air * r
    a_bar = _barrier_attenuation(source, rx_x, rx_y, z_receiver, barriers)

    lwa = source.lwa_total if source.lwa_total is not None else (source.estimated_lwa_from_bands() or 0.0)
    return lwa - a_div - a_atm - a_bar


def extract_isophones(x_coords: np.ndarray, y_coords: np.ndarray, levels: np.ndarray, step: float = 2.0) -> list[tuple[float, np.ndarray]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin = float(np.nanmin(levels))
    vmax = float(np.nanmax(levels))
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    contour_levels = np.arange(start, end + step, step)

    fig, ax = plt.subplots()
    cs = ax.contour(x_coords, y_coords, levels, levels=contour_levels)
    lines: list[tuple[float, np.ndarray]] = []
    for lvl, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            lines.append((float(lvl), seg.copy()))
    plt.close(fig)
    return lines


def calculate_grid_map(
    sources: list[Source],
    barriers: list[Barrier],
    grid: GridSettings,
    alpha_air: float = 0.005,
    progress_cb=None,
) -> CalcResult:
    xs = grid.xmin + (np.arange(grid.nx) + 0.5) * grid.cell_size
    ys = grid.ymin + (np.arange(grid.ny) + 0.5) * grid.cell_size
    levels = np.full((grid.ny, grid.nx), -np.inf, dtype=float)

    total = max(1, grid.nx * grid.ny)
    done = 0
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            contribs = [_source_level_at_receiver(s, x, y, grid.z_receiver, alpha_air, barriers) for s in sources]
            levels[iy, ix] = energy_sum(contribs)
            done += 1
        if progress_cb:
            progress_cb(int((done / total) * 100))

    X, Y = np.meshgrid(xs, ys)
    contours = extract_isophones(X, Y, levels, step=2.0)
    return CalcResult(x_coords=X, y_coords=Y, levels_db=levels, contours=contours)
