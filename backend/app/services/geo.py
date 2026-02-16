from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point


def grid_coordinates(extent: List[float], resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    x_min, y_min, x_max, y_max = extent
    xs = np.arange(x_min + resolution / 2, x_max, resolution)
    ys = np.arange(y_max - resolution / 2, y_min, -resolution)
    return np.meshgrid(xs, ys)


def write_geotiff(path: Path, data: np.ndarray, extent: List[float], resolution: float, epsg: int) -> None:
    x_min, y_min, _, y_max = extent
    transform = from_origin(x_min, y_max, resolution, resolution)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=np.float32,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def sample_dem(dem_path: Path, points: Iterable[Tuple[float, float]], default_z: float = 0.0) -> np.ndarray:
    if not dem_path.exists():
        return np.full(len(list(points)), default_z)
    pts = list(points)
    if not pts:
        return np.array([])
    with rasterio.open(dem_path) as ds:
        vals = [v[0] for v in ds.sample(pts)]
    arr = np.array(vals)
    arr[np.isnan(arr)] = default_z
    return arr


def line_discretize(line: LineString, step: float) -> List[Point]:
    if line.length == 0:
        return [Point(line.coords[0])]
    n = max(1, int(np.ceil(line.length / step)))
    return [line.interpolate(i * line.length / n) for i in range(n + 1)]
