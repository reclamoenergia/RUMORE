from __future__ import annotations

from datetime import datetime

import matplotlib
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from rasterio.transform import from_origin

from app.calc.iso9613 import CalcResult
from app.model.entities import Barrier, GridSettings, Source

matplotlib.use("Agg")


def export_geotiff(path: str, result: CalcResult, grid: GridSettings, epsg: str) -> None:
    transform = from_origin(grid.xmin, grid.ymax, grid.cell_size, grid.cell_size)
    data = np.flipud(result.levels_db).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=epsg,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)


def export_layout_png(path: str, result: CalcResult, grid: GridSettings, epsg: str, sources: list[Source], barriers: list[Barrier]) -> None:
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    im = ax.imshow(
        result.levels_db,
        extent=[grid.xmin, grid.xmax, grid.ymin, grid.ymax],
        origin="lower",
        cmap="viridis",
        alpha=0.85,
    )
    for lvl, seg in result.contours:
        ax.plot(seg[:, 0], seg[:, 1], linewidth=0.8, color="white", alpha=0.8)
        if len(seg) > 0:
            ax.text(seg[0, 0], seg[0, 1], f"{lvl:.0f} dB", fontsize=6, color="white")

    ax.scatter([s.x for s in sources], [s.y for s in sources], c="red", s=25, label="Sorgenti")
    for s in sources:
        ax.text(s.x, s.y, str(s.source_id), fontsize=7, color="red")

    for b in barriers:
        pts = np.array(b.points)
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=2)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("dBA")
    ax.set_title("Mappa acustica ISO 9613 - V1")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    info = f"EPSG: {epsg} | cell_size: {grid.cell_size} m | buffer: {grid.buffer} m | {now}"
    fig.text(0.01, 0.01, info, fontsize=8)
    fig.text(0.93, 0.08, "N", fontsize=12, weight="bold")
    ax.annotate("", xy=(0.95, 0.2), xytext=(0.95, 0.1), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=2))

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", label="Sorgenti", markersize=6),
        Line2D([0], [0], color="black", lw=2, label="Barriere"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # simple scale bar
    length = max(10, int((grid.xmax - grid.xmin) / 10))
    x0, y0 = grid.xmin + 0.05 * (grid.xmax - grid.xmin), grid.ymin + 0.05 * (grid.ymax - grid.ymin)
    ax.plot([x0, x0 + length], [y0, y0], color="black", linewidth=3)
    ax.text(x0, y0 + length * 0.02, f"{length} m", fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
