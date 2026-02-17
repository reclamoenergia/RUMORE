from __future__ import annotations

import json

from app.model.entities import Barrier, GridSettings, ProjectData, Source


def save_project(path: str, project: ProjectData) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project.to_dict(), f, indent=2)


def load_project(path: str) -> ProjectData:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    grid = GridSettings(**data["grid"])
    sources = [Source(**s) for s in data["sources"]]
    barriers = [Barrier(**b) for b in data["barriers"]]
    return ProjectData(epsg=data["epsg"], grid=grid, sources=sources, barriers=barriers)
