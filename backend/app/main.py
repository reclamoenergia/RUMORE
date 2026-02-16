from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from shapely.geometry import LineString, shape

from app.models import CalculationResponse, ContributionResponse, ProjectModel, ScenarioUpdateRequest, SectionRequest
from app.services.acoustics import (
    cache_invalidated,
    combine_active,
    compute_source_contributions,
    energy_to_db,
    project_fingerprint,
)
from app.services.geo import grid_coordinates, write_geotiff

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PROJECTS = DATA / "projects"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
CACHE_DIR = DATA / "cache"
for p in [PROJECTS, UPLOADS, OUTPUTS, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Rumore API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _project_path(project_id: str) -> Path:
    return PROJECTS / f"{project_id}.json"


def _load_project(project_id: str) -> ProjectModel:
    path = _project_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectModel.model_validate_json(path.read_text())


def _cache_paths(project_id: str):
    project_cache = CACHE_DIR / project_id
    project_cache.mkdir(parents=True, exist_ok=True)
    return project_cache, project_cache / "meta.json"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/projects/{project_id}")
def save_project(project_id: str, project: ProjectModel):
    path = _project_path(project_id)
    path.write_text(project.model_dump_json(indent=2))
    return {"project_id": project_id, "path": str(path)}


@app.get("/projects/{project_id}", response_model=ProjectModel)
def get_project(project_id: str):
    return _load_project(project_id)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    dst = UPLOADS / file.filename
    dst.write_bytes(await file.read())
    return {"filename": file.filename, "path": str(dst)}


@app.post("/projects/{project_id}/calculate", response_model=CalculationResponse)
def run_calculation(project_id: str):
    project = _load_project(project_id)
    rec_x, rec_y = grid_coordinates(project.settings.extent, project.settings.resolution)
    cache_path, meta_path = _cache_paths(project_id)

    contributions = []
    if cache_invalidated(meta_path, project):
        contributions = compute_source_contributions(project, rec_x, rec_y)
        for c in contributions:
            np.save(cache_path / f"{c.source_id}.npy", c.energy.astype(np.float32))
        meta_path.write_text(json.dumps({"fingerprint": project_fingerprint(project)}))
    else:
        for npy in cache_path.glob("*.npy"):
            contributions.append({"source_id": npy.stem, "energy": np.load(npy)})

    if contributions and isinstance(contributions[0], dict):
        energy = np.sum([c["energy"] for c in contributions], axis=0)
    else:
        energy = np.sum([c.energy for c in contributions], axis=0) if contributions else np.zeros_like(rec_x)
    levels = energy_to_db(energy)
    scenario_path = OUTPUTS / f"{project_id}_scenario.tif"
    write_geotiff(scenario_path, levels, project.settings.extent, project.settings.resolution, project.crs_epsg)

    iso_path = OUTPUTS / f"{project_id}_isophones.geojson"
    # Simplified pseudo-isophones as rectangular classes for MVP.
    classes = [45, 50, 55, 60, 65]
    feats = []
    for c in classes:
        mask = levels >= c
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        x_coords = rec_x[ys, xs]
        y_coords = rec_y[ys, xs]
        bbox = [float(x_coords.min()), float(y_coords.min()), float(x_coords.max()), float(y_coords.max())]
        geom = {
            "type": "Polygon",
            "coordinates": [[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]],
        }
        feats.append({"type": "Feature", "properties": {"level": c}, "geometry": geom})
    iso_path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

    return CalculationResponse(scenario_raster=str(scenario_path), isophones_geojson=str(iso_path), cache_id=project_fingerprint(project))


@app.post("/projects/{project_id}/scenario")
def update_scenario(project_id: str, req: ScenarioUpdateRequest):
    project = _load_project(project_id)
    rec_x, _ = grid_coordinates(project.settings.extent, project.settings.resolution)
    cache_path, _ = _cache_paths(project_id)
    energies = []
    for sid in req.active_source_ids:
        p = cache_path / f"{sid}.npy"
        if p.exists():
            energies.append(np.load(p))
    total = np.sum(energies, axis=0) if energies else np.zeros_like(rec_x)
    levels = energy_to_db(total)
    scenario_path = OUTPUTS / f"{project_id}_scenario_active.tif"
    write_geotiff(scenario_path, levels, project.settings.extent, project.settings.resolution, project.crs_epsg)
    return {"scenario_raster": str(scenario_path)}


@app.get("/projects/{project_id}/contribution/{source_id}", response_model=ContributionResponse)
def get_contribution(project_id: str, source_id: str):
    project = _load_project(project_id)
    rec_x, _ = grid_coordinates(project.settings.extent, project.settings.resolution)
    cache_path, _ = _cache_paths(project_id)
    p = cache_path / f"{source_id}.npy"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Contribution cache not found")
    levels = energy_to_db(np.load(p))
    out = OUTPUTS / f"{project_id}_{source_id}.tif"
    write_geotiff(out, levels, project.settings.extent, project.settings.resolution, project.crs_epsg)
    return ContributionResponse(source_id=source_id, raster=str(out))


@app.post("/projects/{project_id}/section")
def run_section(project_id: str, req: SectionRequest):
    project = _load_project(project_id)
    feature = next((f for f in project.sections.features if f.properties.get("id") == req.section_feature_id), None)
    if not feature:
        raise HTTPException(status_code=404, detail="Section not found")
    line = shape(feature.geometry.model_dump())
    if not isinstance(line, LineString):
        raise HTTPException(status_code=400, detail="Section geometry must be LineString")

    step_s = feature.properties.get("step_s", 25.0)
    z_min = feature.properties.get("z_min", 0.0)
    z_max = feature.properties.get("z_max", 50.0)
    z_step = feature.properties.get("z_step", 2.0)
    s_vals = np.arange(0, line.length + step_s, step_s)
    z_vals = np.arange(z_min, z_max + z_step, z_step)

    from app.services.acoustics import compute_source_contributions

    curtain = np.zeros((len(z_vals), len(s_vals)), dtype=np.float64)
    for i, s in enumerate(s_vals):
        pt = line.interpolate(min(s, line.length))
        rec_x = np.array([[pt.x]])
        rec_y = np.array([[pt.y]])
        contributions = compute_source_contributions(project, rec_x, rec_y)
        base = sum(c.energy[0, 0] for c in contributions)
        for j, z in enumerate(z_vals):
            curtain[j, i] = max(base - z * 0.01, 1e-9)

    levels = energy_to_db(curtain)
    png_path = OUTPUTS / f"{project_id}_section_{req.section_feature_id}.png"
    csv_path = OUTPUTS / f"{project_id}_section_{req.section_feature_id}.csv"

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.imshow(levels, origin="lower", aspect="auto", extent=[0, line.length, z_min, z_max], cmap="inferno")
    plt.colorbar(label="dB")
    plt.xlabel("s [m]")
    plt.ylabel("z [m]")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["s", "z", "L"])
        for i, s in enumerate(s_vals):
            for j, z in enumerate(z_vals):
                writer.writerow([float(s), float(z), float(levels[j, i])])

    return {"png": str(png_path), "csv": str(csv_path)}


@app.get("/files")
def get_file(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(p)
