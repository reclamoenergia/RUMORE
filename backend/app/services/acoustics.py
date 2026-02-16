from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, shape

from app.models import Feature, ProjectModel
from app.services.geo import line_discretize


@dataclass
class SourceContribution:
    source_id: str
    energy: np.ndarray


def db_to_energy(db: np.ndarray | float) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def energy_to_db(energy: np.ndarray, floor_db: float = 0.0) -> np.ndarray:
    out = np.full_like(energy, floor_db, dtype=np.float64)
    mask = energy > 0
    out[mask] = 10.0 * np.log10(energy[mask])
    return out


def stable_db_sum(levels_db: np.ndarray, axis: int = 0) -> np.ndarray:
    levels_db = np.asarray(levels_db, dtype=np.float64)
    lmax = np.max(levels_db, axis=axis, keepdims=True)
    lsum = lmax + 10.0 * np.log10(np.sum(np.power(10.0, (levels_db - lmax) / 10.0), axis=axis, keepdims=True))
    return np.squeeze(lsum, axis=axis)


def air_absorption_coeff(temp_c: float = 20.0, humidity: float = 70.0) -> float:
    return 0.003 + max(0.0, (humidity - 50.0)) * 0.00002 + max(0.0, (temp_c - 20.0)) * 0.00003


def barrier_attenuation(src: Point, rec_x: np.ndarray, rec_y: np.ndarray, barriers: List[LineString]) -> np.ndarray:
    if not barriers:
        return np.zeros_like(rec_x, dtype=np.float64)
    attenuation = np.zeros_like(rec_x, dtype=np.float64)
    src_pt = np.array([src.x, src.y])
    for barrier in barriers:
        b = np.array(barrier.coords)
        b0, b1 = b[0], b[-1]
        v = b1 - b0
        w = np.stack([rec_x - b0[0], rec_y - b0[1]], axis=-1)
        cross = v[0] * w[..., 1] - v[1] * w[..., 0]
        src_side = v[0] * (src_pt[1] - b0[1]) - v[1] * (src_pt[0] - b0[0])
        intersect_mask = (cross * src_side) < 0
        attenuation = np.maximum(attenuation, np.where(intersect_mask, 8.0, 0.0))
    return attenuation


def propagation_level(lwa: float, src: Point, rec_x: np.ndarray, rec_y: np.ndarray, barriers: List[LineString], alpha_air: float, ground_factor: float) -> np.ndarray:
    dist = np.sqrt((rec_x - src.x) ** 2 + (rec_y - src.y) ** 2)
    dist = np.maximum(dist, 1.0)
    a_div = 20.0 * np.log10(dist) + 11.0
    a_air = alpha_air * dist
    a_gr = ground_factor * np.log10(dist)
    a_bar = barrier_attenuation(src, rec_x, rec_y, barriers)
    lp = lwa - (a_div + a_air + a_gr + a_bar)
    return lp


def project_fingerprint(project: ProjectModel) -> str:
    data = json.dumps(project.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _to_lines(features: Iterable[Feature]) -> List[LineString]:
    out = []
    for f in features:
        geom = shape(f.geometry.model_dump())
        if isinstance(geom, LineString):
            out.append(geom)
    return out


def compute_source_contributions(project: ProjectModel, rec_x: np.ndarray, rec_y: np.ndarray) -> List[SourceContribution]:
    barriers = _to_lines(project.barriers.features)
    alpha_air = air_absorption_coeff(project.settings.temperature_c, project.settings.humidity)
    contributions: List[SourceContribution] = []

    for feature in project.point_sources.features:
        props = feature.properties
        src = shape(feature.geometry.model_dump())
        if not isinstance(src, Point):
            continue
        lp = propagation_level(props["lwa"], src, rec_x, rec_y, barriers, alpha_air, project.settings.ground_factor)
        contributions.append(SourceContribution(props["id"], db_to_energy(lp)))

    discretization_step = max(5.0, project.settings.resolution)
    for feature in project.line_sources.features:
        props = feature.properties
        line = shape(feature.geometry.model_dump())
        if not isinstance(line, LineString):
            continue
        samples = line_discretize(line, discretization_step)
        energies = []
        lwa_per_m = props["lwa_per_m"]
        point_lwa = lwa_per_m + 10.0 * math.log10(max(1.0, line.length / max(1, len(samples))))
        for sample in samples:
            lp = propagation_level(point_lwa, sample, rec_x, rec_y, barriers, alpha_air, project.settings.ground_factor)
            energies.append(db_to_energy(lp))
        contributions.append(SourceContribution(props["id"], np.sum(energies, axis=0)))
    return contributions


def combine_active(contributions: List[SourceContribution], active_ids: List[str]) -> np.ndarray:
    selected = [c.energy for c in contributions if c.source_id in active_ids]
    if not selected:
        return np.zeros_like(contributions[0].energy) if contributions else np.array([])
    return np.sum(selected, axis=0)


def cache_invalidated(cache_meta_path: Path, project: ProjectModel) -> bool:
    if not cache_meta_path.exists():
        return True
    current = project_fingerprint(project)
    previous = json.loads(cache_meta_path.read_text()).get("fingerprint")
    return current != previous
