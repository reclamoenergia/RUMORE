from __future__ import annotations

import re
from typing import Iterable

from .entities import BAND_ORDER, Source


COORD_SPLIT_RE = re.compile(r"\s*[;,]\s*|\s*,\s*")


def parse_coord_string(coord_text: str) -> tuple[float, float, float]:
    if not coord_text or not coord_text.strip():
        raise ValueError("Coordinata vuota. Usa formato 'X, Y, Z'.")
    parts = [p for p in COORD_SPLIT_RE.split(coord_text.strip()) if p != ""]
    if len(parts) != 3:
        raise ValueError(f"Coordinate non valide '{coord_text}'. Atteso: X, Y, Z.")
    try:
        x, y, z = (float(p.replace(" ", "")) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Coordinate non numeriche '{coord_text}'.") from exc
    return x, y, z


def source_from_row(row_index: int, row: list[str]) -> tuple[Source | None, list[str]]:
    errors: list[str] = []
    rid = row_index + 1
    try:
        x, y, z = parse_coord_string(row[1])
    except Exception as exc:
        errors.append(f"Riga {rid}: campo Coord non valido ({exc}).")
        return None, errors

    bands: dict[int, float] = {}
    for i, freq in enumerate(BAND_ORDER, start=2):
        val = row[i].strip() if i < len(row) else ""
        if not val:
            continue
        try:
            bands[freq] = float(val)
        except ValueError:
            errors.append(f"Riga {rid}: banda {freq}Hz non numerica.")

    lwa_total = None
    if len(row) > 10 and row[10].strip():
        try:
            lwa_total = float(row[10])
        except ValueError:
            errors.append(f"Riga {rid}: LwA_tot non numerico.")

    if not bands and lwa_total is None:
        errors.append(f"Riga {rid}: inserire almeno LwA_tot oppure una o più bande.")

    src = Source(source_id=rid, x=x, y=y, z=z, bands=bands, lwa_total=lwa_total)
    est = src.estimated_lwa_from_bands()
    if src.lwa_total is None and est is not None:
        src.lwa_total = est
    ok, msg = src.validate_lwa_consistency()
    if not ok and msg:
        errors.append(msg)

    return (None if errors else src), errors


def rows_to_sources(rows: Iterable[list[str]]) -> tuple[list[Source], list[str]]:
    sources: list[Source] = []
    errors: list[str] = []
    for idx, row in enumerate(rows):
        if len(row) < 2 or not any(c.strip() for c in row):
            continue
        src, err = source_from_row(idx, row)
        if err:
            errors.extend(err)
        if src:
            sources.append(src)
    return sources, errors
