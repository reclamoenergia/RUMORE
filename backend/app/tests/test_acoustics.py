import numpy as np

from app.models import Feature, FeatureCollection, GeoJSONGeometry, ProjectModel
from app.services.acoustics import cache_invalidated, energy_to_db, stable_db_sum


def test_stable_db_sum_known_case():
    levels = np.array([[50.0, 50.0]])
    total = stable_db_sum(levels, axis=1)
    assert np.isclose(total[0], 53.0103, atol=1e-3)


def test_energy_to_db_zero_floor():
    arr = np.array([0.0, 10.0])
    out = energy_to_db(arr)
    assert out[0] == 0.0
    assert np.isclose(out[1], 10.0)


def test_schema_and_cache_invalidation(tmp_path):
    project = ProjectModel(
        point_sources=FeatureCollection(
            features=[
                Feature(
                    geometry=GeoJSONGeometry(type="Point", coordinates=[0, 0]),
                    properties={"id": "p1", "lwa": 90.0, "active": True},
                )
            ]
        )
    )
    meta = tmp_path / "meta.json"
    assert cache_invalidated(meta, project)
    meta.write_text('{"fingerprint": "old"}')
    assert cache_invalidated(meta, project)
