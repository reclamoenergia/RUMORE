from app.model.entities import Source, energy_sum
from app.model.parsing import parse_coord_string


def test_parse_coordinate_string():
    x, y, z = parse_coord_string("100.5, 200.0, 3")
    assert (x, y, z) == (100.5, 200.0, 3.0)
    x2, y2, z2 = parse_coord_string("100.5;200.0;3")
    assert (x2, y2, z2) == (100.5, 200.0, 3.0)


def test_energy_sum_two_sources():
    out = energy_sum([50.0, 50.0])
    assert abs(out - 53.01) < 0.02


def test_lwa_bands_consistency_threshold():
    s = Source(
        source_id=1,
        x=0,
        y=0,
        z=1,
        bands={63: 90, 125: 90, 250: 90, 500: 90, 1000: 90, 2000: 90, 4000: 90, 8000: 90},
        lwa_total=110,
    )
    ok, msg = s.validate_lwa_consistency(1.0)
    assert not ok
    assert msg is not None
