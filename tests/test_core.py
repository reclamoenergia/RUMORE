import numpy as np

from iso9613_lpa_raster.core.iso9613_core import (
    compute_adiv,
    compute_agr_simplified,
    compute_lpa_for_receptors_points,
)


def test_adiv_100m():
    assert abs(float(compute_adiv(100.0)) - 51.0) < 1e-6


def test_single_source_flat():
    rec_xy = np.array([[100.0, 0.0]], dtype=np.float64)
    rec_z = np.array([0.0], dtype=np.float64)
    sources = [(0.0, 0.0, 0.0, 100.0)]
    out = compute_lpa_for_receptors_points(
        rec_xy=rec_xy,
        rec_z=rec_z,
        sources=sources,
        alpha_atm=0.0,
        enable_ground=False,
        g_value=0.5,
        d_min=1.0,
    )
    assert abs(float(out[0]) - 49.0) < 1e-6


def test_energy_sum():
    rec_xy = np.array([[100.0, 0.0]], dtype=np.float64)
    rec_z = np.array([0.0], dtype=np.float64)
    one = [(0.0, 0.0, 0.0, 100.0)]
    two = [(0.0, 0.0, 0.0, 100.0), (0.0, 0.0, 0.0, 100.0)]

    out_one = compute_lpa_for_receptors_points(rec_xy, rec_z, one, 0.0, False, 0.5, 1.0)
    out_two = compute_lpa_for_receptors_points(rec_xy, rec_z, two, 0.0, False, 0.5, 1.0)
    assert abs(float(out_two[0] - out_one[0]) - 3.0103) < 1e-3


def test_d_min_clamp():
    rec_xy = np.array([[0.0, 0.0]], dtype=np.float64)
    rec_z = np.array([0.0], dtype=np.float64)
    sources = [(0.0, 0.0, 0.0, 100.0)]
    out = compute_lpa_for_receptors_points(rec_xy, rec_z, sources, 0.0, False, 0.5, 1.0)
    assert np.isfinite(out[0])


def test_ground_toggle():
    d = 1000.0
    off = compute_agr_simplified(False, 0.8, d)
    on = compute_agr_simplified(True, 0.8, d)
    assert off == 0.0
    assert on > 0.0
