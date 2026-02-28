import numpy as np

from iso9613_lpa_raster.core.iso9613_core import compute_lpa_from_sources_grid


def test_raster_point_consistency_regression():
    x = np.array([[100.0, 200.0]], dtype=np.float64)
    y = np.array([[0.0, 0.0]], dtype=np.float64)
    z = np.array([[0.0, 0.0]], dtype=np.float64)
    nodata = np.array([[False, False]])
    sources = [(0.0, 0.0, 0.0, 100.0)]

    out = compute_lpa_from_sources_grid(
        x_grid=x,
        y_grid=y,
        z_rec=z,
        sources=sources,
        alpha_atm=0.0,
        enable_ground=False,
        g_value=0.5,
        d_min=1.0,
        nodata_mask=nodata,
    )

    expected_100 = 49.0
    expected_200 = 100.0 - (20.0 * np.log10(200.0) + 11.0)
    assert abs(out[0, 0] - expected_100) < 1e-6
    assert abs(out[0, 1] - expected_200) < 1e-6


def test_raster_nodata_mask_regression():
    x = np.array([[100.0]], dtype=np.float64)
    y = np.array([[0.0]], dtype=np.float64)
    z = np.array([[0.0]], dtype=np.float64)
    nodata = np.array([[True]])
    sources = [(0.0, 0.0, 0.0, 100.0)]

    out = compute_lpa_from_sources_grid(x, y, z, sources, 0.0, False, 0.5, 1.0, nodata)
    assert np.isnan(out[0, 0])
