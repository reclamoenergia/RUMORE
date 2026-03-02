import numpy as np

from iso9613_lpa_raster.core.iso9613_core import (
    A_WEIGHT_DB,
    BANDS,
    WIND_TURBINE_SHAPE_DB,
    alpha_iso9613_1,
    build_band_lw_from_shape_scaled_to_LwA,
    build_source_spectrum,
    compute_adiv,
    compute_agr_iso9613_2_octave,
    compute_agr_simplified,
    dz_iso_single_screen,
    kmet_iso,
    abar_from_dz,
    compute_lpa_for_receptors_points,
    compute_lpa_from_sources_grid,
    reconstruct_lwa_total_from_unweighted,
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


def test_agr_disabled_is_zero():
    rec_xy = np.array([[250.0, 0.0]], dtype=np.float64)
    rec_z = np.array([0.0], dtype=np.float64)
    sources = [(0.0, 0.0, 0.0, 100.0)]
    source_spec = [{"x": 0.0, "y": 0.0, "z": 0.0, "h_src": 2.0, "lwa": 100.0, "lw_band": build_source_spectrum(100.0, "FLAT_FROM_LWA")}]
    out_g0 = compute_lpa_for_receptors_points(
        rec_xy=rec_xy,
        rec_z=rec_z,
        sources=sources,
        alpha_atm=0.0,
        enable_ground=False,
        g_value=0.0,
        d_min=1.0,
        use_bands=True,
        sources_spectra=source_spec,
        receiver_height_m=4.0,
    )
    out_g1 = compute_lpa_for_receptors_points(
        rec_xy=rec_xy,
        rec_z=rec_z,
        sources=sources,
        alpha_atm=0.0,
        enable_ground=False,
        g_value=1.0,
        d_min=1.0,
        use_bands=True,
        sources_spectra=source_spec,
        receiver_height_m=4.0,
    )
    assert np.isfinite(out_g0[0])
    assert np.isfinite(out_g1[0])
    assert abs(float(out_g0[0] - out_g1[0])) < 1e-12


def test_agr_changes_with_g():
    agr0 = compute_agr_iso9613_2_octave(1000, 500.0, 2.0, 4.0, 0.0)
    agr1 = compute_agr_iso9613_2_octave(1000, 500.0, 2.0, 4.0, 1.0)
    assert np.isfinite(agr0)
    assert np.isfinite(agr1)
    assert not np.isclose(agr0, agr1)


def test_agr_frequency_dependence():
    agr_63 = compute_agr_iso9613_2_octave(63, 500.0, 2.0, 4.0, 0.5)
    agr_4000 = compute_agr_iso9613_2_octave(4000, 500.0, 2.0, 4.0, 0.5)
    assert np.isfinite(agr_63)
    assert np.isfinite(agr_4000)
    assert not np.isclose(agr_63, agr_4000)


def test_agr_vectorized_support():
    d = np.array([100.0, 500.0, 1000.0], dtype=np.float64)
    hs = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    hr = np.array([4.0, 4.0, 4.0], dtype=np.float64)
    agr = compute_agr_iso9613_2_octave(500, d, hs, hr, 0.7)
    assert isinstance(agr, np.ndarray)
    assert agr.shape == d.shape
    assert np.all(np.isfinite(agr))


def test_agr_invalid_band_raises():
    try:
        compute_agr_iso9613_2_octave(315, 500.0, 2.0, 4.0, 0.5)
    except ValueError:
        return
    assert False, 'Expected ValueError for unsupported octave band'


def test_a_weight_reconstruction():
    lw_flat = {freq: 95.0 for freq in BANDS}
    reconstructed = reconstruct_lwa_total_from_unweighted(lw_flat)
    expected = 10.0 * np.log10(np.sum([10.0 ** ((95.0 + A_WEIGHT_DB[f]) / 10.0) for f in BANDS]))
    assert abs(reconstructed - expected) < 1e-9


def test_flat_from_lwa():
    lw_band = build_source_spectrum(lwa_total=100.0, mode="FLAT_FROM_LWA")
    reconstructed = reconstruct_lwa_total_from_unweighted(lw_band)
    assert abs(reconstructed - 100.0) <= 0.01


def test_shape_scaled_matches_total_lwa():
    lw_band = build_band_lw_from_shape_scaled_to_LwA(100.0, WIND_TURBINE_SHAPE_DB, A_WEIGHT_DB)
    lwa_rec = 10.0 * np.log10(np.sum([10.0 ** ((lw_band[f] + A_WEIGHT_DB[f]) / 10.0) for f in BANDS]))
    assert abs(lwa_rec - 100.0) < 1e-6


def test_alpha_iso_monotonic_frequency():
    a125 = alpha_iso9613_1(125, 10.0, 70.0, 101.325)
    a1000 = alpha_iso9613_1(1000, 10.0, 70.0, 101.325)
    a8000 = alpha_iso9613_1(8000, 10.0, 70.0, 101.325)
    assert a125 > 0.0
    assert a1000 > 0.0
    assert a8000 > a125


def test_use_bands_output_is_finite():
    rec_xy = np.array([[50.0, 0.0]], dtype=np.float64)
    rec_z = np.array([0.0], dtype=np.float64)
    sources = [(0.0, 0.0, 0.0, 100.0)]
    sources_spectra = [{"x": 0.0, "y": 0.0, "z": 0.0, "h_src": 2.0, "lwa": 100.0, "lw_band": build_source_spectrum(100.0, "TURBINE_SHAPE_SCALED")}]
    out = compute_lpa_for_receptors_points(
        rec_xy=rec_xy,
        rec_z=rec_z,
        sources=sources,
        alpha_atm=0.0,
        enable_ground=False,
        g_value=0.5,
        d_min=1.0,
        use_bands=True,
        sources_spectra=sources_spectra,
        temperature_c=10.0,
        relative_humidity=70.0,
        pressure_kpa=101.325,
        receiver_height_m=4.0,
    )
    assert np.isfinite(out[0])


def test_no_change_when_use_bands_false_regression():
    x = np.array([[100.0]], dtype=np.float64)
    y = np.array([[0.0]], dtype=np.float64)
    z = np.array([[0.0]], dtype=np.float64)
    nodata = np.array([[False]])
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
        use_bands=False,
    )
    assert abs(float(out[0, 0]) - 49.0) <= 0.1


def test_kmet_bounds():
    k1 = kmet_iso(50.0, 60.0, 100.0, 2.0)
    assert 0.0 < k1 <= 1.0
    assert kmet_iso(50.0, 60.0, 100.0, 0.0) == 1.0
    assert kmet_iso(50.0, 60.0, 100.0, -1.0) == 1.0


def test_dz_zero_when_z_le_0():
    assert dz_iso_single_screen(1000, 50.0, 60.0, 100.0, 0.0) == 0.0


def test_dz_increases_with_z():
    z1 = dz_iso_single_screen(1000, 50.0, 60.0, 100.0, 0.1)
    z2 = dz_iso_single_screen(1000, 50.0, 60.0, 100.0, 1.0)
    assert z2 > z1


def test_abar_non_negative():
    assert abar_from_dz(1.0, 3.0) >= 0.0
    assert abar_from_dz(5.0, 2.0) >= 0.0
