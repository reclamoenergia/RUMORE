import math

import numpy as np

BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

A_WEIGHT_DB = {
    63: -26.2,
    125: -16.1,
    250: -8.6,
    500: -3.2,
    1000: 0.0,
    2000: 1.2,
    4000: 1.0,
    8000: -1.1,
}

# Template relativo di forma spettrale aerogeneratore (offset in dB).
# Viene sempre riscalato sul totale LwA utente tramite build_band_lw_from_shape_scaled_to_LwA.
WIND_TURBINE_SHAPE_DB = {
    63: -8.0,
    125: -5.0,
    250: -2.5,
    500: 0.0,
    1000: 1.0,
    2000: 0.0,
    4000: -3.0,
    8000: -8.0,
}

ISO9613_REFERENCE_T_K = 293.15
ISO9613_TRIPLE_POINT_T_K = 273.16
ISO9613_REFERENCE_P_PA = 101325.0


def _source_tuple(src):
    if isinstance(src, dict):
        return float(src["x_s"]), float(src["y_s"]), float(src["z_s"]), float(src["lwa"])
    if len(src) >= 5:
        return float(src[1]), float(src[2]), float(src[3]), float(src[4])
    return float(src[0]), float(src[1]), float(src[2]), float(src[3])


def _source_spectrum_tuple(src):
    if isinstance(src, dict):
        return float(src["x"]), float(src["y"]), float(src["z"]), float(src["lwa"]), src.get("lw_band")
    return float(src[0]), float(src[1]), float(src[2]), float(src[3]), src[4] if len(src) > 4 else None


def _sum_db(levels_db):
    levels = np.asarray(levels_db, dtype=np.float64)
    if levels.size == 0:
        return -np.inf
    powers = np.power(10.0, levels / 10.0)
    total = np.sum(powers)
    if total <= 0.0:
        return -np.inf
    return float(10.0 * np.log10(total))


def _normalize_offsets(user_offsets_db):
    if not user_offsets_db:
        return {}
    return {int(freq): float(value) for freq, value in user_offsets_db.items() if int(freq) in BANDS}


def to_unweighted_band_lw(lwa_band_dict):
    return {int(freq): float(lwa_band_dict[freq]) - A_WEIGHT_DB[int(freq)] for freq in BANDS}


def _reconstruct_lwa_total_from_unweighted(lw_band_dict):
    return _sum_db([float(lw_band_dict[freq]) + A_WEIGHT_DB[freq] for freq in BANDS])


def reconstruct_lwa_total_from_unweighted(lw_band_dict):
    return _reconstruct_lwa_total_from_unweighted(lw_band_dict)


def _flat_spectrum_from_lwa_total(lwa_total):
    a_power_sum = np.sum([np.power(10.0, A_WEIGHT_DB[freq] / 10.0) for freq in BANDS])
    base = float(lwa_total) - 10.0 * np.log10(a_power_sum)
    return {freq: base for freq in BANDS}


def build_band_lw_from_shape_scaled_to_LwA(lwa_total, shape_db, a_weight_db=None):
    a_weight = A_WEIGHT_DB if a_weight_db is None else a_weight_db
    k_sum = np.sum([np.power(10.0, (float(shape_db[freq]) + float(a_weight[freq])) / 10.0) for freq in BANDS])
    k = 10.0 * np.log10(k_sum)
    s = float(lwa_total) - k
    return {freq: s + float(shape_db[freq]) for freq in BANDS}


def build_source_spectrum(lwa_total, mode, lw_band_fields=None, user_offsets_db=None):
    offsets = _normalize_offsets(user_offsets_db)

    if lw_band_fields and all(int(freq) in lw_band_fields for freq in BANDS):
        base = {freq: float(lw_band_fields[freq]) for freq in BANDS}
    elif mode == "TURBINE_SHAPE_SCALED":
        base = build_band_lw_from_shape_scaled_to_LwA(lwa_total, WIND_TURBINE_SHAPE_DB, A_WEIGHT_DB)
    else:
        base = _flat_spectrum_from_lwa_total(lwa_total)

    return {freq: base[freq] + offsets.get(freq, 0.0) for freq in BANDS}


def saturation_vapor_pressure(T_K):
    exponent = -6.8346 * np.power(ISO9613_TRIPLE_POINT_T_K / T_K, 1.261) + 4.6151
    return ISO9613_REFERENCE_P_PA * np.power(10.0, exponent)


def molar_concentration_water_vapor(relative_humidity, p_sat_pa, p_atm_pa):
    rh = np.clip(float(relative_humidity), 0.1, 100.0)
    return (rh / 100.0) * (p_sat_pa / p_atm_pa)


def relaxation_frequencies_oxygen_nitrogen(T_K, RH, p_atm_pa):
    h = molar_concentration_water_vapor(RH, saturation_vapor_pressure(T_K), p_atm_pa)
    p_ratio = p_atm_pa / ISO9613_REFERENCE_P_PA
    t_ratio = T_K / ISO9613_REFERENCE_T_K

    fr_o = p_ratio * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = p_ratio * np.power(t_ratio, -0.5) * (9.0 + 280.0 * h * np.exp(-4.170 * (np.power(t_ratio, -1.0 / 3.0) - 1.0)))
    return float(fr_o), float(fr_n)


def alpha_iso9613_1(f_hz, T_C, RH_percent, p_kPa=101.325):
    f = float(f_hz)
    t_k = float(T_C) + 273.15
    p_pa = float(p_kPa) * 1000.0
    rh = np.clip(float(RH_percent), 0.1, 100.0)

    fr_o, fr_n = relaxation_frequencies_oxygen_nitrogen(t_k, rh, p_pa)
    t_ratio = t_k / ISO9613_REFERENCE_T_K

    classical = 1.84e-11 * (ISO9613_REFERENCE_P_PA / p_pa) * np.sqrt(t_ratio)
    oxygen = 0.01275 * np.exp(-2239.1 / t_k) / (fr_o + (f * f) / fr_o)
    nitrogen = 0.1068 * np.exp(-3352.0 / t_k) / (fr_n + (f * f) / fr_n)
    molecular = np.power(t_ratio, -2.5) * (oxygen + nitrogen)

    alpha = 8.686 * (f * f) * (classical + molecular)
    return float(alpha)


def compute_adiv(d):
    return 20.0 * np.log10(d) + 11.0


def compute_aatm_broadband(alpha_atm, d):
    return alpha_atm * d


def compute_agr_simplified(enable_ground, g_value, d):
    if not enable_ground:
        if np.isscalar(d):
            return 0.0
        return np.zeros_like(d, dtype=np.float64)

    d0 = 200.0
    agr_max = 3.0
    k = agr_max * g_value
    agr = k * (1.0 - np.exp(-d / d0))
    return np.clip(agr, 0.0, agr_max)


def compute_lpa_from_sources_grid(
    x_grid,
    y_grid,
    z_rec,
    sources,
    alpha_atm,
    enable_ground,
    g_value,
    d_min,
    nodata_mask,
    use_bands=False,
    sources_spectra=None,
    temperature_c=10.0,
    relative_humidity=70.0,
    pressure_kpa=101.325,
):
    p_tot = np.zeros_like(x_grid, dtype=np.float64)

    if use_bands:
        alpha_per_band = {freq: alpha_iso9613_1(freq, temperature_c, relative_humidity, pressure_kpa) for freq in BANDS}
        active_sources = sources_spectra or []
        if not active_sources:
            for src in sources:
                x_s, y_s, z_s, lwa = _source_tuple(src)
                active_sources.append({"x": x_s, "y": y_s, "z": z_s, "lwa": lwa, "lw_band": _flat_spectrum_from_lwa_total(lwa)})

        for src in active_sources:
            x_s, y_s, z_s, lwa, lw_band = _source_spectrum_tuple(src)
            bands = lw_band or _flat_spectrum_from_lwa_total(lwa)
            dxy = np.hypot(x_grid - x_s, y_grid - y_s)
            dz = z_rec - z_s
            d = np.sqrt(dxy * dxy + dz * dz)
            d = np.maximum(d, d_min)

            adiv = compute_adiv(d)
            agr = compute_agr_simplified(enable_ground, g_value, d)

            for freq in BANDS:
                aatm_band = alpha_per_band[freq] * d
                lp_band = float(bands[freq]) - (adiv + aatm_band + agr)
                p_tot += np.power(10.0, (lp_band + A_WEIGHT_DB[freq]) / 10.0)

        out = np.full_like(x_grid, np.nan, dtype=np.float64)
        positive = p_tot > 0.0
        out[positive] = 10.0 * np.log10(p_tot[positive])
        out[nodata_mask] = np.nan
        return out

    for src in sources:
        x_s, y_s, z_s, lwa = _source_tuple(src)
        dxy = np.hypot(x_grid - x_s, y_grid - y_s)
        dz = z_rec - z_s
        d = np.sqrt(dxy * dxy + dz * dz)
        d = np.maximum(d, d_min)

        adiv = compute_adiv(d)
        aatm = compute_aatm_broadband(alpha_atm, d)
        agr = compute_agr_simplified(enable_ground, g_value, d)

        lp = lwa - (adiv + aatm + agr)
        p_tot += np.power(10.0, lp / 10.0)

    out = np.full_like(x_grid, np.nan, dtype=np.float64)
    positive = p_tot > 0.0
    out[positive] = 10.0 * np.log10(p_tot[positive])
    out[nodata_mask] = np.nan
    return out


def compute_lpa_for_receptors_points(
    rec_xy,
    rec_z,
    sources,
    alpha_atm,
    enable_ground,
    g_value,
    d_min,
    use_bands=False,
    sources_spectra=None,
    temperature_c=10.0,
    relative_humidity=70.0,
    pressure_kpa=101.325,
):
    rec_xy = np.asarray(rec_xy, dtype=np.float64)
    rec_z = np.asarray(rec_z, dtype=np.float64)

    n = rec_xy.shape[0]
    p_tot = np.zeros(n, dtype=np.float64)
    valid = np.isfinite(rec_xy[:, 0]) & np.isfinite(rec_xy[:, 1]) & np.isfinite(rec_z)

    if use_bands:
        alpha_per_band = {freq: alpha_iso9613_1(freq, temperature_c, relative_humidity, pressure_kpa) for freq in BANDS}
        active_sources = sources_spectra or []
        if not active_sources:
            for src in sources:
                x_s, y_s, z_s, lwa = _source_tuple(src)
                active_sources.append({"x": x_s, "y": y_s, "z": z_s, "lwa": lwa, "lw_band": _flat_spectrum_from_lwa_total(lwa)})

        for src in active_sources:
            x_s, y_s, z_s, lwa, lw_band = _source_spectrum_tuple(src)
            bands = lw_band or _flat_spectrum_from_lwa_total(lwa)
            dxy = np.hypot(rec_xy[:, 0] - x_s, rec_xy[:, 1] - y_s)
            dz = rec_z - z_s
            d = np.sqrt(dxy * dxy + dz * dz)
            d = np.maximum(d, d_min)

            adiv = compute_adiv(d)
            agr = compute_agr_simplified(enable_ground, g_value, d)

            for freq in BANDS:
                aatm_band = alpha_per_band[freq] * d
                lp_band = float(bands[freq]) - (adiv + aatm_band + agr)
                contrib = np.power(10.0, (lp_band + A_WEIGHT_DB[freq]) / 10.0)
                p_tot += np.where(valid, contrib, 0.0)

        out = np.full(n, np.nan, dtype=np.float64)
        pos = (p_tot > 0.0) & valid
        out[pos] = 10.0 * np.log10(p_tot[pos])
        return out

    for src in sources:
        x_s, y_s, z_s, lwa = _source_tuple(src)
        dxy = np.hypot(rec_xy[:, 0] - x_s, rec_xy[:, 1] - y_s)
        dz = rec_z - z_s
        d = np.sqrt(dxy * dxy + dz * dz)
        d = np.maximum(d, d_min)

        adiv = compute_adiv(d)
        aatm = compute_aatm_broadband(alpha_atm, d)
        agr = compute_agr_simplified(enable_ground, g_value, d)

        lp = lwa - (adiv + aatm + agr)
        contrib = np.power(10.0, lp / 10.0)
        p_tot += np.where(valid, contrib, 0.0)

    out = np.full(n, np.nan, dtype=np.float64)
    pos = (p_tot > 0.0) & valid
    out[pos] = 10.0 * np.log10(p_tot[pos])
    return out
