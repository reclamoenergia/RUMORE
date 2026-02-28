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

WIND_TURBINE_STD = {
    8: {63: 84.0, 125: 88.0, 250: 92.0, 500: 95.0, 1000: 96.0, 2000: 94.0, 4000: 90.0, 8000: 84.0},
    10: {63: 87.0, 125: 91.0, 250: 95.0, 500: 98.0, 1000: 100.0, 2000: 98.0, 4000: 94.0, 8000: 88.0},
    12: {63: 89.0, 125: 93.0, 250: 97.0, 500: 100.0, 1000: 102.0, 2000: 100.0, 4000: 96.0, 8000: 90.0},
}


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


def _nearest_wind_bin(wind_bin):
    available = sorted(WIND_TURBINE_STD.keys())
    if wind_bin in WIND_TURBINE_STD:
        return int(wind_bin)
    return min(available, key=lambda b: abs(b - int(wind_bin)))


def nearest_wind_bin(wind_bin):
    return _nearest_wind_bin(wind_bin)


def build_source_spectrum(mode, lwa_total, lw_band_fields=None, wind_bin=None, user_offsets_db=None):
    offsets = _normalize_offsets(user_offsets_db)
    if lw_band_fields and all(int(freq) in lw_band_fields for freq in BANDS):
        return {freq: float(lw_band_fields[freq]) + offsets.get(freq, 0.0) for freq in BANDS}

    if mode == "WIND_TURBINE_STD":
        selected_bin = _nearest_wind_bin(10 if wind_bin is None else wind_bin)
        lwa_band = {
            freq: float(WIND_TURBINE_STD[selected_bin][freq]) + offsets.get(freq, 0.0)
            for freq in BANDS
        }
        return to_unweighted_band_lw(lwa_band)

    lw_band = _flat_spectrum_from_lwa_total(lwa_total)
    return {freq: lw_band[freq] + offsets.get(freq, 0.0) for freq in BANDS}


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
):
    p_tot = np.zeros_like(x_grid, dtype=np.float64)

    if use_bands:
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
            aatm = compute_aatm_broadband(alpha_atm, d)
            agr = compute_agr_simplified(enable_ground, g_value, d)

            for freq in BANDS:
                lp_band = float(bands[freq]) - (adiv + aatm + agr)
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
):
    rec_xy = np.asarray(rec_xy, dtype=np.float64)
    rec_z = np.asarray(rec_z, dtype=np.float64)

    n = rec_xy.shape[0]
    p_tot = np.zeros(n, dtype=np.float64)
    valid = np.isfinite(rec_xy[:, 0]) & np.isfinite(rec_xy[:, 1]) & np.isfinite(rec_z)

    if use_bands:
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
            aatm = compute_aatm_broadband(alpha_atm, d)
            agr = compute_agr_simplified(enable_ground, g_value, d)

            for freq in BANDS:
                lp_band = float(bands[freq]) - (adiv + aatm + agr)
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
