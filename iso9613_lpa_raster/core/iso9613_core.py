import math

import numpy as np

BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]


def _source_tuple(src):
    if isinstance(src, dict):
        return float(src["x_s"]), float(src["y_s"]), float(src["z_s"]), float(src["lwa"])
    if len(src) >= 5:
        return float(src[1]), float(src[2]), float(src[3]), float(src[4])
    return float(src[0]), float(src[1]), float(src[2]), float(src[3])


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
):
    p_tot = np.zeros_like(x_grid, dtype=np.float64)

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
):
    rec_xy = np.asarray(rec_xy, dtype=np.float64)
    rec_z = np.asarray(rec_z, dtype=np.float64)

    n = rec_xy.shape[0]
    p_tot = np.zeros(n, dtype=np.float64)
    valid = np.isfinite(rec_xy[:, 0]) & np.isfinite(rec_xy[:, 1]) & np.isfinite(rec_z)

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
