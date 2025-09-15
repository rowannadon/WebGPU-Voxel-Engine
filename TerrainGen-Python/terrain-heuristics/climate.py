import math
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = [
    "latitude_degrees",
    "prevailing_wind_3cell",
    "prevailing_wind",
    "directional_slope",
    "temperature_from_lat_elev",
    "precipitation_lat_bands",
    "compute_rain_shadow_advanced",
    "precipitation_orographic_advanced",
    "potential_evapotranspiration",
    "actual_evapotranspiration",
]


def latitude_degrees(h: int) -> np.ndarray:
    """Return 1D latitude array spanning 90..-90 degrees for map rows."""
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    lat = 90.0 - 180.0 * ys
    return lat.astype(np.float32)


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def prevailing_wind_3cell(
    lat_deg: np.ndarray,
    eq_blend_deg: float = 5.0,
    ferrel_tilt: float = 0.12,
    polar_tilt: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth 3-cell atmospheric circulation model."""
    h = lat_deg.shape[0]
    lat = lat_deg.reshape(h, 1).astype(np.float32)
    a = np.abs(lat)

    w_tr = 1.0 - _smoothstep(25.0, 35.0, a)
    w_po = _smoothstep(55.0, 65.0, a)
    w_we = np.clip(1.0 - w_tr - w_po, 0.0, 1.0)

    s = np.tanh(np.radians(lat) / np.radians(eq_blend_deg))

    u_tr = -s
    v_tr = s

    u_we = np.ones_like(lat)
    v_we = ferrel_tilt * s

    u_po = -np.ones_like(lat)
    v_po = -polar_tilt * s

    u = w_tr * u_tr + w_we * u_we + w_po * u_po
    v = w_tr * v_tr + w_we * v_we + w_po * v_po

    mag = np.sqrt(u * u + v * v)
    tiny = 1e-8
    mask = mag < tiny
    if np.any(mask):
        pref = np.sign(w_we - w_po)
        pref = np.where(pref == 0, 1.0, pref)
        u[mask] = pref[mask]
        v[mask] = (ferrel_tilt - polar_tilt) * 0.5 * s[mask]
        mag[mask] = np.sqrt(u[mask] * u[mask] + v[mask] * v[mask])

    u /= mag
    v /= mag
    return u.astype(np.float32), v.astype(np.float32)


def prevailing_wind(lat_deg: np.ndarray, eq_blend_deg: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return constant north-to-south wind (legacy helper)."""
    h = lat_deg.shape[0]
    lat_deg.reshape(h, 1)  # shape alignment only
    u = np.ones((h, 1), dtype=np.float32)
    v = np.ones_like(u)
    mag = np.sqrt(u * u + v * v)
    mag[mag == 0] = 1.0
    u /= mag
    v /= mag
    return u.astype(np.float32), v.astype(np.float32)


def directional_slope(dzdx: np.ndarray, dzdy: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project gradient onto wind direction."""
    return (dzdx * u + dzdy * v).astype(np.float32)


def temperature_from_lat_elev(
    lat_deg: np.ndarray,
    elev: np.ndarray,
    lapse_c_per_km: float,
    t_equator_c: float,
    t_pole_c: float,
) -> np.ndarray:
    lat_abs = np.abs(lat_deg)[:, None]
    coslat = np.cos(np.radians(lat_abs))
    t0 = t_pole_c + (t_equator_c - t_pole_c) * (coslat ** 1.0)
    temp = t0 - (lapse_c_per_km * (elev / 1000.0))
    return temp.astype(np.float32)


def precipitation_lat_bands(lat_deg: np.ndarray, base_mm: float = 1200.0) -> np.ndarray:
    lat_abs = np.abs(lat_deg)[:, None]

    def g(center, sigma, sign=1.0):
        return sign * np.exp(-0.5 * ((lat_abs - center) / sigma) ** 2)

    patt = 1.0
    patt += 0.9 * g(0.0, 12.0)
    patt += 0.6 * g(60.0, 10.0)
    patt += -0.9 * g(30.0, 10.0)
    patt += -0.5 * g(85.0, 5.0)
    P = base_mm * np.clip(patt, 0.1, None)
    return P.astype(np.float32)


def compute_rain_shadow_advanced(
    elev: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    cellsize: float,
    max_distance_km: float = 50.0,
    shadow_decay_km: float = 20.0,
    height_threshold_m: float = 100.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Vectorized rain-shadow approximation based on average wind."""
    h, w = elev.shape
    u = wind_u[:, 0] if wind_u.shape[1] == 1 else wind_u.mean(axis=1)
    v = wind_v[:, 0] if wind_v.shape[1] == 1 else wind_v.mean(axis=1)
    avg_u = np.mean(u)
    avg_v = np.mean(v)
    wind_mag = np.sqrt(avg_u ** 2 + avg_v ** 2)
    if wind_mag < 1e-6:
        return np.ones((h, w), dtype=np.float32)
    avg_u /= wind_mag
    avg_v /= wind_mag

    n_steps = min(20, int(max_distance_km * 1000 / cellsize))
    step_size = max(1, int(cellsize / 1000))
    shadow_acc = np.zeros((h, w), dtype=np.float32)

    for step in range(1, n_steps + 1):
        shift_x = int(round(step * step_size * avg_u))
        shift_y = int(round(step * step_size * avg_v))
        upwind_elev = np.full_like(elev, -9999.0)

        src_x_start = max(0, -shift_x)
        src_x_end = min(w, w - shift_x)
        src_y_start = max(0, -shift_y)
        src_y_end = min(h, h - shift_y)

        dst_x_start = max(0, shift_x)
        dst_x_end = min(w, w + shift_x)
        dst_y_start = max(0, shift_y)
        dst_y_end = min(h, h + shift_y)

        if (
            src_x_end > src_x_start
            and src_y_end > src_y_start
            and dst_x_end > dst_x_start
            and dst_y_end > dst_y_start
        ):
            upwind_elev[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = elev[
                src_y_start:src_y_end, src_x_start:src_x_end
            ]

        valid = upwind_elev > -9999.0
        height_diff = np.zeros_like(elev)
        height_diff[valid] = np.maximum(0.0, upwind_elev[valid] - elev[valid])

        significant = height_diff > height_threshold_m
        dist_km = step * step_size * cellsize / 1000.0
        decay = np.exp(-dist_km / shadow_decay_km)
        shadow_contrib = (
            significant
            * np.minimum(0.3, height_diff / 1000.0)
            * decay
            * max(0.0, strength)
        )
        shadow_acc = np.minimum(0.8, shadow_acc + shadow_contrib)

    shadow_mult = 1.0 - shadow_acc
    shadow_mult = gaussian_filter(shadow_mult, sigma=1.0)
    return np.clip(shadow_mult, 0.2, 1.0).astype(np.float32)


def precipitation_orographic_advanced(
    P_lat: np.ndarray,
    elev: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    dzdx: np.ndarray,
    dzdy: np.ndarray,
    dist_coast_m: np.ndarray,
    cellsize: float,
    alpha: float = 2.0,
    beta: float = 0.15,
    coast_decay_m: float = 150000.0,
    coast_min_frac: float = 0.75,
    use_advanced_shadow: bool = True,
    shadow_max_distance_km: float = 400.0,
    shadow_decay_km: float = 150.0,
    shadow_height_threshold_m: float = 150.0,
    shadow_strength: float = 1.0,
) -> np.ndarray:
    """Combine latitudinal precipitation with orographic lift and rain shadow."""
    dir_slope = (dzdx * wind_u + dzdy * wind_v).astype(np.float32)
    lift = 1.0 + alpha * np.maximum(0.0, dir_slope)

    if use_advanced_shadow:
        shadow_multiplier = compute_rain_shadow_advanced(
            elev,
            wind_u,
            wind_v,
            cellsize,
            max_distance_km=shadow_max_distance_km,
            shadow_decay_km=shadow_decay_km,
            height_threshold_m=shadow_height_threshold_m,
            strength=shadow_strength,
        )
    else:
        shadow_multiplier = 1.0 / (1.0 + beta * np.maximum(0.0, -dir_slope))

    coast = coast_min_frac + (1.0 - coast_min_frac) * np.exp(
        -dist_coast_m / max(1.0, coast_decay_m)
    )

    P = P_lat * lift * shadow_multiplier * coast
    return np.clip(P, 0.0, None).astype(np.float32)


def potential_evapotranspiration(
    temp_c: np.ndarray,
    lat_deg: np.ndarray,
    k: float = 20.0,
) -> np.ndarray:
    coslat = np.cos(np.radians(np.abs(lat_deg)))[:, None]
    coslat = np.clip(coslat, 0.2, 1.0)
    term = np.clip(temp_c + 5.0, 0.0, None)
    return (k * coslat * term).astype(np.float32)


def actual_evapotranspiration(P_mm: np.ndarray, PET_mm: np.ndarray) -> np.ndarray:
    eps = 1e-6
    return ((P_mm * PET_mm) / (P_mm + PET_mm + eps)).astype(np.float32)
