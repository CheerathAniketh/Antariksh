"""Light-curve-only features for one BLS candidate. Inputs are the two-pass flattened curve and the BLS ephemeris."""
import numpy as np


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _depth(flux, mask, base, sigma):
    n = int(mask.sum())
    if n < 2 or not np.isfinite(sigma):
        return np.nan, np.nan, n
    return base - float(np.mean(flux[mask])), sigma / np.sqrt(n), n


def extract_features(time, flux, period, t0, dur, depth_ppm, snr, n_transits) -> dict:
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    epoch = np.floor((time - t0 + 0.5 * period) / period).astype(int)

    intr = np.abs(phase) < 0.5 * dur
    oot = np.abs(phase) > 1.5 * dur
    sigma = _mad(flux[oot]) if oot.sum() > 10 else np.nan
    base = float(np.median(flux[oot])) if oot.sum() > 10 else 1.0

    d_all, e_all, _ = _depth(flux, intr, base, sigma)
    d_odd, e_odd, _ = _depth(flux, intr & (epoch % 2 == 1), base, sigma)
    d_even, e_even, _ = _depth(flux, intr & (epoch % 2 == 0), base, sigma)

    # secondary eclipse window: centred on phase 0.5
    phase2 = ((time - t0) % period) - 0.5 * period
    d_sec, e_sec, _ = _depth(flux, np.abs(phase2) < 0.5 * dur, base, sigma)

    # V vs U shape: central third depth over full-duration depth (~1 = flat-bottomed, >1 = V)
    d_cen, _, _ = _depth(flux, np.abs(phase) < dur / 6, base, sigma)

    sig_oe = np.sqrt(e_odd**2 + e_even**2) if np.isfinite(e_odd) and np.isfinite(e_even) else np.nan
    dur_h = dur * 24
    return {
        "period_d": period,
        "duration_h": dur_h,
        "duty_cycle": dur / period,
        "bls_depth_ppm": depth_ppm,
        "bls_snr": snr,
        "n_transits": n_transits,
        "depth_meas_ppm": d_all * 1e6 if np.isfinite(d_all) else np.nan,
        "depth_per_hour_ppm": depth_ppm / dur_h,
        "oot_scatter_ppm": sigma * 1e6,
        "odd_even_sigma": abs(d_odd - d_even) / sig_oe if np.isfinite(sig_oe) and sig_oe > 0 else np.nan,
        "odd_even_rel": abs(d_odd - d_even) / abs(d_all) if np.isfinite(d_all) and d_all != 0 else np.nan,
        "sec_depth_ppm": d_sec * 1e6 if np.isfinite(d_sec) else np.nan,
        "sec_snr": d_sec / e_sec if np.isfinite(e_sec) and e_sec > 0 else np.nan,
        "sec_ratio": d_sec / d_all if np.isfinite(d_all) and d_all != 0 else np.nan,
        "v_ratio": d_cen / d_all if np.isfinite(d_all) and d_all != 0 else np.nan,
    }
