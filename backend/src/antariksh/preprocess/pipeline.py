"""Fixed preprocessing order (see CONTRACT.md):
quality mask (applied at download) -> drop NaN -> median normalize
-> flatten (Savitzky-Golay, optionally with detected transits masked)
-> upper sigma clip -> stitch (no-op, one sector per target).
"""
from dataclasses import dataclass

import lightkurve as lk
import numpy as np

from antariksh.config import CONTRACT, RAW_DIR
from antariksh.detect.snr import in_transit_mask


@dataclass
class Curve:
    time: np.ndarray
    flux: np.ndarray        # flattened, normalized to ~1
    flux_err: np.ndarray
    raw_time: np.ndarray
    raw_flux: np.ndarray    # normalized, NOT flattened (kept for plots)


def load_raw(tic: int) -> lk.LightCurve:
    return lk.read(RAW_DIR / f"TIC{int(tic)}.fits")


def _arr(x) -> np.ndarray:
    x = getattr(x, "value", x)
    return np.asarray(x, dtype=float)


def preprocess(lc: lk.LightCurve, ephem=None) -> Curve:
    """ephem = (period_d, t0, duration_d) of a detected signal. If given, those in-transit
    points (plus a margin) are excluded from the flatten fit so the trend cannot absorb the dip."""
    cfg = CONTRACT["preprocess"]
    lc = lc.remove_nans().normalize()
    raw = lc.copy()

    mask = None
    if ephem is not None:
        period, t0, dur = ephem
        margin = cfg["flatten"].get("two_pass_mask_margin", 1.5)
        mask = in_transit_mask(_arr(lc.time), period, t0, margin * dur)

    win = cfg["flatten"]["window_length"]
    if win >= len(lc):                      # very short curve: shrink window, keep it odd
        win = (len(lc) // 2) * 2 - 1
    flat = lc.flatten(window_length=win, polyorder=cfg["flatten"]["polyorder"], mask=mask)
    flat = flat.remove_outliers(sigma_upper=cfg["clip"]["sigma_upper"], sigma_lower=float("inf"))

    t, f, e = _arr(flat.time), _arr(flat.flux), _arr(flat.flux_err)
    if not np.isfinite(e).any():
        e = np.full_like(f, np.nan)
    e = np.where(np.isfinite(e), e, np.nanmedian(e) if np.isfinite(e).any() else np.std(f))
    return Curve(t, f, e, _arr(raw.time), _arr(raw.flux))
