import numpy as np


def in_transit_mask(time, period, t0, duration_d):
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    return np.abs(phase) < 0.5 * duration_d


def transit_snr(time, flux, period, t0, duration_d, depth):
    """SNR = depth / (sigma_oot / sqrt(n_in_transit)); sigma_oot is a robust (MAD) estimate."""
    m = in_transit_mask(time, period, t0, duration_d)
    n_in = int(m.sum())
    if n_in < 2 or (~m).sum() < 10:
        return 0.0
    oot = flux[~m]
    sigma = 1.4826 * np.median(np.abs(oot - np.median(oot)))
    if sigma <= 0:
        return 0.0
    return float(depth / (sigma / np.sqrt(n_in)))
