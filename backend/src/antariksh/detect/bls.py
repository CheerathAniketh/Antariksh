from dataclasses import dataclass, field

import numpy as np
from astropy.timeseries import BoxLeastSquares

from antariksh.config import CONTRACT
from antariksh.detect.snr import transit_snr


@dataclass
class BLSResult:
    period_d: float
    t0: float
    duration_d: float
    depth_ppm: float
    snr: float
    n_transits: int
    periods: np.ndarray = field(repr=False)
    power: np.ndarray = field(repr=False)
    stats: dict = field(default_factory=dict, repr=False)   # odd/even, secondary, etc. for features


def run_bls(time, flux, flux_err=None) -> BLSResult:
    c = CONTRACT["bls"]
    durations = np.asarray(c["durations_h"], dtype=float) / 24.0
    max_duty = float(c.get("max_duty_cycle", 1.0))

    model = BoxLeastSquares(time, flux, dy=flux_err)
    res = model.autopower(
        durations,
        minimum_period=c["period_min_d"],
        maximum_period=c["period_max_d"],
        frequency_factor=c["frequency_factor"],
    )

    ok = (np.asarray(res.duration) / np.asarray(res.period)) <= max_duty
    power = np.where(ok, np.asarray(res.power), -np.inf)
    i = int(np.argmax(power)) if ok.any() else int(np.argmax(res.power))

    period, dur, t0 = float(res.period[i]), float(res.duration[i]), float(res.transit_time[i])
    depth = float(res.depth[i])

    stats = model.compute_stats(period, dur, t0)
    n_tr = int(np.sum(np.asarray(stats["per_transit_count"]) > 0))
    snr = transit_snr(np.asarray(time), np.asarray(flux), period, t0, dur, depth)

    return BLSResult(period, t0, dur, depth * 1e6, snr, n_tr, res.period, res.power, stats)
