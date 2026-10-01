import lightkurve as lk
import numpy as np

from antariksh.pipeline import run_two_pass


def make_curve(with_transit=True, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 27, 2 / 60 / 24)                       # 27 d at 2-min cadence
    f = 1 + rng.normal(0, 1000e-6, t.size)                  # 1000 ppm white noise
    f += 0.003 * np.sin(2 * np.pi * t / 5.0)                # slow stellar variability
    if with_transit:
        P, t0, dur, depth = 3.7, 1.2, 2.5 / 24, 5000e-6
        ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
        f[np.abs(ph) < dur / 2] -= depth
    return t, f, np.full_like(t, 1000e-6)


def test_recovers_injected_transit():
    t, f, e = make_curve(True)
    c, r, _ = run_two_pass(lk.LightCurve(time=t, flux=f, flux_err=e))
    assert abs(r.period_d - 3.7) / 3.7 < 0.01
    assert abs(r.depth_ppm - 5000) / 5000 < 0.30
    assert r.snr > 10


def test_noise_only_is_low_snr():
    t, f, e = make_curve(False)
    c, r, _ = run_two_pass(lk.LightCurve(time=t, flux=f, flux_err=e))
    assert r.snr < 8
