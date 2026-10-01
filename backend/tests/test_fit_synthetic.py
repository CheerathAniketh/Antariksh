import batman
import numpy as np

from antariksh.fit.transit_fit import U_LD, duration_depth, fit_transit


def test_fit_recovers_injected_params():
    rng = np.random.default_rng(1)
    t = np.arange(0, 27, 2 / 60 / 24)
    P, t0, rp, a, b = 3.7, 1.2, 0.07, 11.0, 0.3
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = t0, P, rp, a
    p.inc = float(np.degrees(np.arccos(b / a)))
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, U_LD, "quadratic"
    f = batman.TransitModel(p, t).light_curve(p) + rng.normal(0, 500e-6, t.size)
    e = np.full_like(t, 500e-6)

    T_true, D_true = duration_depth(t0, P, rp, a, b)
    r = fit_transit(t, f, e, period=3.7012, t0=1.205, duration_d=T_true / 24 * 1.2, depth_ppm=D_true * 0.8)
    assert r["fit_ok"]
    assert abs(r["fit_period_d"] - P) / P < 5e-4
    assert abs(r["fit_depth_ppm"] - D_true) / D_true < 0.10
    assert abs(r["fit_duration_h"] - T_true) / T_true < 0.10
    assert r["fit_depth_err_ppm"] > 0 and r["fit_duration_err_h"] > 0
