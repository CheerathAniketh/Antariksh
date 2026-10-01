"""Transit fit: batman + scipy least_squares around a BLS ephemeris.
Assumptions (stated in the report): circular orbit, quadratic limb darkening fixed to generic TESS-band
values, flattened (two-pass) curve. Uncertainties: covariance from the Jacobian scaled by the residual
variance, propagated to depth and duration by Monte Carlo draws."""
import batman
import numpy as np
from scipy.optimize import least_squares

U_LD = [0.35, 0.25]
N_MC = 300


def _params(t0, per, rp, a, b):
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = float(t0), float(per), float(rp), float(a)
    p.inc = float(np.degrees(np.arccos(np.clip(b / a, 0.0, 1.0))))
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, list(U_LD), "quadratic"
    return p


def duration_depth(t0, per, rp, a, b):
    """(T14 in hours, mid-transit depth in ppm). T14 is the first-to-fourth contact duration."""
    sini = np.sqrt(max(1 - (b / a) ** 2, 1e-12))
    arg = np.sqrt(max((1 + rp) ** 2 - b ** 2, 0.0)) / (a * sini)
    t14 = per / np.pi * np.arcsin(min(arg, 1.0)) * 24
    p = _params(t0, per, rp, a, b)
    depth = (1 - batman.TransitModel(p, np.array([t0], float)).light_curve(p)[0]) * 1e6
    return float(t14), float(depth)


def fit_transit(time, flux, flux_err, period, t0, duration_d, depth_ppm, seed=42) -> dict:
    time, flux, flux_err = (np.asarray(x, float) for x in (time, flux, flux_err))
    # reference epoch near the middle of the data decorrelates period and t0
    t0m = t0 + np.round((np.mean(time) - t0) / period) * period
    win = max(3 * duration_d, 0.25)
    ph = ((time - t0m + 0.5 * period) % period) - 0.5 * period
    sel = np.abs(ph) < win
    if sel.sum() < 50:
        return {"fit_ok": False, "fit_error": "too few in-window points"}
    t, f, e = time[sel], flux[sel], flux_err[sel]

    dt = max(duration_d, 0.05)
    lo = np.array([t0m - dt, period * 0.99, 1e-3, 1.5, 0.0])
    hi = np.array([t0m + dt, period * 1.01, 0.6, 300.0, 1.0])
    rp0 = float(np.clip(np.sqrt(max(depth_ppm, 50.0) * 1e-6), 0.005, 0.5))
    a0 = float(np.clip(period / (np.pi * duration_d), 2.0, 150.0))

    p = _params(t0m, period, rp0, a0, 0.3)
    m = batman.TransitModel(p, t)

    def resid(x):
        p.t0, p.per, p.rp, p.a = x[0], x[1], x[2], x[3]
        p.inc = float(np.degrees(np.arccos(np.clip(x[4] / x[3], 0.0, 1.0))))
        return f - m.light_curve(p)

    best = None
    for b0 in (0.1, 0.5, 0.85):
        r = least_squares(resid, [t0m, period, rp0, a0, b0], bounds=(lo, hi),
                          x_scale=[0.002, 0.0005, 0.01, 2.0, 0.2])
        if best is None or r.cost < best.cost:
            best = r
    x = best.x
    if not np.all(np.isfinite(x)):
        return {"fit_ok": False, "fit_error": "non-finite solution"}

    dof = max(len(t) - 5, 1)
    cov = np.linalg.pinv(best.jac.T @ best.jac) * (np.sum(best.fun ** 2) / dof)
    err = np.sqrt(np.clip(np.diag(cov), 0, None))

    rng = np.random.default_rng(seed)
    draws = np.clip(rng.multivariate_normal(x, cov, N_MC, check_valid="ignore"), lo, hi)
    dd = np.array([duration_depth(*d) for d in draws])
    d_err = float(np.nanstd(dd[:, 1])) if np.isfinite(dd[:, 1]).any() else np.nan
    t_err = float(np.nanstd(dd[:, 0])) if np.isfinite(dd[:, 0]).any() else np.nan

    t14, depth = duration_depth(*x)
    return {
        "fit_ok": True, "fit_error": "",
        "fit_period_d": float(x[1]), "fit_period_err_d": float(err[1]),
        "fit_t0": float(x[0]), "fit_rp": float(x[2]), "fit_a": float(x[3]), "fit_b": float(x[4]),
        "fit_depth_ppm": depth, "fit_depth_err_ppm": d_err,
        "fit_duration_h": t14, "fit_duration_err_h": t_err,
        "fit_rchi2": float(np.sum((best.fun / e) ** 2) / dof), "fit_n_points": int(len(t)),
    }


def model_flux(time, fit):
    """Best-fit batman model flux at `time`, from a fit_transit() result dict."""
    p = _params(fit["fit_t0"], fit["fit_period_d"], fit["fit_rp"], fit["fit_a"], fit["fit_b"])
    return batman.TransitModel(p, np.asarray(time, float)).light_curve(p)
