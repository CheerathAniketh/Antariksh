"""Two-pass detection, plus the full single-light-curve analysis shared by the API and the figure script."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from antariksh.classify.baseline import prep_lr
from antariksh.config import ARTIFACTS_DIR
from antariksh.detect.bls import run_bls
from antariksh.features.extract import extract_features
from antariksh.fit.transit_fit import fit_transit
from antariksh.preprocess.pipeline import preprocess

SNR_DETECT = 7.0      # conventional detection threshold, fixed a priori
MIN_TRANSITS = 2


def run_two_pass(lc):
    """-> (final curve, final BLS result, pass-1 BLS result)."""
    c1 = preprocess(lc)
    r1 = run_bls(c1.time, c1.flux, c1.flux_err)
    c2 = preprocess(lc, ephem=(r1.period_d, r1.t0, r1.duration_d))
    r2 = run_bls(c2.time, c2.flux, c2.flux_err)
    return c2, r2, r1


@lru_cache(maxsize=1)
def _models():
    return joblib.load(ARTIFACTS_DIR / "models" / "baseline.joblib")


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _flags(f, snr, n_tr):
    """Fixed, physically motivated vetting flags. These are rules, not a trained classifier."""
    out = []
    if snr < SNR_DETECT:
        out.append("low_snr")
    if n_tr < 3:
        out.append("few_transits")
    if np.isfinite(f["odd_even_sigma"]) and f["odd_even_sigma"] > 3 and f["odd_even_rel"] > 0.2:
        out.append("odd_even_depth_mismatch")
    if np.isfinite(f["sec_snr"]) and f["sec_snr"] > 3 and f["sec_ratio"] > 0.1:
        out.append("secondary_eclipse")
    return out


def analyze_lightcurve(lc) -> dict:
    c, r, _ = run_two_pass(lc)
    c = preprocess(lc, ephem=(r.period_d, r.t0, r.duration_d))
    f = extract_features(c.time, c.flux, r.period_d, r.t0, r.duration_d, r.depth_ppm, r.snr, r.n_transits)
    m = _models()
    X = pd.DataFrame([f])
    p_gb = float(m["grad_boost"].predict_proba(X[m["features"]])[0, 1])
    p_lr = float(m["logreg"].predict_proba(prep_lr(X))[0, 1])
    fit = fit_transit(c.time, c.flux, c.flux_err, r.period_d, r.t0, r.duration_d, r.depth_ppm)
    return {
        "detected": bool(r.snr >= SNR_DETECT and r.n_transits >= MIN_TRANSITS),
        "bls": {"period_d": _num(r.period_d), "t0": _num(r.t0), "duration_h": _num(r.duration_d * 24),
                "depth_ppm": _num(r.depth_ppm), "snr": _num(r.snr), "n_transits": int(r.n_transits)},
        "fit": {"ok": bool(fit.get("fit_ok")), "error": fit.get("fit_error", ""),
                "period_d": _num(fit.get("fit_period_d")), "period_err_d": _num(fit.get("fit_period_err_d")),
                "depth_ppm": _num(fit.get("fit_depth_ppm")), "depth_err_ppm": _num(fit.get("fit_depth_err_ppm")),
                "duration_h": _num(fit.get("fit_duration_h")), "duration_err_h": _num(fit.get("fit_duration_err_h")),
                "reduced_chi2": _num(fit.get("fit_rchi2"))},
        "classification": {
            "label": "planet_candidate" if p_gb >= 0.5 else "non_planet",
            "score_planet": p_gb, "score_planet_logreg": p_lr,
            "note": "Binary baseline (planet vs non-planet: EB, blend, systematics). Scores are uncalibrated model outputs in [0, 1], not probabilities; they are not validated as likelihoods.",
        },
        "flags": _flags(f, r.snr, r.n_transits),
        "_curve": c, "_bls": r, "_fit": fit,
    }
