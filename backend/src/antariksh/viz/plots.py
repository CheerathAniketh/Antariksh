"""4-panel diagnostic: raw, flattened, BLS periodogram, phase-folded with the batman fit."""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from antariksh.fit.transit_fit import model_flux


def _fold(time, period, t0):
    return ((np.asarray(time) - t0 + 0.5 * period) % period) - 0.5 * period


def _bin(x, y, nb=40):
    edges = np.linspace(x.min(), x.max(), nb + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    xs, ys = [], []
    for i in range(nb):
        m = idx == i
        if m.sum() >= 3:
            xs.append(x[m].mean())
            ys.append(np.median(y[m]))
    return np.array(xs), np.array(ys)


def make_figure(curve, bls, fit, title=""):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    a1, a2, a3, a4 = axes.ravel()

    a1.plot(curve.raw_time, curve.raw_flux, ".", ms=1, color="0.4")
    a1.set(title="Raw (PDCSAP, normalized)", xlabel="Time [BTJD]", ylabel="Flux")

    a2.plot(curve.time, curve.flux, ".", ms=1, color="0.4")
    n = np.arange(np.floor((curve.time.min() - bls.t0) / bls.period_d),
                  np.ceil((curve.time.max() - bls.t0) / bls.period_d) + 1)
    for tc in bls.t0 + n * bls.period_d:
        if curve.time.min() <= tc <= curve.time.max():
            a2.axvline(tc, color="C3", alpha=0.3, lw=0.8)
    a2.set(title="Flattened (transits masked in the trend fit); red = predicted transits",
           xlabel="Time [BTJD]", ylabel="Flux")

    a3.plot(bls.periods, bls.power, lw=0.8)
    a3.axvline(bls.period_d, color="C3", ls="--", label=f"P = {bls.period_d:.4f} d")
    a3.set(title="BLS periodogram", xlabel="Period [d]", ylabel="Power", xscale="log")
    a3.legend(loc="upper right")

    ok = bool(fit.get("fit_ok"))
    per = fit["fit_period_d"] if ok else bls.period_d
    t0 = fit["fit_t0"] if ok else bls.t0
    dur_h = fit["fit_duration_h"] if ok else bls.duration_d * 24
    ph_h = _fold(curve.time, per, t0) * 24
    half = max(3 * dur_h, 2.0)
    w = np.abs(ph_h) < half
    a4.plot(ph_h[w], curve.flux[w], ".", ms=2, color="0.7")
    if w.sum() > 20:
        bx, by = _bin(ph_h[w], curve.flux[w])
        a4.plot(bx, by, "o", ms=4, color="C0", label="binned median")
    if ok:
        tt = t0 + np.linspace(-half, half, 2000) / 24
        a4.plot((tt - t0) * 24, model_flux(tt, fit), color="C3", lw=2, label="batman fit")
    a4.set(title="Phase-folded", xlabel="Hours from mid-transit", ylabel="Flux", xlim=(-half, half))
    a4.legend(loc="lower right")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
