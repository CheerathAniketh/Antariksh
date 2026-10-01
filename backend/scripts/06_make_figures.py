"""4-panel figures for val targets: random BLS hits, BLS misses, and grad-boost misclassifications (seed 42)."""
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from antariksh.config import ARTIFACTS_DIR, CONFIGS_DIR
from antariksh.pipeline import analyze_lightcurve
from antariksh.preprocess.pipeline import load_raw
from antariksh.viz.plots import make_figure

warnings.filterwarnings("ignore")
K = 4


def pick():
    pe = pd.read_csv(ARTIFACTS_DIR / "metrics" / "param_errors_per_target.csv")
    pe = pe[pe.split == "val"]
    mc = pd.read_csv(ARTIFACTS_DIR / "failures" / "misclassified_val.csv")
    mc = mc[mc.model == "grad_boost"]
    sel = []
    for tag, df in (("hit", pe[pe.category == "hit"]), ("bls_miss", pe[pe.category != "hit"]), ("misclassified", mc)):
        for tic in df.sample(min(K, len(df)), random_state=42).tic_id:
            sel.append((int(tic), tag))
    return sel


def main():
    t = pd.read_csv(CONFIGS_DIR / "targets.csv").set_index("tic_id")
    out = ARTIFACTS_DIR / "figures" / "val"
    out.mkdir(parents=True, exist_ok=True)
    for tic, tag in pick():
        res = analyze_lightcurve(load_raw(tic))
        ref = t.loc[tic]
        title = (f"TIC{tic} [{tag}] label={ref.label} ({ref.disposition}) ref P={ref.period_d:.3f} d | "
                 f"BLS P={res['bls']['period_d']:.3f} d SNR={res['bls']['snr']:.1f} | "
                 f"P(planet)={res['classification']['prob_planet']:.2f}")
        fig = make_figure(res["_curve"], res["_bls"], res["_fit"], title)
        fig.savefig(out / f"TIC{tic}_{tag}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out / f"TIC{tic}_{tag}.png", flush=True)


if __name__ == "__main__":
    main()
