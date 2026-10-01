"""BLS recovery + parameter errors (fit and BLS-only vs TOI) on planets. Nothing is dropped from the table."""
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from antariksh.classify.evaluate import classify_period
from antariksh.config import ARTIFACTS_DIR, CONFIGS_DIR, PROCESSED_DIR


def work(r):
    warnings.filterwarnings("ignore")
    from antariksh.fit.transit_fit import fit_transit
    from antariksh.preprocess.pipeline import load_raw, preprocess
    tic = int(r["tic_id"])
    try:
        eph = (r["bls_period_d"], r["bls_t0"], r["bls_duration_d"])
        c = preprocess(load_raw(tic), ephem=eph)
        return {"tic_id": tic, **fit_transit(c.time, c.flux, c.flux_err, *eph, r["bls_depth_ppm"])}
    except Exception as e:  # noqa: BLE001
        return {"tic_id": tic, "fit_ok": False, "fit_error": f"{type(e).__name__}: {e}"[:200]}


def rel(a, b):
    return (a - b) / b


def main():
    t = pd.read_csv(CONFIGS_DIR / "targets.csv")
    b = pd.read_csv(PROCESSED_DIR / "bls_results.csv")
    d = t.merge(b[b.error.isna()].drop(columns="error"), on="tic_id")
    d["category"] = [classify_period(x, y) for x, y in zip(d.bls_period_d, d.period_d)]

    pl = d[d.label == 1].copy()
    with ProcessPoolExecutor(6) as ex:
        fits = pd.DataFrame(list(ex.map(work, pl.to_dict("records"))))
    pl = pl.merge(fits, on="tic_id", how="left")

    pl["period_rel_err"] = rel(pl.fit_period_d, pl.period_d)
    pl["depth_rel_err"] = rel(pl.fit_depth_ppm, pl.depth_ppm)
    pl["duration_rel_err"] = rel(pl.fit_duration_h, pl.duration_h)
    pl["bls_depth_rel_err"] = rel(pl.bls_depth_ppm, pl.depth_ppm)
    pl["bls_duration_rel_err"] = rel(pl.bls_duration_d * 24, pl.duration_h)
    (ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    pl.to_csv(ARTIFACTS_DIR / "metrics" / "param_errors_per_target.csv", index=False)

    cols = ["period_rel_err", "depth_rel_err", "duration_rel_err", "bls_depth_rel_err", "bls_duration_rel_err"]
    summ = {}
    for split in ("val", "train"):
        for name, sub in (("bls_hits", pl[(pl.split == split) & (pl.category == "hit")]),
                          ("all_planets", pl[pl.split == split])):
            s = sub[sub.fit_ok == True]  # noqa: E712
            summ[f"{split}/{name}"] = {"n": int(len(sub)), "n_fit_ok": int(len(s)),
                                       **{c: float(s[c].abs().median()) for c in cols}}
    rec = d.groupby(["split", "label"]).category.value_counts().unstack(fill_value=0)
    out = {"median_abs_rel_error": summ, "bls_recovery_counts": json.loads(rec.to_json(orient="index"))}
    json.dump(out, open(ARTIFACTS_DIR / "metrics" / "param_summary.json", "w"), indent=2)

    print(pd.DataFrame(summ).T.round(3).to_string())
    print("\nfit failures:", int((pl.fit_ok != True).sum()), "| errors:", pl.fit_error.dropna().unique()[:3])  # noqa: E712
    print("\nBLS-hit planets, worst 5 by |duration error|:")
    h = pl[(pl.category == "hit") & (pl.fit_ok == True)].copy()  # noqa: E712
    h["abs_dur"] = h.duration_rel_err.abs()
    print(h.nlargest(5, "abs_dur")[["tic_id", "split", "duration_h", "fit_duration_h", "depth_ppm", "fit_depth_ppm", "bls_snr"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
