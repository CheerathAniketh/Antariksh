"""Feature table from saved BLS ephemerides: one preprocess pass per target, no BLS rerun."""
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from antariksh.config import PROCESSED_DIR

OUT = PROCESSED_DIR / "features.csv"


def work(r):
    warnings.filterwarnings("ignore")
    from antariksh.features.extract import extract_features
    from antariksh.preprocess.pipeline import load_raw, preprocess
    tic = int(r["tic_id"])
    try:
        eph = (r["bls_period_d"], r["bls_t0"], r["bls_duration_d"])
        c = preprocess(load_raw(tic), ephem=eph)
        f = extract_features(c.time, c.flux, *eph, r["bls_depth_ppm"], r["bls_snr"], r["bls_n_transits"])
        return {"tic_id": tic, "error": "", **f}
    except Exception as e:  # noqa: BLE001
        return {"tic_id": tic, "error": f"{type(e).__name__}: {e}"[:200]}


if __name__ == "__main__":
    b = pd.read_csv(PROCESSED_DIR / "bls_results.csv")
    b = b[b.error.isna()]
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(work, b.to_dict("records")))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows, {(df.error != '').sum()} errors")
    print(df.drop(columns=["tic_id", "error"]).describe().T[["count", "mean", "min", "max"]].round(2))
