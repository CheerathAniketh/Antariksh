"""Two-pass preprocess + BLS on every downloaded target. Resumable; failures are logged, never dropped."""
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from antariksh.config import ARTIFACTS_DIR, CONFIGS_DIR, PROCESSED_DIR, RAW_DIR

OUT = PROCESSED_DIR / "bls_results.csv"
COLS = ["tic_id", "error", "bls_period_d", "bls_t0", "bls_duration_d", "bls_depth_ppm",
        "bls_snr", "bls_n_transits", "p1_period_d", "p1_depth_ppm", "p1_snr"]


def classify(p_bls, p_ref, tol=0.01):
    ratio = p_bls / p_ref
    if abs(ratio - 1) < tol:
        return "hit"
    for h in (0.5, 2.0, 1 / 3, 3.0):
        if abs(ratio / h - 1) < tol:
            return "harmonic"
    return "miss"


def work(tic):
    warnings.filterwarnings("ignore")
    from antariksh.pipeline import run_two_pass
    from antariksh.preprocess.pipeline import load_raw
    try:
        _, r, r1 = run_two_pass(load_raw(tic))
        return {"tic_id": tic, "error": "", "bls_period_d": r.period_d, "bls_t0": r.t0,
                "bls_duration_d": r.duration_d, "bls_depth_ppm": r.depth_ppm, "bls_snr": r.snr,
                "bls_n_transits": r.n_transits, "p1_period_d": r1.period_d,
                "p1_depth_ppm": r1.depth_ppm, "p1_snr": r1.snr}
    except Exception as e:  # noqa: BLE001
        return {"tic_id": tic, "error": f"{type(e).__name__}: {e}"[:200]}


def summarize():
    t = pd.read_csv(CONFIGS_DIR / "targets.csv")
    r = pd.read_csv(OUT)
    d = t.merge(r, on="tic_id")
    fail = d[d.error.notna()]
    ok = d[d.error.isna()].copy()
    ok["category"] = [classify(a, b) for a, b in zip(ok.bls_period_d, ok.period_d)]
    fdir = ARTIFACTS_DIR / "failures"
    fdir.mkdir(parents=True, exist_ok=True)
    ok[ok.category != "hit"].to_csv(fdir / "bls_misses.csv", index=False)
    fail.to_csv(fdir / "bls_errors.csv", index=False)
    print(f"\nprocessed {len(d)} | errors {len(fail)}")
    print(pd.crosstab([ok.split, ok.label], ok.category, margins=True))
    for lab in (1, 0):
        s = ok[ok.label == lab]
        if len(s):
            print(f"label={lab} recovery (hit): {(s.category == 'hit').mean():.2f}  n={len(s)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        tics = pd.read_csv(CONFIGS_DIR / "targets.csv")["tic_id"].astype(int).tolist()
        done = set(pd.read_csv(OUT)["tic_id"]) if OUT.exists() else set()
        todo = [x for x in tics if (RAW_DIR / f"TIC{x}.fits").exists() and x not in done]
        if a.limit:
            todo = todo[: a.limit]
        print(f"{len(todo)} to process, {len(done)} already done", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(work, x) for x in todo]
            for k, f in enumerate(as_completed(futs), 1):
                row = f.result()
                pd.DataFrame([row], columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                print(f"[{k}/{len(todo)}] TIC{row['tic_id']} {row['error']}", flush=True)
    summarize()


if __name__ == "__main__":
    main()
