"""Resumable TESS SPOC light curve downloader (one sector per target)."""
import csv
import os
import time
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd

from antariksh.config import CONFIGS_DIR, CONTRACT, RAW_DIR

LOG = RAW_DIR / "download_log.csv"


def _log(row: dict) -> None:
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tic_id", "status", "sector", "error"])
        if new:
            w.writeheader()
        w.writerow(row)


def download_target(tic: int, retries: int = 3) -> dict:
    d = CONTRACT["data"]
    out = RAW_DIR / f"TIC{tic}.fits"
    if out.exists():
        return {"tic_id": tic, "status": "cached", "sector": "", "error": ""}

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            sr = lk.search_lightcurve(
                f"TIC {tic}", mission=d["mission"], author=d["author"], exptime=d["exptime_s"]
            )
            if len(sr) == 0:
                return {"tic_id": tic, "status": "no_data", "sector": "", "error": ""}

            seq = np.asarray(sr.table["sequence_number"], dtype=int)
            i = int(np.argmin(seq))                       # lowest available sector (frozen rule)
            lc = sr[i].download(
                flux_column=d["flux_column"], quality_bitmask=d["quality_bitmask"]
            )
            if lc is None:
                raise RuntimeError("download returned None")

            tmp = out.with_suffix(".part")
            lc.to_fits(path=str(tmp), overwrite=True)
            os.replace(tmp, out)                          # atomic: no half-written files
            return {"tic_id": tic, "status": "ok", "sector": int(seq[i]), "error": ""}
        except Exception as e:                            # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"[:200]
            time.sleep(2 * attempt)
    return {"tic_id": tic, "status": "failed", "sector": "", "error": last_err}


def download_all(limit: int | None = None, sleep_s: float = 0.5) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(CONFIGS_DIR / "targets.csv")
    tics = targets["tic_id"].astype(int).tolist()
    if limit:
        tics = tics[:limit]

    for k, tic in enumerate(tics, 1):
        row = download_target(tic)
        _log(row)
        print(f"[{k}/{len(tics)}] TIC{tic}: {row['status']} {row['error']}", flush=True)
        if row["status"] != "cached":
            time.sleep(sleep_s)