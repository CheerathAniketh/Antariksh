import pandas as pd
from antariksh.config import CONFIGS_DIR
from antariksh.ingest.mast import download_target, _log

tics = pd.read_csv(CONFIGS_DIR / "targets.csv")["tic_id"].astype(int).tolist()[::-1]
for k, tic in enumerate(tics, 1):
    row = download_target(tic)
    _log(row)
    print(f"[rev {k}/{len(tics)}] TIC{tic}: {row['status']} {row['error']}", flush=True)
