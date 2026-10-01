"""Build the frozen target list from the NASA Exoplanet Archive TOI table."""
import io

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from antariksh.config import CONFIGS_DIR, CONTRACT

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
QUERY = (
    "select toi,tid,tfopwg_disp,pl_orbper,pl_trandep,pl_trandurh,st_tmag "
    "from toi"
)
SNAPSHOT = CONFIGS_DIR / "toi_snapshot.csv"
OUT = CONFIGS_DIR / "targets.csv"


def fetch_toi(refresh: bool = False) -> pd.DataFrame:
    """Cache the TOI table so the target list stays reproducible after the archive updates."""
    if SNAPSHOT.exists() and not refresh:
        return pd.read_csv(SNAPSHOT)
    r = requests.get(TAP_URL, params={"query": QUERY, "format": "csv"}, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.to_csv(SNAPSHOT, index=False)
    return df


def build_targets(refresh: bool = False) -> pd.DataFrame:
    cfg = CONTRACT["targets"]
    seed = CONTRACT["seed"]
    pmin, pmax = cfg.get("period_range_d", [0.5, 15.0])

    df = fetch_toi(refresh)
    print("TOI columns:", list(df.columns))
    print("TOI rows:", len(df))

    df = df.dropna(subset=["tid", "tfopwg_disp", "pl_orbper", "pl_trandep", "pl_trandurh"])
    df["tfopwg_disp"] = df["tfopwg_disp"].str.strip().str.upper()

    pos, neg = cfg["positive_dispositions"], cfg["negative_dispositions"]
    df = df[df["tfopwg_disp"].isin(pos + neg)].copy()
    df["label"] = df["tfopwg_disp"].isin(pos).astype(int)

    # keep periods BLS can actually search
    df = df[(df["pl_orbper"] >= pmin) & (df["pl_orbper"] <= pmax)]

    # one row per star (lowest TOI number) so the split is grouped by TIC by construction
    df = df.sort_values("toi").drop_duplicates("tid", keep="first")

    n = cfg["max_per_class"]
    parts = [g.sample(min(len(g), n), random_state=seed) for _, g in df.groupby("label")]
    df = pd.concat(parts).reset_index(drop=True)

    train_idx, val_idx = train_test_split(
        df.index, test_size=cfg["split"]["val_frac"], stratify=df["label"], random_state=seed
    )
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"

    out = df.rename(
        columns={
            "tid": "tic_id",
            "tfopwg_disp": "disposition",
            "pl_orbper": "period_d",
            "pl_trandep": "depth_ppm",
            "pl_trandurh": "duration_h",
            "st_tmag": "tmag",
        }
    )[["tic_id", "toi", "label", "split", "disposition", "period_d", "depth_ppm", "duration_h", "tmag"]]
    out["tic_id"] = out["tic_id"].astype(int)

    out.to_csv(OUT, index=False)
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(f"wrote {OUT}")
    return out