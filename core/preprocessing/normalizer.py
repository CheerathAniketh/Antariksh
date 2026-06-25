import numpy as np
import os
import logging
from core.preprocessing.outlier_rejection import clean_outliers
from core.preprocessing.detrending import flatten_lightcurve

logger = logging.getLogger("Antariksh.Normalizer")

def process_raw_file(filepath: str):
    """Loads a saved .npz file, runs preprocessing, and prints statistics."""
    logger.info(f"Loading raw file for preprocessing: {filepath}")
    
    data = np.load(filepath)
    t, f, fe = data['time'], data['flux'], data['flux_err']
    
    # 1. Strip Outliers
    t_clean, f_clean, fe_clean = clean_outliers(t, f, fe)
    
    # 2. Flatten trends
    f_flat = flatten_lightcurve(t_clean, f_clean)
    
    print("\n--- Preprocessing Success ---")
    print(f"Raw points: {len(f)} -> Cleaned points: {len(f_flat)}")
    print(f"Flattened Flux Center Metric (Median): {np.median(f_flat):.4f}")
    return t_clean, f_flat

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    target_file = "data/raw/tic_86396382_sec_20.npz"
    
    if os.path.exists(target_file):
        process_raw_file(target_file)
    else:
        print(f"Error: Run the ingestion step first! Missing file: {target_file}")