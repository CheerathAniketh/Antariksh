import os
import numpy as np
import lightkurve as lk
import logging

logger = logging.getLogger("Antariksh.Parser")

def save_raw_lightcurve(lc: lk.LightCurve, output_dir: str = "data/raw") -> str:
    """
    Extracts core time-series arrays from a LightCurve object and 
    saves them to a clean compressed numpy file in our local data lake.
    """
    if lc is None:
        return None
        
    tic_id = str(lc.meta.get('OBJECT', 'unknown')).replace("TIC", "").strip()
    sector = lc.sector
    
    # Create target directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define a clean file name scheme
    filename = f"tic_{tic_id}_sec_{sector}.npz"
    filepath = os.path.join(output_dir, filename)
    
    # Extract raw arrays
    # We use PDCSAP flux as our baseline
    time = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value
    
    logger.info(f"Extracting arrays for TIC {tic_id}...")
    
    # Save as compressed numpy file
    np.savez_compressed(
        filepath,
        time=time,
        flux=flux,
        flux_err=flux_err,
        tic_id=tic_id,
        sector=sector
    )
    
    logger.info(f"Raw array binary saved locally to: {filepath}")
    return filepath


if __name__ == "__main__":
    from core.ingestion.fits_parser import save_raw_lightcurve
    from core.ingestion.downloader import fetch_lightcurve
    # 1. Download WASP-12 data from NASA
    lc = fetch_lightcurve("86396382") 
    
    if lc:
        print("\n--- Download Success ---")
        print(f"Observed Sector: {lc.sector}")
        
        # 2. Extract and save the raw arrays to Antariksh/data/raw/
        raw_binary_path = save_raw_lightcurve(lc)
        print(f"Extraction Step Complete!")