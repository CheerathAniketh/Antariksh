import os
import time
import logging
import lightkurve as lk
import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Antariksh.MassDownloader")

TARGET_LIST = [
    {"tic_id": "86396382", "sector": 20}, # WASP-12
    {"tic_id": "236361519", "sector": 1}, # WASP-4 
    {"tic_id": "261136679", "sector": 1}, # Pi Mensae 
    {"tic_id": "149603524", "sector": 4}  # WASP-46 
]

def download_with_retry(search_name, sector, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Querying NASA MAST (Attempt {attempt}/{max_retries})...")
            search_result = lk.search_lightcurve(search_name, sector=sector, author="SPOC")
            if not search_result or len(search_result) == 0:
                return None
            return search_result[0].download()
        except Exception as e:
            logger.warning(f"⚠️ Connection glitch on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise e
    return None

def download_target_batch(targets: list, save_dir: str = "data/raw"):
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Starting mass download for {len(targets)} targets...")
    
    for target in targets:
        tic = target["tic_id"]
        sector = target["sector"]
        search_name = f"TIC {tic}"
        output_file = f"{save_dir}/tic_{tic}_sec_{sector}.npz"
        
        # We REMOVE the skip check temporarily for this run so it overwrites 
        # and repairs the corrupted files we just downloaded!
        logger.info(f"Starting pipeline for {search_name} (Sector {sector})...")
        
        try:
            lc = download_with_retry(search_name, sector)
            if lc is None:
                logger.warning(f"No valid data returned for {search_name}. Skipping.")
                continue
            
            time_arr = lc.time.value
            flux_arr = lc.pdcsap_flux.value
            
            mask = ~np.isnan(time_arr) & ~np.isnan(flux_arr)
            clean_time = time_arr[mask]
            clean_flux = flux_arr[mask]
            
            median_flux = np.median(clean_flux)
            clean_flux = clean_flux / median_flux
            
            # FIX: Create a matching error array so core.preprocessing doesn't crash!
            clean_flux_err = np.ones_like(clean_flux) * 0.001
            
            # Save all 3 arrays into the archive archive
            np.savez(output_file, time=clean_time, flux=clean_flux, flux_err=clean_flux_err)
            logger.info(f"✅ Successfully saved and fixed {output_file}")
            
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Permanently failed {search_name}: {e}")

if __name__ == "__main__":
    download_target_batch(TARGET_LIST)
    print("\nBatch Download Complete! Run scripts.batch_pipeline next.")