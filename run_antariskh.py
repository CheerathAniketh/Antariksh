import os
import time
import logging
import lightkurve as lk
import numpy as np
print('Imported modules')
# Import your core pipeline modules
from core.preprocessing.normalizer import process_raw_file
from core.signal_search.bls_search import run_bls_search
from core.signal_search.phase_folder import phase_fold_and_bin
from core.signal_search.snr_estimator import calculate_transit_snr
from core.classification.inference import predict_exoplanet
print('Imported from CORE')
# Silence the heavy background logging to keep our terminal clean
logging.getLogger("Antariksh.API").setLevel(logging.ERROR)
logging.getLogger("Antariksh.SNR").setLevel(logging.ERROR)
logging.getLogger("Antariksh.Normalizer").setLevel(logging.ERROR)
logging.getLogger("Antariksh.Preprocessing").setLevel(logging.ERROR)
logging.getLogger("Antariksh.SignalSearch").setLevel(logging.ERROR)
logging.getLogger("Antariksh.Inference").setLevel(logging.ERROR)

def auto_fetch_from_nasa(tic_id: str, sector: int, save_dir: str = "data/raw") -> str:
    """Queries NASA, downloads the data, and returns the local file path."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/tic_{tic_id}_sec_{sector}.npz"
    
    # If we already have it, don't waste time downloading!
    if os.path.exists(file_path):
        print(f"Local data found for TIC {tic_id}. Skipping download.")
        return file_path
        
    print(f"📡 Connecting to NASA MAST Archive for TIC {tic_id} (Sector {sector})...")
    search_name = f"TIC {tic_id}"
    
    # Simple retry block for network hiccups
    for attempt in range(1, 4):
        try:
            search_result = lk.search_lightcurve(search_name, sector=sector, author="SPOC")
            if not search_result or len(search_result) == 0:
                raise Exception("No SPOC pipeline data found for this target.")
                
            lc = search_result[0].download()
            if lc is None:
                raise Exception("NASA returned an empty file.")
                
            # Rip out the math columns
            time_arr = lc.time.value
            flux_arr = lc.pdcsap_flux.value
            
            # Filter out NaNs
            mask = ~np.isnan(time_arr) & ~np.isnan(flux_arr)
            clean_time = time_arr[mask]
            clean_flux = flux_arr[mask]
            
            # Normalize baseline to 1.0
            median_flux = np.median(clean_flux)
            clean_flux = clean_flux / median_flux
            clean_flux_err = np.ones_like(clean_flux) * 0.001
            
            # Compress and save
            np.savez(file_path, time=clean_time, flux=clean_flux, flux_err=clean_flux_err)
            print(f"✅ Auto-fetch success! Saved to {file_path}")
            return file_path
            
        except Exception as e:
            print(f"⚠️ NASA connection warning (Attempt {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(3)
            else:
                raise e

def run_pipeline():
    print("\n========================================================")
    print("         PROJECT ANTARIKSH: UNIFIED HUNTING RIG         ")
    print("========================================================\n")
    
    # Ask the user for inputs interactively via terminal
    tic_id = input("Enter NASA TIC Star ID (e.g., 86396382): ").strip()
    sector_input = input("Enter TESS Sector Number (e.g., 20): ").strip()
    
    if not tic_id or not sector_input:
        print("❌ Error: Star ID and Sector cannot be empty!")
        return
        
    try:
        sector = int(sector_input)
    except ValueError:
        print("Error: Sector must be a valid number!")
        return

    print(f"\nInitializing target analysis sequence...")
    
    try:
        # Step 1: Auto-fetch data if missing
        file_path = auto_fetch_from_nasa(tic_id, sector)
        
        # Step 2: Run the Physics Preprocessing Engine
        print("Running physics preprocessing & detrending...")
        time_data, flux_data = process_raw_file(file_path)
        flux_err = np.ones_like(flux_data) * 0.001
        
        # Step 3: Compute Ephemeris & Physical SNR
        print("🧮 Sweeping frequency spectra for period peaks (BLS)...")
        period, epoch, depth, _ = run_bls_search(time_data, flux_data, flux_err)
        snr = calculate_transit_snr(time_data, flux_data, period, epoch, depth)
        
        # Step 4: Fold the data into the uniform AI tensor
        print("Executing phase-folding geometry matrix...")
        cnn_tensor = phase_fold_and_bin(time_data, flux_data, period, epoch)
        
        # Step 5: Feed the Deep Learning Model Weights
        print("Evaluating matrix with 1D Convolutional Neural Network...")
        ai_result = predict_exoplanet(cnn_tensor)
        
        # Step 6: Print out the ultimate scientific discovery layout
        print("\n========================================================")
        print("                  TARGET DISCOVERY REPORT               ")
        print("========================================================")
        print(f" Target Star ID       : TIC {tic_id}")
        print(f" Observation Sector   : Sector {sector}")
        print(f" Detected Orbit (Year): {period:.4f} Days")
        print(f" Shadow Transit Depth : {(depth * 100):.4f}%")
        print(f" Physical Physics SNR : {snr:.2f}")
        print("--------------------------------------------------------")
        print(f" AI CLASSIFICATION : {ai_result['prediction']}")
        print(f" AI CONFIDENCE     : {round(ai_result['confidence'] * 100, 2)}%")
        print("========================================================\n")
        
    except Exception as e:
        print(f"\nPipeline Aborted: {e}\n")
print('Starting')

if __name__ == "__main__":
    while True:
        run_pipeline()
        choice = input("Do you want to check another star? (y/n): ").strip().lower()
        if choice != 'y':
            print("\nClear skies, Commander! Exiting Antariksh Rig. \n")
            break