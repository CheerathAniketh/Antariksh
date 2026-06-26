import os
import logging
import numpy as np
import pandas as pd
from core.preprocessing.normalizer import process_raw_file
from core.signal_search.bls_search import run_bls_search
from core.signal_search.phase_folder import phase_fold_and_bin
from core.signal_search.snr_estimator import calculate_transit_snr
from core.classification.inference import predict_exoplanet

# Direct all standard pipeline logs to silence to keep console clean for tabular layout
logging.getLogger("Antariksh.API").setLevel(logging.ERROR)
logging.getLogger("Antariksh.SNR").setLevel(logging.ERROR)
logging.getLogger("Antariksh.Normalizer").setLevel(logging.ERROR)
logging.getLogger("Antariksh.Preprocessing").setLevel(logging.ERROR)
logging.getLogger("Antariksh.SignalSearch").setLevel(logging.ERROR)
logging.getLogger("Antariksh.Inference").setLevel(logging.ERROR)

def run_mass_search(data_directory: str = "data/raw"):
    """
    Scans the local data directory, processes every star file found,
    and compiles a master spreadsheet of physical and AI discoveries.
    """
    print("\n========================================================")
    print("      ANTARIKSH AUTOMATED MASS DETECTOR STARTING        ")
    print("========================================================\n")
    
    master_results = []
    
    # 1. Find all star files in the raw data folder
    all_files = [f for f in os.listdir(data_directory) if f.endswith('.npz')]
    
    if not all_files:
        print(f"No star data found in {data_directory}. Please run mass_downloader.py first.")
        return

    print(f"Found {len(all_files)} star targets ready for processing.\n")
    
    for file_name in all_files:
        # Extract TIC ID and Sector from filename format (e.g., "tic_86396382_sec_20.npz")
        parts = file_name.replace(".npz", "").split("_")
        if len(parts) < 4:
            continue
            
        tic_id = parts[1]
        sector = parts[3]
        
        print(f"🔄 Analyzing Target: TIC {tic_id} (Sector {sector})...")
        full_path = os.path.join(data_directory, file_name)
        
        try:
            # 2. Extract and Detrend Raw Signal
            time, flux = process_raw_file(full_path)
            flux_err = np.ones_like(flux) * 0.001
            
            # 3. Compute Physical Ephemeris
            period, epoch, depth, _ = run_bls_search(time, flux, flux_err)
            snr = calculate_transit_snr(time, flux, period, epoch, depth)
            
            # 4. Run Phase-Folding Origami to 2001 Grid
            cnn_tensor = phase_fold_and_bin(time, flux, period, epoch)
            
            # 5. Evaluate Matrix using 1D CNN Model Weights
            ai_result = predict_exoplanet(cnn_tensor)
            
            # 6. Append Summary Row
            target_summary = {
                "TIC ID": tic_id,
                "Sector": sector,
                "Period (Days)": round(float(period), 4),
                "Transit Depth": round(float(depth), 4),
                "Physics SNR": round(float(snr), 2),
                "AI Classification": ai_result["prediction"],
                "AI Confidence %": round(ai_result["confidence"] * 100, 2)
            }
            master_results.append(target_summary)
            print(f"✅ Result: {ai_result['prediction']} ({round(ai_result['confidence'] * 100, 2)}%)\n")
            
        except Exception as e:
            print(f"❌ Failed to process TIC {tic_id}: {e}\n")
            continue
            
    # 7. Format output using Pandas DataFrame layout
    if master_results:
        df = pd.DataFrame(master_results)
        
        print("\n========================================================")
        print("                FINAL DISCOVERY CATALOG                 ")
        print("========================================================")
        print(df.to_string(index=False))
        print("========================================================\n")
        
        # Save output to disk
        output_path = "discovery_catalog.csv"
        df.to_csv(output_path, index=False)
        print(f"Spreadsheet exported successfully to: {output_path}")
    else:
        print("No targets were successfully processed.")

if __name__ == "__main__":
    run_mass_search()