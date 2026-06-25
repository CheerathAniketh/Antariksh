import numpy as np
import logging
from astropy.timeseries import BoxLeastSquares

logger = logging.getLogger("Antariksh.SignalSearch")

def run_bls_search(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray):
    """
    Runs a Box Least Squares periodogram search to detect periodic transit dips.
    Returns the best period, epoch (transit time), depth, and the periodogram object.
    """
    logger.info("Initializing Box Least Squares (BLS) periodogram search...")
    
    # Instantiate the BLS model using astropy
    bls = BoxLeastSquares(time, flux, dy=flux_err)
    
    # We define a search grid for the periods.
    # For testing WASP-12, we know its period is ~1.09 days, so we search between 0.5 and 5 days.
    # duration_grid defines how wide the transit box can be (typically 0.05 to 0.15 days)
    durations = np.linspace(0.05, 0.15, 5)
    
    logger.info("Sweeping frequency grid for periodic power peaks...")
    periodogram = bls.power(
        period=np.arange(0.5, 3.0, 0.001), 
        duration=durations
    )
    
    # Extract the highest peak in the power spectrum
    best_index = np.argmax(periodogram.power)
    best_period = periodogram.period[best_index]
    best_transit_time = periodogram.transit_time[best_index]
    best_depth = periodogram.depth[best_index]
    
    logger.info(f"BLS Hit Found! Detected Period: {best_period:.4f} days, Depth: {best_depth:.4f}")
    
    return best_period, best_transit_time, best_depth, periodogram

if __name__ == "__main__":
    import os
    from core.preprocessing.normalizer import process_raw_file
    from core.signal_search.phase_folder import phase_fold_and_bin
    
    logging.basicConfig(level=logging.INFO)
    raw_file = "data/raw/tic_86396382_sec_20.npz"
    
    if os.path.exists(raw_file):
        # 1. Clean and flatten using our Phase 2 engine
        time, flux = process_raw_file(raw_file)
        # Standard dummy error array for baseline weights
        flux_err = np.ones_like(flux) * 0.001 
        
        # 2. Run the BLS Search
        period, epoch, depth, _ = run_bls_search(time, flux, flux_err)
        
        # 3. Phase-fold and create our final PyTorch-ready vector
        cnn_tensor = phase_fold_and_bin(time, flux, period, epoch)
        
        print("\n--- Signal Search & Folding Success ---")
        print(f"Final Tensor Shape for PyTorch: {cnn_tensor.shape}")
        print(f"Minimum Value in Tensor (Transit Depth Peak): {np.min(cnn_tensor):.4f}")