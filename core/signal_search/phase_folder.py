import numpy as np
import logging

logger = logging.getLogger("Antariksh.SignalSearch")

def phase_fold_and_bin(time: np.ndarray, flux: np.ndarray, period: float, epoch: float, num_bins: int = 2001) -> np.ndarray:
    """
    Folds the timeline at the detected orbital period and downsamples it into a 
    fixed-length 1D tensor suitable for PyTorch CNN input.
    """
    logger.info(f"Phase-folding timeline around period {period:.4f} days...")
    
    # Calculate phase for each time step: centered at 0 (mid-transit)
    folded_phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    folded_phase = folded_phase / period  # Normalize phase between -0.5 and 0.5
    
    # Sort the phases and flux arrays accordingly
    sort_idx = np.argsort(folded_phase)
    sorted_phase = folded_phase[sort_idx]
    sorted_flux = flux[sort_idx]
    
    logger.info(f"Binning folded light curve into {num_bins} uniform bins...")
    # Create uniform bins from -0.5 to 0.5
    bin_edges = np.linspace(-0.5, 0.5, num_bins + 1)
    
    # Digitizer assigns each sorted phase to its matching bin index
    bin_assignments = np.digitize(sorted_phase, bin_edges) - 1
    
    # Initialize a fixed array tracking baseline flux (1.0)
    binned_flux = np.ones(num_bins, dtype=np.float32)
    
    # Compute the median flux value inside each bin boundary
    for i in range(num_bins):
        mask = (bin_assignments == i)
        if np.any(mask):
            binned_flux[i] = np.median(sorted_flux[mask])
            
    return binned_flux