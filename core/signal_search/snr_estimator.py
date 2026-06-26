import numpy as np
import logging

logger = logging.getLogger("Antariksh.SNR")

def calculate_transit_snr(time: np.ndarray, flux: np.ndarray, period: float, epoch: float, depth: float) -> float:
    """
    Calculates the physical Signal-to-Noise Ratio (SNR) of the transit.
    This acts as our mathematical sanity check against the AI's confidence score.
    """
    logger.info("Calculating physics SNR metric...")
    
    # 1. Calculate the background "fuzz" (Noise)
    # Since our detrending flattened the star's baseline to 1.0, 
    # the standard deviation of the flux perfectly represents the noise floor.
    noise = np.std(flux)
    
    if noise == 0:
        return 0.0
        
    # 2. Calculate the raw ratio (Signal / Noise)
    # Depth is how deep the planet's shadow is (the Signal).
    raw_snr = depth / noise
    
    logger.info(f"Calculated SNR: {raw_snr:.2f}")
    
    return float(raw_snr)