import numpy as np
import logging
from astropy.stats import sigma_clip

logger = logging.getLogger("Antariksh.Preprocessing")

def clean_outliers(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, sigma: float = 5.0):
    """
    Applies aggressive sigma clipping to strip out massive non-astrophysical spikes 
    (e.g., cosmic ray impacts on the telescope CCD).
    """
    logger.info(f"Applying {sigma}-sigma outlier clipping...")
    
    # astropy's sigma_clip returns a masked array where True means it's an outlier
    clipped_flux = sigma_clip(flux, sigma=sigma, maxiters=5, stdfunc=np.nanstd)
    mask = ~clipped_flux.mask
    
    # Filter arrays using the mask to keep only valid entries, stripping NaNs simultaneously
    nan_mask = ~np.isnan(time) & ~np.isnan(flux) & mask
    
    logger.info(f"Removed {len(flux) - np.sum(nan_mask)} outlier/NaN cadences.")
    return time[nan_mask], flux[nan_mask], flux_err[nan_mask]