import numpy as np
import logging
from wotan import flatten

logger = logging.getLogger("Antariksh.Preprocessing")

def flatten_lightcurve(time: np.ndarray, flux: np.ndarray, window_length: float = 0.3) -> np.ndarray:
    """
    Removes long-term stellar variability and systemic trends using a biweight filter.
    Returns a perfectly flattened, relative flux tracking around 1.0.
    """
    logger.info(f"Flattening light curve trends using Wotan (window_length={window_length} days)...")
    
    # robust biweight filter handles sharp planetary dips smoothly without breaking them
    flat_flux, trend_flux = flatten(
        time, 
        flux, 
        method='biweight', 
        window_length=window_length, 
        return_trend=True
    )
    
    return flat_flux