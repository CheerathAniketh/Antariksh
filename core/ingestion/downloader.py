import lightkurve as lk
import logging
from typing import Optional

# Setup structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Antariksh.Ingestion")

def fetch_lightcurve(tic_id: str, sector: Optional[int] = None) -> lk.LightCurve:
    """
    Downloads high-cadence (2-minute) TESS lightcurve data.
    
    Args:
        tic_id (str): The TIC identifier (e.g., "86396382")
        sector (int, optional): Specific mission sector to fetch.
        
    Returns:
        lk.LightCurve: The downloaded SPOC lightcurve object.
    """
    logger.info(f"Searching MAST for target: TIC {tic_id}")
    
    try:
        # Target 'SPOC' short-cadence data
        search_result = lk.search_lightcurve(f"TIC {tic_id}", author="SPOC", exptime=120)
        
        if len(search_result) == 0:
            logger.error(f"No SPOC data found for TIC {tic_id}.")
            raise ValueError(f"Target {tic_id} not found in TESS archives.")

        # Filter by sector if specified by checking the 'mission' strings (e.g., "TESS Sector 14")
        if sector is not None:
            sector_string = f"TESS Sector {sector}"
            # Select rows where the mission column matches our target sector string
            mask = [sector_string in m for m in search_result.mission]
            search_result = search_result[mask]
            
            if len(search_result) == 0:
                logger.error(f"Sector {sector} not available for TIC {tic_id}.")
                return None

        # Download the first available matching lightcurve
        logger.info(f"Downloading light curve data for TIC {tic_id}...")
        lc = search_result[0].download()
        
        return lc

    except Exception as e:
        logger.exception(f"Failed to download data for TIC {tic_id}: {e}")
        raise

if __name__ == "__main__":
    # WASP-12 true TESS Catalog ID
    lc = fetch_lightcurve("86396382") 
    if lc:
        print("\n--- Download Success ---")
        print(f"Object Metadata Name: {lc.meta.get('OBJECT', 'Unknown')}")
        print(f"Observed Sector: {lc.sector}")
        print(f"Data Shape: {lc.flux.shape}")
        print("\nFirst 5 Rows of Photometry:")
        print(lc.head())
        
        # Try plotting. If headless, comment this out.
        lc.plot()