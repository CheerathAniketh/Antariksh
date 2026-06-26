from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import numpy as np

# Import our Antariksh pipeline modules
from core.preprocessing.normalizer import process_raw_file
from core.signal_search.bls_search import run_bls_search
from core.signal_search.phase_folder import phase_fold_and_bin
from core.signal_search.snr_estimator import calculate_transit_snr
from core.classification.inference import predict_exoplanet

# Setup logging and initialize the server
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Antariksh.API")
app = FastAPI(title="Antariksh Exoplanet API", version="1.0")

# Define the JSON structure we expect from the user
class TargetRequest(BaseModel):
    tic_id: str
    sector: int

@app.get("/")
def health_check():
    """Simple ping to check if the server is awake."""
    return {"status": "online", "message": "Antariksh Deep Learning Pipeline Active"}

@app.post("/analyze")
def analyze_target(request: TargetRequest):
    """
    Receives a target ID, finds its local data, runs the entire physics 
    and AI pipeline, and returns the classification.
    """
    logger.info(f"Received request for TIC {request.tic_id} Sector {request.sector}")
    
    # 1. Locate the file we downloaded earlier
    file_path = f"data/raw/tic_{request.tic_id}_sec_{request.sector}.npz"
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Data for TIC {request.tic_id} Sector {request.sector} not found. Ensure it is in {file_path}."
        )
    
    try:
        # 2. Run the Physics Engine (Scrub, Search, Fold)
        logger.info("Initializing Physics Engine...")
        time, flux = process_raw_file(file_path)
        flux_err = np.ones_like(flux) * 0.001
        
        period, epoch, depth, _ = run_bls_search(time, flux, flux_err)
        snr = calculate_transit_snr(time, flux, period, epoch, depth)
        
        # Slicing the stack into our 2001 AI tensor
        cnn_tensor = phase_fold_and_bin(time, flux, period, epoch)
        
        # 3. Run the AI Brain
        logger.info("Running Deep Learning Inference...")
        ai_result = predict_exoplanet(cnn_tensor)
        
        # 4. Return the ultimate JSON package
        logger.info("Analysis complete! Returning results.")
        return {
            "target": {
                "tic_id": request.tic_id,
                "sector": request.sector
            },
            "physics_metrics": {
                "orbital_period_days": round(float(period), 4),
                "transit_depth": round(float(depth), 4),
                "signal_to_noise_ratio": round(float(snr), 2)
            },
            "ai_classification": {
                "prediction": ai_result["prediction"],
                "confidence": round(ai_result["confidence"] * 100, 2),
                "full_distribution": ai_result["all_probabilities"]
            }
        }
        
    except Exception as e:
        logger.error(f"Pipeline crashed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))