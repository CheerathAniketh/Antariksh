import torch
import numpy as np
import logging
from core.classification.model_builder import Exoplanet1DCNN

logger = logging.getLogger("Antariksh.Inference")

# The 4 target categories our model tracks
CLASS_MAPPING = {
    0: "Confirmed Planet (Transit)",
    1: "Eclipsing Binary (False Positive)",
    2: "Stellar Variability / Noise",
    3: "Quiet Background Baseline"
}

def predict_exoplanet(binned_flux: np.ndarray, weights_path: str = "models/weights/best_cnn.pt") -> dict:
    """
    Loads the trained 1D-CNN, evaluates a binned light curve tensor, 
    and returns class probabilities with the top prediction.
    """
    logger.info("Initializing inference engine and loading model weights...")
    
    # 1. Instantiate model structure and load weights
    model = Exoplanet1DCNN(num_classes=4)
    
    try:
        # Load the weights safely onto the CPU
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval() # Switch to evaluation mode (turns off dropout/batchnorm updates)
    except Exception as e:
        logger.error(f"Failed to load model weights from {weights_path}: {e}")
        raise

    # 2. Prepare the tensor input shape (Batch, 1, 2001)
    tensor_input = torch.tensor(binned_flux, dtype=torch.float32).unsqueeze(0)
    
    # 3. Run forward pass without calculating gradients (saves RAM/computation)
    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        
    top_class_idx = int(np.argmax(probabilities))
    prediction_label = CLASS_MAPPING[top_class_idx]
    confidence_score = float(probabilities[top_class_idx])
    
    logger.info(f"Inference Complete. Predicted: {prediction_label} ({confidence_score*100:.2f}%)")
    
    # Return structured metadata for the future FastAPI response
    return {
        "prediction": prediction_label,
        "confidence": confidence_score,
        "all_probabilities": {CLASS_MAPPING[i]: float(probabilities[i]) for i in range(4)}
    }


if __name__ == "__main__":
    import os
    from core.preprocessing.normalizer import process_raw_file
    from core.signal_search.bls_search import run_bls_search
    from core.signal_search.phase_folder import phase_fold_and_bin
    
    logging.basicConfig(level=logging.INFO)
    raw_file = "data/raw/tic_86396382_sec_20.npz"
    
    if os.path.exists(raw_file):
        # 1. Load and Preprocess 
        time, flux = process_raw_file(raw_file)
        flux_err = np.ones_like(flux) * 0.001
        
        # 2. Signal Processing Physics
        period, epoch, depth, _ = run_bls_search(time, flux, flux_err)
        cnn_tensor = phase_fold_and_bin(time, flux, period, epoch)
        
        # 3. Deep Learning Inference
        result = predict_exoplanet(cnn_tensor)
        
        print("\n--- Complete Local Pipeline Verification Success ---")
        print(f"Target System Prediction: {result['prediction']}")
        print(f"AI Confidence Score: {result['confidence'] * 100:.2f}%")
        print("Full Softmax Distribution:", result['all_probabilities'])