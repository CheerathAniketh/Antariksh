from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]      # repo root
CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"


def load_contract() -> dict:
    with open(CONFIGS_DIR / "contract.yaml") as f:
        return yaml.safe_load(f)


CONTRACT = load_contract()