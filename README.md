# Antariksh

Takes a TESS light curve, searches it for transit signals, scores whether the signal looks like a planet, and fits transit parameters. Python backend (FastAPI) with a small React frontend.

TESS light curve -> BLS transit search -> planet / non-planet classifier -> batman transit fit -> FastAPI + diagnostic plots.

Built to the evaluation contract in `CONTRACT.md` (machine-readable: `configs/contract.yaml`). The contract text was frozen in commit `cb48b94`; later amendments are logged in its changelog and visible with `git diff cb48b94 -- CONTRACT.md configs/contract.yaml`. Frozen target list: `configs/targets.csv`. Results, failures and limitations: `RESULTS.md`.

## Results

Validation split: 83 stars (45 planets, 38 non-planets), from 276 evaluated targets. 24 of the 300 frozen targets have no SPOC 2-min light curve and are excluded. Labels are TOI dispositions, so the errors measure agreement with the catalogue, not ground truth. Full tables, failure cases and limitations are in `RESULTS.md`.

| Stage | Validation result |
|---|---|
| BLS period recovery (within 1% of the TOI period) | 34/45 planets (76%), 36/38 non-planets (95%) |
| Classifier: gradient boosting (primary model) | PR-AUC 0.85 [0.74, 0.94], precision 0.71, recall 0.80 |
| Classifier: logistic regression (predeclared baseline) | PR-AUC 0.74 [0.61, 0.88], precision 0.75, recall 0.73 |
| BLS SNR alone (reference) | PR-AUC 0.72 |
| Transit fit vs TOI, 34 BLS-hit planets (median error) | depth 3.5%, duration 2.3% (raw BLS box: 7.4% and 14.4%) |

- The classifier intervals overlap heavily. 83 stars cannot establish that gradient boosting beats the baseline or SNR alone.
- Gradient boosting was chosen as the primary model after seeing validation PR-AUC, and it overfits (train PR-AUC 0.999 vs 0.85 on validation). Scores are uncalibrated model outputs, not probabilities.
- A post-hoc label-shuffle check (200 permutations, not preregistered) gives p of about 0.005 for gradient boosting and about 0.09 for logistic regression.
- BLS misses 11 of 45 validation planets, mostly from stellar variability or noise-level peaks. Examples are in `RESULTS.md` and `artifacts/figures/val/`.

## Setup

Tested on Fedora with Python 3.14.

    python3 -m venv venv
    source venv/bin/activate
    pip install -r backend/requirements.lock.txt
    pip install -e backend

`backend/requirements.lock.txt` holds the exact versions behind the reported results; the saved model (`artifacts/models/baseline.joblib`) was pickled with them. `backend/requirements.txt` lists the unpinned dependencies.
batman needs a C compiler (`sudo dnf install gcc python3-devel` on Fedora) and `setuptools` (for `distutils`).

## Reproduce the reported results

Each step reads the previous step's output. `data/` is gitignored, so a fresh clone must download and recompute it.

    python backend/scripts/01_build_targets.py      # configs/targets.csv from configs/toi_snapshot.csv (regenerates identically)
    python backend/scripts/02_download.py           # TESS SPOC 2-min light curves -> data/raw (resumable, slow)
    python backend/scripts/03_run_detection.py      # two-pass preprocess + BLS -> data/processed/bls_results.csv
    python backend/scripts/03b_features.py          # feature table
    python backend/scripts/04_train.py              # baselines -> artifacts/metrics, artifacts/models
    python backend/scripts/05_evaluate.py           # parameter errors vs TOI -> artifacts/metrics
    python backend/scripts/06_make_figures.py       # 4-panel figures -> artifacts/figures/val
    python backend/scripts/07_label_shuffle_posthoc.py   # post-hoc leakage diagnostic (not part of the frozen contract)
    python backend/scripts/08_api_smoke.py          # API smoke receipt -> artifacts/receipts/api_smoke.json

Targets with no SPOC 2-min light curve are skipped and logged in `data/raw/download_log.csv`.

## API

    python -m uvicorn antariksh.api.main:app --port 8000

    curl -s -X POST localhost:8000/analyze -H "Content-Type: application/json" \
      -d '{"tic_id": 393831507, "include_plot": true}'

Request: `tic_id` (downloaded and cached if missing), or `time` + `flux` (+ `flux_err`). Response: detection flag, BLS signal with SNR, batman fit with uncertainties, uncalibrated classifier score (`score_planet`, not a probability), vetting flags, optional base64 PNG of the 4-panel plot.

## Frontend

A basic React + Vite + TypeScript UI in `frontend/` posts to `/analyze` and shows the BLS signal, the transit fit and the classifier score. It is a thin client over the API and is not deployed. The Vite dev server proxies `/analyze` and `/health` to `127.0.0.1:8000`, so the API must be running locally.

    cd frontend
    npm install
    npm run dev

## Tests

    python -m pytest backend/tests -q

Six tests (3 API, 2 BLS, 1 transit fit). The BLS and fit tests inject synthetic signals with known truth. The API test uses the committed `artifacts/models/baseline.joblib`.

## Scope

Binary classifier (planet vs non-planet) trained on a few hundred TOI-labelled stars. Light-curve only: no centroid or pixel-level blend vetting. See `RESULTS.md` for limitations and failure cases.
