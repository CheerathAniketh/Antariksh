# Evidence manifest: Nexus #69, M1 baseline slice

Scope: a reproducible, light-curve-only TESS planet-candidate baseline (BLS detection, planet / non-planet classifier, batman fit, FastAPI). Not a replacement for astronomer vetting.

## Revisions

| item | value |
|---|---|
| Commit reviewed at first handoff | `086665d1019e16400a0aa07727cf784fc264c7e6` |
| Contract freeze commit | `cb48b94` |
| Closeout revision | git tag `m1-closeout`. A file cannot contain its own commit hash, so the exact SHA is given in the closeout email and by `git rev-parse m1-closeout`. |

Everything after `086665d` is listed in the `CONTRACT.md` changelog (entries dated 2026-10-03): API field rename, post-hoc diagnostic, smoke script, committed copies of logs and tables. No model, metric or threshold changed.

## Artifacts

| evidence | location |
|---|---|
| Frozen contract | `CONTRACT.md`, `configs/contract.yaml` |
| Frozen target list (300) | `configs/targets.csv` |
| Download log | `artifacts/metrics/download_log.csv` (copy of `data/raw/download_log.csv`) |
| 24 targets with no SPOC 2-min data | `artifacts/metrics/excluded_no_data.csv` |
| Classifier metrics and 95% CIs | `artifacts/metrics/metrics.json` |
| Validation misclassifications | `artifacts/failures/misclassified_val.csv` |
| BLS recovery table | `RESULTS.md` section 1; counts in `artifacts/metrics/param_summary.json`; per-target `artifacts/metrics/bls_results.csv`; misses `artifacts/failures/bls_misses.csv` |
| Parameter-error table | `artifacts/metrics/param_errors_per_target.csv`, `artifacts/metrics/param_summary.json` |
| Six-test receipt | `artifacts/receipts/pytest.txt` |
| API smoke receipt | `artifacts/receipts/api_smoke.json` |
| Post-hoc label-shuffle diagnostic | `artifacts/metrics/label_shuffle_posthoc.json` |

## Post-hoc label-shuffle diagnostic

Post-hoc, not preregistered, run after the validation results were known, and not used to change any feature, model or threshold. Method: permute the train labels 200 times, refit each fixed-hyperparameter model, and score validation against the true labels. Validation prevalence (chance PR-AUC) is 0.542.

Result (200 permutations, seed 42): no train/val star overlap. Shuffled-label validation PR-AUC centres near chance (logistic regression mean 0.572, gradient boosting mean 0.556, prevalence 0.542). Gradient boosting's observed 0.855 exceeds its shuffled 95th percentile (0.685; empirical p = 0.005), so the diagnostic gives no evidence of leakage. Logistic regression's observed 0.740 is below its shuffled 95th percentile (0.768; empirical p = 0.090), so with n = 83 it cannot be distinguished from shuffled-label chance. This is a post-hoc diagnostic on a small validation set and does not establish absence of all leakage.

## Claim table

| status | claim |
|---|---|
| Supported | The pipeline (ingest, preprocess, BLS, classifier, batman fit, API) runs end to end on real TESS SPOC 2-min data and reproduced identically from a cleared `data/processed/`. |
| Supported | BLS recovers the TOI period within 1% for 34 of 45 validation planets (76%) and 36 of 38 validation non-planets. All misses are retained in `artifacts/failures/`. |
| Supported | On the 34 BLS-hit validation planets, median absolute relative error vs the TOI catalogue after the batman fit: depth 3.5%, duration 2.3%. These measure agreement with the catalogue, not ground truth. |
| Supported | Validation PR-AUC (n=83): logistic regression 0.74 [0.61, 0.88], gradient boosting 0.85 [0.74, 0.94], BLS SNR alone 0.72. |
| Supported | Failure cases, the 24 of 300 targets without SPOC data, and the light-curve-only, binary-classifier scope are documented. |
| Unsupported | That gradient boosting beats logistic regression or the SNR-only reference. The intervals overlap, n=83, and gradient boosting was chosen as the API model after seeing validation PR-AUC. Logistic regression is the predeclared baseline. |
| Unsupported | That the API scores are calibrated probabilities. They are uncalibrated model outputs; no calibration study was done. |
| Unsupported | That results generalise to all 300 targets (24 lack SPOC 2-min data), to other sectors, or to stars without TOI labels. |
| Unsupported | That this replaces astronomer vetting. There is no centroid or pixel-level blend vetting, and the classifier separates only planet from non-planet. |
| Exploratory | The gradient-boosting vs logistic-regression PR-AUC ordering (post-selection observation). |
| Exploratory | The label-shuffle diagnostic (post-hoc). |
| Exploratory | The 1/3x and 3x harmonic categories. The frozen text names only 0.5x and 2x; the wider set was documented after the freeze, and the 1% hit rule is unchanged. |

## Retained limitations

- Gradient boosting overfits: train PR-AUC 0.999 vs 0.85 on validation. Settings were fixed before evaluation and not tuned.
- Labels are TOI dispositions and reference parameters are TOI catalogue values (SPOC fits to the same data).
- SNR assumes white noise, so correlated noise inflates it. One signal per star. One sector per target.
