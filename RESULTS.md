# Results

All numbers are from `artifacts/metrics/` and reproducible with the steps in `README.md`. Evaluation contract: `CONTRACT.md`.
Reference parameters are TOI-catalogue values (SPOC fits to the same TESS data), so errors measure **agreement with the catalogue**, not ground truth.

## Data

- 300 frozen targets (150 planets: TOI dispositions CP/KP; 150 non-planets: FP), split 70/30 grouped by TIC, seed 42.
- 24 targets have no SPOC 2-min light curve on MAST and are excluded from every metric. Evaluated: **276** targets.
- The excluded targets are listed in `artifacts/metrics/excluded_no_data.csv`.
- Train: 193 (104 planets, 89 non-planets). Validation: **83** (45 planets, 38 non-planets). Intervals are wide at this size.

## 1. BLS detection (period within 1% of the TOI period)

| split | class | n | hit | harmonic | miss | hit rate |
|---|---|---|---|---|---|---|
| train | planet | 104 | 83 | 1 | 20 | 80% |
| val | planet | 45 | 34 | 0 | 11 | 76% |
| train | non-planet | 89 | 67 | 12 | 10 | 75% |
| val | non-planet | 38 | 36 | 0 | 2 | 95% |

Harmonics are 0.5x, 2x, 1/3x and 3x of the reference and are not counted as hits. All misses are listed in `artifacts/failures/bls_misses.csv`.

## 2. Classification (planet vs non-planet, validation, n=83)

Fixed hyperparameters, threshold 0.5, 95% bootstrap CIs (1000 resamples).

| model | PR-AUC | precision | recall | F1 | confusion [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|
| logistic regression | 0.74 [0.61, 0.88] | 0.75 | 0.73 | 0.74 | [[27,11],[12,33]] |
| gradient boosting | 0.85 [0.74, 0.94] | 0.71 | 0.80 | 0.75 | [[23,15],[9,36]] |
| BLS SNR alone (reference) | 0.72 | | | | |

- The intervals overlap heavily. Gradient boosting looks better than SNR alone, but 83 stars cannot establish that. Logistic regression is barely above the SNR-only reference.
- Gradient boosting overfits: train PR-AUC 0.999 vs 0.85 on validation. Settings were fixed before evaluation and not tuned.
- Gradient boosting is the primary model in the API. That choice was made after seeing validation PR-AUC (logged in `CONTRACT.md`).
- Labels are the TOI dispositions, and features come from whatever BLS found. Planets whose BLS detection failed (section 1) are still scored, so this evaluates the whole pipeline and not just the classifier.
- Misclassifications: `artifacts/failures/misclassified_val.csv`.

## 3. Parameter errors vs TOI (BLS-hit planets)

Median absolute relative error after the batman fit. "BLS box" is the raw BLS estimate for comparison.

| split | n | period | depth (fit) | depth (BLS box) | duration (fit) | duration (BLS box) |
|---|---|---|---|---|---|---|
| val | 34 | 5.5e-5 | 3.5% | 7.4% | 2.3% | 14.4% |
| train | 83 | 6.6e-5 | 4.3% | 13.2% | 3.3% | 12.4% |

| split | depth within 10% | duration within 10% | 90th percentile \|depth error\| |
|---|---|---|---|
| val (n=34) | 82% | 74% | 13% |
| train (n=83) | 71% | 87% | 24% |

- Fit uncertainties: scaled Jacobian covariance, with depth and duration errors from 300 Monte Carlo draws.
- Per-target table: `artifacts/metrics/param_errors_per_target.csv`. The "all planets" rows in `param_summary.json` include BLS misses, where the fit is on the wrong signal. Use the BLS-hit rows.

## 4. Failure cases (figures in `artifacts/figures/val/`)

- **Stellar variability beats the planet** (TIC437011608). Rotation at about +-0.4% survives the flatten step, and BLS picks 13.36 d at SNR 27 instead of the real 4.24 d planet (about 840 ppm).
- **No real peak** (TIC150428135, TIC26584326, TIC420112589). SNR is 5-10 and the periodogram power is noise. BLS power rises with period, which biases weak detections toward long periods (14.8 d at the grid edge in one case).
- **A systematic, not a transit** (TIC404421005). The "dip" is a ramp at the start of the sector, with SNR 34. This is a known planet that the classifier scores 0.29, so it is a false negative.
- **Few transits** (TIC49705089, FP, called planet). Two transits only, so the odd/even and secondary-eclipse features have almost no power.
- **Indistinguishable from a planet in the light curve** (TIC362709886, FP, called planet). Clean 6500 ppm dip, 6 transits, SNR 119.
- **A clean planet scored low** (TIC159951311, 0.455): four deep transits at SNR 70, still under the 0.5 threshold. Classifier probabilities do not track how planet-like a curve looks.

## 5. Limitations

- **Binary classifier.** The problem statement asks for transits, eclipses, blends and others. This pipeline separates planet from non-planet; vetting flags (odd/even, secondary eclipse, low SNR, few transits) are fixed rules, not a trained multi-class model. No blend classification.
- **Light-curve only.** No centroid or pixel-level vetting.
- **SNR assumes white noise**, so correlated noise inflates it (spurious detections reach SNR 27-34). Red-noise-aware significance is future work.
- **Probabilities are uncalibrated.**
- **Small sample.** 83 validation stars, one sector per target, 276 of 300 evaluated.
- **One signal per star.** Multi-planet systems: BLS reports the strongest signal only.
- **Contract timing.** The contract text was committed in cb48b94 (2026-09-30) before any download or training. The frozen target list (`configs/targets.csv`) was empty at that commit and is committed with the results, and the pre-training amendments are logged in `CONTRACT.md`; `git diff cb48b94 -- configs/contract.yaml` shows them.
- **Correction logged in `CONTRACT.md` (2026-10-01):** the API originally computed features from a curve flattened with the pass-1 ephemeris, while training used the final one. This affected 21 of 276 targets and is fixed; stored and API probabilities now match.

## 6. Not done

- CNN-LSTM comparison (optional in the brief).
- Hyperparameter tuning, probability calibration, bigger training set, frontend.
