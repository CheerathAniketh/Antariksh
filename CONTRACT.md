# Antariksh Evaluation Contract (frozen before any training)

Tracked in the-bu1ld-nexus#69. Machine-readable version: configs/contract.yaml.
Any change after the freeze commit must be logged in the Changelog below, not silently edited.

## Data
- TESS SPOC 2-min PDCSAP light curves via Lightkurve/MAST, one sector per target (lowest available).
- Labels from the NASA Exoplanet Archive TOI table dispositions.
  Positive: CP, KP. Negative: FP. PC/APC/FA excluded (no reliable label).
- Reference parameters are the TOI catalogue values, not independent literature values.
  Errors are therefore agreement with the catalogue, not ground truth.

## Split
Stratified by label, grouped by TIC ID, seed 42, 70/30 train/val. Frozen list in configs/targets.csv.

## Preprocessing (fixed order)
quality mask -> drop NaN -> median normalize -> flatten (Savitzky-Golay, 721 cadences, order 2)
-> upper sigma clip (4 sigma) -> stitch.

## Detection
astropy BoxLeastSquares, periods 0.5-15 d, durations 0.5-6 h.
Hit = period within 1% of the reference. Harmonics (0.5x, 2x) are counted as misses and reported separately.

## Classification
Binary planet vs non-planet. Baselines: logistic regression and gradient boosting, fixed hyperparameters.
Metrics: precision, recall, F1, PR-AUC, confusion matrix on the validation split.
Optional CNN-LSTM only as a same-contract comparison after the baseline works.

## Parameter error
Relative error of period, depth, duration vs TOI values; median absolute relative error plus per-target table.

## Known limitations
- Non-planet class is a mix (EBs, blends, systematics); no sub-classification in M1.
- No centroid/pixel-level blend vetting (light curve only).
- Small sample; results reported with that caveat.

## Changelog
- Freeze commit: cb48b94 (2026-09-30). The contract text was committed there before any download or training. The target list and the amendments below came afterwards; see `git diff cb48b94 -- CONTRACT.md configs/contract.yaml`.
- 2026-09-30 (pre-training): added bls.max_duty_cycle = 0.1 (duration/period). BLS peaks with a
  duty cycle above 10% are rejected as non-transit. Motivated by smoke-test failures on 2 targets, not tuned on validation labels.
- 2026-09-30 (pre-training): flatten is now two-pass. Pass 1 runs BLS on the flattened curve; pass 2
  re-flattens with the detected transits (1.5x duration margin) excluded from the trend fit, then re-runs BLS.
  Motivated by a synthetic injection test where single-pass flattening removed ~70% of the injected depth. No validation labels used.
- 2026-09-30 (pre-training): 24 of 300 frozen targets have no SPOC 2-min light curve on MAST and are excluded from every metric. targets.csv is unchanged; the evaluated set is the 276 targets with data (see data/raw/download_log.csv).
- 2026-09-30 (pre-training): baseline hyperparameters fixed before any val evaluation. LogReg: C=1, class_weight=balanced, median impute, standardize, log10 on heavy-tailed positive columns. HistGradientBoosting: max_depth 3, lr 0.05, 200 iters, l2 1.0. Threshold 0.5. 95% CIs by bootstrap (1000 resamples of val, seed 42). Features are light-curve derived only; no TOI values or magnitudes.
- 2026-10-01 (pre-evaluation): parameter-fit settings fixed before any error table was produced. batman, circular orbit, quadratic limb darkening fixed at u=[0.35, 0.25] (generic TESS band), free t0/period/rp/a/b, window max(3*BLS duration, 0.25 d), 3 starts in b, uncertainties from the scaled Jacobian covariance plus 300 Monte Carlo draws. Reported duration is T14. Headline error summary is on BLS-hit planets in val; all planets are reported alongside.
- 2026-10-01: API thresholds fixed a priori: detection requires BLS SNR >= 7 and >= 2 transits; vetting flags are rules (odd/even mismatch > 3 sigma and > 20% relative; secondary eclipse > 3 sigma and > 10% of primary depth; low_snr; few_transits). Flags are not trained. The API reports gradient boosting as the primary probability, a choice made after seeing val PR-AUC (logreg is returned alongside).
- 2026-10-01 (correction): features (03b), parameter fits (05) and now the API/figures all use the curve re-flattened with the final BLS ephemeris (a third flatten after pass 2). The API previously used the pass-2 input curve (flattened with the pass-1 ephemeris), which gave different features whenever the two passes disagreed. Found when TIC404421005's API probability contradicted its stored prediction. Serving code changed to match training; no model, metric or threshold changed.
- 2026-10-03 (clarification): the harmonic set used in evaluation is 0.5x, 2x, 1/3x and 3x of the reference period (see classify_period). The Detection section above lists only 0.5x and 2x. No metric changed.
