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
- (freeze commit SHA goes here)