"""Fixed-hyperparameter baselines (contract: no tuning). Features are light-curve derived only."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "period_d", "duration_h", "duty_cycle", "bls_depth_ppm", "bls_snr", "n_transits",
    "depth_meas_ppm", "depth_per_hour_ppm", "oot_scatter_ppm", "odd_even_sigma",
    "odd_even_rel", "sec_depth_ppm", "sec_snr", "sec_ratio", "v_ratio",
]
LOG_COLS = ["period_d", "duration_h", "bls_depth_ppm", "bls_snr", "depth_meas_ppm",
            "depth_per_hour_ppm", "oot_scatter_ppm"]


def prep_lr(X):
    """log10 of the heavy-tailed positive columns; trees do not need this."""
    X = X[FEATURES].copy()
    for c in LOG_COLS:
        X[c] = np.log10(X[c].clip(lower=1e-3))
    return X


def build_lr(seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)),
    ])


def build_gb(seed):
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=seed
    )
