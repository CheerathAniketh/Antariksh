"""POST-HOC label-shuffle leakage diagnostic (not part of the frozen contract).

Added after the baseline results were seen. It is a sanity check, not a result:
  * it is NOT preregistered,
  * it must NOT be used to tune features, models or thresholds,
  * it does NOT touch metrics.json or baseline.joblib (writes only label_shuffle_posthoc.json).

Method: keep the frozen split and the fixed-hyperparameter models from 04_train.py. Randomly permute the
TRAIN labels, refit, and score the validation stars against their TRUE labels. If the features carry no
information that survives destroying the train label<->feature link, validation PR-AUC should collapse to
about the validation prevalence. A shuffled-label model that still scores well on validation points to
leakage (e.g. duplicated stars across splits, or features that encode the label).
In-sample train PR-AUC under shuffled labels is also recorded; gradient boosting can memorise noise, so it is
expected to be high and is not informative about leakage.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from antariksh.classify.baseline import FEATURES, build_gb, build_lr, prep_lr
from antariksh.config import ARTIFACTS_DIR, CONFIGS_DIR, CONTRACT, PROCESSED_DIR

warnings.filterwarnings("ignore")
SEED = CONTRACT["seed"]
N_PERM = 200


def summarize(null, observed, prevalence):
    null = np.asarray(null)
    return {
        "observed_val_pr_auc_true_labels": float(observed),
        "val_prevalence_chance_level": float(prevalence),
        "shuffled_val_pr_auc_mean": float(null.mean()),
        "shuffled_val_pr_auc_p95": float(np.quantile(null, 0.95)),
        "shuffled_val_pr_auc_max": float(null.max()),
        # +1 smoothing so the empirical p-value is never exactly 0
        "empirical_p_ge_observed": float((1 + (null >= observed).sum()) / (1 + len(null))),
    }


def main():
    t = pd.read_csv(CONFIGS_DIR / "targets.csv")
    f = pd.read_csv(PROCESSED_DIR / "features.csv")
    d = t.merge(f[f.error.isna()].drop(columns="error"), on="tic_id", suffixes=("_ref", ""))
    tr, va = d[d.split == "train"], d[d.split == "val"]
    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()
    prev = float(yva.mean())
    print(f"POST-HOC diagnostic | train n={len(tr)} val n={len(va)} val prevalence={prev:.3f} | permutations={N_PERM}")

    # split-integrity check: no star on both sides of the split
    overlap = set(tr.tic_id) & set(va.tic_id)
    print(f"TIC ids in both train and val: {len(overlap)}")

    obs = {
        "logreg": average_precision_score(yva, build_lr(SEED).fit(prep_lr(tr), ytr).predict_proba(prep_lr(va))[:, 1]),
        "grad_boost": average_precision_score(yva, build_gb(SEED).fit(tr[FEATURES], ytr).predict_proba(va[FEATURES])[:, 1]),
    }

    rng = np.random.default_rng(SEED)
    null = {"logreg": [], "grad_boost": []}
    train_in_sample = {"logreg": [], "grad_boost": []}
    for _ in range(N_PERM):
        yp = rng.permutation(ytr)
        lr = build_lr(SEED).fit(prep_lr(tr), yp)
        gb = build_gb(SEED).fit(tr[FEATURES], yp)
        null["logreg"].append(average_precision_score(yva, lr.predict_proba(prep_lr(va))[:, 1]))
        null["grad_boost"].append(average_precision_score(yva, gb.predict_proba(va[FEATURES])[:, 1]))
        train_in_sample["logreg"].append(average_precision_score(yp, lr.predict_proba(prep_lr(tr))[:, 1]))
        train_in_sample["grad_boost"].append(average_precision_score(yp, gb.predict_proba(tr[FEATURES])[:, 1]))

    out = {
        "status": "post-hoc diagnostic; not preregistered; not used for tuning; frozen contract unchanged",
        "n_permutations": N_PERM,
        "seed": SEED,
        "tic_ids_in_both_splits": len(overlap),
        "models": {m: {**summarize(null[m], obs[m], prev),
                       "shuffled_train_in_sample_pr_auc_mean": float(np.mean(train_in_sample[m]))}
                   for m in null},
    }
    for m, r in out["models"].items():
        print(f"\n{m}: observed val PR-AUC {r['observed_val_pr_auc_true_labels']:.3f} | "
              f"shuffled mean {r['shuffled_val_pr_auc_mean']:.3f}, p95 {r['shuffled_val_pr_auc_p95']:.3f}, "
              f"max {r['shuffled_val_pr_auc_max']:.3f} | chance {prev:.3f} | p>=obs {r['empirical_p_ge_observed']:.4f}")

    (ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "metrics" / "label_shuffle_posthoc.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nsaved artifacts/metrics/label_shuffle_posthoc.json")


if __name__ == "__main__":
    main()
