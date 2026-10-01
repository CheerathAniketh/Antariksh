"""Train baselines on the frozen train split, evaluate once on val, save metrics + misclassifications."""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from antariksh.classify.baseline import FEATURES, build_gb, build_lr, prep_lr
from antariksh.config import ARTIFACTS_DIR, CONFIGS_DIR, CONTRACT, PROCESSED_DIR

warnings.filterwarnings("ignore")
SEED = CONTRACT["seed"]
N_BOOT = 1000


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return {"precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh, zero_division=0),
            "f1": f1_score(y, yh, zero_division=0),
            "pr_auc": average_precision_score(y, p)}


def bootstrap_ci(y, p):
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        rows.append(metrics(y[i], p[i]))
    b = pd.DataFrame(rows)
    return {k: [float(b[k].quantile(0.025)), float(b[k].quantile(0.975))] for k in b}


def main():
    t = pd.read_csv(CONFIGS_DIR / "targets.csv")
    f = pd.read_csv(PROCESSED_DIR / "features.csv")
    d = t.merge(f[f.error.isna()].drop(columns="error"), on="tic_id", suffixes=("_ref", ""))
    tr, va = d[d.split == "train"], d[d.split == "val"]
    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()
    print(f"train n={len(tr)} (pos {ytr.sum()}) | val n={len(va)} (pos {yva.sum()})")

    lr, gb = build_lr(SEED).fit(prep_lr(tr), ytr), build_gb(SEED).fit(tr[FEATURES], ytr)
    scores = {"logreg": (lr.predict_proba(prep_lr(va))[:, 1], lr.predict_proba(prep_lr(tr))[:, 1]),
              "grad_boost": (gb.predict_proba(va[FEATURES])[:, 1], gb.predict_proba(tr[FEATURES])[:, 1])}

    out, bad = {}, []
    for name, (pv, ptr) in scores.items():
        m, ci = metrics(yva, pv), bootstrap_ci(yva, pv)
        cm = confusion_matrix(yva, (pv >= 0.5).astype(int), labels=[0, 1]).tolist()
        out[name] = {"val": m, "val_ci95": ci, "confusion_[[TN,FP],[FN,TP]]": cm,
                     "train_pr_auc_in_sample": float(average_precision_score(ytr, ptr))}
        print(f"\n{name}  val: " + "  ".join(f"{k}={v:.3f} [{ci[k][0]:.2f},{ci[k][1]:.2f}]" for k, v in m.items()))
        print(f"  confusion [[TN,FP],[FN,TP]] = {cm} | train PR-AUC (in-sample) = {out[name]['train_pr_auc_in_sample']:.3f}")
        w = va.assign(model=name, prob_planet=pv)
        bad.append(w[(pv >= 0.5).astype(int) != yva][
            ["tic_id", "model", "label", "disposition", "prob_planet", "period_d_ref", "period_d", "bls_snr"]])

    # trivial reference: rank by BLS SNR alone
    out["snr_only_reference"] = {"val_pr_auc": float(average_precision_score(yva, va.bls_snr))}
    print(f"\nSNR-only reference PR-AUC = {out['snr_only_reference']['val_pr_auc']:.3f}  (prevalence = {yva.mean():.3f})")

    (ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "failures").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    json.dump({"n_train": len(tr), "n_val": len(va), **out}, open(ARTIFACTS_DIR / "metrics" / "metrics.json", "w"), indent=2)
    pd.concat(bad).to_csv(ARTIFACTS_DIR / "failures" / "misclassified_val.csv", index=False)
    joblib.dump({"logreg": lr, "grad_boost": gb, "features": FEATURES}, ARTIFACTS_DIR / "models" / "baseline.joblib")
    print("\nsaved metrics.json, misclassified_val.csv, baseline.joblib")


if __name__ == "__main__":
    main()
