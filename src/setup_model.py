"""
setup_model.py
==============
SETUP — retrain the XGBoost_cw fraud-detection pipeline that the three
extensions depend on, then save it.

Recreates the exact training pipeline from the original notebook:
  * Stratified 80/20 split (random_state=42)
  * Stratified 60,000-row subsample
  * Stratified 75/25 train/val split inside the subsample
  * XGBClassifier with the parameters specified by the brief
  * imblearn.Pipeline([StandardScaler, XGBClassifier])
  * Decision threshold = 0.4949

Outputs written:
  * outputs/xgb_fraud_model.pkl   (full fitted imblearn.Pipeline)
  * outputs/scaler.pkl            (the fitted StandardScaler from the pipeline)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Force UTF-8 stdout so '£' and '✓' render on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# Make this script runnable both as `python src/setup_model.py` and as a
# module import from another script in src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline import prepare_splits, outputs_dir, RNG, PALETTE  # noqa: E402

warnings.filterwarnings("ignore")

THRESHOLD = 0.4949   # decision threshold from the original notebook


def build_xgb_pipeline(scale_pos_weight: float) -> ImbPipeline:
    """
    XGBoost_cw pipeline — the exact configuration specified in the brief.

    The pipeline wraps StandardScaler + XGBClassifier so the scaler is fit
    only on the training data (no leakage from validation/test).
    """
    # XGBoost >= 2.0 deprecated `use_label_encoder`; pass it only if accepted.
    xgb_kwargs = dict(
        max_depth=4,
        min_child_weight=5,
        subsample=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RNG,
        n_jobs=-1,
        tree_method="hist",
    )
    try:
        clf = XGBClassifier(use_label_encoder=False, **xgb_kwargs)
    except TypeError:
        # Newer XGBoost has removed the kwarg entirely
        clf = XGBClassifier(**xgb_kwargs)

    return ImbPipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", clf),
    ])


def main() -> int:
    print("=" * 70)
    print("SETUP — retraining XGBoost_cw fraud-detection model")
    print("=" * 70)

    splits = prepare_splits(verbose=True)
    X_train, y_train = splits.X_train, splits.y_train
    X_val,   y_val   = splits.X_val,   splits.y_val
    X_test,  y_test  = splits.X_test,  splits.y_test

    # scale_pos_weight = (neg / pos) on the train set
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)
    print(f"\nscale_pos_weight = {n_neg:,} / {n_pos} = {spw:.2f}")

    # Build + fit pipeline
    pipe = build_xgb_pipeline(scale_pos_weight=spw)
    print("\nFitting imblearn.Pipeline([StandardScaler, XGBClassifier(...)])"
          f"  on {len(X_train):,} rows ...")
    pipe.fit(X_train, y_train)
    print("  ✓ Pipeline fitted.")

    # ---- Evaluate on TEST ---------------------------------------------------
    y_proba_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_test  = (y_proba_test >= THRESHOLD).astype(int)

    pr_auc    = average_precision_score(y_test, y_proba_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall    = recall_score(y_test, y_pred_test)
    f1        = f1_score(y_test, y_pred_test, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

    print("\nTest-set results @ threshold = {:.4f}".format(THRESHOLD))
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  CM        : TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    # ---- Business £ summary -------------------------------------------------
    amounts_test = splits.amounts_test
    fraud_total   = float(amounts_test[y_test == 1].sum())
    fraud_caught  = float(amounts_test[(y_test == 1) & (y_pred_test == 1)].sum())
    fp_friction   = fp * 5.0
    print(f"\nBusiness impact:")
    print(f"  Fraud value in test window : £{fraud_total:,.2f}")
    print(f"  Recovered by model         : £{fraud_caught:,.2f} "
          f"({fraud_caught/max(fraud_total,1):.1%})")
    print(f"  FP friction (£5 per FP)    : £{fp_friction:,.2f}")
    print(f"  Net benefit                : £{fraud_caught - fp_friction:,.2f}")

    print("\nSO WHAT: PR-AUC of {:.4f} on a base-rate of 0.17% means the model "
          "is roughly {:.0f}x better than random at ranking fraud."
          .format(pr_auc, pr_auc / (y_test.mean())))

    # ---- Persist ------------------------------------------------------------
    out = outputs_dir()
    model_path  = out / "xgb_fraud_model.pkl"
    scaler_path = out / "scaler.pkl"
    joblib.dump(pipe, model_path)
    joblib.dump(pipe.named_steps["scaler"], scaler_path)

    # Also save a tiny metadata file the extensions can read
    meta = {
        "threshold": THRESHOLD,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fraud_total": fraud_total, "fraud_caught": fraud_caught,
        "fp_friction": fp_friction,
        "feature_names": splits.feature_names,
        "scale_pos_weight": spw,
    }
    joblib.dump(meta, out / "model_metadata.pkl")

    print(f"\nSaved:")
    print(f"  {model_path}")
    print(f"  {scaler_path}")
    print(f"  {out / 'model_metadata.pkl'}")
    print("\nSETUP COMPLETE ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
