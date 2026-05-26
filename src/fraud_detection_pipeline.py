"""
Credit Card Fraud Detection Pipeline (BUSM131 coursework)
=========================================================
A methodologically sound, end-to-end pipeline:
  - Stratified train/test split BEFORE any scaling (no leakage)
  - sklearn / imblearn Pipelines for preprocessing + resampling
  - Class imbalance handled two ways: class_weight='balanced' AND SMOTE
  - Three models: Logistic Regression, Random Forest, Gradient Boosting (XGBoost optional)
  - Stratified k-fold CV with hyperparameter tuning (RandomizedSearchCV)
  - Imbalance-appropriate metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
  - Decision-threshold optimisation on the validation PR curve
  - Business interpretation focused on recall and expected financial loss avoided
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    pos_rate = y.mean()
    print(f"Loaded {len(df):,} rows | fraud rate = {pos_rate:.4%} "
          f"({y.sum():,} of {len(y):,})")
    return X, y


# ---------------------------------------------------------------------------
# 2. Pipeline factories
# ---------------------------------------------------------------------------
def build_pipelines() -> Dict[str, ImbPipeline]:
    """
    Return a dict of named pipelines.

    Each pipeline has the form:  StandardScaler -> [SMOTE] -> classifier.
    StandardScaler sits INSIDE the pipeline so it is fit only on training
    folds during cross-validation -> no leakage from test rows.

    Two imbalance strategies are exercised:
      * `*_cw`     : class_weight='balanced' (cost-sensitive learning)
      * `*_smote`  : SMOTE oversampling on the training fold only
    """
    pipelines: Dict[str, ImbPipeline] = {}

    # --- Logistic Regression -------------------------------------------------
    pipelines["LogReg_cw"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear", class_weight="balanced",
            max_iter=2000, random_state=RNG)),
    ])
    pipelines["LogReg_smote"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RNG)),
        ("clf", LogisticRegression(
            solver="liblinear", max_iter=2000, random_state=RNG)),
    ])

    # --- Random Forest -------------------------------------------------------
    pipelines["RandomForest_cw"] = ImbPipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(
            n_estimators=300, n_jobs=-1,
            class_weight="balanced", random_state=RNG)),
    ])
    pipelines["RandomForest_smote"] = ImbPipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("smote", SMOTE(random_state=RNG)),
        ("clf", RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RNG)),
    ])

    # --- Gradient Boosting ---------------------------------------------------
    if HAS_XGB:
        # XGBoost handles imbalance via scale_pos_weight (cost-sensitive)
        pipelines["XGBoost_cw"] = ImbPipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.1,
                eval_metric="aucpr", tree_method="hist",
                n_jobs=-1, random_state=RNG)),
        ])
        pipelines["XGBoost_smote"] = ImbPipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("smote", SMOTE(random_state=RNG)),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.1,
                eval_metric="aucpr", tree_method="hist",
                n_jobs=-1, random_state=RNG)),
        ])
    else:
        pipelines["GradBoost_cw"] = ImbPipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.1,
                random_state=RNG)),
        ])
        pipelines["GradBoost_smote"] = ImbPipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("smote", SMOTE(random_state=RNG)),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.1,
                random_state=RNG)),
        ])

    return pipelines


def param_grid(name: str) -> Dict:
    """Compact RandomizedSearchCV grid per model family."""
    if name.startswith("LogReg"):
        return {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    if name.startswith("RandomForest"):
        return {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 8, 16],
            "clf__min_samples_split": [2, 10],
        }
    if name.startswith("XGBoost"):
        # scale_pos_weight is the imbalance lever for the *_cw variant
        return {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.05, 0.1],
            "clf__scale_pos_weight": [1, 100, 577],  # ~neg/pos ratio
        }
    if name.startswith("GradBoost"):
        return {
            "clf__n_estimators": [150, 300],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
        }
    return {}


# ---------------------------------------------------------------------------
# 3. Tuning + evaluation
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    name: str
    best_params: Dict
    cv_pr_auc: float
    threshold: float
    metrics: Dict[str, float]
    y_proba: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    pipeline: ImbPipeline = field(repr=False)


def tune_model(name: str, pipe: ImbPipeline,
               X_train: pd.DataFrame, y_train: pd.Series,
               n_iter: int = 6) -> Tuple[ImbPipeline, Dict, float]:
    """RandomizedSearchCV optimised for PR-AUC (avg precision)."""
    grid = param_grid(name)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)

    if not grid:
        pipe.fit(X_train, y_train)
        return pipe, {}, np.nan

    search = RandomizedSearchCV(
        pipe, param_distributions=grid,
        n_iter=min(n_iter, np.prod([len(v) for v in grid.values()])),
        scoring="average_precision",  # PR-AUC: best for heavy imbalance
        cv=cv, n_jobs=-1, random_state=RNG, refit=True, verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                   min_recall: float = 0.85) -> float:
    """
    Pick the smallest threshold whose recall >= min_recall, breaking ties
    by maximising F1. If the recall floor is unreachable, fall back to the
    F1-maximising threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one extra point; align lengths
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)

    eligible = recall >= min_recall
    if eligible.any():
        idx_pool = np.where(eligible)[0]
        idx = idx_pool[np.argmax(f1[idx_pool])]
    else:
        idx = int(np.argmax(f1))
    return float(thresholds[idx])


def evaluate(name: str, pipe: ImbPipeline,
             X_test: pd.DataFrame, y_test: pd.Series,
             threshold: float) -> Dict[str, float]:
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
        "PR-AUC":    average_precision_score(y_test, y_proba),
    }, y_proba, y_pred


# ---------------------------------------------------------------------------
# 4. Plot helpers
# ---------------------------------------------------------------------------
def plot_curves(results: List[ModelResult], y_test: np.ndarray,
                outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    # ROC
    plt.figure(figsize=(7, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r.y_proba)
        plt.plot(fpr, tpr, label=f"{r.name} (AUC={r.metrics['ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curves.png"), dpi=150)
    plt.close()

    # Precision-Recall
    plt.figure(figsize=(7, 6))
    for r in results:
        p, rec, _ = precision_recall_curve(y_test, r.y_proba)
        plt.plot(rec, p, label=f"{r.name} (AP={r.metrics['PR-AUC']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall curves"); plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_curves.png"), dpi=150)
    plt.close()

    # Confusion matrix for the best model (highest PR-AUC)
    best = max(results, key=lambda r: r.metrics["PR-AUC"])
    cm = confusion_matrix(y_test, best.y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"])
    plt.title(f"Confusion Matrix — {best.name} @ thr={best.threshold:.3f}")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix_best.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 5. Business framing
# ---------------------------------------------------------------------------
def business_summary(best: ModelResult, y_test: pd.Series,
                     amounts_test: pd.Series) -> str:
    """
    Translate confusion-matrix counts into £ exposure.
    Assumption: the fraud amount is fully lost if missed (FN), and a
    blocked-but-legit transaction (FP) costs ~£5 in friction (review +
    customer churn proxy). These constants are illustrative.
    """
    cm = confusion_matrix(y_test, best.y_pred)
    tn, fp, fn, tp = cm.ravel()

    fraud_total   = float(amounts_test[y_test == 1].sum())
    fraud_caught  = float(amounts_test[(y_test == 1) & (best.y_pred == 1)].sum())
    fraud_missed  = float(amounts_test[(y_test == 1) & (best.y_pred == 0)].sum())
    fp_friction   = fp * 5.0

    return (
        f"\n--- Business impact ({best.name}) ---\n"
        f"Total fraud value in test window : £{fraud_total:,.2f}\n"
        f"Recovered by model (TP £)        : £{fraud_caught:,.2f}\n"
        f"Missed by model    (FN £)        : £{fraud_missed:,.2f}\n"
        f"Loss-prevention ratio            : {fraud_caught / max(fraud_total,1):.2%}\n"
        f"False-positive friction cost     : £{fp_friction:,.2f} "
        f"({fp:,} legit txns flagged)\n"
        f"Net benefit (recovered - friction): "
        f"£{fraud_caught - fp_friction:,.2f}\n"
    )


# ---------------------------------------------------------------------------
# 6. Driver
# ---------------------------------------------------------------------------
def main(csv_path: str, outdir: str = "outputs") -> pd.DataFrame:
    X, y = load_data(csv_path)

    # ---- 6.1 Stratified split BEFORE any fitting --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RNG
    )
    print(f"Train: {len(X_train):,}  fraud={y_train.sum():,}   "
          f"Test: {len(X_test):,}  fraud={y_test.sum():,}")

    amounts_test = X_test["Amount"].copy()

    # ---- 6.2 Build, tune, evaluate ---------------------------------------
    results: List[ModelResult] = []
    for name, pipe in build_pipelines().items():
        print(f"\n>>> {name}")
        best_pipe, best_params, cv_score = tune_model(name, pipe, X_train, y_train)
        print(f"    best params: {best_params}")
        print(f"    CV PR-AUC  : {cv_score:.4f}")

        # threshold tuned on a held-out slice of TRAIN to avoid touching test
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_train, y_train, test_size=0.20,
            stratify=y_train, random_state=RNG
        )
        best_pipe.fit(X_tr2, y_tr2)
        val_proba = best_pipe.predict_proba(X_val)[:, 1]
        thr = tune_threshold(y_val.values, val_proba, min_recall=0.85)

        # final fit on full train, then evaluate on untouched test
        best_pipe.fit(X_train, y_train)
        metrics, y_proba, y_pred = evaluate(name, best_pipe, X_test, y_test, thr)
        print(f"    threshold  : {thr:.4f}")
        print(f"    test       : "
              + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        results.append(ModelResult(
            name=name, best_params=best_params, cv_pr_auc=cv_score,
            threshold=thr, metrics=metrics,
            y_proba=y_proba, y_pred=y_pred, pipeline=best_pipe,
        ))

    # ---- 6.3 Comparison table --------------------------------------------
    table = pd.DataFrame([
        {"Model": r.name, "Threshold": round(r.threshold, 4),
         "CV PR-AUC": round(r.cv_pr_auc, 4), **{k: round(v, 4) for k, v in r.metrics.items()}}
        for r in results
    ]).sort_values("PR-AUC", ascending=False).reset_index(drop=True)

    print("\n=== Model Comparison (sorted by PR-AUC) ===")
    print(table.to_string(index=False))

    os.makedirs(outdir, exist_ok=True)
    table.to_csv(os.path.join(outdir, "model_comparison.csv"), index=False)

    # ---- 6.4 Plots + business summary ------------------------------------
    plot_curves(results, y_test.values, outdir)
    best = max(results, key=lambda r: r.metrics["PR-AUC"])
    summary = business_summary(best, y_test, amounts_test)
    print(summary)
    with open(os.path.join(outdir, "report_notes.txt"), "w") as fh:
        fh.write("Best model: " + best.name + "\n")
        fh.write(table.to_string(index=False))
        fh.write("\n" + summary)

        fh.write("""
Why recall (not accuracy) drives this problem
---------------------------------------------
Frauds are ~0.17% of transactions. A trivial 'predict legit' classifier
scores ~99.83% accuracy and catches zero fraud. Accuracy is therefore
uninformative under heavy imbalance.

The asymmetry of costs is what matters:
  * False Negative (missed fraud)  -> direct chargeback loss = £amount
  * False Positive (blocked legit) -> review cost + customer friction
A missed £200 fraud typically costs the issuer 40x more than wrongly
challenging a £200 legit purchase. We therefore optimise PR-AUC during
training and pick the operating threshold that holds recall >= 0.85,
maximising F1 inside that constraint. ROC-AUC is reported for context
but is misleadingly optimistic on imbalanced data because the FPR
denominator is dominated by the huge legit class.

Connecting to financial risk reduction
--------------------------------------
The 'Business impact' block above quantifies fraud £ recovered vs
false-positive friction cost. This is the figure to put in front of a
risk committee — it converts model performance into expected loss
avoided per period, which is the language the business uses.
""")
    print(f"\nArtifacts written to: {os.path.abspath(outdir)}")
    return table


if __name__ == "__main__":
    DATA = "creditcard.csv"
    if not os.path.exists(DATA):
        DATA = ("D:/Masterclass/credit-card-fraud-detection-main/"
                "credit-card-fraud-detection-main/creditcard.csv")
    main(DATA, outdir="outputs")
