"""
calibration_analysis.py
=======================
EXTENSION 2 — Probability calibration + cost-sensitive threshold tuning.

The original group report flagged: "Wrap the best estimator in
CalibratedClassifierCV so the output is a true probability."  This
script implements that and shows the before/after business impact.

Generates three charts under outputs/:
  1. calibration_curve.png         — reliability diagram, before vs after
  2. calibration_distribution.png  — predicted-prob histograms by class
  3. threshold_optimisation.png    — 4-panel cost-sensitive sweep
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# UTF-8 stdout for Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline import prepare_splits, outputs_dir, PALETTE, RNG  # noqa: E402
from setup_model import build_xgb_pipeline, THRESHOLD  # noqa: E402

warnings.filterwarnings("ignore")

FP_COST = 5.0          # £ friction per false positive
DPI = 150


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  saved -> {path.relative_to(path.parent.parent)}")


def _confusion_at(threshold: float, y_true: np.ndarray,
                  proba: np.ndarray) -> tuple:
    pred = (proba >= threshold).astype(int)
    return confusion_matrix(y_true, pred).ravel()  # tn, fp, fn, tp


def cost_sensitive_sweep(y_true: np.ndarray, proba: np.ndarray,
                         amounts: np.ndarray, fp_cost: float = FP_COST):
    thresholds = np.arange(0.01, 1.00, 0.01)
    rows = []
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        tp_amount = float(amounts[(y_true == 1) & (pred == 1)].sum())
        fp_count  = int(((y_true == 0) & (pred == 1)).sum())
        fp_friction = fp_count * fp_cost
        net = tp_amount - fp_friction
        rows.append({
            "threshold": thr,
            "tp_amount": tp_amount,
            "fp_count": fp_count,
            "fp_friction": fp_friction,
            "net": net,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 70)
    print("EXTENSION 2 — Probability calibration & cost-sensitive threshold")
    print("=" * 70)

    out = outputs_dir()
    splits = prepare_splits(verbose=True)
    X_train, y_train = splits.X_train, splits.y_train
    X_test,  y_test  = splits.X_test, splits.y_test
    amounts_test = splits.amounts_test.values

    # ---- Load uncalibrated pipeline ----------------------------------------
    pipe = joblib.load(out / "xgb_fraud_model.pkl")
    print("\nLoaded uncalibrated XGBoost pipeline.")

    # Uncalibrated predictions on test
    proba_uncal = pipe.predict_proba(X_test)[:, 1]

    # ---- Fit CalibratedClassifierCV ----------------------------------------
    # Use a freshly built (unfitted) pipeline so CalibratedClassifierCV
    # can do its own cv=5 refit. sklearn 1.6+ requires the estimator to
    # be unfitted when cv != "prefit".
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)
    unfitted_pipe = build_xgb_pipeline(scale_pos_weight=spw)

    print("\nFitting CalibratedClassifierCV(method='isotonic', cv=5) ...")
    calibrated = CalibratedClassifierCV(
        unfitted_pipe, method="isotonic", cv=5
    )
    calibrated.fit(X_train, y_train)
    proba_cal = calibrated.predict_proba(X_test)[:, 1]
    joblib.dump(calibrated, out / "xgb_fraud_model_calibrated.pkl")
    print("  ✓ Calibrated model fitted and saved.")

    # ---- Calibration statistics --------------------------------------------
    mask_fraud = (y_test.values == 1)
    print("\nMean predicted P(fraud) for the FRAUD class:")
    print(f"  Uncalibrated : {proba_uncal[mask_fraud].mean():.4f}")
    print(f"  Calibrated   : {proba_cal[mask_fraud].mean():.4f}")
    print("Mean predicted P(fraud) for the LEGIT class:")
    print(f"  Uncalibrated : {proba_uncal[~mask_fraud].mean():.4f}")
    print(f"  Calibrated   : {proba_cal[~mask_fraud].mean():.4f}")

    brier_uncal = brier_score_loss(y_test, proba_uncal)
    brier_cal   = brier_score_loss(y_test, proba_cal)
    prauc_uncal = average_precision_score(y_test, proba_uncal)
    prauc_cal   = average_precision_score(y_test, proba_cal)
    print(f"\nBrier score  uncalibrated={brier_uncal:.6f}  "
          f"calibrated={brier_cal:.6f}  "
          f"(lower is better, improvement={brier_uncal-brier_cal:+.6f})")
    print(f"PR-AUC       uncalibrated={prauc_uncal:.4f}    "
          f"calibrated={prauc_cal:.4f}")

    # =====================================================================
    # CHART 6 — Reliability diagram (calibration curve)
    # =====================================================================
    print("\n[1/3] Reliability diagram ...")
    n_bins = 10
    frac_pos_uncal, mean_pred_uncal = calibration_curve(
        y_test, proba_uncal, n_bins=n_bins, strategy="quantile"
    )
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_test, proba_cal, n_bins=n_bins, strategy="quantile"
    )

    # 95% Wilson interval for fraction of positives in each bin
    def wilson(p, n, z=1.96):
        if n == 0:
            return 0.0, 0.0
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        return max(0, centre - half), min(1, centre + half)

    # Approximate bin sizes
    bin_size = max(1, len(y_test) // n_bins)
    bands_u = np.array([wilson(p, bin_size) for p in frac_pos_uncal])
    bands_c = np.array([wilson(p, bin_size) for p in frac_pos_cal])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1,
            ls="--", label="Perfect calibration")
    ax.fill_between(mean_pred_uncal, bands_u[:, 0], bands_u[:, 1],
                    color=PALETTE["fraud"], alpha=0.12)
    ax.plot(mean_pred_uncal, frac_pos_uncal,
            color=PALETTE["fraud"], lw=2, ls="--", marker="o",
            label=f"Uncalibrated  (Brier={brier_uncal:.5f})")
    ax.fill_between(mean_pred_cal, bands_c[:, 0], bands_c[:, 1],
                    color=PALETTE["accent"], alpha=0.18)
    ax.plot(mean_pred_cal, frac_pos_cal,
            color=PALETTE["accent"], lw=2.5, marker="s",
            label=f"Calibrated (isotonic)  (Brier={brier_cal:.5f})")
    ax.set_xlabel("Mean predicted P(fraud)  [quantile bins]")
    ax.set_ylabel("Fraction of positives in bin")
    ax.set_title("Probability calibration — before vs after isotonic regression")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    _save(out / "calibration_curve.png")

    print("\nSO WHAT: Before calibration, a predicted score of 0.7 did not "
          "mean 70% probability of fraud. After isotonic regression it does, "
          "enabling cost-sensitive threshold setting based on actual £ "
          "values per transaction rather than arbitrary F1 optimisation.")

    # =====================================================================
    # CHART 7 — Probability distribution shift (log y)
    # =====================================================================
    print("\n[2/3] Probability distribution shift ...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, proba, title in [
        (axes[0], proba_uncal, "Before calibration"),
        (axes[1], proba_cal,   "After calibration"),
    ]:
        ax.hist(proba[~mask_fraud], bins=50, range=(0, 1),
                color=PALETTE["legit"], alpha=0.7, label="Legit",
                edgecolor="white")
        ax.hist(proba[mask_fraud], bins=50, range=(0, 1),
                color=PALETTE["fraud"], alpha=0.85, label="Fraud",
                edgecolor="white")
        ax.set_yscale("log")
        ax.set_xlabel("Predicted P(fraud)")
        ax.set_title(title)
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Count (log scale)")
    fig.suptitle("Predicted probability distributions before/after calibration",
                 fontsize=13, y=1.02)
    _save(out / "calibration_distribution.png")

    # Separation metric — mean probability gap between classes
    gap_uncal = proba_uncal[mask_fraud].mean() - proba_uncal[~mask_fraud].mean()
    gap_cal   = proba_cal[mask_fraud].mean()   - proba_cal[~mask_fraud].mean()
    print(f"\nSO WHAT: mean P(fraud) gap between classes "
          f"(fraud − legit) = {gap_uncal:.3f} uncalibrated, "
          f"{gap_cal:.3f} calibrated. A larger gap means cleaner separation "
          f"in probability space.")

    # =====================================================================
    # CHART 8 — Cost-sensitive threshold optimisation (4 panels)
    # =====================================================================
    print("\n[3/3] Cost-sensitive threshold sweep on calibrated model ...")
    sweep = cost_sensitive_sweep(y_test.values, proba_cal,
                                 amounts=amounts_test, fp_cost=FP_COST)

    thr_cost_opt = float(sweep.loc[sweep["net"].idxmax(), "threshold"])
    net_opt      = float(sweep["net"].max())
    thr_f1_opt   = float(sweep.loc[sweep["f1"].idxmax(),  "threshold"])
    f1_opt       = float(sweep["f1"].max())

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # Panel 1 — Net benefit vs threshold
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sweep["threshold"], sweep["net"],
             color=PALETTE["accent"], lw=2.2)
    ax1.fill_between(sweep["threshold"], 0, sweep["net"],
                     where=(sweep["net"] > 0),
                     color=PALETTE["accent"], alpha=0.15)
    for thr, label, col in [
        (THRESHOLD,  f"Original  {THRESHOLD:.4f}", PALETTE["neutral"]),
        (thr_f1_opt, f"F1-optimal {thr_f1_opt:.2f}", PALETTE["legit"]),
        (thr_cost_opt, f"£-optimal {thr_cost_opt:.2f}", PALETTE["fraud"]),
    ]:
        ax1.axvline(thr, color=col, lw=1.4, ls="--", alpha=0.85, label=label)
    ax1.scatter([thr_cost_opt], [net_opt], color=PALETTE["fraud"],
                s=80, zorder=5, edgecolor="white")
    ax1.annotate(f"£{net_opt:,.0f}", xy=(thr_cost_opt, net_opt),
                 xytext=(8, 8), textcoords="offset points",
                 color=PALETTE["fraud"], fontweight="bold")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Net benefit (£ recovered − £ FP friction)")
    ax1.set_title("Panel 1 — Net £ benefit vs threshold")
    ax1.legend(loc="lower center", frameon=False, fontsize=8, ncol=3)
    ax1.grid(alpha=0.25)

    # Panel 2 — £ recovered vs £ friction (dual axis)
    ax2a = fig.add_subplot(gs[0, 1])
    ax2a.plot(sweep["threshold"], sweep["tp_amount"],
              color=PALETTE["accent"], lw=2, label="£ fraud recovered")
    ax2a.set_xlabel("Threshold")
    ax2a.set_ylabel("£ fraud recovered (TP £)",
                    color=PALETTE["accent"])
    ax2a.tick_params(axis="y", labelcolor=PALETTE["accent"])
    ax2b = ax2a.twinx()
    ax2b.plot(sweep["threshold"], sweep["fp_friction"],
              color=PALETTE["fraud"], lw=2, ls="--",
              label="£ FP friction (cnt × £5)")
    ax2b.set_ylabel("£ FP friction", color=PALETTE["fraud"])
    ax2b.tick_params(axis="y", labelcolor=PALETTE["fraud"])
    ax2a.set_title("Panel 2 — £ recovered vs £ friction")
    ax2a.grid(alpha=0.25)
    # Combine legends
    lines, labels = ax2a.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines + lines2, labels + labels2, loc="center right",
                frameon=False, fontsize=9)

    # Panel 3 — Precision / Recall / F1
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(sweep["threshold"], sweep["precision"],
             color=PALETTE["legit"], lw=2, label="Precision")
    ax3.plot(sweep["threshold"], sweep["recall"],
             color=PALETTE["fraud"], lw=2, label="Recall")
    ax3.plot(sweep["threshold"], sweep["f1"],
             color=PALETTE["accent"], lw=2.5, label="F1")
    ax3.axvline(thr_f1_opt, color=PALETTE["accent"], lw=1.4, ls="--",
                alpha=0.7, label=f"F1-optimal {thr_f1_opt:.2f}")
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Score")
    ax3.set_title("Panel 3 — Precision, Recall, F1 vs threshold")
    ax3.legend(loc="lower center", frameon=False, fontsize=8, ncol=2)
    ax3.grid(alpha=0.25)

    # Panel 4 — Confusion matrix at cost-optimal threshold
    ax4 = fig.add_subplot(gs[1, 1])
    tn, fp, fn, tp = _confusion_at(thr_cost_opt, y_test.values, proba_cal)
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax4.imshow(cm, cmap="Blues")
    ax4.set_xticks([0, 1]); ax4.set_xticklabels(["Pred Legit", "Pred Fraud"])
    ax4.set_yticks([0, 1]); ax4.set_yticklabels(["True Legit", "True Fraud"])
    for i in range(2):
        for j in range(2):
            colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax4.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color=colour, fontsize=13, fontweight="bold")
    ax4.set_title(f"Panel 4 — Confusion matrix @ £-optimal "
                  f"threshold = {thr_cost_opt:.2f}")

    fig.suptitle("Cost-sensitive threshold optimisation (calibrated model)",
                 fontsize=14, y=1.00)
    _save(out / "threshold_optimisation.png")

    # ---- Compare key thresholds --------------------------------------------
    orig_row = sweep.iloc[(sweep["threshold"] - THRESHOLD).abs().idxmin()]
    f1_row   = sweep.loc[sweep["f1"].idxmax()]
    cost_row = sweep.loc[sweep["net"].idxmax()]

    delta_pts = abs(f1_row["threshold"] - cost_row["threshold"]) * 100
    extra_recovered = cost_row["tp_amount"] - f1_row["tp_amount"]
    extra_friction  = cost_row["fp_friction"] - f1_row["fp_friction"]
    extra_net       = cost_row["net"] - f1_row["net"]

    print(f"\nSO WHAT: F1-optimal and £-optimal thresholds differ by "
          f"{delta_pts:.0f} points. At the £-optimal threshold "
          f"({cost_row['threshold']:.2f}) the model recovers "
          f"£{extra_recovered:+,.0f} more fraud than at F1-optimal "
          f"({f1_row['threshold']:.2f}), with £{extra_friction:+,.0f} "
          f"extra friction — a net benefit of £{extra_net:+,.0f}. The "
          f"operations team should review this lever quarterly.")

    # ---- Summary report ----------------------------------------------------
    def row_summary(label, row, proba):
        tn_, fp_, fn_, tp_ = _confusion_at(row["threshold"], y_test.values, proba)
        return (f"  {label:<14} thr={row['threshold']:.4f}  "
                f"P={row['precision']:.3f}  R={row['recall']:.3f}  "
                f"F1={row['f1']:.3f}  TP=£{row['tp_amount']:,.0f}  "
                f"FP={fp_}  net=£{row['net']:,.0f}")

    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY REPORT")
    print("=" * 70)
    print("Uncalibrated XGBoost:")
    print(f"  PR-AUC = {prauc_uncal:.4f}   Brier = {brier_uncal:.6f}")
    print("Calibrated XGBoost (isotonic, cv=5):")
    print(f"  PR-AUC = {prauc_cal:.4f}   Brier = {brier_cal:.6f}")
    print("\nThreshold comparison (all numbers on calibrated probabilities):")
    print(row_summary("original   ", orig_row, proba_cal))
    print(row_summary("F1-optimal ", f1_row,   proba_cal))
    print(row_summary("£-optimal  ", cost_row, proba_cal))

    if brier_cal < brier_uncal:
        rec = ("DEPLOY the CALIBRATED model. It produces probability scores "
               "that can be combined with £ amounts to make cost-aware "
               "decisions, and Brier score improved.")
    else:
        rec = ("DEPLOY the UNCALIBRATED model. Calibration did not improve "
               "Brier on this test split — likely the raw scores already "
               "rank fraud well enough.")
    print(f"\nRecommendation: {rec}")
    print("\nEXTENSION 2 COMPLETE ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
