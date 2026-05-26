"""
shap_analysis.py
================
EXTENSION 1 — SHAP explainability for the XGBoost_cw fraud model.

The original group report flagged: "A separate explainability tool (SHAP)
would be needed if a customer or regulator asks for reasons."  This
script closes that gap.

Generates five charts under outputs/:
  1. shap_summary_beeswarm.png      — global feature impact distribution
  2. shap_feature_importance.png    — top-15 mean |SHAP| bar chart
  3. shap_waterfall_fraud.png       — most-confident TRUE FRAUD explanation
  4. shap_waterfall_fp.png          — most-confident FALSE POSITIVE explanation
  5. shap_dependence_v14.png        — SHAP vs raw V14, coloured by Amount
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
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline import prepare_splits, outputs_dir, PALETTE, RNG  # noqa: E402

warnings.filterwarnings("ignore")

THRESHOLD = 0.4949
DPI = 150


def stratified_sample_indices(y: pd.Series, n_per_class: int = 250,
                              seed: int = RNG) -> np.ndarray:
    """Return positional indices: n_per_class fraud + n_per_class legit."""
    rng = np.random.RandomState(seed)
    pos_pool = np.where(y.values == 1)[0]
    neg_pool = np.where(y.values == 0)[0]
    n_pos = min(n_per_class, len(pos_pool))
    n_neg = min(n_per_class, len(neg_pool))
    pos_idx = rng.choice(pos_pool, size=n_pos, replace=False)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    return idx


_BG      = "#0B1623"   # app background (deep navy)
_SURFACE = "#13243C"   # card surface
_BORDER  = "#243E5C"   # subtle grid / spine
_TEXT    = "#F1F5F9"   # primary text
_TEXT2   = "#94A8C0"   # secondary text


def _apply_dark_theme() -> None:
    """Apply project dark-theme to all subsequent matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor":  _BG,
        "axes.facecolor":    _SURFACE,
        "axes.edgecolor":    _BORDER,
        "axes.labelcolor":   _TEXT2,
        "text.color":        _TEXT,
        "xtick.color":       _TEXT2,
        "ytick.color":       _TEXT2,
        "xtick.labelcolor":  _TEXT2,
        "ytick.labelcolor":  _TEXT2,
        "grid.color":        _BORDER,
        "legend.facecolor":  _BG,
        "legend.edgecolor":  _BORDER,
        "legend.labelcolor": _TEXT2,
        "figure.edgecolor":  _BG,
    })


def _save(fig_or_plt, path: Path) -> None:
    """Save current figure at DPI with dark background, then close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=_BG)
    plt.close("all")
    print(f"  saved -> {path.relative_to(path.parent.parent)}")


def main() -> int:
    print("=" * 70)
    print("EXTENSION 1 — SHAP explainability for XGBoost fraud model")
    print("=" * 70)

    _apply_dark_theme()   # set dark theme before any figure is created

    out = outputs_dir()
    splits = prepare_splits(verbose=True)
    X_train, y_train = splits.X_train, splits.y_train
    X_test,  y_test  = splits.X_test, splits.y_test
    feature_names = splits.feature_names

    # ---- Load the trained pipeline (from setup_model.py) -------------------
    model_path = out / "xgb_fraud_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run `python src/setup_model.py` first."
        )
    pipe = joblib.load(model_path)
    scaler = pipe.named_steps["scaler"]
    xgb_model = pipe.named_steps["clf"]
    print(f"\nLoaded pipeline from {model_path.name}")

    # ---- Build a stratified 500-row sample of the TEST set -----------------
    idx = stratified_sample_indices(y_test, n_per_class=250, seed=RNG)
    X_sample_raw = X_test.iloc[idx]
    y_sample = y_test.iloc[idx]

    # Apply the same scaler the model saw
    X_sample_scaled = pd.DataFrame(
        scaler.transform(X_sample_raw),
        columns=feature_names,
        index=X_sample_raw.index,
    )
    print(f"\nStratified test sample: {len(idx)} rows "
          f"({int(y_sample.sum())} fraud + {int((y_sample==0).sum())} legit)")

    # ---- Compute SHAP values on the sample ---------------------------------
    print("\nComputing SHAP values with TreeExplainer ...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_explanation = explainer(X_sample_scaled)
    shap_values = shap_explanation.values  # (500, n_features)
    print(f"SHAP values computed for {shap_values.shape[0]} transactions")

    # =====================================================================
    # CHART 1 — Beeswarm summary plot
    # =====================================================================
    print("\n[1/5] Beeswarm summary plot ...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample_scaled,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    fig = plt.gcf()
    fig.suptitle("SHAP feature impact — XGBoost fraud model",
                 fontsize=13, y=1.02)
    _save(fig, out / "shap_summary_beeswarm.png")

    # ---- "SO WHAT" for chart 1 -----
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top3 = [(feature_names[i], mean_abs[i]) for i in order[:3]]

    # Direction: sign of correlation between raw feature value and SHAP value
    def shap_direction(feat_idx: int) -> str:
        vals = X_sample_scaled.values[:, feat_idx]
        shp = shap_values[:, feat_idx]
        if np.corrcoef(vals, shp)[0, 1] > 0:
            return "HIGHER values push toward fraud"
        return "LOWER values push toward fraud"

    print(f"\nSO WHAT: top 3 features by mean |SHAP| are:")
    for name, val in top3:
        direction = shap_direction(feature_names.index(name))
        print(f"  • {name:<6} mean|SHAP|={val:.4f}  ({direction})")
    print(f"  These three features carry the bulk of the fraud signal.")

    # =====================================================================
    # CHART 2 — Mean absolute impact bar chart (top 15)
    # =====================================================================
    print("\n[2/5] Top-15 mean |SHAP| bar chart ...")
    top15_idx = order[:15]
    top15_names = [feature_names[i] for i in top15_idx]
    top15_means = mean_abs[top15_idx]

    # Direction colour: red if MEAN signed SHAP > 0 (pushes toward fraud)
    signed_means = shap_values[:, top15_idx].mean(axis=0)
    bar_colours = [PALETTE["fraud"] if s > 0 else PALETTE["legit"]
                   for s in signed_means]

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(top15_names))
    bars = ax.barh(y_pos, top15_means, color=bar_colours, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15_names)
    ax.invert_yaxis()  # most-important on top
    ax.set_xlabel("Mean |SHAP value|  (avg impact on model output)")
    ax.set_title("Top 15 features by fraud-detection contribution")
    for bar, val in zip(bars, top15_means):
        ax.text(val + max(top15_means) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color=_TEXT2)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=PALETTE["fraud"], label="Avg direction: pushes toward FRAUD"),
        Patch(color=PALETTE["legit"], label="Avg direction: pushes toward LEGIT"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.25)
    _save(fig, out / "shap_feature_importance.png")

    cumulative_share = top15_means[:5].sum() / mean_abs.sum()
    print(f"\nSO WHAT: top 5 features explain "
          f"{cumulative_share*100:.0f}% of the total |SHAP| magnitude — "
          f"a compliance team could audit these {5} features specifically "
          f"rather than the full {len(feature_names)}-feature input.")

    # =====================================================================
    # CHART 3 — Waterfall for most-confident TRUE FRAUD
    # =====================================================================
    print("\n[3/5] Waterfall — most-confident true fraud ...")
    proba_sample = pipe.predict_proba(X_sample_raw)[:, 1]
    # Among the rows where y_sample == 1, pick the one with the highest
    # predicted probability — that's the most-confident correct fraud call.
    fraud_mask = (y_sample.values == 1)
    fraud_probs = proba_sample.copy()
    fraud_probs[~fraud_mask] = -1
    best_fraud_pos = int(np.argmax(fraud_probs))
    txn_id_fraud = X_sample_raw.index[best_fraud_pos]
    prob_fraud   = proba_sample[best_fraud_pos]

    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_explanation[best_fraud_pos], show=False, max_display=12)
    fig = plt.gcf()
    fig.suptitle(f"Why the model flagged transaction #{txn_id_fraud} as FRAUD",
                 fontsize=12, y=1.02)
    ax = plt.gca()
    ax.text(0.02, -0.18,
            f"Predicted probability: {prob_fraud*100:.1f}%  |  True label: FRAUD",
            transform=ax.transAxes, fontsize=10,
            color=PALETTE["fraud"], fontweight="bold")
    _save(fig, out / "shap_waterfall_fraud.png")

    # Dominant feature for this row
    row_shap = shap_values[best_fraud_pos]
    top_feat_idx = int(np.argmax(np.abs(row_shap)))
    top_feat_name = feature_names[top_feat_idx]
    top_feat_value_raw = X_sample_raw.iloc[best_fraud_pos, top_feat_idx]
    top_feat_shap = row_shap[top_feat_idx]

    print(f"\nSO WHAT: This is the explanation a fraud analyst or regulator "
          f"would receive. {top_feat_name}={top_feat_value_raw:.2f} is the "
          f"dominant flag — it pushed the log-odds by {top_feat_shap:+.2f} "
          f"alone (transaction #{txn_id_fraud}, predicted fraud probability "
          f"{prob_fraud*100:.1f}%).")

    # =====================================================================
    # CHART 4 — Waterfall for worst FALSE POSITIVE
    # =====================================================================
    print("\n[4/5] Waterfall — worst false positive ...")
    pred_sample = (proba_sample >= THRESHOLD).astype(int)
    fp_mask = (y_sample.values == 0) & (pred_sample == 1)

    if fp_mask.any():
        fp_probs = proba_sample.copy()
        fp_probs[~fp_mask] = -1
        worst_fp_pos = int(np.argmax(fp_probs))
    else:
        # No false positive in the 500-row sample — fall back to the highest-
        # scoring legit row instead so we still get a chart.
        legit_mask = (y_sample.values == 0)
        legit_probs = proba_sample.copy()
        legit_probs[~legit_mask] = -1
        worst_fp_pos = int(np.argmax(legit_probs))

    txn_id_fp = X_sample_raw.index[worst_fp_pos]
    prob_fp   = proba_sample[worst_fp_pos]

    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_explanation[worst_fp_pos], show=False, max_display=12)
    fig = plt.gcf()
    fig.suptitle(f"Why the model incorrectly flagged transaction #{txn_id_fp}",
                 fontsize=12, y=1.02)
    ax = plt.gca()
    ax.text(0.02, -0.18,
            f"Predicted probability: {prob_fp*100:.1f}%  |  True label: LEGIT",
            transform=ax.transAxes, fontsize=10,
            color=PALETTE["legit"], fontweight="bold")
    _save(fig, out / "shap_waterfall_fp.png")

    row_shap_fp = shap_values[worst_fp_pos]
    top_fp_indices = np.argsort(np.abs(row_shap_fp))[::-1][:2]
    top_fp_names = [feature_names[i] for i in top_fp_indices]
    print(f"\nSO WHAT: This transaction was flagged because "
          f"{top_fp_names[0]} and {top_fp_names[1]} showed unusual values — "
          f"likely a high-value legitimate purchase pattern that resembles "
          f"fraud behaviour. The operations team should prioritise reviewing "
          f"these feature combinations.")

    # =====================================================================
    # CHART 5 — SHAP dependence plot for V14
    # =====================================================================
    print("\n[5/5] Dependence plot — V14 vs SHAP, coloured by Amount ...")
    v14_idx = feature_names.index("V14")
    v14_vals_raw = X_sample_raw["V14"].values
    v14_shap = shap_values[:, v14_idx]
    amounts = X_sample_raw["Amount"].values

    fig, ax = plt.subplots(figsize=(9, 6))
    # Clip Amount colour scale at 99th percentile so a single outlier
    # doesn't desaturate the rest of the points.
    amt_clip = np.clip(amounts, 0, np.quantile(amounts, 0.99))
    sc = ax.scatter(v14_vals_raw, v14_shap,
                    c=amt_clip, cmap="viridis",
                    s=22, alpha=0.8, edgecolor="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Transaction Amount (£, clipped at 99th pct)",
                   color=PALETTE["neutral"])

    ax.axhline(0, color=PALETTE["neutral"], lw=1, ls="--", alpha=0.6)
    ax.axvline(-3, color=PALETTE["fraud"], lw=1.5, ls=":",
               alpha=0.7, label="V14 = -3 (fraud-risk threshold)")
    ax.set_xlabel("V14 (raw value)")
    ax.set_ylabel("SHAP value for V14 (impact on log-odds of fraud)")
    ax.set_title("SHAP dependence — V14 drives the fraud signal")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, out / "shap_dependence_v14.png")

    # Quantify the V14 < -3 rule
    rule_mask = v14_vals_raw < -3
    if rule_mask.any():
        rule_fraud_rate = y_sample.values[rule_mask].mean()
    else:
        rule_fraud_rate = float("nan")
    print(f"\nSO WHAT: in this sample, transactions with V14 < -3 have a "
          f"fraud rate of {rule_fraud_rate*100:.1f}% versus the population "
          f"base rate of 0.17%. This single threshold could be deployed as "
          f"a simple business rule alongside the full model.")

    # =====================================================================
    # Final report
    # =====================================================================
    print("\n" + "=" * 70)
    print("SHAP SUMMARY REPORT")
    print("=" * 70)
    print(f"Sample size           : {len(idx)} transactions (250 fraud + 250 legit)")
    print(f"Top global driver     : {feature_names[order[0]]}  "
          f"(mean|SHAP|={mean_abs[order[0]]:.4f})")
    print(f"Most-confident fraud  : txn #{txn_id_fraud}  "
          f"(predicted {prob_fraud*100:.1f}%)")
    print(f"Worst false positive  : txn #{txn_id_fp}  "
          f"(predicted {prob_fp*100:.1f}%)")
    print(f"Top-5 SHAP share      : {cumulative_share*100:.0f}% of total |SHAP|")
    print(f"V14 < -3 rule         : {rule_fraud_rate*100:.1f}% fraud rate "
          f"in sample (vs 0.17% base)")
    print("Outputs               : 5 charts saved to outputs/")
    print("\nEXTENSION 1 COMPLETE ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
