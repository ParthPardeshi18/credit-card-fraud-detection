"""
fraud_app.py
============
EXTENSION 3 — Streamlit fraud risk scoring app.

Run with:   streamlit run src/fraud_app.py

Three tabs:
  1. Transaction risk scorer  — live SHAP explanation per transaction
  2. Model performance        — KPI cards, ROC/PR, confusion matrix,
                                model comparison table, £ impact
  3. Global SHAP explainability — pre-generated charts from Ext. 1
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    average_precision_score, roc_auc_score,
)

# Make `from data_pipeline import ...` work no matter how Streamlit invokes us
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_pipeline import prepare_splits, project_root, PALETTE  # noqa: E402

# --------------------------------------------------------------------------
# Page config + theme
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Risk Scorer",
    layout="wide",
    page_icon="🛡️",
)

# Custom dark theme via CSS
DARK_BG = "#0D1B2A"
CARD_BG = "#0F2338"
ACCENT  = "#00B4D8"
TEXT    = "#F2F4F7"

st.markdown(f"""
<style>
    .stApp {{ background-color: {DARK_BG}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{ background-color: {CARD_BG}; }}
    .kpi-card {{
        background: {CARD_BG};
        padding: 18px 16px;
        border-radius: 10px;
        border: 1px solid #1B3553;
        text-align: center;
    }}
    .kpi-label {{ font-size: 0.85rem; color: #94A8C0; margin-bottom: 4px; }}
    .kpi-value {{ font-size: 1.7rem; font-weight: 700; color: {ACCENT}; }}
    .risk-badge {{
        display: inline-block;
        padding: 10px 24px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 8px 0;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background: {CARD_BG}; padding: 8px 16px; border-radius: 6px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {ACCENT} !important; color: {DARK_BG} !important;
    }}
    h1, h2, h3 {{ color: {TEXT}; }}
    .business-box {{
        background: {CARD_BG}; padding: 18px 22px; border-radius: 10px;
        border-left: 4px solid {PALETTE['accent']};
    }}
</style>
""", unsafe_allow_html=True)

# Matplotlib dark theme
plt.rcParams.update({
    "figure.facecolor":  CARD_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    "#3A526E",
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "savefig.facecolor": CARD_BG,
})


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_metadata():
    meta_path = project_root() / "outputs" / "model_metadata.pkl"
    return joblib.load(meta_path) if meta_path.exists() else None


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(project_root() / "outputs" / "xgb_fraud_model.pkl")


@st.cache_resource(show_spinner=False)
def load_explainer(_pipe):
    xgb = _pipe.named_steps["clf"]
    return shap.TreeExplainer(xgb)


@st.cache_data(show_spinner=False)
def load_splits():
    return prepare_splits(verbose=False)


@st.cache_data(show_spinner=False)
def load_model_comparison():
    path = project_root() / "outputs" / "model_comparison.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<h2 style='color:{ACCENT}'>🛡️ Fraud Risk Scorer</h2>",
                unsafe_allow_html=True)
    st.markdown("**QMUL BUSM131 Masterclass**")
    st.caption("Model: XGBoost + Class Weighting")
    st.caption("Dataset: 284,807 transactions | 492 fraud cases")
    st.caption("Collaboration: QMUL BUSM131 Masterclass")
    st.markdown("---")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10, max_value=0.90, value=0.4949, step=0.01,
        help="Updates the risk-tier boundaries in real time.",
    )
    st.caption(f"Current threshold: **{threshold:.4f}**")
    st.markdown("---")
    st.caption("Extensions: SHAP explainability, probability calibration, "
               "live risk scoring.")


# --------------------------------------------------------------------------
# Load core artifacts (with friendly error if setup wasn't run)
# --------------------------------------------------------------------------
try:
    pipe = load_model()
    metadata = load_metadata() or {}
    feature_names = list(pipe.feature_names_in_) if hasattr(
        pipe, "feature_names_in_") else metadata.get(
        "feature_names", ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
except FileNotFoundError:
    st.error("Model not found. Run `python src/setup_model.py` first.")
    st.stop()


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Transaction Risk Scorer",
    "📊 Model Performance",
    "🔍 Global SHAP Explainability",
])


# ==========================================================================
# TAB 1 — Live transaction scorer
# ==========================================================================
with tab1:
    st.markdown("### Live fraud risk scorer — XGBoost model")
    st.markdown(
        "<p style='color:#94A8C0'>Enter transaction details to get a real-time "
        "fraud probability and explanation.</p>",
        unsafe_allow_html=True,
    )

    # Build input form
    with st.form("scorer_form", border=False):
        col_l, col_r = st.columns(2)
        with col_l:
            amount = st.number_input(
                "Amount (£)", min_value=0.01, max_value=25000.0,
                value=89.00, step=1.0)
            time_hr = st.slider(
                "Time (hours since stream start)",
                min_value=0, max_value=48, value=12)
            v14 = st.slider("V14 (top fraud signal)",
                            -10.0, 5.0, 0.0, 0.1)
            v17 = st.slider("V17", -10.0, 5.0, 0.0, 0.1)
            v12 = st.slider("V12", -10.0, 5.0, 0.0, 0.1)
        with col_r:
            v10 = st.slider("V10", -10.0, 5.0, 0.0, 0.1)
            v11 = st.slider("V11", -10.0, 5.0, 0.0, 0.1)
            v4  = st.slider("V4",   -5.0, 10.0, 0.0, 0.1)
            st.info("Remaining PCA features (V1–V28 not shown) "
                    "set to population mean (0.0).")

        submitted = st.form_submit_button(
            "🔍  Score Transaction", use_container_width=True, type="primary")

    if submitted:
        # Build the feature vector
        row = {f: 0.0 for f in feature_names}
        row["Time"]   = time_hr * 3600.0
        row["Amount"] = amount
        row["V4"]   = v4
        row["V10"]  = v10
        row["V11"]  = v11
        row["V12"]  = v12
        row["V14"]  = v14
        row["V17"]  = v17
        x_input = pd.DataFrame([row])[feature_names]

        proba = float(pipe.predict_proba(x_input)[0, 1])

        # Risk tier (uses sidebar threshold for the decision text, but the
        # tier badge uses the fixed 0.3 / 0.6 boundaries from the brief).
        if proba >= 0.6:
            tier, tier_col, tier_text = "AUTO-BLOCK", "#E24B4A", "white"
        elif proba >= 0.3:
            tier, tier_col, tier_text = "MANUAL REVIEW", "#F4A261", "black"
        else:
            tier, tier_col, tier_text = "AUTO-APPROVE", "#1D9E75", "white"

        decision = "BLOCKED" if proba >= threshold else "APPROVED"

        # Results panel
        st.markdown("---")
        rcol1, rcol2 = st.columns([1, 1])
        with rcol1:
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-label'>Fraud probability</div>"
                f"<div class='kpi-value' style='font-size:3rem'>"
                f"{proba*100:.1f}%</div></div>",
                unsafe_allow_html=True,
            )
        with rcol2:
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-label'>Risk tier</div>"
                f"<div class='risk-badge' "
                f"style='background:{tier_col};color:{tier_text};"
                f"font-size:1.4rem;'>{tier}</div></div>",
                unsafe_allow_html=True,
            )

        # Probability bar
        bar_fig, bar_ax = plt.subplots(figsize=(10, 1.1))
        bar_ax.barh([0], [1.0], color="#1B3553", height=0.6)
        bar_ax.barh([0], [proba], color=tier_col, height=0.6)
        bar_ax.axvline(threshold, color=ACCENT, ls="--", lw=1.5)
        bar_ax.text(threshold, 0.55, f" thr={threshold:.3f}",
                    color=ACCENT, fontsize=9)
        bar_ax.set_xlim(0, 1); bar_ax.set_ylim(-0.5, 0.7)
        bar_ax.set_yticks([])
        bar_ax.set_xticks(np.linspace(0, 1, 6))
        bar_ax.set_xlabel("P(fraud)")
        for spine in ("top", "right", "left"):
            bar_ax.spines[spine].set_visible(False)
        st.pyplot(bar_fig, transparent=True)
        plt.close(bar_fig)

        st.info(f"At threshold **{threshold:.4f}**, this transaction "
                f"would be: **{decision}** "
                f"({'fraud probability above threshold' if proba >= threshold else 'below threshold'}).")

        # ---- SHAP explanation for this single transaction ------------------
        st.markdown("### Why the model scored this transaction "
                    f"{proba*100:.1f}%")

        explainer = load_explainer(pipe)
        scaler = pipe.named_steps["scaler"]
        x_scaled = pd.DataFrame(scaler.transform(x_input),
                                columns=feature_names)
        sv = explainer(x_scaled)
        shap_vals = sv.values[0]                # (n_features,)

        # Top-5 features by absolute SHAP magnitude
        order = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_names = [feature_names[i] for i in order]
        top_vals  = shap_vals[order]
        top_raw   = x_input.values[0, order]
        colours   = ["#E24B4A" if v > 0 else "#1D9E75" for v in top_vals]

        ex_col1, ex_col2 = st.columns([1.1, 1])
        with ex_col1:
            efig, eax = plt.subplots(figsize=(7, 4))
            y_pos = np.arange(len(top_names))[::-1]
            eax.barh(y_pos, top_vals, color=colours, edgecolor="white")
            eax.set_yticks(y_pos)
            eax.set_yticklabels(top_names)
            eax.axvline(0, color="#3A526E", lw=1)
            eax.set_xlabel("SHAP value (impact on log-odds of fraud)")
            eax.set_title("Top 5 feature contributions")
            for spine in ("top", "right"):
                eax.spines[spine].set_visible(False)
            st.pyplot(efig, transparent=True)
            plt.close(efig)
            st.caption("🟢 Green = pushing toward LEGIT   "
                       "🔴 Red = pushing toward FRAUD")

        with ex_col2:
            df_ex = pd.DataFrame({
                "Feature": top_names,
                "Value":   [f"{v:.3f}" for v in top_raw],
                "SHAP impact": [f"{v:+.3f}" for v in top_vals],
                "Direction": ["toward FRAUD" if v > 0 else "toward LEGIT"
                              for v in top_vals],
            })
            st.dataframe(df_ex, hide_index=True, use_container_width=True)


# ==========================================================================
# TAB 2 — Model performance
# ==========================================================================
with tab2:
    st.markdown("### Model validation results")

    meta = metadata
    pr_auc    = meta.get("pr_auc",    0.7868)
    f1_score_ = meta.get("f1",        0.7900)
    recall    = meta.get("recall",    0.8061)
    tp        = int(meta.get("tp", 79))
    fp_       = int(meta.get("fp", 23))
    fn        = int(meta.get("fn", 19))
    tn        = int(meta.get("tn", 56841))
    fraud_total  = meta.get("fraud_total",  10644.93)
    fraud_caught = meta.get("fraud_caught", 8598.05)
    fp_friction  = meta.get("fp_friction",  115.0)

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    fpr = fp_ / max(fp_ + tn, 1)
    recovered_pct = fraud_caught / max(fraud_total, 1) * 100
    for col, label, val in [
        (k1, "PR-AUC",                f"{pr_auc:.4f}"),
        (k2, "F1 Score",              f"{f1_score_:.4f}"),
        (k3, "Fraud Recovered",       f"{recovered_pct:.1f}%"),
        (k4, "False Positive Rate",   f"{fpr*100:.3f}%"),
    ]:
        with col:
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-label'>{label}</div>"
                f"<div class='kpi-value'>{val}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("&nbsp;")

    # ROC + PR + confusion matrix — rebuild from test set
    with st.spinner("Computing curves on held-out test set ..."):
        splits = load_splits()
        proba_test = pipe.predict_proba(splits.X_test)[:, 1]
        fpr_arr, tpr_arr, _ = roc_curve(splits.y_test, proba_test)
        prec_arr, rec_arr, _ = precision_recall_curve(splits.y_test, proba_test)
        roc_auc = roc_auc_score(splits.y_test, proba_test)
        ap = average_precision_score(splits.y_test, proba_test)

    ccol1, ccol2 = st.columns(2)
    with ccol1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.plot(fpr_arr, tpr_arr, color=PALETTE["accent"], lw=2,
                 label=f"ROC (AUC={roc_auc:.3f})")
        ax1.plot(rec_arr, prec_arr, color=PALETTE["fraud"], lw=2,
                 label=f"PR  (AP={ap:.3f})", ls="--")
        ax1.plot([0, 1], [0, 1], color="#3A526E", lw=1, ls=":",
                 alpha=0.5)
        ax1.set_xlabel("FPR  /  Recall"); ax1.set_ylabel("TPR / Precision")
        ax1.set_title("ROC and PR curves — XGBoost_cw")
        ax1.legend(loc="lower left", frameon=False); ax1.grid(alpha=0.2)
        st.pyplot(fig1, transparent=True); plt.close(fig1)

    with ccol2:
        y_pred = (proba_test >= threshold).astype(int)
        cm = confusion_matrix(splits.y_test, y_pred)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im = ax2.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                col = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax2.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                         color=col, fontsize=14, fontweight="bold")
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Pred Legit", "Pred Fraud"])
        ax2.set_yticks([0, 1]); ax2.set_yticklabels(["True Legit", "True Fraud"])
        ax2.set_title(f"Confusion matrix @ thr = {threshold:.4f}")
        st.pyplot(fig2, transparent=True); plt.close(fig2)

    st.markdown("&nbsp;")

    # Model comparison
    st.markdown("#### Model comparison (all variants)")
    comp = load_model_comparison()
    if comp is not None and not comp.empty:
        rec_model = "XGBoost_cw"
        # Highlight best PR-AUC row in green and recommended row note
        def _style_pr(val, col_max):
            if col_max is None or pd.isna(val):
                return ""
            return ("background-color: rgba(29,158,117,0.35); "
                    "color: white; font-weight: 700;"
                    if val == col_max else "")
        try:
            col_max = comp["PR-AUC"].max() if "PR-AUC" in comp else None
            styled = comp.style.format(
                {c: "{:.4f}" for c in comp.columns if comp[c].dtype.kind == "f"}
            )
            if "PR-AUC" in comp:
                styled = styled.applymap(
                    lambda v: _style_pr(v, col_max), subset=["PR-AUC"])
            if "Model" in comp:
                styled = styled.apply(
                    lambda r: ["color:#00B4D8;font-weight:700"
                               if str(r["Model"]) == rec_model else ""
                               for _ in r],
                    axis=1)
            st.dataframe(styled, hide_index=True, use_container_width=True)
        except Exception:
            st.dataframe(comp, hide_index=True, use_container_width=True)
    else:
        st.caption("Model comparison table not available "
                   "(outputs/model_comparison.csv missing).")

    st.markdown("&nbsp;")

    # Business impact box
    net_benefit = fraud_caught - fp_friction
    st.markdown(
        f"""<div class='business-box'>
        <h4 style='margin-top:0; color:{PALETTE['accent']}'>
            Business impact — held-out test window ({tn+fp_+fn+tp:,} transactions)
        </h4>
        <ul style='line-height:1.8'>
          <li>✓ <b>£{fraud_caught:,.0f}</b> of <b>£{fraud_total:,.0f}</b> fraud value recovered</li>
          <li>✓ <b>{recovered_pct:.1f}%</b> loss prevention rate</li>
          <li>✓ ~<b>£{fp_friction:,.0f}</b> false-positive friction cost</li>
          <li>✓ <b>Net benefit: £{net_benefit:,.0f}</b></li>
        </ul>
        </div>""",
        unsafe_allow_html=True,
    )


# ==========================================================================
# TAB 3 — Global SHAP explainability
# ==========================================================================
with tab3:
    st.markdown("### Model explainability — what drives fraud detection")
    out = project_root() / "outputs"

    charts = [
        ("shap_feature_importance.png",
         "Mean |SHAP| per feature (top 15) — the average magnitude of "
         "each feature's effect on the fraud score."),
        ("shap_summary_beeswarm.png",
         "Distribution of SHAP values per feature. Each dot is one "
         "transaction; colour shows whether the raw feature value was "
         "high (red) or low (blue)."),
        ("shap_waterfall_fraud.png",
         "Explanation for the most-confident TRUE FRAUD case — the "
         "stack of features that pushed the score above threshold."),
        ("shap_waterfall_fp.png",
         "Explanation for the worst FALSE POSITIVE — the legitimate "
         "transaction the model wrongly flagged. The features driving "
         "the false flag are the ones the operations team should review."),
    ]
    for fname, caption in charts:
        path = out / fname
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
        else:
            st.warning(f"{fname} not found — run `python src/shap_analysis.py`.")

    st.markdown(
        f"""<div class='business-box' style='margin-top:18px'>
        <h4 style='margin-top:0; color:{PALETTE['accent']}'>
            How to read these charts (for the risk committee)
        </h4>
        <ul style='line-height:1.8'>
          <li><b>Each bar/dot represents the model's reasoning</b> — not a single rule,
              but how much each feature shifted the fraud score for a given transaction.</li>
          <li><b>Red colours push toward fraud, blue toward legit.</b>
              The model approves when the blue weight outweighs the red.</li>
          <li><b>The same feature can do either job</b> depending on its value —
              that is why a simple "if V14 &lt; -3 then flag" rule under-performs
              the full model, but is still useful as a sanity check.</li>
        </ul>
        </div>""",
        unsafe_allow_html=True,
    )
