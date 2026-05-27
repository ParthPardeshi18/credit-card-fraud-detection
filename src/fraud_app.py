"""
fraud_app.py
============
EXTENSION 3 — Streamlit fraud risk scoring app.

Queen Mary University of London · BUSM131 Business Analytics Masterclass
Personal extension built on top of the group submission.

Run with:   streamlit run src/fraud_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    average_precision_score, roc_auc_score,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_pipeline import prepare_splits, project_root, st_load_dataframe  # noqa: E402


# ==========================================================================
# Page config
# ==========================================================================
st.set_page_config(
    page_title="Fraud Risk Scorer · QMUL BUSM131",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️",
)

# Brand palette — aligned with the project's other charts
C = {
    "bg":        "#0B1623",
    "surface":   "#13243C",
    "surface_2": "#1B3553",
    "border":    "#243E5C",
    "accent":    "#00B4D8",
    "accent_2":  "#0096C7",
    "fraud":     "#E24B4A",
    "legit":     "#378ADD",
    "success":   "#1D9E75",
    "warning":   "#F4A261",
    "neutral":   "#888780",
    "text":      "#F1F5F9",
    "text_2":    "#94A8C0",
    "text_3":    "#6B7B92",
}

# ==========================================================================
# Global CSS — typography, surfaces, controls, spacing
# ==========================================================================
st.markdown(f"""
<style>
/* ---------- base ---------- */
html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "Inter", "Helvetica Neue", Arial, sans-serif;
}}
.stApp {{ background-color: {C["bg"]}; color: {C["text"]}; }}
section.main > div.block-container {{
    padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px;
}}

/* ---------- headings ---------- */
h1, h2, h3, h4 {{ color: {C["text"]}; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.75rem; font-weight: 700; }}
h2 {{ font-size: 1.30rem; font-weight: 600; margin-top: 1.2rem; }}
h3 {{ font-size: 1.05rem; font-weight: 600; color: {C["text"]}; }}
p, li, label, .stMarkdown {{ color: {C["text_2"]}; }}

hr {{ border-color: {C["border"]}; opacity: 0.5; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
    background-color: {C["surface"]};
    border-right: 1px solid {C["border"]};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: {C["text"]}; }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; background: transparent; border-bottom: 1px solid {C["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: {C["text_2"]};
    padding: 10px 18px; border-radius: 6px 6px 0 0;
    font-weight: 500; border: none;
}}
.stTabs [aria-selected="true"] {{
    background: transparent !important; color: {C["accent"]} !important;
    border-bottom: 2px solid {C["accent"]} !important;
}}

/* ---------- metric cards ---------- */
div[data-testid="stMetric"] {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 14px 18px;
}}
div[data-testid="stMetric"] label {{
    color: {C["text_3"]} !important; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;
}}
div[data-testid="stMetricValue"] {{
    color: {C["accent"]} !important; font-size: 1.75rem; font-weight: 700;
}}
div[data-testid="stMetricDelta"] {{ color: {C["text_2"]} !important; }}

/* ---------- form controls ----------
   Slider track/thumb colour comes from .streamlit/config.toml
   (primaryColor = #00B4D8). We only tune the surrounding container. */
.stNumberInput input, .stTextInput input {{
    background: {C["surface"]} !important; color: {C["text"]} !important;
    border: 1px solid {C["border"]} !important; border-radius: 6px;
}}
.stNumberInput button {{
    background: {C["surface_2"]} !important; color: {C["text"]} !important;
    border: 1px solid {C["border"]} !important;
}}

/* ---------- primary button (cyan with high-contrast dark text) ---------- */
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"] {{
    background: {C["accent"]} !important;
    color: #0B1623 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    padding: 12px 22px !important;
    transition: all 0.18s ease;
    box-shadow: 0 1px 0 rgba(255,255,255,0.06) inset,
                0 2px 8px rgba(0,180,216,0.18);
}}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover {{
    background: {C["accent_2"]} !important;
    color: #0B1623 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(0,180,216,0.28);
}}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primaryFormSubmit"] p,
.stFormSubmitButton > button[kind="primary"] p,
.stFormSubmitButton > button[kind="primaryFormSubmit"] p,
button[data-testid="stBaseButton-primary"] p,
button[data-testid="stBaseButton-primaryFormSubmit"] p,
button[data-testid="stBaseButton-primary"] div,
button[data-testid="stBaseButton-primaryFormSubmit"] div,
button[data-testid="stBaseButton-primary"] span,
button[data-testid="stBaseButton-primaryFormSubmit"] span {{
    color: #0B1623 !important; font-weight: 700 !important;
}}
/* Secondary buttons (e.g. download) — keep readable on dark surfaces */
.stButton > button[kind="secondary"] {{
    background: {C["surface_2"]} !important;
    color: {C["text"]} !important;
    border: 1px solid {C["border"]} !important;
}}

/* ---------- dataframe ---------- */
[data-testid="stDataFrame"] {{
    border: 1px solid {C["border"]}; border-radius: 8px; overflow: hidden;
}}

/* ---------- expander (visible chevron + readable header) ---------- */
[data-testid="stExpander"] details {{
    background: {C["surface"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary {{
    background: {C["surface"]} !important;
    color: {C["text"]} !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary:hover {{
    background: {C["surface_2"]} !important;
}}
[data-testid="stExpander"] summary svg {{
    fill: {C["accent"]} !important;
    color: {C["accent"]} !important;
}}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
    background: {C["surface"]} !important;
    border-top: 1px solid {C["border"]} !important;
    padding: 14px 16px 16px 16px !important;
}}

/* ---------- alerts ---------- */
.stAlert {{ border-radius: 8px; border: 1px solid {C["border"]}; }}

/* ---------- custom helper classes ---------- */
.brand-tag {{
    display: inline-block;
    background: {C["surface_2"]};
    color: {C["text_2"]};
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}}
.section-caption {{
    color: {C["text_3"]}; font-size: 0.90rem; margin-top: -4px;
    margin-bottom: 0.8rem;
}}
.risk-card {{
    background: {C["surface"]}; border: 1px solid {C["border"]};
    border-radius: 10px; padding: 18px 22px; height: 100%;
}}
.risk-tier {{
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-weight: 700; letter-spacing: 0.04em; font-size: 0.78rem;
    text-transform: uppercase;
}}
.muted {{ color: {C["text_3"]}; font-size: 0.85rem; }}

footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# Plotly common layout
# ==========================================================================
def _layout(title: str | None = None, height: int = 360):
    d: dict = dict(
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(family="Inter, Segoe UI, sans-serif",
                  color=C["text_2"], size=12),
        height=height,
        margin=dict(l=50, r=20, t=50 if title else 20, b=40),
        xaxis=dict(gridcolor=C["surface_2"], linecolor=C["border"],
                   zerolinecolor=C["surface_2"]),
        yaxis=dict(gridcolor=C["surface_2"], linecolor=C["border"],
                   zerolinecolor=C["surface_2"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                    borderwidth=0, font=dict(color=C["text_2"], size=11)),
    )
    # Only inject title key when there is a value — passing title=None in
    # Plotly ≥6 renders the JavaScript string "undefined" on the chart.
    if title:
        d["title"] = dict(text=title, x=0.01,
                          font=dict(size=14, color=C["text"]))
    return d


# ==========================================================================
# Cached loaders
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_metadata():
    p = project_root() / "outputs" / "model_metadata.pkl"
    return joblib.load(p) if p.exists() else {}


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(project_root() / "outputs" / "xgb_fraud_model.pkl")


@st.cache_resource(show_spinner=False)
def load_explainer(_pipe):
    return shap.TreeExplainer(_pipe.named_steps["clf"])


@st.cache_data(show_spinner=False)
def load_splits(_df: "pd.DataFrame"):
    # Receive the already-cached DataFrame so prepare_splits never
    # triggers a second file-read / download within the same session.
    # The leading underscore tells @st.cache_data not to hash the df arg
    # (hashing a 57k-row DataFrame on every rerun is expensive).
    return prepare_splits(verbose=False, df=_df)


@st.cache_data(show_spinner=False)
def load_test_predictions(_df: "pd.DataFrame"):
    pipe_ = load_model()
    s = load_splits(_df)
    proba = pipe_.predict_proba(s.X_test)[:, 1]
    return s.X_test, s.y_test, proba


@st.cache_data(show_spinner=False)
def load_model_comparison():
    p = project_root() / "outputs" / "model_comparison.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


# ==========================================================================
# Load core artefacts — all errors surface as user-visible messages so
# the process never crashes silently (which causes the Streamlit Cloud
# health-check to report "connection refused").
# ==========================================================================
try:
    pipe = load_model()
    metadata = load_metadata()
    feature_names = (list(pipe.feature_names_in_)
                     if hasattr(pipe, "feature_names_in_")
                     else metadata.get("feature_names",
                                       ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]))
except FileNotFoundError:
    st.error(
        "⚠️ **Model file not found** — `outputs/xgb_fraud_model.pkl`.\n\n"
        "Run `python src/setup_model.py` locally, commit the `.pkl` to the "
        "`outputs/` folder, and redeploy."
    )
    st.stop()
except Exception as _exc:
    st.error(f"⚠️ **Unexpected error loading model:** `{_exc}`")
    st.stop()

# Dataset — load after the model so a missing dataset shows a clear message
# rather than a blank screen or connection-refused health-check failure.
try:
    _dataset_df = st_load_dataframe()   # cached; downloads from HF if needed
except FileNotFoundError as _exc:
    st.error(
        "⚠️ **Dataset not available.**\n\n"
        "The app could not find `data/creditcard.parquet` locally and the "
        "HuggingFace download failed.\n\n"
        "**Fix:** follow `docs/DATASET_HOSTING.md` to upload the Parquet file "
        "to HuggingFace Hub, then update `HF_DATASET_REPO` in "
        "`src/data_pipeline.py` and redeploy."
    )
    st.stop()
except Exception as _exc:
    st.error(
        f"⚠️ **Dataset load error:** `{_exc}`\n\n"
        "Check `HF_DATASET_REPO` in `src/data_pipeline.py` and that the "
        "HuggingFace repo is **public** (or that `HF_TOKEN` is set in Secrets)."
    )
    st.stop()


# ==========================================================================
# Sidebar
# ==========================================================================
with st.sidebar:
    st.markdown(
        f"""<div style='padding: 4px 0 14px 0;'>
            <h2 style='margin:0; color:{C["accent"]}; font-size:1.35rem'>
                🛡️ Fraud Risk Scorer
            </h2>
            <div class='muted' style='margin-top:2px'>
                Live XGBoost scoring · SHAP explainability
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<span class='brand-tag'>About</span>",
                unsafe_allow_html=True)
    st.markdown(
        f"""<div style='margin-top:6px; line-height:1.55'>
            <b style='color:{C["text"]}'>Queen Mary University of London</b><br>
            <span class='muted'>BUSM131 · Business Analytics Masterclass</span><br>
            <span class='muted'>Year-long project · personal extensions</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("&nbsp;")
    st.markdown("<span class='brand-tag'>Production model</span>",
                unsafe_allow_html=True)
    st.markdown(
        f"""<div style='margin-top:6px; line-height:1.55'>
            <b style='color:{C["text"]}'>XGBoost + Class Weighting</b><br>
            <span class='muted'>Trained on 284,807 transactions</span><br>
            <span class='muted'>492 fraud cases (0.17% base rate)</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10, max_value=0.90, value=0.4949, step=0.01,
        help="Above this score → BLOCK. Updates risk tiers in real time.",
    )
    st.markdown(
        f"<div class='muted'>Active threshold "
        f"<b style='color:{C['accent']}'>{threshold:.4f}</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("<span class='brand-tag'>Personal extensions</span>",
                unsafe_allow_html=True)
    st.markdown(
        """<ul style='padding-left: 18px; line-height:1.7; margin-top:6px'>
            <li>SHAP explainability</li>
            <li>Probability calibration</li>
            <li>Live risk scorer</li>
        </ul>""", unsafe_allow_html=True)


# ==========================================================================
# Header
# ==========================================================================
st.markdown(
    f"""<div style='display:flex; align-items:baseline; gap:14px;
                    margin-bottom: 4px;'>
        <h1 style='margin:0'>Live Fraud Risk Scorer</h1>
        <span class='brand-tag'>BUSM131 · QMUL</span>
    </div>
    <p class='section-caption' style='font-size:0.95rem'>
        Score any credit-card transaction in real time and inspect
        exactly which features moved the decision.
    </p>""",
    unsafe_allow_html=True,
)

tab_score, tab_perf, tab_global = st.tabs([
    "🎯  Risk Scorer",
    "📊  Model Performance",
    "🔍  Global Explainability",
])


# ==========================================================================
# TAB 1 — Live transaction risk scorer
# ==========================================================================
with tab_score:
    st.markdown("### Score a transaction")
    st.markdown(
        "<p class='section-caption'>Enter transaction details below. "
        "The model returns a fraud probability, a recommended action, "
        "and an explanation of which features drove the score.</p>",
        unsafe_allow_html=True,
    )

    with st.form("scorer_form", border=False):
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.markdown("**Transaction profile**")
            amount = st.number_input(
                "Amount (£)", min_value=0.01, max_value=25000.0,
                value=89.00, step=1.0, format="%.2f",
            )
            time_hr = st.slider(
                "Time of day (hours since stream start)",
                min_value=0, max_value=48, value=12,
                help="0–48h since the first transaction in the dataset.",
            )

        with col_r:
            st.markdown("**Key signal features (PCA)**")
            v14 = st.slider("V14 — top fraud signal",
                            -10.0, 5.0, 0.0, 0.1,
                            help="Strongest single fraud signal; very "
                                 "negative values strongly indicate fraud.")
            v17 = st.slider("V17", -10.0, 5.0, 0.0, 0.1)
            v12 = st.slider("V12", -10.0, 5.0, 0.0, 0.1)

        with st.expander("Additional PCA features (V4, V10, V11)", expanded=False):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                v10 = st.slider("V10", -10.0, 5.0, 0.0, 0.1)
            with ec2:
                v11 = st.slider("V11", -10.0, 5.0, 0.0, 0.1)
            with ec3:
                v4 = st.slider("V4",  -5.0, 10.0, 0.0, 0.1)
            st.caption("Remaining PCA features V1–V28 (not shown) are set "
                       "to the population mean of 0.")

        submitted = st.form_submit_button(
            "🔍  Score Transaction", use_container_width=True, type="primary"
        )

    if submitted:
        row = {f: 0.0 for f in feature_names}
        row["Time"]   = time_hr * 3600.0
        row["Amount"] = amount
        row["V4"], row["V10"], row["V11"] = v4, v10, v11
        row["V12"], row["V14"], row["V17"] = v12, v14, v17
        x_input = pd.DataFrame([row])[feature_names]

        proba = float(pipe.predict_proba(x_input)[0, 1])

        if proba >= 0.6:
            tier, tier_col, tier_text = "AUTO-BLOCK", C["fraud"], "Block the transaction immediately."
        elif proba >= 0.3:
            tier, tier_col, tier_text = "MANUAL REVIEW", C["warning"], "Route to a human reviewer."
        else:
            tier, tier_col, tier_text = "AUTO-APPROVE", C["success"], "Approve automatically."

        live_decision = "BLOCKED" if proba >= threshold else "APPROVED"
        live_decision_col = C["fraud"] if proba >= threshold else C["success"]

        st.markdown("---")
        st.markdown("### Risk assessment")

        g1, g2 = st.columns([1.25, 1], gap="large")
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "valueformat": ".1f",
                        "font": {"size": 46, "color": tier_col}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                             "tickcolor": C["text_3"],
                             "tickfont": {"color": C["text_2"], "size": 10}},
                    "bar": {"color": tier_col, "thickness": 0.28},
                    "bgcolor": C["surface"],
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  30], "color": "rgba(29,158,117,0.20)"},
                        {"range": [30, 60], "color": "rgba(244,162,97,0.22)"},
                        {"range": [60,100], "color": "rgba(226,75,74,0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": C["accent"], "width": 3},
                        "thickness": 0.85,
                        "value": threshold * 100,
                    },
                },
                domain={"x": [0, 1], "y": [0, 1]},
            ))
            gauge.update_layout(**_layout(height=290))
            gauge.update_layout(margin=dict(l=20, r=20, t=20, b=10))
            st.plotly_chart(gauge, use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown(
                f"<div class='muted' style='text-align:center; "
                f"margin-top:-10px'>"
                f"Fraud probability · cyan tick marks the active "
                f"threshold ({threshold:.4f})"
                f"</div>",
                unsafe_allow_html=True,
            )

        with g2:
            st.markdown(
                f"""<div class='risk-card'>
                    <div class='muted'>RECOMMENDED ACTION</div>
                    <div class='risk-tier' style='background:{tier_col};
                        color:#0B1623; margin: 8px 0 12px 0'>{tier}</div>
                    <div style='color:{C["text_2"]}; line-height:1.55'>
                        {tier_text}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("&nbsp;")
            st.markdown(
                f"""<div class='risk-card'>
                    <div class='muted'>LIVE THRESHOLD DECISION</div>
                    <div style='font-size:1.4rem; font-weight:700;
                                color:{live_decision_col}; margin-top:6px'>
                        {live_decision}
                    </div>
                    <div class='muted' style='margin-top:2px'>
                        At threshold {threshold:.4f},
                        score {proba*100:.2f}% is
                        {"above" if proba >= threshold else "below"} cutoff.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("&nbsp;")
        st.markdown(
            "### Why the model scored this transaction "
            f"<span style='color:{tier_col}'>{proba*100:.1f}%</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='section-caption'>Top 5 features by absolute SHAP "
            "contribution. A positive (red) value pushes the score "
            "toward FRAUD; a negative (green) value pushes toward LEGIT.</p>",
            unsafe_allow_html=True,
        )

        explainer = load_explainer(pipe)
        scaler = pipe.named_steps["scaler"]
        x_scaled = pd.DataFrame(scaler.transform(x_input),
                                columns=feature_names)
        sv = explainer(x_scaled)
        shap_vals = sv.values[0]

        order = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_names = [feature_names[i] for i in order]
        top_vals  = shap_vals[order]
        top_raw   = x_input.values[0, order]
        bar_colours = [C["fraud"] if v > 0 else C["success"] for v in top_vals]

        sc1, sc2 = st.columns([1.3, 1], gap="large")
        with sc1:
            bar = go.Figure(go.Bar(
                x=top_vals, y=top_names, orientation="h",
                marker=dict(color=bar_colours, line=dict(width=0)),
                text=[f"{v:+.3f}" for v in top_vals],
                textposition="outside",
                textfont=dict(color=C["text"], size=11),
                hovertemplate="%{y}: <b>%{x:+.4f}</b><extra></extra>",
            ))
            bar.update_layout(**_layout(height=320))
            bar.update_layout(
                xaxis_title="SHAP value (impact on log-odds of fraud)",
                yaxis=dict(autorange="reversed", gridcolor=C["surface_2"],
                           linecolor=C["border"]),
                showlegend=False,
                bargap=0.35,
            )
            bar.add_vline(x=0, line=dict(color=C["border"], width=1))
            st.plotly_chart(bar, use_container_width=True,
                            config={"displayModeBar": False})

        with sc2:
            df_ex = pd.DataFrame({
                "Feature":  top_names,
                "Value":    [f"{v:.3f}" for v in top_raw],
                "Impact":   [f"{v:+.3f}" for v in top_vals],
                "Pushes →": ["FRAUD" if v > 0 else "LEGIT" for v in top_vals],
            })
            st.dataframe(df_ex, hide_index=True, use_container_width=True)
            st.caption(
                "🔴 Red impact = pushes toward fraud · "
                "🟢 Green impact = pushes toward legit"
            )


# ==========================================================================
# TAB 2 — Model performance
# ==========================================================================
with tab_perf:
    st.markdown("### Model validation results")
    st.markdown(
        "<p class='section-caption'>Performance on the held-out test set "
        "(56,962 transactions, untouched during model selection).</p>",
        unsafe_allow_html=True,
    )

    pr_auc       = metadata.get("pr_auc", 0.7868)
    f1_metric    = metadata.get("f1", 0.7900)
    recall_val   = metadata.get("recall", 0.8061)
    precision    = metadata.get("precision", 0.7745)
    tp = int(metadata.get("tp", 79))
    fp_ = int(metadata.get("fp", 23))
    fn = int(metadata.get("fn", 19))
    tn = int(metadata.get("tn", 56841))
    fraud_total  = metadata.get("fraud_total",  10644.93)
    fraud_caught = metadata.get("fraud_caught",  8598.05)
    fp_friction  = metadata.get("fp_friction",   115.0)
    fpr = fp_ / max(fp_ + tn, 1)
    recovered_pct = fraud_caught / max(fraud_total, 1) * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("PR-AUC",              f"{pr_auc:.4f}",
              help="Area under the Precision-Recall curve.")
    k2.metric("F1 Score",            f"{f1_metric:.4f}",
              help=f"Precision {precision:.3f} · Recall {recall_val:.3f}.")
    k3.metric("Fraud Recovered",     f"{recovered_pct:.1f}%",
              help=f"£{fraud_caught:,.0f} of £{fraud_total:,.0f}.")
    k4.metric("False Positive Rate", f"{fpr*100:.3f}%",
              help=f"{fp_} of {fp_+tn:,} legit transactions flagged.")

    st.markdown("&nbsp;")

    X_test, y_test, proba_test = load_test_predictions(_dataset_df)
    fpr_arr, tpr_arr, _ = roc_curve(y_test, proba_test)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, proba_test)
    roc_auc = roc_auc_score(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)

    cc1, cc2 = st.columns(2, gap="large")
    with cc1:
        st.markdown("#### ROC curve")
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(
            x=fpr_arr, y=tpr_arr, mode="lines",
            line=dict(color=C["accent"], width=2.5),
            name=f"ROC  (AUC = {roc_auc:.3f})",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=C["border"], width=1, dash="dot"),
            name="Random baseline",
            hoverinfo="skip",
        ))
        roc_fig.update_layout(**_layout(height=340))
        roc_fig.update_layout(
            xaxis_title="False positive rate",
            yaxis_title="True positive rate (recall)",
        )
        st.plotly_chart(roc_fig, use_container_width=True,
                        config={"displayModeBar": False})

    with cc2:
        st.markdown("#### Precision–Recall curve")
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(
            x=rec_arr, y=prec_arr, mode="lines",
            line=dict(color=C["fraud"], width=2.5),
            name=f"PR  (AP = {ap:.3f})",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ))
        pr_fig.add_hline(y=float(y_test.mean()),
                         line=dict(color=C["border"], width=1, dash="dot"),
                         annotation_text="Base rate",
                         annotation_font=dict(color=C["text_3"], size=10),
                         annotation_position="bottom right")
        pr_fig.update_layout(**_layout(height=340))
        pr_fig.update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
        )
        st.plotly_chart(pr_fig, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("&nbsp;")
    st.markdown("#### Confusion matrix")
    st.caption(f"Recomputed at the active decision threshold "
               f"({threshold:.4f}). Adjust the sidebar slider to see "
               f"the trade-off live.")

    y_pred = (proba_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm / cm.sum()

    cmf = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted Legit", "Predicted Fraud"],
        y=["True Legit", "True Fraud"],
        colorscale=[
            [0.0, C["surface"]],
            [0.4, "#0A5C7A"],
            [1.0, C["accent"]],
        ],
        showscale=False,
        text=[[f"{cm[i,j]:,}<br><span style='font-size:11px;opacity:0.7'>"
               f"{cm_norm[i,j]*100:.2f}%</span>"
               for j in range(2)] for i in range(2)],
        texttemplate="%{text}",
        textfont=dict(color=C["text"], size=18),
        hovertemplate="%{y} → %{x}<br>count: %{z:,}<extra></extra>",
    ))
    cmf.update_layout(**_layout(height=340))
    cmf.update_layout(
        xaxis=dict(side="bottom", tickfont=dict(color=C["text"])),
        yaxis=dict(autorange="reversed", tickfont=dict(color=C["text"])),
        margin=dict(l=120, r=80, t=30, b=40),
    )
    cmcol_l, cmcol_c, cmcol_r = st.columns([1, 2, 1])
    with cmcol_c:
        st.plotly_chart(cmf, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("&nbsp;")
    st.markdown("#### Model comparison")
    st.caption("All variants from the group submission, ranked by PR-AUC. "
               "**XGBoost_cw** is the recommended production model.")

    comp = load_model_comparison()
    if comp is not None and not comp.empty:
        priority = ["Model", "PR-AUC", "F1", "Precision",
                    "Recall", "Threshold", "ROC-AUC", "Accuracy"]
        ordered = [c for c in priority if c in comp.columns]
        rest = [c for c in comp.columns if c not in ordered]
        comp = comp[ordered + rest].copy()

        if "PR-AUC" in comp.columns:
            comp = comp.sort_values("PR-AUC", ascending=False)\
                       .reset_index(drop=True)

        rec_model = "XGBoost_cw"
        if "Model" in comp.columns and (comp["Model"] == rec_model).any():
            comp.insert(1, " ", np.where(
                comp["Model"] == rec_model, "⭐ Recommended", ""))

        col_config = {}
        if "PR-AUC" in comp.columns:
            col_config["PR-AUC"] = st.column_config.ProgressColumn(
                "PR-AUC", format="%.4f", min_value=0.0, max_value=1.0)
        for c in ("F1", "Precision", "Recall", "ROC-AUC",
                  "Accuracy", "Threshold"):
            if c in comp.columns:
                col_config[c] = st.column_config.NumberColumn(c, format="%.4f")

        st.dataframe(
            comp, hide_index=True, use_container_width=True,
            column_config=col_config,
            height=min(56 + 38 * len(comp), 420),
        )
    else:
        st.info("Model comparison table not available "
                "(outputs/model_comparison.csv missing).")

    st.markdown("&nbsp;")
    st.markdown("#### Business impact")
    st.caption("Translating model performance into £ recovered and friction "
               "cost (assumed at £5 per false positive).")
    net = fraud_caught - fp_friction

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Fraud value in test", f"£{fraud_total:,.0f}",
               help=f"{tp+fn} fraud transactions in the held-out set.")
    bc2.metric("Recovered by model",  f"£{fraud_caught:,.0f}",
               delta=f"{recovered_pct:.1f}% of total")
    bc3.metric("Friction cost",       f"£{fp_friction:,.0f}",
               delta=f"{fp_} false positives", delta_color="inverse")
    bc4.metric("Net benefit",         f"£{net:,.0f}",
               delta="recovered − friction")


# ==========================================================================
# TAB 3 — Global SHAP explainability
# ==========================================================================
with tab_global:
    st.markdown("### Global explainability")
    st.markdown(
        "<p class='section-caption'>What the XGBoost model has learned about "
        "fraud, globally — the patterns and feature ranges that drive its "
        "decisions across the whole test set.</p>",
        unsafe_allow_html=True,
    )

    _shap_dir = project_root() / "outputs"

    def _shap_img(fname: str, caption: str) -> None:
        path = _shap_dir / fname
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.caption(caption)
        else:
            st.warning(f"`{fname}` not found — run `python src/shap_analysis.py`.")

    # --- Row 1: beeswarm (60%) | feature importance (40%) ---
    col_bees, col_imp = st.columns([0.6, 0.4])
    with col_bees:
        _shap_img(
            "shap_summary_beeswarm.png",
            "Each dot = one transaction. Red = high feature value, "
            "Blue = low. Points left of centre reduce fraud score, "
            "right increase it.",
        )
    with col_imp:
        _shap_img(
            "shap_feature_importance.png",
            "V14 alone contributes 1.86× more than V10 — "
            "the strongest single fraud signal in the model.",
        )

    st.markdown("&nbsp;")

    # --- Row 2: true positive waterfall | false positive waterfall ---
    col_wf, col_wfp = st.columns(2)
    with col_wf:
        _shap_img(
            "shap_waterfall_fraud.png",
            "Transaction correctly flagged — V14=−8.67 "
            "drove the score to 100% fraud probability.",
        )
    with col_wfp:
        _shap_img(
            "shap_waterfall_fp.png",
            "False positive — V14=−9.25 mimicked fraud "
            "patterns despite being a legitimate transaction.",
        )

    st.markdown("---")
    st.markdown("#### How to read these charts")
    rd1, rd2, rd3 = st.columns(3)
    with rd1:
        st.markdown(
            f"""<div class='risk-card'>
                <div style='color:{C["accent"]}; font-weight:700;
                            margin-bottom: 6px'>1 · The model isn't a rule</div>
                <div class='muted'>Each chart shows how much every feature
                <i>shifted</i> the fraud score for a transaction — not a
                fixed if/then rule.</div>
            </div>""", unsafe_allow_html=True)
    with rd2:
        st.markdown(
            f"""<div class='risk-card'>
                <div style='color:{C["accent"]}; font-weight:700;
                            margin-bottom: 6px'>2 · Direction matters</div>
                <div class='muted'>Red contributions push toward fraud,
                blue/green push toward legit. The score is approved when
                the legit weight outweighs the fraud weight.</div>
            </div>""", unsafe_allow_html=True)
    with rd3:
        st.markdown(
            f"""<div class='risk-card'>
                <div style='color:{C["accent"]}; font-weight:700;
                            margin-bottom: 6px'>3 · One feature, two roles</div>
                <div class='muted'>The same PCA feature can push either
                way depending on its value. That's why a simple
                "if V14 &lt; −3 flag" rule under-performs the full model
                but is useful as a sanity check.</div>
            </div>""", unsafe_allow_html=True)
