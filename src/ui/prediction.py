from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ml.predict import CreditRiskPredictor
from src.ui.styles import (
    configure_plotly_chart,
    create_metric_card,
    get_svg_icon,
    render_executive_briefing,
)
from src.utils.config import get_settings


@st.cache_resource
def load_predictor() -> CreditRiskPredictor:
    settings = get_settings()
    return CreditRiskPredictor(settings.model_dir / "ebm_bundle.joblib")


DEFAULT_PROFILES = {
    "Prime Low Risk (Strong Scores & Clean History)": {
        "EXT_SOURCE_3": 0.72,
        "EXT_SOURCE_2": 0.68,
        "EXT_SOURCE_1": 0.65,
        "AMT_INCOME_TOTAL": 220000.0,
        "AMT_CREDIT": 450000.0,
        "AMT_ANNUITY": 24000.0,
        "AMT_GOODS_PRICE": 400000.0,
        "AGE_YEARS": 48.0,
        "DAYS_EMPLOYED": -4500.0,
        "DAYS_EMPLOYED_CLEAN": 4500.0,
        "DAYS_BIRTH": -17520.0,
        "DAYS_ID_PUBLISH": -3000.0,
        "DAYS_REGISTRATION": -5000.0,
        "DAYS_LAST_PHONE_CHANGE": -1200.0,
        "REGION_POPULATION_RELATIVE": 0.025,
        "OWN_CAR_AGE": 5.0,
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "ACTIVE_CREDIT_COUNT": 2.0,
        "PRIOR_CREDIT_COUNT": 5.0,
        "PREV_REFUSAL_RATE": 0.0,
        "INSTALLMENT_ROWS": 24.0,
        "PAID_AMOUNT": 180000.0,
        "SCHEDULED_AMOUNT": 180000.0,
        "INST_PAYMENT_RATIO": 1.0,
        "INST_LATE_RATE": 0.0,
    },
    "Moderate Medium Risk (Average Scores & Moderate Debt)": {
        "EXT_SOURCE_3": 0.44,
        "EXT_SOURCE_2": 0.48,
        "EXT_SOURCE_1": 0.40,
        "AMT_INCOME_TOTAL": 140000.0,
        "AMT_CREDIT": 550000.0,
        "AMT_ANNUITY": 28000.0,
        "AMT_GOODS_PRICE": 500000.0,
        "AGE_YEARS": 34.0,
        "DAYS_EMPLOYED": -1800.0,
        "DAYS_EMPLOYED_CLEAN": 1800.0,
        "DAYS_BIRTH": -12410.0,
        "DAYS_ID_PUBLISH": -2000.0,
        "DAYS_REGISTRATION": -3000.0,
        "DAYS_LAST_PHONE_CHANGE": -500.0,
        "REGION_POPULATION_RELATIVE": 0.018,
        "OWN_CAR_AGE": 10.0,
        "ORGANIZATION_TYPE": "Self-employed",
        "ACTIVE_CREDIT_COUNT": 4.0,
        "PRIOR_CREDIT_COUNT": 6.0,
        "PREV_REFUSAL_RATE": 0.15,
        "INSTALLMENT_ROWS": 30.0,
        "PAID_AMOUNT": 150000.0,
        "SCHEDULED_AMOUNT": 160000.0,
        "INST_PAYMENT_RATIO": 0.94,
        "INST_LATE_RATE": 0.08,
    },
    "Subprime High Risk (Young Renter, Overdue & Late History)": {
        "EXT_SOURCE_3": 0.18,
        "EXT_SOURCE_2": 0.22,
        "EXT_SOURCE_1": 0.20,
        "AMT_INCOME_TOTAL": 90000.0,
        "AMT_CREDIT": 600000.0,
        "AMT_ANNUITY": 36000.0,
        "AMT_GOODS_PRICE": 520000.0,
        "AGE_YEARS": 24.0,
        "DAYS_EMPLOYED": -500.0,
        "DAYS_EMPLOYED_CLEAN": 500.0,
        "DAYS_BIRTH": -8760.0,
        "DAYS_ID_PUBLISH": -800.0,
        "DAYS_REGISTRATION": -1200.0,
        "DAYS_LAST_PHONE_CHANGE": -100.0,
        "REGION_POPULATION_RELATIVE": 0.010,
        "OWN_CAR_AGE": 15.0,
        "ORGANIZATION_TYPE": "Other",
        "ACTIVE_CREDIT_COUNT": 6.0,
        "PRIOR_CREDIT_COUNT": 8.0,
        "PREV_REFUSAL_RATE": 0.50,
        "INSTALLMENT_ROWS": 18.0,
        "PAID_AMOUNT": 60000.0,
        "SCHEDULED_AMOUNT": 85000.0,
        "INST_PAYMENT_RATIO": 0.70,
        "INST_LATE_RATE": 0.35,
    },
}


def build_applicant_row(base_profile: dict, overrides: dict) -> pd.DataFrame:
    d = {**base_profile, **overrides}
    income = max(1.0, float(d["AMT_INCOME_TOTAL"]))
    credit = float(d["AMT_CREDIT"])
    annuity = float(d["AMT_ANNUITY"])
    goods = max(1.0, float(d["AMT_GOODS_PRICE"]))
    age = max(18.0, float(d["AGE_YEARS"]))
    emp = float(d["DAYS_EMPLOYED_CLEAN"])

    d["CREDIT_INCOME_RATIO"] = credit / income
    d["ANNUITY_INCOME_RATIO"] = annuity / income
    d["CREDIT_TERM"] = annuity / max(1.0, credit)
    d["CREDIT_GOODS_RATIO"] = credit / goods
    d["EMPLOYED_AGE_RATIO"] = (emp / 365.25) / age
    return pd.DataFrame([d])


def render_prediction_tab():
    # Executive Briefing Callout
    render_executive_briefing(
        title="Production Underwriting Engine & Real-Time Sensitivity Analysis",
        description=(
            "This module performs real-time credit underwriting using the production Calibrated Explainable Boosting Machine (EBM). "
            "Applicant attributes are mapped to a calibrated default probability and an institutional 0–1000 risk score segmented into "
            "3 internal model risk bands: Low (<65), Medium (65–158), and High (>158). Users can select realistic baseline archetypes "
            "or adjust parameters to simulate counterfactual what-if sensitivity scenarios."
        ),
        takeaways=[
            "Engine: Calibrated EBM",
            "Score Scale: 0–1000",
            "Monotonic Risk Bands: Low, Med, High",
            "Interactive Sensitivity Sliders",
        ],
        icon="bullseye",
    )

    predictor = load_predictor()

    col_input, col_output = st.columns([1.15, 0.85])

    with col_input:
        st.markdown(
            f"""
            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 8px; display:flex; align-items:center; gap:6px;">
                {get_svg_icon("user-tag", "#38bdf8", 14)} <span>Baseline Applicant Archetype</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_profile = st.selectbox(
            "Select Archetype:",
            list(DEFAULT_PROFILES.keys()),
            label_visibility="collapsed",
        )
        base = DEFAULT_PROFILES[selected_profile]

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 8px; display:flex; align-items:center; gap:6px;">
                {get_svg_icon("sliders", "#38bdf8", 14)} <span>External Credit Bureau Scores</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Row 1: 3 External Bureau Scores with Tooltips
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            ext_3 = st.slider(
                "Bureau Score C (Top)",
                0.01,
                0.95,
                float(base["EXT_SOURCE_3"]),
                0.01,
                help="Normalized credit bureau score C (EXT_SOURCE_3). Single most influential predictive factor in the entire portfolio.",
            )
        with sc2:
            ext_2 = st.slider(
                "Bureau Score B",
                0.01,
                0.95,
                float(base["EXT_SOURCE_2"]),
                0.01,
                help="Normalized credit bureau score B (EXT_SOURCE_2). Strong secondary creditworthiness indicator.",
            )
        with sc3:
            ext_1 = st.slider(
                "Bureau Score A",
                0.01,
                0.95,
                float(base["EXT_SOURCE_1"]),
                0.01,
                help="Normalized external inquiry bureau score A (EXT_SOURCE_1).",
            )

        st.markdown(
            f"""
            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-top: 14px; margin-bottom: 8px; display:flex; align-items:center; gap:6px;">
                {get_svg_icon("money", "#38bdf8", 14)} <span>Loan & Income Parameters</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Row 2: Financial Amounts with Formatted Display
        fn1, fn2, fn3 = st.columns(3)
        with fn1:
            credit_amt = st.number_input("Credit Amount (dataset currency units)", 10000.0, 2000000.0, float(base["AMT_CREDIT"]), 10000.0, help="Total loan principal applied for; the source dataset does not specify a display currency.")
            st.caption(f"Formatted: **{credit_amt:,.0f} units**")
        with fn2:
            income_amt = st.number_input("Annual Income (dataset currency units)", 10000.0, 2000000.0, float(base["AMT_INCOME_TOTAL"]), 5000.0, help="Declared annual gross income; the source dataset does not specify a display currency.")
            st.caption(f"Formatted: **{income_amt:,.0f} units**")
        with fn3:
            annuity_amt = st.number_input("Loan Annuity (dataset currency units)", 1000.0, 200000.0, float(base["AMT_ANNUITY"]), 1000.0, help="Scheduled annual loan repayment amount; the source dataset does not specify a display currency.")
            st.caption(f"Formatted: **{annuity_amt:,.0f} units**")

        st.markdown(
            f"""
            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-top: 14px; margin-bottom: 8px; display:flex; align-items:center; gap:6px;">
                {get_svg_icon("users", "#38bdf8", 14)} <span>Demographic & Payment Behaviour</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Row 3: Demographics & Behaviour with Tooltips
        ps1, ps2 = st.columns(2)
        with ps1:
            age_val = st.slider("Applicant Age (Years)", 20, 75, int(base["AGE_YEARS"]), help="Current age of applicant in years.")
        with ps2:
            late_rate = st.slider("Installment Late Rate (%)", 0.0, 1.0, float(base["INST_LATE_RATE"]), 0.01, help="Historical frequency of late loan installment payments (e.g. 0.08 = 8% delinquency rate).")

        overrides = {
            "EXT_SOURCE_3": ext_3,
            "EXT_SOURCE_2": ext_2,
            "EXT_SOURCE_1": ext_1,
            "AMT_CREDIT": credit_amt,
            "AMT_INCOME_TOTAL": income_amt,
            "AMT_ANNUITY": annuity_amt,
            "AGE_YEARS": float(age_val),
            "DAYS_BIRTH": -float(age_val) * 365.25,
            "INST_LATE_RATE": late_rate,
        }

        applicant_df = build_applicant_row(base, overrides)
        st.session_state["scored_applicant"] = applicant_df

    with col_output:
        st.markdown(
            f"""
            <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 10px; display:flex; align-items:center; gap:6px;">
                {get_svg_icon("gauge", "#38bdf8", 14)} <span>Model Underwriting Scorecard</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        result = predictor.predict(applicant_df)
        prob = float(result.iloc[0]["default_probability"])
        score = int(result.iloc[0]["risk_score"])
        band = str(result.iloc[0]["risk_band"])

        badge_class = f"badge-{band.lower()}"
        if band == "Low":
            icon_svg = get_svg_icon("shield-check", "#34d399", 14, "margin-right:4px;")
            color_accent = "#34d399"
        elif band == "Medium":
            icon_svg = get_svg_icon("warning", "#fbbf24", 14, "margin-right:4px;")
            color_accent = "#fbbf24"
        else:
            icon_svg = get_svg_icon("warning", "#f43f5e", 14, "margin-right:4px;")
            color_accent = "#f43f5e"

        band_html = f'<span class="badge {badge_class}" style="font-size: 0.95rem; padding: 6px 14px; display:inline-flex; align-items:center;">{icon_svg} {band} RISK BAND</span>'

        # Unified Underwriting Card
        st.markdown(
            f"""
            <div class="metric-card" style="text-align: center; padding: 22px 24px 18px 24px;">
                <div style="margin-bottom: 10px;">{band_html}</div>
                <div class="metric-label" style="font-size: 0.76rem; letter-spacing:0.04em;">MODEL-ESTIMATED DEFAULT PROBABILITY</div>
                <div class="metric-value" style="font-size: 2.6rem; color: {color_accent}; margin: 4px 0; text-shadow: 0 0 24px rgba(56, 189, 248, 0.35);">{prob*100:.2f}%</div>
                <div class="metric-delta" style="justify-content: center; color: #94a3b8; font-size: 0.86rem;">
                    Internal Risk Score: <b style="color: #f8fafc; font-size: 1rem; margin-left: 5px;">{score}</b> / 1000
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Plotly Gauge Chart with Calibrated Spacing to Prevent Overlap
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 0.92]},
                gauge={
                    "axis": {
                        "range": [0, 1000],
                        "tickcolor": "#334155",
                        "tickfont": {"size": 10, "color": "#64748b"},
                    },
                    "bar": {"color": "#38bdf8", "thickness": 0.22},
                    "bgcolor": "#0b0f19",
                    "steps": [
                        {"range": [0, 65], "color": "rgba(16, 185, 129, 0.2)"},
                        {"range": [65, 158], "color": "rgba(245, 158, 11, 0.2)"},
                        {"range": [158, 1000], "color": "rgba(244, 63, 94, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f8fafc", "width": 2},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            )
        )
        configure_plotly_chart(fig_gauge, title="Payment Difficulty Risk Gauge (0-1000)", height=205, margin_t=32, margin_b=10, margin_l=25, margin_r=25)
        st.plotly_chart(fig_gauge, width="stretch")

        # Financial affordability chips
        cir = float(applicant_df["CREDIT_INCOME_RATIO"].iloc[0])
        air = float(applicant_df["ANNUITY_INCOME_RATIO"].iloc[0])
        r1, r2 = st.columns(2)
        with r1:
            create_metric_card("Credit / Income Ratio", f"{cir:.2f}x", "Benchmark: 3.55x", "neutral", icon="scale")
        with r2:
            create_metric_card("Annuity / Income Burden", f"{air*100:.1f}%", "Debt Service Ratio", "neutral", icon="receipt")
