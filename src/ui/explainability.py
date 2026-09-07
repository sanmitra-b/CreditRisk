from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui.styles import (
    configure_plotly_chart,
    create_metric_card,
    get_feature_label,
    get_svg_icon,
    render_executive_briefing,
)
from src.utils.config import get_settings


@st.cache_data
def load_explanation_artifacts():
    settings = get_settings()
    eval_dir = settings.artifact_dir / "evaluation"
    
    global_imp = pd.read_csv(eval_dir / "ebm_global_importance.csv")
    shap_summary = pd.read_csv(eval_dir / "lightgbm_shap_summary.csv")
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    
    return global_imp, shap_summary, metrics


def render_explainability_tab():
    # Executive Briefing Callout
    render_executive_briefing(
        title="Glass-Box Model Governance & Applicant-Level Explanations",
        description=(
            "This module supports model review with transparent global patterns and applicant-level score contributions. "
            "Unlike post-hoc explanations for a shadow gradient-boosting model, the production Explainable Boosting Machine (EBM) "
            "is additive by design. Its contributions are useful evidence for governance review, but they are not, by themselves, "
            "a legal compliance determination or finalized adverse-action reason codes."
        ),
        takeaways=[
            "Governance: Auditable Contributions",
            "Exact Mathematical Additivity",
            "EBM (0.764 AUC) vs Shadow LightGBM (0.778 AUC)",
            "De-Jargonized Reason Codes",
        ],
        icon="shield-halved",
    )

    global_imp, shap_summary, metrics = load_explanation_artifacts()

    # Section 1: Benchmark Snapshot
    c1, c2, c3 = st.columns(3)
    with c1:
        lr = metrics.get("logistic_regression", {})
        create_metric_card(
            "Interpretable Baseline (Logistic)",
            f"{lr.get('roc_auc', 0.759):.3f} AUC",
            f"Brier: {lr.get('brier_score', 0.199):.3f}",
            "neutral",
            icon="compass",
        )
    with c2:
        lgb = metrics.get("lightgbm_shadow", {})
        create_metric_card(
            "Shadow Ceiling (LightGBM)",
            f"{lgb.get('roc_auc', 0.778):.3f} AUC",
            f"Brier: {lgb.get('brier_score', 0.174):.3f}",
            "neutral",
            icon="layer-group",
        )
    with c3:
        ebm = metrics.get("ebm", {})
        create_metric_card(
            "Production Model (Calibrated EBM)",
            f"{ebm.get('roc_auc', 0.764):.3f} AUC",
            f"Brier: {ebm.get('brier_score', 0.068):.3f} (Calibrated)",
            "positive",
            icon="shield-halved",
        )

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Section 2: Global EBM Term Importance vs LightGBM SHAP (Distinct Color Palettes)
    col_global, col_shap = st.columns(2)

    with col_global:
        st.markdown(
            f"""
            <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px; display:flex; align-items:center; gap:6px;">
                {get_svg_icon("chart-column", "#38bdf8", 16)} <span>Production EBM Global Term Importance</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        top_ebm = global_imp.head(15).sort_values("importance", ascending=True)
        top_ebm["display_name"] = [get_feature_label(t) for t in top_ebm["term"]]
        fig_ebm = go.Figure()
        fig_ebm.add_trace(
            go.Bar(
                x=top_ebm["importance"],
                y=top_ebm["display_name"],
                orientation="h",
                marker_color="#38bdf8",
                text=[f"{val:.3f}" for val in top_ebm["importance"]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Mean Score Contribution: %{x:.3f}<extra></extra>",
            )
        )
        fig_ebm.update_layout(xaxis_title="Mean Absolute Score Contribution", yaxis_title="")
        configure_plotly_chart(fig_ebm, height=440, margin_l=195, margin_r=20, margin_t=10, margin_b=45)
        st.plotly_chart(fig_ebm, width="stretch")

    with col_shap:
        st.markdown(
            f"""
            <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px; display:flex; align-items:center; gap:6px;">
                {get_svg_icon("network-wired", "#a855f7", 16)} <span>Shadow LightGBM Sampled Tree SHAP Summary</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        top_shap = shap_summary.head(15).sort_values("mean_abs_shap", ascending=True)
        top_shap["display_name"] = [get_feature_label(f) for f in top_shap["feature"]]
        fig_shap = go.Figure()
        fig_shap.add_trace(
            go.Bar(
                x=top_shap["mean_abs_shap"],
                y=top_shap["display_name"],
                orientation="h",
                marker_color="#a855f7",
                text=[f"{val:.3f}" for val in top_shap["mean_abs_shap"]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Mean |SHAP Value|: %{x:.3f}<extra></extra>",
            )
        )
        fig_shap.update_layout(xaxis_title="Mean |SHAP Value|", yaxis_title="")
        configure_plotly_chart(fig_shap, height=440, margin_l=195, margin_r=20, margin_t=10, margin_b=45)
        st.plotly_chart(fig_shap, width="stretch")

    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)

    # Section 3: Applicant-Level Local Adverse Action Explanation
    st.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; display:flex; align-items:center; gap:8px;">
                {get_svg_icon("list-check", "#38bdf8", 18)} <span>Local Adverse-Action Explanation (Current Scored Applicant)</span>
            </div>
            <div style="font-size: 0.83rem; color: #94a3b8; margin-top: 2px;">
                Exact additive score contributions show how model inputs move predicted log-odds; compliance teams must validate any customer-facing reason codes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    from src.ui.prediction import DEFAULT_PROFILES, build_applicant_row, load_predictor
    predictor = load_predictor()

    if "scored_applicant" not in st.session_state:
        default_profile_name = list(DEFAULT_PROFILES.keys())[0]
        st.session_state["scored_applicant"] = build_applicant_row(DEFAULT_PROFILES[default_profile_name], {})

    applicant_df = st.session_state["scored_applicant"]

    try:
        local_exp = predictor.local_explanation(applicant_df).data(0)
        names = local_exp["names"]
        scores = local_exp["scores"]
        values = local_exp.get("values", ["" for _ in names])

        def _clean_val(v):
            try:
                f = float(v)
                if abs(f) < 1.0:
                    return f"{f:.3f}"
                return f"{f:,.1f}"
            except (ValueError, TypeError):
                return str(v)

        contrib_df = pd.DataFrame({
            "term": names,
            "score": scores,
            "value": [_clean_val(v) for v in values],
        })
        contrib_df["display_name"] = [get_feature_label(t) for t in contrib_df["term"]]
        contrib_df["abs_score"] = contrib_df["score"].abs()
        top_contrib = contrib_df.sort_values("abs_score", ascending=False).head(12).sort_values("score", ascending=True)

        colors = ["#f43f5e" if s > 0 else "#10b981" for s in top_contrib["score"]]
        fig_local = go.Figure(
            go.Bar(
                x=top_contrib["score"],
                y=[f"{t} ({v})" for t, v in zip(top_contrib["display_name"], top_contrib["value"])],
                orientation="h",
                marker_color=colors,
                text=[f"{s:+.2f}" for s in top_contrib["score"]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Score Impact: %{x:+.3f}<extra></extra>",
            )
        )
        fig_local.update_layout(
            xaxis_title="Additive Score Impact (Red = Risk Escalators, Green = Risk Mitigators)",
            yaxis_title="",
        )
        configure_plotly_chart(fig_local, height=400, margin_l=230, margin_r=25, margin_t=10, margin_b=45)
        st.plotly_chart(fig_local, width="stretch")

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

        # Plain-English Adverse Action Factors
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"""
                <div style="font-size: 0.92rem; font-weight: 700; color: #f43f5e; margin-bottom: 10px; display:flex; align-items:center; gap:6px;">
                    {get_svg_icon("trend-up", "#f43f5e", 15)} <span>Top Risk Escalators (Adverse-Action Factors)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            risk_escalators = contrib_df[contrib_df["score"] > 0].sort_values("score", ascending=False).head(3)
            if not risk_escalators.empty:
                for _, r in risk_escalators.iterrows():
                    human_term = get_feature_label(r['term'])
                    score_pts = int(round(r['score'] * 100))
                    st.markdown(
                        f"""
                        <div class="insight-card" style="border-left-color: #f43f5e; padding: 12px 16px; margin-bottom: 8px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                                <span style="font-weight:700; color:#f8fafc; font-size:0.9rem;">{human_term}</span>
                                <span style="font-size:0.75rem; font-weight:700; background:rgba(244,63,94,0.2); color:#f43f5e; padding:3px 10px; border-radius:6px; border:1px solid rgba(244,63,94,0.4);">+{score_pts} PTS RISK</span>
                            </div>
                            <div style="font-size:0.82rem; color:#cbd5e1;">Observed Value: <code>{r['value']}</code> &bull; Directionally increases default probability.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<div class="insight-card" style="padding: 12px 16px; color: #94a3b8;">No risk escalators identified.</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown(
                f"""
                <div style="font-size: 0.92rem; font-weight: 700; color: #10b981; margin-bottom: 10px; display:flex; align-items:center; gap:6px;">
                    {get_svg_icon("trend-down", "#10b981", 15)} <span>Top Risk Mitigators (Protective Factors)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            risk_mitigators = contrib_df[contrib_df["score"] < 0].sort_values("score", ascending=True).head(3)
            if not risk_mitigators.empty:
                for _, r in risk_mitigators.iterrows():
                    human_term = get_feature_label(r['term'])
                    score_pts = int(round(abs(r['score']) * 100))
                    st.markdown(
                        f"""
                        <div class="insight-card" style="border-left-color: #10b981; padding: 12px 16px; margin-bottom: 8px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                                <span style="font-weight:700; color:#f8fafc; font-size:0.9rem;">{human_term}</span>
                                <span style="font-size:0.75rem; font-weight:700; background:rgba(16,185,129,0.2); color:#10b981; padding:3px 10px; border-radius:6px; border:1px solid rgba(16,185,129,0.4);">-{score_pts} PTS RISK</span>
                            </div>
                            <div style="font-size:0.82rem; color:#cbd5e1;">Observed Value: <code>{r['value']}</code> &bull; Protective credit factor mitigating default likelihood.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<div class="insight-card" style="padding: 12px 16px; color: #94a3b8;">No risk mitigators identified.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Could not compute live local explanation: {e}")
