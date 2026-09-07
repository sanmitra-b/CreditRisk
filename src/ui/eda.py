from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import Engine, text

from src.ui.styles import (
    configure_plotly_chart,
    create_metric_card,
    get_svg_icon,
    render_executive_briefing,
)


@st.cache_data
def load_eda_data(_engine: Engine):
    with _engine.connect() as conn:
        portfolio = pd.read_sql_query(text("SELECT * FROM analytics.portfolio_summary"), conn)
        age_bands = pd.read_sql_query(text("SELECT * FROM analytics.age_band_summary ORDER BY age_band"), conn)
        housing = pd.read_sql_query(text("SELECT * FROM analytics.housing_summary ORDER BY default_rate DESC"), conn)
        education = pd.read_sql_query(text("SELECT * FROM analytics.education_summary ORDER BY default_rate DESC"), conn)
        repayment = pd.read_sql_query(text("SELECT * FROM analytics.repayment_history_summary"), conn)
    return portfolio, age_bands, housing, education, repayment


def render_eda_tab(engine: Engine):
    portfolio, age_bands, housing, education, repayment = load_eda_data(engine)
    p_row = portfolio.iloc[0] if not portfolio.empty else {"applicants": 307511, "default_rate": 0.0807, "average_credit": 599026, "average_income": 168798}
    benchmark_rate = float(p_row["default_rate"]) * 100

    # Executive Briefing Callout
    render_executive_briefing(
        title="Portfolio Risk Exploration & Baseline Macro Profiling",
        description=(
            "This module establishes the macroeconomic credit foundation across 307,511 retail applicants. "
            "Because non-defaults outnumber defaults by 11.4:1 (8.07% baseline), raw accuracy is deceptive—underwriting teams "
            "must isolate structural segment lift (e.g. rented apartments at 1.53x baseline), repayment distress transitions, and "
            "affordability ratios to detect systemic risk before deploying machine learning models."
        ),
        takeaways=[
            "Portfolio Baseline: 8.07%",
            "Severe Imbalance: 11.4:1",
            "Highest Lift: Rented (1.53x)",
            "Strongest Signal: Bureau History",
        ],
        icon="chart-line",
    )

    # Main Hero Section
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom: 16px;">
            <div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em; display:flex; align-items:center; gap:8px;">
                    {get_svg_icon("sliders", "#38bdf8", 20)} <span>Primary Risk Dimensions & Benchmark Trajectory</span>
                </div>
                <div style="font-size: 0.83rem; color: #94a3b8; margin-top: 2px;">
                    Select cross-sectional risk dimensions to inspect distribution trajectories against the 8.07% portfolio benchmark.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_control, col_chart = st.columns([3.4, 8.6])

    with col_control:
        st.markdown(
            f"""
            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 8px; display:flex; align-items:center; gap:6px;">
                {get_svg_icon("sliders", "#38bdf8", 14)} <span>Filter Risk Dimension</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_view = st.selectbox(
            "Primary Risk Dimension",
            ["Age Cohorts (20 to 60+)", "Housing Segment Lift", "Repayment History Distress"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        create_metric_card(
            label="Total Portfolio",
            value=f"{int(p_row['applicants']):,}",
            delta="11.4:1 Non-Default Ratio",
            delta_type="neutral",
            icon="users",
        )

        create_metric_card(
            label="Portfolio Default Rate",
            value=f"{p_row['default_rate']*100:.2f}%",
            delta="Benchmark Baseline",
            delta_type="warning",
            icon="percent",
        )

        # Mini comparative cards
        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 6px;">
                <div class="metric-card" style="padding: 14px; margin-bottom: 0;">
                    <div class="metric-label" style="font-size: 0.68rem;">Highest Risk Cohort</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #f43f5e; margin: 3px 0;">20–29</div>
                    <div style="font-size: 0.75rem; color: #f43f5e; font-weight: 600; display:flex; align-items:center; gap:4px;">
                        {get_svg_icon("trend-up", "#f43f5e", 12)} <span>11.44% Default</span>
                    </div>
                </div>
                <div class="metric-card" style="padding: 14px; margin-bottom: 0;">
                    <div class="metric-label" style="font-size: 0.68rem;">Lowest Risk Cohort</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #10b981; margin: 3px 0;">60+</div>
                    <div style="font-size: 0.75rem; color: #10b981; font-weight: 600; display:flex; align-items:center; gap:4px;">
                        {get_svg_icon("trend-down", "#10b981", 12)} <span>4.92% Default</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_chart:
        if "Age" in metric_view:
            fig_main = go.Figure()

            fig_main.add_trace(
                go.Scatter(
                    x=age_bands["age_band"],
                    y=age_bands["default_rate"] * 100,
                    mode="lines+markers",
                    name="Observed Default Rate",
                    line=dict(color="#38bdf8", width=3, shape="spline", smoothing=1.1),
                    marker=dict(size=9, color="#38bdf8", symbol="circle"),
                    fill="tozeroy",
                    fillcolor="rgba(56, 189, 248, 0.08)",
                    hovertemplate="<b>Age: %{x}</b><br>Default Rate: %{y:.2f}%<extra></extra>",
                )
            )

            fig_main.add_hline(
                y=benchmark_rate,
                line_dash="dash",
                line_color="#f59e0b",
                line_width=1.5,
                annotation_text=f"Portfolio Baseline: {benchmark_rate:.2f}%",
                annotation_position="top right",
                annotation_font=dict(color="#f59e0b", size=11),
            )

            fig_main.update_layout(
                yaxis_title="Default Rate (%)",
                xaxis_title="Age Cohort",
                yaxis=dict(ticksuffix="%"),
            )
            configure_plotly_chart(fig_main, title="Age Cohort Default Trajectory (Monotonic Decrease from 20 to 60+)", height=390)
            st.plotly_chart(fig_main, width="stretch")

        elif "Housing" in metric_view:
            housing["lift"] = housing["default_rate"] / float(p_row["default_rate"])
            fig_house = go.Figure()
            fig_house.add_trace(
                go.Bar(
                    x=housing["housing_type"],
                    y=housing["default_rate"] * 100,
                    name="Default Rate (%)",
                    marker_color=["#f43f5e" if "Rented" in h else "#38bdf8" for h in housing["housing_type"]],
                    text=[f"{v*100:.1f}%" for v in housing["default_rate"]],
                    textposition="auto",
                )
            )
            fig_house.add_hline(y=benchmark_rate, line_dash="dash", line_color="#f59e0b", annotation_text=f"Baseline: {benchmark_rate:.1f}%")
            fig_house.update_layout(yaxis_title="Default Rate (%)", yaxis=dict(ticksuffix="%"))
            configure_plotly_chart(fig_house, title="Housing Type Default Rate (Rented Apartment 1.53x Lift)", height=390)
            st.plotly_chart(fig_house, width="stretch")

        else:
            fig_rep = px.bar(
                repayment,
                x="signal",
                y="default_rate",
                color="segment",
                barmode="group",
                text=[f"{v*100:.1f}%" for v in repayment["default_rate"]],
                color_discrete_map={"Observed": "#f43f5e", "Not observed": "#10b981"},
            )
            fig_rep.update_layout(yaxis_title="Observed Default Rate", yaxis=dict(ticksuffix=""))
            configure_plotly_chart(fig_rep, title="Repayment Distress Lift by Historical Behaviour", height=390)
            st.plotly_chart(fig_rep, width="stretch")

    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)

    # Spacious 2x2 Segment Analysis Grid
    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em; display:flex; align-items:center; gap:8px;">
                {get_svg_icon("table-cells", "#38bdf8", 20)} <span>Segment Analysis vs Portfolio Baseline (2 × 2 Deep Dive)</span>
            </div>
            <div style="font-size: 0.83rem; color: #94a3b8; margin-top: 2px;">
                Detailed sub-segment cross-tabulations isolating risk concentrations relative to the 8.07% portfolio benchmark.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    housing_map = {
        "House / apartment": "House / Apt",
        "Rented apartment": "Rented Apt",
        "With parents": "With Parents",
        "Municipal apartment": "Municipal Apt",
        "Office apartment": "Office Apt",
        "Co-op apartment": "Co-op Apt",
    }
    edu_map = {
        "Secondary / secondary special": "Secondary",
        "Higher education": "Higher Education",
        "Incomplete higher": "Incomplete Higher",
        "Lower secondary": "Lower Secondary",
        "Academic degree": "Academic Degree",
    }

    # Row 1 of 2x2 Grid
    r1_c1, r1_c2 = st.columns(2)

    with r1_c1:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">Housing Type vs Baseline</div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 10px;">Observed default rate across residential property ownership categories.</div>
                """,
                unsafe_allow_html=True,
            )
            h_labels = [housing_map.get(h, h) for h in housing["housing_type"]]
            fig_c1 = go.Figure()
            fig_c1.add_trace(go.Bar(
                x=h_labels,
                y=housing["default_rate"] * 100,
                marker_color="#f43f5e",
                name="Housing Rate",
                hovertemplate="<b>%{x}</b>: %{y:.2f}%<extra></extra>",
                text=[f"{v*100:.1f}%" for v in housing["default_rate"]],
                textposition="auto",
            ))
            fig_c1.add_hline(y=benchmark_rate, line_dash="dash", line_color="#f59e0b", line_width=1.5, annotation_text="Baseline (8.07%)")
            fig_c1.update_layout(showlegend=False, yaxis=dict(ticksuffix="%"), xaxis=dict(tickangle=-15, tickfont=dict(size=10, color="#94a3b8")))
            configure_plotly_chart(fig_c1, height=250, margin_l=40, margin_r=15, margin_t=15, margin_b=45)
            st.plotly_chart(fig_c1, width="stretch")

    with r1_c2:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">Housing Lift Delta (%)</div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 10px;">Net spread above or below the 8.07% portfolio benchmark in percentage points.</div>
                """,
                unsafe_allow_html=True,
            )
            housing["delta"] = (housing["default_rate"] * 100) - benchmark_rate
            fig_c2 = go.Figure()
            fig_c2.add_trace(go.Scatter(
                x=h_labels,
                y=housing["delta"],
                mode="lines+markers",
                line=dict(color="#38bdf8", width=2.5),
                marker=dict(size=7, color="#38bdf8"),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.2)",
                name="Delta",
                hovertemplate="<b>%{x}</b>: %{y:+.2f} pp<extra></extra>",
            ))
            fig_c2.add_hline(y=0, line_color="#64748b", line_width=1.5)
            fig_c2.update_layout(showlegend=False, yaxis=dict(ticksuffix=" pp"), xaxis=dict(tickangle=-15, tickfont=dict(size=10, color="#94a3b8")))
            configure_plotly_chart(fig_c2, height=250, margin_l=45, margin_r=15, margin_t=15, margin_b=45)
            st.plotly_chart(fig_c2, width="stretch")

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    # Row 2 of 2x2 Grid
    r2_c1, r2_c2 = st.columns(2)

    with r2_c1:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">Repayment Behaviour Distress Lift</div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 10px;">Default rate for applicants with observed delinquency vs clean history.</div>
                """,
                unsafe_allow_html=True,
            )
            rep_obs = repayment[repayment["segment"] == "Observed"]
            fig_c3 = go.Figure()
            fig_c3.add_trace(go.Bar(
                x=["Bureau Overdue", "Previous Loan Refusal", "Installment Delinquency"],
                y=rep_obs["default_rate"] * 100,
                marker_color="#f59e0b",
                name="Observed Distress",
                hovertemplate="<b>%{x}</b>: %{y:.2f}%<extra></extra>",
                text=[f"{v*100:.1f}%" for v in rep_obs["default_rate"]],
                textposition="auto",
            ))
            fig_c3.add_hline(y=benchmark_rate, line_dash="dash", line_color="#64748b", line_width=1.5, annotation_text="Baseline (8.07%)")
            fig_c3.update_layout(showlegend=False, yaxis=dict(ticksuffix="%"), xaxis=dict(tickangle=0, tickfont=dict(size=10, color="#94a3b8")))
            configure_plotly_chart(fig_c3, height=250, margin_l=40, margin_r=15, margin_t=15, margin_b=45)
            st.plotly_chart(fig_c3, width="stretch")

    with r2_c2:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">Education Qualification Risk Spread</div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 10px;">Monotonic relationship between formal educational tier and default probability.</div>
                """,
                unsafe_allow_html=True,
            )
            e_labels = [edu_map.get(e, e) for e in education["education_level"]]
            fig_c4 = go.Figure()
            fig_c4.add_trace(go.Scatter(
                x=e_labels,
                y=education["default_rate"] * 100,
                mode="lines+markers",
                line=dict(color="#10b981", width=2.5),
                marker=dict(size=8, color="#10b981"),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.15)",
                name="Education Rate",
                hovertemplate="<b>%{x}</b>: %{y:.2f}%<extra></extra>",
            ))
            fig_c4.add_hline(y=benchmark_rate, line_dash="dash", line_color="#64748b", line_width=1.5, annotation_text="Baseline (8.07%)")
            fig_c4.update_layout(showlegend=False, yaxis=dict(ticksuffix="%"), xaxis=dict(tickangle=-15, tickfont=dict(size=10, color="#94a3b8")))
            configure_plotly_chart(fig_c4, height=250, margin_l=40, margin_r=15, margin_t=15, margin_b=45)
            st.plotly_chart(fig_c4, width="stretch")

    st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)

    # Confirmed Canonical EDA Findings (Natural Left-to-Right Row Reading Order: 1, 2, 3 then 4, 5, 6)
    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em; display:flex; align-items:center; gap:8px;">
                {get_svg_icon("bookmark", "#38bdf8", 20)} <span>Confirmed Canonical EDA Insights</span>
            </div>
            <div style="font-size: 0.83rem; color: #94a3b8; margin-top: 2px;">
                Validated empirical principles governing loan underwriting and portfolio governance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 1: Insights 1, 2, 3
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1:
        st.markdown(
            f"""
            <div class="insight-card">
                <b>{get_svg_icon("scale", "#38bdf8", 15, "margin-right:6px;")} 1. Severe Class Imbalance (8.07%)</b><br>
                11.4:1 ratio across 307,511 applicants. Evaluation must strictly prioritize ROC-AUC, PR-AUC, and Brier calibration over raw accuracy metrics.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with row1_c2:
        st.markdown(
            f"""
            <div class="insight-card" style="border-left-color: #f43f5e;">
                <b>{get_svg_icon("user-group", "#f43f5e", 15, "margin-right:6px;")} 2. Youth Risk Concentration (11.4%)</b><br>
                Applicants aged 20–29 display the highest observed default rate (11.44%). This is an association, not a causal conclusion; age is excluded from the surrogate policy rules.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with row1_c3:
        st.markdown(
            f"""
            <div class="insight-card" style="border-left-color: #f59e0b;">
                <b>{get_svg_icon("house", "#f59e0b", 15, "margin-right:6px;")} 3. Rented Housing Segment Lift (1.53x)</b><br>
                Rented apartments exhibit a 12.3% default rate (1.53x portfolio baseline lift), reflecting residential instability and liquidity friction.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Row 2: Insights 4, 5, 6
    row2_c1, row2_c2, row2_c3 = st.columns(3)
    with row2_c1:
        st.markdown(
            f"""
            <div class="insight-card">
                <b>{get_svg_icon("trend-up", "#38bdf8", 15, "margin-right:6px;")} 4. Affordability Superiority</b><br>
                Credit-to-income and annuity-to-income debt service ratios separate distressed applicants far more cleanly than raw credit or income figures.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with row2_c2:
        st.markdown(
            f"""
            <div class="insight-card" style="border-left-color: #f43f5e;">
                <b>{get_svg_icon("star", "#f43f5e", 15, "margin-right:6px;")} 5. External Bureau Dominance</b><br>
                Third-party normalized bureau score (EXT_SOURCE_3) is the single most predictive feature. Missing-value indicators are maintained to capture omission signal.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with row2_c3:
        st.markdown(
            f"""
            <div class="insight-card" style="border-left-color: #10b981;">
                <b>{get_svg_icon("clock", "#10b981", 15, "margin-right:6px;")} 6. Behaviour Over Demographics</b><br>
                Historical bureau overdues (19.8% default) and past loan refusals (11.9% default) provide far stronger risk signals than demographic traits.
            </div>
            """,
            unsafe_allow_html=True,
        )
