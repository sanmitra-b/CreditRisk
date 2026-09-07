from __future__ import annotations

import streamlit as st

from src.db.connection import create_database_engine, database_is_ready
from src.ui.chatbot import render_chatbot_tab
from src.ui.eda import render_eda_tab
from src.ui.explainability import render_explainability_tab
from src.ui.prediction import render_prediction_tab
from src.ui.rules import render_rules_tab
from src.ui.styles import apply_theme, get_svg_icon

# Wide layout configuration
st.set_page_config(
    page_title="NeoStats | Credit Risk Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_db_engine():
    return create_database_engine(read_only=True)


def main():
    apply_theme()
    engine = get_db_engine()
    db_ok = database_is_ready(engine)

    # Sidebar
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
                {get_svg_icon("building-columns", "#38bdf8", 24)}
                <span style="font-size: 1.3rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">NeoStats Credit</span>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 16px;">Enterprise Credit Risk Intelligence</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Database Status Badge
        if db_ok:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px; background: rgba(16, 185, 129, 0.1); padding:8px 12px; border-radius:10px; border:1px solid rgba(16, 185, 129, 0.3); backdrop-filter: blur(12px);">
                    {get_svg_icon("circle", "#34d399", 10, "filter: drop-shadow(0 0 6px #34d399);")}
                    <span style="font-size: 0.82rem; color: #e2e8f0; font-weight: 500;">PostgreSQL Analytics Online</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px; background: rgba(244, 63, 94, 0.1); padding:8px 12px; border-radius:10px; border:1px solid rgba(244, 63, 94, 0.3); backdrop-filter: blur(12px);">
                    {get_svg_icon("circle", "#f43f5e", 10)}
                    <span style="font-size: 0.82rem; color: #f43f5e; font-weight: 500;">PostgreSQL Analytics Offline</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 8px;">
                Platform Architecture
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.8;">
                <div style="display:flex; align-items:center; gap:8px;">{get_svg_icon("shield-halved", "#38bdf8", 14)} <span><b>Production</b>: Calibrated EBM</span></div>
                <div style="margin-left: 22px; color:#94a3b8; font-size:0.78rem;">ROC-AUC: 0.764 | Brier: 0.068</div>
                <div style="display:flex; align-items:center; gap:8px;">{get_svg_icon("layer-group", "#94a3b8", 14)} <span><b>Shadow</b>: LightGBM (0.778)</span></div>
                <div style="display:flex; align-items:center; gap:8px;">{get_svg_icon("compass", "#94a3b8", 14)} <span><b>Baseline</b>: Logistic (0.759)</span></div>
                <div style="display:flex; align-items:center; gap:8px;">{get_svg_icon("database", "#10b981", 14)} <span><b>Storage</b>: PostgreSQL 17</span></div>
                <div style="display:flex; align-items:center; gap:8px;">{get_svg_icon("robot", "#f59e0b", 14)} <span><b>Chat</b>: LangGraph + Gemini</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("NeoStats AI Labs • Enterprise Edition")

    # Main Glassmorphic Header Banner
    st.markdown(
        f"""
        <div class="metric-card" style="padding: 24px 32px; margin-bottom: 28px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 14px; width: 54px; height: 54px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 24px rgba(56, 189, 248, 0.25);">
                    {get_svg_icon("chart-line", "#38bdf8", 28)}
                </div>
                <div>
                    <div style="font-size: 1.85rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; line-height: 1.25;">Credit Risk Intelligence Platform</div>
                    <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 4px;">Auditable Underwriting, Glass-Box Explainability, and Multi-Tool Agentic Analytics</div>
                </div>
            </div>
            <div style="display: flex; gap: 12px; align-items: center;">
                <span class="badge badge-low" style="display:inline-flex; align-items:center; gap:6px; padding: 7px 16px;">{get_svg_icon("shield-halved", "#34d399", 14)} Model: EBM (Explainable Boosting Machine)</span>
                <span class="badge" style="display:inline-flex; align-items:center; gap:6px; padding: 7px 16px; background: rgba(56, 189, 248, 0.14); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); box-shadow: 0 0 14px rgba(56, 189, 248, 0.2);">{get_svg_icon("database", "#38bdf8", 14)} DB: PostgreSQL served from Supabase</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Five Tabs (Clean, institutional names without emojis)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Agentic Assistant",
        "Portfolio EDA",
        "Risk Prediction",
        "Explainability",
        "Surrogate Rules",
    ])

    with tab1:
        render_chatbot_tab()

    with tab2:
        render_eda_tab(engine)

    with tab3:
        render_prediction_tab()

    with tab4:
        render_explainability_tab()

    with tab5:
        render_rules_tab()


if __name__ == "__main__":
    main()
