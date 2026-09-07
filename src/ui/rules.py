from __future__ import annotations

import json
import streamlit as st

from src.ui.styles import (
    create_metric_card,
    get_svg_icon,
    render_executive_briefing,
)
from src.utils.config import get_settings


@st.cache_data
def load_rules_data():
    settings = get_settings()
    rules_file = settings.artifact_dir / "rules" / "business_rules.json"
    return json.loads(rules_file.read_text(encoding="utf-8"))


# Curated structured representation of the 11 surrogate decision paths
DECISION_PATHS = [
    {
        "tier": "Prime",
        "prob": 0.0244,
        "action": "Lower-Risk Segment",
        "description": "High credit bureau scores across multiple bureaus; stellar creditworthiness.",
        "steps": [
            ("Bureau Score C (Top)", "> 0.536", "#34d399"),
            ("Bureau Score B", "> 0.618", "#34d399"),
        ],
    },
    {
        "tier": "Prime",
        "prob": 0.0400,
        "action": "Lower-Risk Segment",
        "description": "High Bureau C score with moderate-to-high Bureau B score; low historical default.",
        "steps": [
            ("Bureau Score C (Top)", "> 0.536", "#34d399"),
            ("Bureau Score B", "0.399 – 0.618", "#38bdf8"),
        ],
    },
    {
        "tier": "Prime",
        "prob": 0.0472,
        "action": "Lower-Risk Segment",
        "description": "Moderate Bureau C score backed by very strong Bureau B score.",
        "steps": [
            ("Bureau Score C (Top)", "0.316 – 0.536", "#38bdf8"),
            ("Bureau Score B", "> 0.616", "#34d399"),
        ],
    },
    {
        "tier": "Moderate",
        "prob": 0.0609,
        "action": "Review Affordability Signals",
        "description": "High Bureau C, low Bureau B, but clean repayment history without late payments.",
        "steps": [
            ("Bureau Score C (Top)", "> 0.545", "#34d399"),
            ("Bureau Score B", "<= 0.399", "#fbbf24"),
            ("Late Installment Rate", "<= 7.5%", "#34d399"),
        ],
    },
    {
        "tier": "Moderate",
        "prob": 0.0735,
        "action": "Senior Underwriter Review",
        "description": "Weak Bureau C compensated by solid Bureau B and high primary Bureau A score.",
        "steps": [
            ("Bureau Score C (Top)", "<= 0.316", "#f43f5e"),
            ("Bureau Score B", "> 0.417", "#38bdf8"),
            ("Bureau Score A", "> 0.531", "#34d399"),
        ],
    },
    {
        "tier": "Moderate",
        "prob": 0.0815,
        "action": "Standard Review",
        "description": "Moderate scores across both Bureau C and Bureau B near the portfolio median.",
        "steps": [
            ("Bureau Score C (Top)", "0.316 – 0.536", "#38bdf8"),
            ("Bureau Score B", "0.399 – 0.616", "#38bdf8"),
        ],
    },
    {
        "tier": "Moderate",
        "prob": 0.1019,
        "action": "Behavioural Override / Extra Verification",
        "description": "Good Bureau C score, but late installment payments trigger an elevated risk flag.",
        "steps": [
            ("Bureau Score C (Top)", "> 0.545", "#34d399"),
            ("Bureau Score B", "<= 0.399", "#fbbf24"),
            ("Late Installment Rate", "> 7.5% Delinquent", "#f43f5e"),
        ],
    },
    {
        "tier": "High Distress",
        "prob": 0.1247,
        "action": "Elevated-Risk Review",
        "description": "Weak Bureau C and Bureau A scores; elevated risk requiring co-signers.",
        "steps": [
            ("Bureau Score C (Top)", "0.183 – 0.316", "#fbbf24"),
            ("Bureau Score B", "> 0.417", "#38bdf8"),
            ("Bureau Score A", "<= 0.531", "#f43f5e"),
        ],
    },
    {
        "tier": "High Distress",
        "prob": 0.1900,
        "action": "High-Risk Manual Review",
        "description": "Critical low Bureau C and weak Bureau A; default probability exceeds 2x baseline.",
        "steps": [
            ("Bureau Score C (Top)", "<= 0.183", "#f43f5e"),
            ("Bureau Score B", "> 0.417", "#38bdf8"),
            ("Bureau Score A", "<= 0.531", "#f43f5e"),
        ],
    },
    {
        "tier": "High Distress",
        "prob": 0.2116,
        "action": "High-Risk Manual Review",
        "description": "Low Bureau C coupled with low Bureau B; high probability of default.",
        "steps": [
            ("Bureau Score C (Top)", "<= 0.316", "#f43f5e"),
            ("Bureau Score B", "0.225 – 0.417", "#f43f5e"),
        ],
    },
    {
        "tier": "High Distress",
        "prob": 0.2873,
        "action": "Highest-Risk Manual Review",
        "description": "Severe subprime double-dip: both Bureau C and Bureau B in lowest deciles (3.5x baseline).",
        "steps": [
            ("Bureau Score C (Top)", "<= 0.316", "#f43f5e"),
            ("Bureau Score B", "<= 0.225", "#f43f5e"),
        ],
    },
]


def render_rules_tab():
    # Executive Briefing Callout
    render_executive_briefing(
        title="Surrogate Decision Heuristics & Governance Safeguards",
        description=(
            "This module translates the production Explainable Boosting Machine into shallow, human-auditable decision rules. "
            "A Depth-4 surrogate decision tree distills the complex ensemble into transparent explanatory thresholds for model reviewers."
        ),
        takeaways=[
            "Surrogate Fidelity: R² = 0.525",
            "Risk Band Agreement: 68.0%",
            "Safeguard: Age-Derived Fields Excluded",
            "Auditable Decision Thresholds",
        ],
        icon="scale",
    )

    data = load_rules_data()

    # Top Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        create_metric_card(
            "Surrogate Validation R²",
            f"{data.get('validation_r2', 0.525):.3f}",
            "Score Variance Explained",
            "neutral",
            icon="math",
        )
    with c2:
        create_metric_card(
            "Risk-Band Agreement",
            f"{data.get('risk_band_agreement', 0.680)*100:.1f}%",
            "Concordance with EBM Bands",
            "positive",
            icon="handshake",
        )
    with c3:
        excluded = len(data.get("excluded_policy_sensitive_features", []))
        create_metric_card(
            "Policy-Sensitive Features",
            f"{excluded} Purged",
            "Excluded from Surrogate",
            "positive",
            icon="shield-halved",
        )
    with c4:
        create_metric_card(
            "Surrogate Tree Depth",
            "Depth 4",
            "Transparent Architecture",
            "neutral",
            icon="sitemap",
        )

    # Regulatory Warning Box
    st.markdown(
        f"""
        <div class="insight-card" style="border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.08); margin-bottom: 24px;">
            <div style="color: #fbbf24; font-weight: 700; font-size: 0.9rem; margin-bottom: 4px; display:flex; align-items:center; gap:6px;">
                {get_svg_icon("warning", "#fbbf24", 16)} <span>REGULATORY COMPLIANCE NOTICE</span>
            </div>
            <div style="color: #cbd5e1; font-size: 0.83rem; line-height: 1.6;">
                {data.get('disclaimer', 'Rules approximate EBM scores for explanation only.')}
                These decision paths represent shallow surrogate approximations to aid credit underwriters in understanding general risk thresholds.
                They <b>MUST NOT</b> be utilized as automated adverse-action rules or substitute for holistic credit assessment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual Rule Flow Section
    st.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; display:flex; align-items:center; gap:8px;">
                {get_svg_icon("sitemap", "#38bdf8", 20)} <span>Visual Underwriting Decision Flow (11 Auditable Paths)</span>
            </div>
            <div style="font-size: 0.83rem; color: #94a3b8; margin-top: 2px;">
                Step-by-step decision rules derived from the surrogate decision tree mapping applicant criteria to expected default rates.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Filter for Rules
    rule_filter = st.radio(
        "Filter Decision Paths:",
        ["All Paths (11)", "Prime / Low Risk (<5%)", "Moderate / Review (5% – 12%)", "High Distress (>12%)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    filtered_paths = DECISION_PATHS
    if "Prime" in rule_filter:
        filtered_paths = [p for p in DECISION_PATHS if p["prob"] < 0.05]
    elif "Moderate" in rule_filter:
        filtered_paths = [p for p in DECISION_PATHS if 0.05 <= p["prob"] <= 0.12]
    elif "High" in rule_filter:
        filtered_paths = [p for p in DECISION_PATHS if p["prob"] > 0.12]

    # Render Visual Rule Cards
    for i, path in enumerate(filtered_paths):
        tier = path["tier"]
        prob = path["prob"]
        action = path["action"]
        desc = path["description"]

        if tier == "Prime":
            badge_class = "badge-low"
            border_color = "#10b981"
            icon_svg = get_svg_icon("shield-check", "#34d399", 13)
        elif tier == "Moderate":
            badge_class = "badge-medium"
            border_color = "#f59e0b"
            icon_svg = get_svg_icon("warning", "#fbbf24", 13)
        else:
            badge_class = "badge-high"
            border_color = "#f43f5e"
            icon_svg = get_svg_icon("xmark", "#f43f5e", 13)

        steps_html = ""
        for j, (cond_name, cond_val, cond_col) in enumerate(path["steps"]):
            arrow = f'<span style="color:#64748b; margin: 0 4px;">&rarr;</span>' if j > 0 else ""
            steps_html += f'{arrow}<span class="rule-step"><span>{cond_name}:</span> <code style="color:{cond_col} !important;">{cond_val}</code></span>'

        card_html = f"""
        <div class="rule-card" style="border-left: 4px solid {border_color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="badge {badge_class}" style="padding: 4px 10px; font-size: 0.76rem;">{icon_svg} {tier}</span>
                    <span style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">Rule #{i+1}: {action}</span>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.76rem; color: #94a3b8;">Surrogate Default Rate:</span>
                    <span style="font-size: 1.15rem; font-weight: 800; color: {border_color}; margin-left: 6px;">{prob*100:.1f}%</span>
                </div>
            </div>
            <div style="margin: 10px 0 8px 0; display: flex; flex-wrap: wrap; align-items: center;">
                {steps_html}
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">{desc}</div>
        </div>
        """
        if hasattr(st, "html"):
            st.html(card_html)
        else:
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Collapsible Technical Specification for Compliance Auditors
    with st.expander("Technical Tree Hierarchy & Specification (For Compliance Auditors & Engineers)"):
        st.caption("Raw ASCII decision-tree representation generated by scikit-learn DecisionTreeClassifier(max_depth=4).")
        raw_rules = data.get("rules", "")
        st.code(raw_rules, language="text")
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 10px;">
                <b>Excluded Protected Attributes:</b> <code>{', '.join(data.get('excluded_policy_sensitive_features', []))}</code><br>
                <b>Candidate Features:</b> {len(data.get('features', []))} operational features considered.
            </div>
            """,
            unsafe_allow_html=True,
        )
