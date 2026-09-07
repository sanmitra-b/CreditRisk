from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, text

from src.db.connection import create_database_engine
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)
EMBEDDING_DIMENSIONS = 768
STOP_WORDS = {
    "and", "are", "for", "from", "how", "the", "this", "that", "was",
    "what", "when", "where", "which", "why", "with", "does", "mean",
}


def create_local_embedding(value: str) -> list[float]:
    """Create a deterministic, dependency-free lexical embedding for pgvector search."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    tokens = re.findall(r"[a-z0-9_]+", value.lower())
    expanded = tokens + [part for token in tokens if "_" in token for part in token.split("_")]
    for token in expanded:
        if len(token) < 2 or token in STOP_WORDS:
            continue
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % EMBEDDING_DIMENSIONS
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"

CURATED_KNOWLEDGE_DOCS = [
    {
        "source_type": "eda_findings",
        "source_name": "canonical_eda",
        "heading": "Portfolio Default Rate and Class Imbalance",
        "content": (
            "The overall portfolio default rate is 8.07% (approximately 8.1%), representing an 11.4:1 "
            "majority-to-minority class imbalance across 307,511 applicants. Because of this severe imbalance, "
            "model evaluation must prioritize ROC-AUC, PR-AUC, probability calibration (Brier score), "
            "and risk band stratification rather than raw accuracy."
        ),
        "metadata": {"category": "portfolio_metric", "key_metric": "overall_default_rate", "value": 0.0807},
    },
    {
        "source_type": "eda_findings",
        "source_name": "canonical_eda",
        "heading": "Age Cohort Risk Lift",
        "content": (
            "Applicants aged 20-29 exhibit the highest age-band default rate at 11.4%, compared to the 8.1% portfolio average. "
            "Default risk monotonically decreases with age: 30-39 (9.6%), 40-49 (7.9%), 50-59 (6.3%), and 60+ (5.0%). "
            "This indicates younger borrowers face significantly higher repayment friction, though age must not be used "
            "as an adverse action rule due to fair lending principles."
        ),
        "metadata": {"category": "demographic_risk", "highest_risk_age": "20-29", "rate": 0.114},
    },
    {
        "source_type": "eda_findings",
        "source_name": "canonical_eda",
        "heading": "Housing Type Risk Differentiation",
        "content": (
            "Among demographic segments with adequate statistical support, applicants residing in rented apartments "
            "have the highest observed default rate at 12.3%, yielding an observed lift of 1.53x relative to the "
            "portfolio average (based on 4,881 applicants). House / apartment residents have a baseline default rate of 7.8%."
        ),
        "metadata": {"category": "housing_risk", "top_housing_risk": "Rented apartment", "lift": 1.53},
    },
    {
        "source_type": "eda_findings",
        "source_name": "canonical_eda",
        "heading": "Affordability and Debt Burden Signals",
        "content": (
            "Credit-to-income and annuity-to-income ratios separate repayment outcomes far more effectively than "
            "raw credit or income amounts. High payment-to-income burdens directly correlate with payment distress, "
            "supporting affordability-based engineered features."
        ),
        "metadata": {"category": "affordability", "engineered_features": ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]},
    },
    {
        "source_type": "eda_findings",
        "source_name": "canonical_eda",
        "heading": "External Credit Scores and EXT_SOURCE_3 Importance",
        "content": (
            "EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3 are normalized credit scores from external credit bureaus and third-party data providers. "
            "EXT_SOURCE_3 is the single strongest predictor of loan default across all features in the portfolio. Higher scores indicate lower credit risk. "
            "Explicit missing-value treatment and indicator flags are required to preserve signal when external bureaus lack records."
        ),
        "metadata": {"category": "feature_importance", "top_feature": "EXT_SOURCE_3"},
    },
    {
        "source_type": "eda_findings",
        "source_name": "canonical_eda",
        "heading": "Repayment History Signals",
        "content": (
            "Prior repayment behavior provides the strongest behavioural risk signal. Applicants with overdue credit bureau records show a 19.8% "
            "default rate; applicants with previous loan application refusals show an 11.9% default rate; and applicants with late installment "
            "payments show a 9.2% default rate, compared to clean applicants."
        ),
        "metadata": {"category": "repayment_history", "signals": ["overdue_bureau", "prev_refusal", "late_installment"]},
    },
    {
        "source_type": "model_rationale",
        "source_name": "model_architecture",
        "heading": "Why Explainable Boosting Machine (EBM) Was Selected as Production Model",
        "content": (
            "Explainable Boosting Machine (EBM / GA2M) was selected as the non-negotiable production model because it delivers glass-box "
            "interpretability with tree-level predictive accuracy. Unlike black-box neural networks or XGBoost, EBM calculates exact additive "
            "score contributions for every feature and pairwise interaction. This supports model audit and the documented review of candidate "
            "reason codes; compliance and customer-facing notices still require separate validation. Production Calibrated EBM achieved ROC-AUC of 0.764, PR-AUC of 0.250, and an "
            "outstanding Brier calibration score of 0.068, producing monotonically increasing default rates from Low (3.3%) to Medium (9.8%) "
            "and High (24.3%) risk bands."
        ),
        "metadata": {"category": "model_selection", "model": "EBM", "roc_auc": 0.764, "brier_score": 0.068},
    },
    {
        "source_type": "model_rationale",
        "source_name": "benchmark_models",
        "heading": "Baseline and Shadow Models Comparison",
        "content": (
            "Three models were evaluated on the stratified test split: "
            "1) Logistic Regression: Interpretable baseline floor (ROC-AUC: 0.759, PR-AUC: 0.238, Brier: 0.199). "
            "2) Shadow LightGBM: Shadow ceiling and feature selector (ROC-AUC: 0.778, PR-AUC: 0.265, Brier: 0.174), used to select the top 30 raw features. "
            "3) Calibrated EBM: Production model (ROC-AUC: 0.764, PR-AUC: 0.250, Brier: 0.068), striking the optimal balance between performance and strict governance."
        ),
        "metadata": {"category": "model_comparison"},
    },
    {
        "source_type": "surrogate_rules",
        "source_name": "business_rules",
        "heading": "Surrogate Tree Rules and Governance Disclaimer",
        "content": (
            "Surrogate decision tree rules are shallow rule approximations extracted to explain EBM decision boundaries to credit underwriters. "
            "Validation R2 is 0.525 and risk-band agreement is 68.0%. Key threshold splits involve EXT_SOURCE_3 <= 0.316, EXT_SOURCE_2 <= 0.417, "
            "and INST_LATE_RATE. REGULATORY DISCLAIMER: These rules are approximate explanatory heuristics and MUST NOT be used as rigid credit approval policy."
        ),
        "metadata": {"category": "surrogate_rules", "risk_band_agreement": 0.680},
    },
    {
        "source_type": "risk_policy",
        "source_name": "risk_bands",
        "heading": "Risk Band Definitions and Thresholds",
        "content": (
            "Internal model-estimated payment-difficulty risk bands are calibrated on validation score percentiles: "
            "- Low Risk: Predicted probability < 0.0649 (< P60 percentile). Observed test default rate: 3.3%. "
            "- Medium Risk: Predicted probability between 0.0649 and 0.1579 (P60 to P85). Observed test default rate: 9.8%. "
            "- High Risk: Predicted probability >= 0.1579 (>= P85). Observed test default rate: 24.3%. "
            "These bands are internal risk segments and not RBI-defined credit ratings."
        ),
        "metadata": {"category": "risk_bands", "low_threshold": 0.0649, "high_threshold": 0.1579},
    },
]


def load_column_definitions(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        LOGGER.warning("Column description file not found: %s", csv_path)
        return []
    df = pd.read_csv(csv_path, encoding="latin1")
    chunks: list[dict[str, Any]] = []
    seen = set()
    for _, row in df.iterrows():
        col_name = str(row.get("Row", "")).strip()
        desc = str(row.get("Description", "")).strip()
        table = str(row.get("Table", "")).strip()
        if not col_name or col_name in seen or not desc:
            continue
        seen.add(col_name)
        chunks.append(
            {
                "source_type": "feature_definition",
                "source_name": table,
                "heading": f"Feature Definition: {col_name}",
                "content": f"{col_name} ({table}): {desc}",
                "metadata": {"column": col_name, "table": table, "category": "dictionary"},
            }
        )
    return chunks


def populate_knowledge_base(engine: Engine) -> int:
    settings = get_settings()
    data_dir = settings.data_dir
    csv_path = data_dir / "HomeCredit_columns_description.csv"
    column_docs = load_column_definitions(csv_path)

    all_docs = CURATED_KNOWLEDGE_DOCS + column_docs

    with engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE platform.knowledge_chunks RESTART IDENTITY;"))
        for doc in all_docs:
            connection.execute(
                text(
                    "INSERT INTO platform.knowledge_chunks "
                    "(source_type, source_name, heading, content, metadata, embedding) "
                    "VALUES (:source_type, :source_name, :heading, :content, :metadata, CAST(:embedding AS vector));"
                ),
                {
                    "source_type": doc["source_type"],
                    "source_name": doc["source_name"],
                    "heading": doc.get("heading", ""),
                    "content": doc["content"],
                    "metadata": json.dumps(doc.get("metadata", {})),
                    "embedding": _vector_literal(
                        create_local_embedding(f"{doc.get('heading', '')} {doc['content']}")
                    ),
                },
            )
        count = connection.execute(text("SELECT COUNT(*) FROM platform.knowledge_chunks;")).scalar_one()

    LOGGER.info("Populated platform.knowledge_chunks with %d documents", count)
    return int(count)


def search_knowledge_base(engine: Engine, query: str, limit: int = 5) -> list[dict[str, Any]]:
    query_embedding = _vector_literal(create_local_embedding(query))
    sql_query = """
    SELECT chunk_id, source_type, source_name, heading, content, metadata,
           1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
    FROM platform.knowledge_chunks
    WHERE embedding IS NOT NULL
      AND 1 - (embedding <=> CAST(:query_embedding AS vector)) > 0.05
    ORDER BY embedding <=> CAST(:query_embedding AS vector), chunk_id ASC
    LIMIT :limit;
    """

    with engine.connect() as connection:
        rows = connection.execute(
            text(sql_query),
            {"query_embedding": query_embedding, "limit": limit},
        ).mappings().all()
        results = [dict(row) for row in rows]

    return results


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_database_engine()
    count = populate_knowledge_base(engine)
    print(f"Knowledge base populated successfully with {count} chunks.")

    # Run quick sanity search
    sample = search_knowledge_base(engine, "EXT_SOURCE_3", limit=2)
    print("Sanity search for EXT_SOURCE_3:", json.dumps(sample, indent=2, default=str))


if __name__ == "__main__":
    main()
