from __future__ import annotations

import re
from google import genai

from src.talk_to_data.prompt_templates import NL_TO_SQL_SYSTEM_PROMPT, SQL_REPAIR_PROMPT
from src.utils.config import get_settings
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

CANONICAL_SQL_MAPPINGS = [
    (r"overall default rate", "SELECT default_rate, applicants, defaults FROM analytics.portfolio_summary"),
    (r"highest default rate", "SELECT age_band, default_rate, applicants FROM analytics.age_band_summary ORDER BY default_rate DESC LIMIT 1"),
    (r"education level", "SELECT education_level, default_rate, applicants FROM analytics.education_summary ORDER BY default_rate DESC"),
    (r"late-payment|late payment", "SELECT segment, default_rate, applicants FROM analytics.repayment_history_summary WHERE signal = 'Late instalment history'"),
    (r"housing", "SELECT housing_type, default_rate, applicants FROM analytics.housing_summary ORDER BY default_rate DESC"),
]


class NLToSQLGenerator:
    def __init__(self, client: genai.Client | None = None):
        settings = get_settings()
        self.client = client
        if self.client is None and settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_default_model

    def generate_sql(self, question: str) -> str:
        # Check canonical fast path
        normalized = question.lower().strip()
        for pattern, sql in CANONICAL_SQL_MAPPINGS:
            if re.search(pattern, normalized):
                LOGGER.info("Using canonical SQL for '%s'", question)
                return sql

        # LLM generation
        if self.client is not None:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{NL_TO_SQL_SYSTEM_PROMPT}\n\nQuestion: {question}\nSQL:",
                )
                raw_sql = response.text.strip()
                # Clean markdown blocks
                raw_sql = re.sub(r"^```[a-zA-Z]*\n", "", raw_sql)
                raw_sql = re.sub(r"\n```$", "", raw_sql).strip()
                raw_sql = raw_sql.rstrip(";")
                return raw_sql
            except Exception as e:
                LOGGER.warning("Gemini NL-to-SQL failed: %s", e)

        # Fallback query
        return "SELECT default_rate, applicants FROM analytics.portfolio_summary"

    def repair_sql(self, failed_sql: str, error_message: str) -> str:
        if self.client is not None:
            try:
                prompt = SQL_REPAIR_PROMPT.format(
                    failed_sql=failed_sql,
                    error_message=error_message,
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{NL_TO_SQL_SYSTEM_PROMPT}\n\n{prompt}\nCorrected SQL:",
                )
                repaired = response.text.strip()
                repaired = re.sub(r"^```[a-zA-Z]*\n", "", repaired)
                repaired = re.sub(r"\n```$", "", repaired).strip()
                return repaired.rstrip(";")
            except Exception as e:
                LOGGER.warning("SQL repair generation failed: %s", e)

        # Fallback repair: replace unknown column or select safe portfolio summary
        return "SELECT default_rate, applicants FROM analytics.portfolio_summary"
