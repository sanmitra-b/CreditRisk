from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import Engine, text

from src.db.connection import create_database_engine
from src.talk_to_data.sql_validator import validate_and_format_sql
from src.utils.config import get_settings
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class QueryResult:
    success: bool
    sql: str
    data: pd.DataFrame
    row_count: int
    execution_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "sql": self.sql,
            "row_count": self.row_count,
            "execution_ms": self.execution_ms,
            "error": self.error,
            "preview": self.data.head(10).to_dict(orient="records") if not self.data.empty else [],
        }


class DatabaseQueryRunner:
    def __init__(self, engine: Engine | None = None):
        self.settings = get_settings()
        self.engine = engine or create_database_engine(
            read_only=True,
            statement_timeout_ms=self.settings.sql_timeout_ms,
        )

    def execute_query(self, raw_sql: str) -> QueryResult:
        start_time = time.perf_counter()
        is_valid, validation_err, formatted_sql = validate_and_format_sql(
            raw_sql, max_rows=self.settings.max_sql_rows
        )

        if not is_valid or not formatted_sql:
            elapsed = (time.perf_counter() - start_time) * 1000
            return QueryResult(
                success=False,
                sql=raw_sql,
                data=pd.DataFrame(),
                row_count=0,
                execution_ms=round(elapsed, 2),
                error=f"SQL Validation Error: {validation_err}",
            )

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_query(text(formatted_sql), connection)
            elapsed = (time.perf_counter() - start_time) * 1000
            return QueryResult(
                success=True,
                sql=formatted_sql,
                data=df,
                row_count=len(df),
                execution_ms=round(elapsed, 2),
                error=None,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            LOGGER.warning("Database query execution failed: %s", e)
            return QueryResult(
                success=False,
                sql=formatted_sql,
                data=pd.DataFrame(),
                row_count=0,
                execution_ms=round(elapsed, 2),
                error=f"Database Execution Error: {str(e).strip()}",
            )
