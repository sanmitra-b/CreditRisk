from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    model_dir: Path
    artifact_dir: Path
    random_seed: int
    log_level: str
    database_url: str | None
    analytics_database_url: str | None
    sql_timeout_ms: int
    max_sql_rows: int
    gemini_api_key: str | None
    gemini_default_model: str
    gemini_complex_model: str
    max_web_results: int


def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=_resolve_path(os.getenv("DATA_DIR"), PROJECT_ROOT / "data"),
        model_dir=_resolve_path(os.getenv("MODEL_DIR"), PROJECT_ROOT / "models"),
        artifact_dir=_resolve_path(os.getenv("ARTIFACT_DIR"), PROJECT_ROOT / "artifacts"),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        database_url=os.getenv("DATABASE_URL"),
        analytics_database_url=os.getenv("ANALYTICS_DATABASE_URL"),
        sql_timeout_ms=int(os.getenv("SQL_TIMEOUT_MS", "8000")),
        max_sql_rows=int(os.getenv("MAX_SQL_ROWS", "200")),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_default_model=os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3.5-flash-lite"),
        gemini_complex_model=os.getenv("GEMINI_COMPLEX_MODEL", "gemini-3.8-flash"),
        max_web_results=int(os.getenv("MAX_WEB_RESULTS", "5")),
    )
