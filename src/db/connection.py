from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from src.utils.config import get_settings


def is_transaction_pooler_url(url: str) -> bool:
    """Return True for Supavisor/PgBouncer transaction-mode connections."""
    parsed = make_url(url)
    return parsed.port == 6543 and bool(parsed.host and "pooler.supabase.com" in parsed.host)


def create_database_engine(
    url: str | None = None,
    *,
    read_only: bool = False,
    statement_timeout_ms: int | None = None,
) -> Engine:
    settings = get_settings()
    resolved_url = url or (
        settings.analytics_database_url if read_only else settings.database_url
    )
    if not resolved_url:
        variable = "ANALYTICS_DATABASE_URL" if read_only else "DATABASE_URL"
        raise RuntimeError(f"{variable} is not configured")

    transaction_pooler = is_transaction_pooler_url(resolved_url)
    connect_args = {"connect_timeout": 20}
    if transaction_pooler:
        # Supabase transaction mode does not support server-side prepared
        # statements. NullPool avoids stacking a local pool on Supavisor.
        connect_args["prepare_threshold"] = None
        engine = create_engine(
            resolved_url,
            poolclass=NullPool,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
    else:
        engine = create_engine(
            resolved_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            pool_recycle=1800,
            connect_args=connect_args,
        )
    timeout = statement_timeout_ms or settings.sql_timeout_ms

    if read_only and not transaction_pooler:
        @event.listens_for(engine, "connect")
        def _configure_read_only(dbapi_connection, _connection_record) -> None:
            previous_autocommit = dbapi_connection.autocommit
            dbapi_connection.autocommit = True
            try:
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SET default_transaction_read_only = on")
                    cursor.execute(f"SET statement_timeout = {int(timeout)}")
            finally:
                dbapi_connection.autocommit = previous_autocommit

    return engine


def execute_sql_file(engine: Engine, path: Path) -> None:
    sql_text = path.read_text(encoding="utf-8")
    raw_connection = engine.raw_connection()
    try:
        with raw_connection.cursor() as cursor:
            cursor.execute(sql_text)
        raw_connection.commit()
    except Exception:
        raw_connection.rollback()
        raise
    finally:
        raw_connection.close()


def database_is_ready(engine: Engine) -> bool:
    try:
        with engine.connect() as connection:
            return connection.execute(text("SELECT 1")).scalar_one() == 1
    except Exception:
        return False
