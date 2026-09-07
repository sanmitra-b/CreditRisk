from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from psycopg import sql
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError

from src.db.connection import create_database_engine, execute_sql_file
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger


LOGGER = get_logger(__name__)
ROLE_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")
DEFAULT_APPLICANT_BATCH_SIZE = 500
DEFAULT_LOAD_RETRIES = 4

APPLICANT_COLUMNS = [
    "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "AGE_YEARS", "DAYS_EMPLOYED_ANOM", "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CREDIT_GOODS_RATIO",
    "EMPLOYED_AGE_RATIO", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "PRIOR_CREDIT_COUNT", "ACTIVE_CREDIT_COUNT", "OVERDUE_SHARE",
    "MAX_DAYS_OVERDUE", "HAS_BUREAU_HISTORY", "PREV_APP_COUNT",
    "PREV_REFUSED_COUNT", "PREV_REFUSAL_RATE",
    "HAS_PREVIOUS_APPLICATION_HISTORY", "INSTALLMENT_ROWS",
    "LATE_PAYMENT_COUNT", "UNDERPAYMENT_COUNT", "INST_LATE_RATE",
    "INST_UNDERPAY_RATE", "INST_PAYMENT_RATIO", "HAS_INSTALLMENT_HISTORY",
]

RAW_TABLES = {
    "application_train": ("application_train.csv", ["SK_ID_CURR"]),
    "bureau": ("bureau.csv", ["SK_ID_CURR", "SK_ID_BUREAU"]),
    "previous_application": (
        "previous_application.csv",
        ["SK_ID_CURR", "SK_ID_PREV"],
    ),
}


def prepare_applicant_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(APPLICANT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Applicant artifact is missing columns: {missing}")
    output = frame[APPLICANT_COLUMNS].copy()
    if output["SK_ID_CURR"].isna().any() or output["SK_ID_CURR"].duplicated().any():
        raise ValueError("Applicant artifact must contain one unique, non-null SK_ID_CURR")
    output.columns = [column.lower() for column in output.columns]
    return output


def _record_load(engine: Engine, object_name: str, rows: int, source: Path) -> None:
    statement = text(
        "INSERT INTO platform.data_load_log "
        "(object_name, row_count, source_path) "
        "VALUES (:object_name, :row_count, :source_path)"
    )
    parameters = {
        "object_name": object_name,
        "row_count": rows,
        "source_path": str(source),
    }
    for attempt in range(1, DEFAULT_LOAD_RETRIES + 1):
        try:
            with engine.begin() as connection:
                connection.execute(statement, parameters)
            return
        except OperationalError:
            engine.dispose()
            if attempt == DEFAULT_LOAD_RETRIES:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))


def _applicant_insert_statement():
    columns = [column.lower() for column in APPLICANT_COLUMNS]
    column_sql = ", ".join(columns)
    value_sql = ", ".join(f":{column}" for column in columns)
    return text(
        f"INSERT INTO analytics.applicants ({column_sql}) "
        f"VALUES ({value_sql}) "
        "ON CONFLICT (sk_id_curr) DO NOTHING"
    )


def _insert_applicant_batches(
    engine: Engine,
    applicants: pd.DataFrame,
    batch_size: int,
    max_retries: int,
) -> None:
    """Insert committed, idempotent batches and reconnect after transient failures."""
    statement = _applicant_insert_statement()
    connection = None
    total = len(applicants)

    try:
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            chunk = applicants.iloc[start:stop]
            records = (
                chunk.astype(object)
                .where(pd.notna(chunk), None)
                .to_dict(orient="records")
            )

            for attempt in range(1, max_retries + 1):
                try:
                    if connection is None:
                        connection = engine.connect()
                    with connection.begin():
                        connection.execute(statement, records)
                    break
                except OperationalError as error:
                    if connection is not None:
                        connection.close()
                        connection = None
                    engine.dispose()
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Applicant upload failed after {max_retries} attempts "
                            f"for rows {start + 1:,}-{stop:,}"
                        ) from error
                    delay = min(2 ** (attempt - 1), 8)
                    LOGGER.warning(
                        "Connection dropped for rows %s-%s; retrying in %ss "
                        "(attempt %s/%s)",
                        f"{start + 1:,}",
                        f"{stop:,}",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)

            if stop == total or stop % 10_000 == 0:
                LOGGER.info(
                    "Applicant upload progress: %s / %s (%.1f%%)",
                    f"{stop:,}",
                    f"{total:,}",
                    100 * stop / total,
                )
    finally:
        if connection is not None:
            connection.close()


def load_applicants(
    engine: Engine,
    artifact_path: Path,
    batch_size: int = DEFAULT_APPLICANT_BATCH_SIZE,
    max_retries: int = DEFAULT_LOAD_RETRIES,
    reset_existing: bool = False,
) -> int:
    if batch_size < 1:
        raise ValueError("Applicant batch size must be positive")
    if max_retries < 1:
        raise ValueError("Maximum retries must be positive")
    LOGGER.info("Reading applicant artifact %s", artifact_path)
    applicants = prepare_applicant_frame(pd.read_parquet(artifact_path))
    if reset_existing:
        with engine.begin() as connection:
            connection.execute(text("TRUNCATE TABLE analytics.applicants"))
    else:
        LOGGER.info(
            "Resumable mode enabled; existing applicant IDs will be retained and skipped"
        )
    _insert_applicant_batches(engine, applicants, batch_size, max_retries)
    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as connection:
                loaded = connection.execute(
                    text("SELECT COUNT(*) FROM analytics.applicants")
                ).scalar_one()
            break
        except OperationalError:
            engine.dispose()
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    if loaded != len(applicants):
        raise RuntimeError(f"Expected {len(applicants)} applicants; loaded {loaded}")
    _record_load(engine, "analytics.applicants", loaded, artifact_path)
    LOGGER.info("Loaded %s applicant-grain rows", f"{loaded:,}")
    return loaded


def load_raw_csv(
    engine: Engine,
    table_name: str,
    csv_path: Path,
    id_columns: list[str],
    chunksize: int = 25_000,
) -> int:
    if not csv_path.exists():
        LOGGER.warning("Skipping missing raw file: %s", csv_path)
        return 0
    rows = 0
    for chunk_number, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        chunk.columns = [column.lower() for column in chunk.columns]
        mode = "replace" if chunk_number == 0 else "append"
        chunk.to_sql(
            table_name,
            engine,
            schema="raw",
            if_exists=mode,
            index=False,
            chunksize=2_000,
        )
        rows += len(chunk)
    with engine.begin() as connection:
        for identifier in id_columns:
            normalized = identifier.lower()
            connection.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_{normalized}" '
                    f'ON raw."{table_name}" ("{normalized}")'
                )
            )
    _record_load(engine, f"raw.{table_name}", rows, csv_path)
    LOGGER.info("Loaded %s rows into raw.%s", f"{rows:,}", table_name)
    return rows


def provision_login_role(engine: Engine, role: str, password: str) -> None:
    if not ROLE_PATTERN.fullmatch(role):
        raise ValueError("Database role must be a lowercase SQL identifier")
    raw_connection = engine.raw_connection()
    try:
        with raw_connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
            if cursor.fetchone():
                cursor.execute(
                    sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(
                        sql.Identifier(role), sql.Literal(password)
                    )
                )
            else:
                cursor.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} IN ROLE analytics_ro").format(
                        sql.Identifier(role), sql.Literal(password)
                    )
                )
            cursor.execute(
                sql.SQL("ALTER ROLE {} SET statement_timeout = '8s'").format(
                    sql.Identifier(role)
                )
            )
            cursor.execute(
                sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(
                    sql.Identifier(role)
                )
            )
        raw_connection.commit()
    except Exception:
        raw_connection.rollback()
        raise
    finally:
        raw_connection.close()


def initialize_database(
    engine: Engine,
    artifact_path: Path,
    data_dir: Path,
    include_raw: bool = False,
    analytics_user: str | None = None,
    analytics_password: str | None = None,
    applicant_batch_size: int = DEFAULT_APPLICANT_BATCH_SIZE,
    load_retries: int = DEFAULT_LOAD_RETRIES,
    reset_applicants: bool = False,
) -> None:
    sql_dir = get_settings().project_root / "sql"
    execute_sql_file(engine, sql_dir / "001_extensions_roles.sql")
    execute_sql_file(engine, sql_dir / "002_schema.sql")
    load_applicants(
        engine,
        artifact_path,
        batch_size=applicant_batch_size,
        max_retries=load_retries,
        reset_existing=reset_applicants,
    )
    if include_raw:
        for table_name, (filename, identifiers) in RAW_TABLES.items():
            load_raw_csv(engine, table_name, data_dir / filename, identifiers)
    execute_sql_file(engine, sql_dir / "003_analytics_views.sql")
    execute_sql_file(engine, sql_dir / "004_indexes_grants.sql")
    if analytics_user and analytics_password:
        provision_login_role(engine, analytics_user, analytics_password)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL analytics data.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=settings.artifact_dir / "applicant_features.parquet",
    )
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--analytics-user", default="analytics_user")
    parser.add_argument("--analytics-password", default=os.getenv("ANALYTICS_PASSWORD"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_APPLICANT_BATCH_SIZE,
        help="Applicant rows committed per resumable upload batch.",
    )
    parser.add_argument(
        "--load-retries",
        type=int,
        default=DEFAULT_LOAD_RETRIES,
        help="Attempts per batch after a transient database disconnect.",
    )
    parser.add_argument(
        "--reset-applicants",
        action="store_true",
        help="Clear existing applicants before loading (disabled for resumable cloud loads).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_database_engine()
    initialize_database(
        engine,
        args.artifact,
        settings.data_dir,
        include_raw=args.include_raw,
        analytics_user=args.analytics_user,
        analytics_password=args.analytics_password,
        applicant_batch_size=args.batch_size,
        load_retries=args.load_retries,
        reset_applicants=args.reset_applicants,
    )


if __name__ == "__main__":
    main()
