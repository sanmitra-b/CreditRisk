from pathlib import Path

import pandas as pd
import pytest

from src.db.load_data import APPLICANT_COLUMNS, prepare_applicant_frame
from src.db.connection import is_transaction_pooler_url


def applicant_frame() -> pd.DataFrame:
    values = {column: [None, None] for column in APPLICANT_COLUMNS}
    values["SK_ID_CURR"] = [100001, 100002]
    values["TARGET"] = [0, 1]
    return pd.DataFrame(values)


def test_prepare_applicant_frame_preserves_unique_grain():
    prepared = prepare_applicant_frame(applicant_frame())
    assert len(prepared) == prepared["sk_id_curr"].nunique() == 2
    assert list(prepared.columns) == [name.lower() for name in APPLICANT_COLUMNS]


def test_prepare_applicant_frame_rejects_duplicate_applicants():
    frame = applicant_frame()
    frame.loc[1, "SK_ID_CURR"] = frame.loc[0, "SK_ID_CURR"]
    with pytest.raises(ValueError, match="one unique"):
        prepare_applicant_frame(frame)


def test_database_security_and_pooler_configuration():
    sql_dir = Path(__file__).resolve().parents[2] / "sql"
    role_sql = (sql_dir / "001_extensions_roles.sql").read_text(encoding="utf-8")
    grants_sql = (sql_dir / "004_indexes_grants.sql").read_text(encoding="utf-8")
    assert "REVOKE ALL ON SCHEMA raw FROM PUBLIC" in role_sql
    assert "GRANT USAGE ON SCHEMA raw TO analytics_ro" not in role_sql
    assert "GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_ro" in grants_sql
    assert "INSERT" not in grants_sql
    assert "UPDATE" not in grants_sql

    transaction_url = (
        "postgresql+psycopg://postgres.project:secret@"
        "aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
    )
    session_url = transaction_url.replace(":6543/", ":5432/")
    local_url = "postgresql+psycopg://credit_app:secret@localhost:6543/credit_risk"

    assert is_transaction_pooler_url(transaction_url) is True
    assert is_transaction_pooler_url(session_url) is False
    assert is_transaction_pooler_url(local_url) is False
