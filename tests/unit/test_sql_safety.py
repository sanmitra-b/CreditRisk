import pytest
from src.talk_to_data.sql_validator import validate_and_format_sql


def test_valid_select_query():
    sql = "SELECT default_rate, applicants FROM analytics.portfolio_summary"
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is True
    assert err is None
    assert "LIMIT 200" in formatted


def test_valid_with_select_query():
    sql = """
    WITH high_risk AS (
        SELECT sk_id_curr, amt_credit FROM analytics.applicants WHERE target = 1
    )
    SELECT AVG(amt_credit) FROM high_risk
    """
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is True
    assert err is None
    assert "LIMIT 200" in formatted


def test_reject_drop_table():
    sql = "DROP TABLE analytics.applicants;"
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is False
    assert "Prohibited operation" in err or "Only SELECT" in err


def test_reject_delete_from():
    sql = "DELETE FROM analytics.applicants WHERE sk_id_curr = 1"
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is False
    assert "Prohibited operation" in err or "Only SELECT" in err


def test_reject_update_query():
    sql = "UPDATE analytics.applicants SET target = 0"
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is False


def test_reject_insert_query():
    sql = "INSERT INTO analytics.applicants (sk_id_curr, target) VALUES (999, 1)"
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is False


def test_reject_comments():
    sql = "SELECT * FROM analytics.applicants -- ignore"
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is False
    assert "comments are strictly prohibited" in err

    sql2 = "SELECT * FROM analytics.applicants /* block comment */"
    valid2, err2, formatted2 = validate_and_format_sql(sql2)
    assert valid2 is False


def test_reject_unapproved_schema():
    sql = "SELECT * FROM raw.application_train"
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is False
    assert "schema 'raw' is not permitted" in err


def test_reject_unapproved_table():
    sql = "SELECT * FROM analytics.unknown_table"
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is False
    assert "not in the approved analytics whitelist" in err


def test_reject_disallowed_function():
    sql = "SELECT pg_sleep(10) FROM analytics.portfolio_summary"
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is False
    assert "pg_sleep" in err


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT version()",
        "SELECT current_user",
        "SELECT current_setting('server_version')",
        "SELECT current_user FROM analytics.portfolio_summary",
    ],
)
def test_reject_system_introspection(sql):
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is False
    assert formatted is None


def test_allow_approved_aggregate_functions():
    sql = "SELECT AVG(default_rate), COUNT(*) FROM analytics.portfolio_summary"
    valid, err, formatted = validate_and_format_sql(sql)
    assert valid is True
    assert err is None


def test_clamp_large_limit():
    sql = "SELECT sk_id_curr FROM analytics.applicants LIMIT 10000"
    valid, err, formatted = validate_and_format_sql(sql, max_rows=200)
    assert valid is True
    assert "LIMIT 200" in formatted
