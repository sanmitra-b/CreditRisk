from __future__ import annotations

import re
from typing import Tuple

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {
    "applicants",
    "applicant_risk_features",
    "portfolio_summary",
    "age_band_summary",
    "education_summary",
    "housing_summary",
    "repayment_history_summary",
}

ALLOWED_FUNCTIONS = {
    "avg",
    "ceil",
    "ceiling",
    "coalesce",
    "count",
    "date_trunc",
    "floor",
    "greatest",
    "least",
    "lower",
    "max",
    "min",
    "nullif",
    "round",
    "sum",
    "upper",
}

DISALLOWED_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Merge,
    exp.Command,
)


def validate_and_format_sql(sql_str: str, max_rows: int = 200) -> Tuple[bool, str | None, str | None]:
    if not sql_str or not sql_str.strip():
        return False, "SQL query is empty", None

    cleaned = sql_str.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned).strip()

    # Reject comments
    if "--" in cleaned or "/*" in cleaned:
        return False, "SQL comments are strictly prohibited for safety", None

    try:
        parsed = sqlglot.parse(cleaned, read="postgres")
    except Exception as e:
        return False, f"SQL syntax parsing error: {e}", None

    statements = [stmt for stmt in parsed if stmt is not None]
    if len(statements) == 0:
        return False, "No valid SQL statements found", None
    if len(statements) > 1:
        return False, "Multiple SQL statements are not permitted", None

    stmt = statements[0]

    # Must be Select or With that contains Select
    if isinstance(stmt, exp.Select):
        select_stmt = stmt
    elif isinstance(stmt, exp.With):
        select_stmt = stmt.this
        if not isinstance(select_stmt, exp.Select):
            return False, "WITH query must conclude with a SELECT statement", None
    else:
        return False, f"Only SELECT statements are permitted (found {type(stmt).__name__})", None

    # Ensure no DDL/DML nested inside
    for bad_type in DISALLOWED_EXPRESSIONS:
        if list(stmt.find_all(bad_type)):
            return False, f"Prohibited operation detected: {bad_type.__name__}", None

    # Every chatbot query must read at least one approved analytics object.
    tables = list(stmt.find_all(exp.Table))
    if not tables:
        return False, "Query must read from an approved analytics table/view.", None

    # Check table sources
    for table in tables:
        table_name = table.name.lower()
        schema_name = table.db.lower() if table.db else "analytics"
        
        # Disallow non-analytics schemas
        if schema_name not in ("analytics", ""):
            return False, f"Access to schema '{schema_name}' is not permitted. Only 'analytics' tables/views are allowed.", None

        cte_names = [cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)]
        if table_name in cte_names:
            continue

        if table_name not in ALLOWED_TABLES:
            return False, f"Table/view '{table_name}' is not in the approved analytics whitelist.", None

    # Use a positive function allowlist; PostgreSQL metadata and filesystem helpers
    # are rejected even if sqlglot represents them as specialized expression types.
    for func in (node for node in stmt.walk() if isinstance(node, exp.Func)):
        func_name = (
            func.name if isinstance(func, exp.Anonymous) else func.sql_name()
        ).lower()
        if func_name not in ALLOWED_FUNCTIONS:
            return False, f"Function '{func_name}' is not in the approved function whitelist.", None

    # Enforce row limit cleanly
    limit_clause = stmt.find(exp.Limit)
    if limit_clause is None:
        stmt = stmt.limit(max_rows)
    else:
        try:
            current_limit = int(limit_clause.expression.name)
            if current_limit > max_rows or current_limit <= 0:
                limit_clause.set("expression", exp.Literal.number(max_rows))
        except Exception:
            limit_clause.set("expression", exp.Literal.number(max_rows))

    formatted_sql = stmt.sql("postgres")
    return True, None, formatted_sql
