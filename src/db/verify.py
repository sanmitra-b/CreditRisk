from __future__ import annotations

import json

from sqlalchemy import Engine, text

from src.db.connection import create_database_engine


def verify_analytics(engine: Engine) -> dict[str, int | float | bool]:
    checks: dict[str, int | float | bool] = {}
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT COUNT(*) AS rows, COUNT(DISTINCT sk_id_curr) AS applicants, "
                "AVG(target::DOUBLE PRECISION) AS default_rate "
                "FROM analytics.applicants"
            )
        ).mappings().one()
        checks["rows"] = int(row["rows"])
        checks["distinct_applicants"] = int(row["applicants"])
        checks["default_rate"] = float(row["default_rate"])
        checks["one_row_per_applicant"] = row["rows"] == row["applicants"]
        checks["portfolio_summary_rows"] = int(
            connection.execute(
                text("SELECT COUNT(*) FROM analytics.portfolio_summary")
            ).scalar_one()
        )
        checks["age_segments"] = int(
            connection.execute(
                text("SELECT COUNT(*) FROM analytics.age_band_summary")
            ).scalar_one()
        )
    if not checks["one_row_per_applicant"]:
        raise RuntimeError("analytics.applicants violates the applicant grain")
    if checks["portfolio_summary_rows"] != 1:
        raise RuntimeError("analytics.portfolio_summary must contain exactly one row")
    return checks


def main() -> None:
    result = verify_analytics(create_database_engine(read_only=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

