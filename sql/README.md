# PostgreSQL analytics setup

The default demo load stores the pre-aggregated, one-row-per-applicant artifact in
`analytics.applicants`. This is the preferred chatbot data source because it avoids
duplicate counting across Home Credit's one-to-many history tables.

## Initialize

PostgreSQL must include the `vector` extension. Set `DATABASE_URL` to an
administrator connection, then run:

```powershell
conda run -n NeoStatsCredit python -m src.db.load_data `
  --analytics-user analytics_user `
  --analytics-password "a-strong-local-password"
```

Add `--include-raw` only when raw-table drill-down is required. It imports
`application_train.csv`, `bureau.csv`, and `previous_application.csv` into the
restricted `raw` schema and indexes their primary linkage IDs. The read-only
analytics role cannot access this schema.

## Verify

Set `ANALYTICS_DATABASE_URL` to the read-only login and run:

```powershell
conda run -n NeoStatsCredit python -m src.db.verify
```

The verification fails if the applicant table is not unique by `SK_ID_CURR` or
the portfolio summary does not have exactly one row.

## Exposed analytics objects

- `analytics.applicant_risk_features`
- `analytics.portfolio_summary`
- `analytics.age_band_summary`
- `analytics.education_summary`
- `analytics.housing_summary`
- `analytics.repayment_history_summary`

The `analytics_ro` group has `SELECT` access only to approved analytics objects
and knowledge-base chunks. Login roles also receive an eight-second statement
timeout and read-only transactions.

