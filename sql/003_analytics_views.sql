CREATE OR REPLACE VIEW analytics.applicant_risk_features AS
SELECT
    sk_id_curr,
    target,
    CASE
        WHEN age_years < 30 THEN '20-29'
        WHEN age_years < 40 THEN '30-39'
        WHEN age_years < 50 THEN '40-49'
        WHEN age_years < 60 THEN '50-59'
        ELSE '60+'
    END AS age_band,
    name_income_type,
    name_education_type,
    name_family_status,
    name_housing_type,
    occupation_type,
    amt_income_total,
    amt_credit,
    amt_annuity,
    credit_income_ratio,
    annuity_income_ratio,
    ext_source_1,
    ext_source_2,
    ext_source_3,
    prior_credit_count,
    overdue_share,
    prev_refusal_rate,
    inst_late_rate,
    inst_underpay_rate,
    has_bureau_history,
    has_previous_application_history,
    has_installment_history
FROM analytics.applicants;

CREATE OR REPLACE VIEW analytics.portfolio_summary AS
SELECT
    COUNT(*) AS applicants,
    SUM(target) AS defaults,
    AVG(target::DOUBLE PRECISION) AS default_rate,
    AVG(amt_credit) AS average_credit,
    AVG(amt_income_total) AS average_income
FROM analytics.applicants;

CREATE OR REPLACE VIEW analytics.age_band_summary AS
SELECT
    age_band,
    COUNT(*) AS applicants,
    SUM(target) AS defaults,
    AVG(target::DOUBLE PRECISION) AS default_rate
FROM analytics.applicant_risk_features
GROUP BY age_band;

CREATE OR REPLACE VIEW analytics.education_summary AS
SELECT
    COALESCE(name_education_type, 'Unknown') AS education_level,
    COUNT(*) AS applicants,
    SUM(target) AS defaults,
    AVG(target::DOUBLE PRECISION) AS default_rate
FROM analytics.applicants
GROUP BY COALESCE(name_education_type, 'Unknown');

CREATE OR REPLACE VIEW analytics.housing_summary AS
SELECT
    COALESCE(name_housing_type, 'Unknown') AS housing_type,
    COUNT(*) AS applicants,
    SUM(target) AS defaults,
    AVG(target::DOUBLE PRECISION) AS default_rate
FROM analytics.applicants
GROUP BY COALESCE(name_housing_type, 'Unknown');

CREATE OR REPLACE VIEW analytics.repayment_history_summary AS
SELECT
    'Bureau overdue history'::TEXT AS signal,
    CASE WHEN COALESCE(overdue_share, 0) > 0 THEN 'Observed' ELSE 'Not observed' END AS segment,
    COUNT(*) AS applicants,
    AVG(target::DOUBLE PRECISION) AS default_rate
FROM analytics.applicants
GROUP BY CASE WHEN COALESCE(overdue_share, 0) > 0 THEN 'Observed' ELSE 'Not observed' END
UNION ALL
SELECT
    'Previous refusal history',
    CASE WHEN COALESCE(prev_refused_count, 0) > 0 THEN 'Observed' ELSE 'Not observed' END,
    COUNT(*),
    AVG(target::DOUBLE PRECISION)
FROM analytics.applicants
GROUP BY CASE WHEN COALESCE(prev_refused_count, 0) > 0 THEN 'Observed' ELSE 'Not observed' END
UNION ALL
SELECT
    'Late instalment history',
    CASE WHEN COALESCE(late_payment_count, 0) > 0 THEN 'Observed' ELSE 'Not observed' END,
    COUNT(*),
    AVG(target::DOUBLE PRECISION)
FROM analytics.applicants
GROUP BY CASE WHEN COALESCE(late_payment_count, 0) > 0 THEN 'Observed' ELSE 'Not observed' END;

