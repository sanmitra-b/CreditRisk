CREATE TABLE IF NOT EXISTS analytics.applicants (
    sk_id_curr BIGINT PRIMARY KEY,
    target SMALLINT NOT NULL CHECK (target IN (0, 1)),
    name_contract_type TEXT,
    name_income_type TEXT,
    name_education_type TEXT,
    name_family_status TEXT,
    name_housing_type TEXT,
    occupation_type TEXT,
    organization_type TEXT,
    cnt_children DOUBLE PRECISION,
    cnt_fam_members DOUBLE PRECISION,
    amt_income_total DOUBLE PRECISION,
    amt_credit DOUBLE PRECISION,
    amt_annuity DOUBLE PRECISION,
    amt_goods_price DOUBLE PRECISION,
    age_years DOUBLE PRECISION,
    days_employed_anom SMALLINT,
    credit_income_ratio DOUBLE PRECISION,
    annuity_income_ratio DOUBLE PRECISION,
    credit_term DOUBLE PRECISION,
    credit_goods_ratio DOUBLE PRECISION,
    employed_age_ratio DOUBLE PRECISION,
    ext_source_1 DOUBLE PRECISION,
    ext_source_2 DOUBLE PRECISION,
    ext_source_3 DOUBLE PRECISION,
    prior_credit_count DOUBLE PRECISION,
    active_credit_count DOUBLE PRECISION,
    overdue_share DOUBLE PRECISION,
    max_days_overdue DOUBLE PRECISION,
    has_bureau_history SMALLINT,
    prev_app_count DOUBLE PRECISION,
    prev_refused_count DOUBLE PRECISION,
    prev_refusal_rate DOUBLE PRECISION,
    has_previous_application_history SMALLINT,
    installment_rows DOUBLE PRECISION,
    late_payment_count DOUBLE PRECISION,
    underpayment_count DOUBLE PRECISION,
    inst_late_rate DOUBLE PRECISION,
    inst_underpay_rate DOUBLE PRECISION,
    inst_payment_ratio DOUBLE PRECISION,
    has_installment_history SMALLINT
);

CREATE TABLE IF NOT EXISTS platform.data_load_log (
    load_id BIGSERIAL PRIMARY KEY,
    object_name TEXT NOT NULL,
    row_count BIGINT NOT NULL,
    source_path TEXT NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS platform.knowledge_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS platform.conversation_sessions (
    thread_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

