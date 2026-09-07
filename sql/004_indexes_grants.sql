CREATE INDEX IF NOT EXISTS idx_applicants_target
    ON analytics.applicants (target);
CREATE INDEX IF NOT EXISTS idx_applicants_education
    ON analytics.applicants (name_education_type);
CREATE INDEX IF NOT EXISTS idx_applicants_housing
    ON analytics.applicants (name_housing_type);
CREATE INDEX IF NOT EXISTS idx_applicants_age
    ON analytics.applicants (age_years);
CREATE INDEX IF NOT EXISTS idx_applicants_history
    ON analytics.applicants (has_bureau_history, has_previous_application_history, has_installment_history);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_metadata
    ON platform.knowledge_chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding
    ON platform.knowledge_chunks USING hnsw (embedding vector_cosine_ops);

REVOKE ALL ON ALL TABLES IN SCHEMA analytics FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA platform FROM PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_ro;
GRANT SELECT ON platform.knowledge_chunks TO analytics_ro;

ALTER DEFAULT PRIVILEGES IN SCHEMA analytics
    GRANT SELECT ON TABLES TO analytics_ro;
