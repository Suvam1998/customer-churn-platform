-- ==========================================================================
-- PostgreSQL schema for the Churn & Retention Platform.
-- Mirrors src/db/models.py. The SQLAlchemy layer can also target SQLite for
-- local/CI use (JSONB -> JSON, SERIAL -> INTEGER autoincrement).
-- Apply with:  psql "$DATABASE_URL" -f database/schema.sql
-- ==========================================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id   VARCHAR(32) PRIMARY KEY,
    gender        VARCHAR(16),
    senior_citizen INTEGER,
    partner       VARCHAR(8),
    dependents    VARCHAR(8),
    tenure        INTEGER,
    created_at    TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS subscriptions (
    id                SERIAL PRIMARY KEY,
    customer_id       VARCHAR(32) REFERENCES customers(customer_id),
    contract          VARCHAR(32),
    payment_method    VARCHAR(48),
    paperless_billing VARCHAR(8),
    monthly_charges   DOUBLE PRECISION,
    total_charges     DOUBLE PRECISION,
    internet_service  VARCHAR(24),
    phone_service     VARCHAR(8),
    services          JSONB,
    churn             INTEGER
);
CREATE INDEX IF NOT EXISTS ix_subscriptions_customer ON subscriptions(customer_id);

CREATE TABLE IF NOT EXISTS customer_events (
    id              SERIAL PRIMARY KEY,
    customer_id     VARCHAR(32) REFERENCES customers(customer_id),
    event_type      VARCHAR(32) NOT NULL,
    event_timestamp TIMESTAMP DEFAULT now(),
    value           DOUBLE PRECISION,
    payload         JSONB,
    is_simulated    BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_events_customer ON customer_events(customer_id);
CREATE INDEX IF NOT EXISTS ix_events_ts ON customer_events(event_timestamp);

CREATE TABLE IF NOT EXISTS customer_features (
    id           SERIAL PRIMARY KEY,
    customer_id  VARCHAR(32) REFERENCES customers(customer_id),
    features     JSONB NOT NULL,
    computed_at  TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_features_customer ON customer_features(customer_id);

CREATE TABLE IF NOT EXISTS predictions (
    id                    SERIAL PRIMARY KEY,
    customer_id           VARCHAR(32) REFERENCES customers(customer_id),
    churn_probability     DOUBLE PRECISION NOT NULL,
    risk_level            VARCHAR(16) NOT NULL,
    model_version         VARCHAR(64),
    prediction_latency_ms DOUBLE PRECISION,
    prediction_timestamp  TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_pred_customer ON predictions(customer_id);
CREATE INDEX IF NOT EXISTS ix_pred_ts ON predictions(prediction_timestamp);

CREATE TABLE IF NOT EXISTS risk_scores (
    id                   SERIAL PRIMARY KEY,
    customer_id          VARCHAR(32) REFERENCES customers(customer_id),
    previous_probability DOUBLE PRECISION,
    new_probability      DOUBLE PRECISION NOT NULL,
    risk_change          DOUBLE PRECISION,
    risk_level           VARCHAR(16) NOT NULL,
    revenue_at_risk      DOUBLE PRECISION,
    updated_at           TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_risk_customer ON risk_scores(customer_id);

CREATE TABLE IF NOT EXISTS recommendations (
    id                  SERIAL PRIMARY KEY,
    customer_id         VARCHAR(32) REFERENCES customers(customer_id),
    recommended_action  VARCHAR(48) NOT NULL,
    urgency             VARCHAR(16) NOT NULL,
    reason              TEXT,
    priority_score      DOUBLE PRECISION,
    estimated_net_value DOUBLE PRECISION,
    created_at          TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_rec_customer ON recommendations(customer_id);

CREATE TABLE IF NOT EXISTS model_versions (
    id            SERIAL PRIMARY KEY,
    model_version VARCHAR(64) UNIQUE NOT NULL,
    base_model    VARCHAR(48),
    calibration   VARCHAR(24),
    roc_auc       DOUBLE PRECISION,
    pr_auc        DOUBLE PRECISION,
    f1            DOUBLE PRECISION,
    brier         DOUBLE PRECISION,
    stage         VARCHAR(16) DEFAULT 'development',
    trained_at    TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiments (
    id              SERIAL PRIMARY KEY,
    experiment_name VARCHAR(64) NOT NULL,
    model           VARCHAR(48),
    params          JSONB,
    metrics         JSONB,
    created_at      TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id           SERIAL PRIMARY KEY,
    metric_name  VARCHAR(48) NOT NULL,
    metric_value DOUBLE PRECISION,
    context      JSONB,
    recorded_at  TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_monitoring_name ON monitoring_metrics(metric_name);
CREATE INDEX IF NOT EXISTS ix_monitoring_ts ON monitoring_metrics(recorded_at);
