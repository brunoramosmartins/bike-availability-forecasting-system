-- 006: Anomaly flags table
-- Stores detected anomalies from rule-based and statistical methods.

CREATE TABLE IF NOT EXISTS analytics.anomalies (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    station_id          VARCHAR(20)      NOT NULL,
    detected_at         TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    is_stuck            BOOLEAN          NOT NULL DEFAULT FALSE,
    is_statistical_outlier BOOLEAN       NOT NULL DEFAULT FALSE,
    stuck_duration_hours DOUBLE PRECISION,
    isolation_score     DOUBLE PRECISION,
    resolved_at         TIMESTAMPTZ,
    notes               TEXT,

    CONSTRAINT uq_anomaly_grain
        UNIQUE (station_id, detected_at)
);

CREATE INDEX IF NOT EXISTS idx_anomaly_station
    ON analytics.anomalies (station_id, detected_at DESC);

CREATE INDEX IF NOT EXISTS idx_anomaly_unresolved
    ON analytics.anomalies (resolved_at)
    WHERE resolved_at IS NULL;
