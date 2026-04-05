-- 005 — Predictions table for monitoring and drift detection
-- Stores model predictions alongside actuals for retroactive comparison.

CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS analytics.predictions (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    station_id        VARCHAR(20)      NOT NULL,
    prediction_time   TIMESTAMPTZ      NOT NULL,
    target_time       TIMESTAMPTZ      NOT NULL,
    model_name        VARCHAR(30)      NOT NULL,
    predicted_value   DOUBLE PRECISION NOT NULL,
    actual_value      DOUBLE PRECISION,
    created_at        TIMESTAMPTZ      NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_prediction_grain
        UNIQUE (station_id, target_time, model_name)
);

CREATE INDEX IF NOT EXISTS idx_pred_model_target
    ON analytics.predictions (model_name, target_time);

CREATE INDEX IF NOT EXISTS idx_pred_station_target
    ON analytics.predictions (station_id, target_time);
