-- 001_create_tables.sql
-- Phase 2 — Data Ingestion Pipeline
-- Creates the core tables for GBFS data storage.

-- Station information (reference data, upserted on each run)
CREATE TABLE IF NOT EXISTS station_information (
    station_id   VARCHAR(20) PRIMARY KEY,
    name         TEXT        NOT NULL,
    lat          DOUBLE PRECISION NOT NULL,
    lon          DOUBLE PRECISION NOT NULL,
    capacity     INTEGER     NOT NULL,
    address      TEXT,
    groups       TEXT[],
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_station_information_name
    ON station_information (name);

-- Raw station status (append-only time-series data)
CREATE TABLE IF NOT EXISTS raw_station_status (
    station_id            VARCHAR(20)      NOT NULL,
    num_bikes_available   INTEGER          NOT NULL,
    num_docks_available   INTEGER          NOT NULL,
    num_bikes_disabled    INTEGER          NOT NULL DEFAULT 0,
    num_docks_disabled    INTEGER          NOT NULL DEFAULT 0,
    last_reported         TIMESTAMPTZ      NOT NULL,
    is_renting            BOOLEAN          NOT NULL,
    is_returning          BOOLEAN          NOT NULL,
    status                VARCHAR(30)      NOT NULL,
    ingestion_timestamp   TIMESTAMPTZ      NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_status_station_reported
        UNIQUE (station_id, last_reported)
);

CREATE INDEX IF NOT EXISTS idx_raw_station_status_station_id
    ON raw_station_status (station_id);

CREATE INDEX IF NOT EXISTS idx_raw_station_status_last_reported
    ON raw_station_status (last_reported);

CREATE INDEX IF NOT EXISTS idx_raw_station_status_ingestion
    ON raw_station_status (ingestion_timestamp);
