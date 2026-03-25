-- 002_create_indexes.sql
-- Phase 3 — Data Modeling & Storage (analytics foundation)
-- Additional indexes for station-scoped time-series access patterns.
-- Note: 001_create_tables.sql already creates single-column indexes on
-- raw_station_status(station_id) and raw_station_status(last_reported).

-- Composite index: filter by station and order/ bound by last_reported (typical ML window queries).
CREATE INDEX IF NOT EXISTS idx_raw_station_status_station_last_reported
    ON raw_station_status (station_id, last_reported DESC);
