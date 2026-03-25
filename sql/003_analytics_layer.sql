-- 003_analytics_layer.sql
-- Phase 3 — Curated analytics layer (views, documented grain).
-- Raw tables remain the system of record for ingestion; analytics.* is the
-- stable interface for downstream consumers (ML extracts, BI, monitoring).

CREATE SCHEMA IF NOT EXISTS analytics;

-- Enriched fact-shaped view: one row per (station_id, last_reported) snapshot
-- joined to current dimension attributes (SCD Type 1 semantics on station_information).
CREATE OR REPLACE VIEW analytics.station_status_enriched AS
SELECT
    r.station_id,
    r.last_reported,
    r.ingestion_timestamp,
    r.num_bikes_available,
    r.num_docks_available,
    r.num_bikes_disabled,
    r.num_docks_disabled,
    r.is_renting,
    r.is_returning,
    r.status,
    i.name,
    i.lat,
    i.lon,
    i.capacity,
    i.address,
    i.groups,
    i.updated_at AS station_info_updated_at
FROM raw_station_status AS r
LEFT JOIN station_information AS i ON i.station_id = r.station_id;

COMMENT ON VIEW analytics.station_status_enriched IS
    'Grain: one row per station GBFS status snapshot at last_reported. '
    'Dimension attributes reflect current station_information (Type 1 upsert).';

COMMENT ON COLUMN analytics.station_status_enriched.station_id IS
    'GBFS station identifier (natural key).';
COMMENT ON COLUMN analytics.station_status_enriched.last_reported IS
    'Operator-reported observation time for this snapshot (UTC).';
COMMENT ON COLUMN analytics.station_status_enriched.ingestion_timestamp IS
    'Pipeline write time when this row was ingested (UTC).';

-- Latest snapshot per station by last_reported (ties broken by ingestion_timestamp).
CREATE OR REPLACE VIEW analytics.station_status_latest AS
SELECT DISTINCT ON (r.station_id)
    r.station_id,
    r.last_reported,
    r.ingestion_timestamp,
    r.num_bikes_available,
    r.num_docks_available,
    r.num_bikes_disabled,
    r.num_docks_disabled,
    r.is_renting,
    r.is_returning,
    r.status,
    i.name,
    i.lat,
    i.lon,
    i.capacity,
    i.address,
    i.groups,
    i.updated_at AS station_info_updated_at
FROM raw_station_status AS r
LEFT JOIN station_information AS i ON i.station_id = r.station_id
ORDER BY
    r.station_id,
    r.last_reported DESC,
    r.ingestion_timestamp DESC;

COMMENT ON VIEW analytics.station_status_latest IS
    'Grain: one row per station_id — the latest status by last_reported, '
    'then ingestion_timestamp.';
