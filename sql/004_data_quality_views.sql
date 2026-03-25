-- 004_data_quality_views.sql
-- Phase 3 — Data quality observability (analyst-facing metrics views).
-- Use with run_data_quality (Python) or ad-hoc SQL in Neon.

CREATE SCHEMA IF NOT EXISTS analytics;

-- Rows in raw status with no matching station dimension (should be 0 in steady state).
CREATE OR REPLACE VIEW analytics.v_dq_orphan_status_count AS
SELECT COUNT(*)::bigint AS violation_count
FROM raw_station_status AS r
WHERE NOT EXISTS (
    SELECT 1
    FROM station_information AS s
    WHERE s.station_id = r.station_id
);

COMMENT ON VIEW analytics.v_dq_orphan_status_count IS
    'DQ metric: count of raw_station_status rows whose station_id is missing in station_information.';

-- Negative counts are invalid for bikes/docks available.
CREATE OR REPLACE VIEW analytics.v_dq_negative_availability_count AS
SELECT COUNT(*)::bigint AS violation_count
FROM raw_station_status AS r
WHERE r.num_bikes_available < 0
   OR r.num_docks_available < 0
   OR r.num_bikes_disabled < 0
   OR r.num_docks_disabled < 0;

COMMENT ON VIEW analytics.v_dq_negative_availability_count IS
    'DQ metric: rows with negative bike or dock counts.';

-- Ingestion should not precede operator last_reported (clock skew allowance: none in v1).
CREATE OR REPLACE VIEW analytics.v_dq_ingestion_before_report_count AS
SELECT COUNT(*)::bigint AS violation_count
FROM raw_station_status AS r
WHERE r.ingestion_timestamp < r.last_reported;

COMMENT ON VIEW analytics.v_dq_ingestion_before_report_count IS
    'DQ metric: rows where ingestion_timestamp is strictly before last_reported.';

-- Physical duplicates beyond the unique constraint (expect 0 if constraint holds).
CREATE OR REPLACE VIEW analytics.v_dq_duplicate_grain_count AS
SELECT COUNT(*)::bigint AS violation_count
FROM (
    SELECT r.station_id, r.last_reported
    FROM raw_station_status AS r
    GROUP BY r.station_id, r.last_reported
    HAVING COUNT(*) > 1
) AS d;

COMMENT ON VIEW analytics.v_dq_duplicate_grain_count IS
    'DQ metric: number of (station_id, last_reported) keys with more than one row.';
