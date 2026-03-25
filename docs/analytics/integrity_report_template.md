# Data integrity report — template

Fill after running ingestion and `python -m src.storage.data_quality`.

| Field | Value |
|-------|--------|
| Report date (UTC) | |
| Environment | e.g. Neon production / dev branch |
| `raw_station_status` row count | |
| `station_information` row count | |
| Date span (`min(last_reported)` → `max(last_reported)`) | |
| DQ: orphan status rows | 0 expected |
| DQ: negative availability | 0 expected |
| DQ: ingestion before last_reported | 0 expected |
| DQ: duplicate grain keys | 0 expected |
| Notes | e.g. feed outage, schema migration |

## Optional SQL (Neon SQL Editor)

```sql
SELECT COUNT(*) AS raw_status_rows FROM raw_station_status;
SELECT COUNT(*) AS station_dim_rows FROM station_information;
SELECT MIN(last_reported), MAX(last_reported) FROM raw_station_status;
SELECT * FROM analytics.v_dq_orphan_status_count;
```
