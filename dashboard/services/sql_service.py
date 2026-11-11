"""
AIRWISE Dashboard: SQL & Big Data Service
Provides safe, read-only SQL querying over actual Hive warehouse files via in-process DuckDB,
and delivers authoritative precomputed Spark scalability benchmark metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re
import time
import duckdb
import pandas as pd
import streamlit as st

from dashboard.utils.paths import HIVE_DB_DIR, CHECKPOINT6_SQL_QUERIES_CSV


# Authoritative SQL benchmark queries from Checkpoint 6
CANNED_QUERIES = {
    "Q1: Total Observations Per Site": (
        "SELECT site, count(*) as observation_count\n"
        "FROM sensor_part_site\n"
        "GROUP BY site\n"
        "ORDER BY site;"
    ),
    "Q2: Average BAM Reference Per Site": (
        "SELECT site, round(avg(bam_reference), 2) as mean_bam_pm25,\n"
        "       round(stddev(bam_reference), 2) as std_bam_pm25\n"
        "FROM sensor_part_site\n"
        "GROUP BY site\n"
        "ORDER BY site;"
    ),
    "Q3: Monthly Sensor Readings Across Channels": (
        "SELECT site, year(timestamp) as yr, month(timestamp) as mo,\n"
        "       round(avg(pa_atm), 2) as avg_atm,\n"
        "       round(avg(pa_cf1), 2) as avg_cf1,\n"
        "       round(avg(pa_alt), 2) as avg_alt\n"
        "FROM sensor_part_site\n"
        "GROUP BY site, yr, mo\n"
        "ORDER BY site, yr, mo\n"
        "LIMIT 15;"
    ),
    "Q4: Humidity Regimes Analysis": (
        "SELECT \n"
        "    CASE \n"
        "        WHEN relative_humidity < 30 THEN 'Low (<30%)'\n"
        "        WHEN relative_humidity < 60 THEN 'Moderate (30-60%)'\n"
        "        WHEN relative_humidity < 80 THEN 'High (60-80%)'\n"
        "        ELSE 'Extreme (>80%)'\n"
        "    END as humidity_regime,\n"
        "    count(*) as count,\n"
        "    round(avg(bam_reference), 2) as avg_bam,\n"
        "    round(avg(pa_atm), 2) as avg_atm\n"
        "FROM sensor_part_site\n"
        "GROUP BY 1\n"
        "ORDER BY count DESC;"
    ),
    "Q5: Pollution Regimes & Nephelometer Bias": (
        "SELECT \n"
        "    CASE \n"
        "        WHEN bam_reference <= 30.0 THEN 'Good'\n"
        "        WHEN bam_reference <= 60.0 THEN 'Satisfactory'\n"
        "        WHEN bam_reference <= 90.0 THEN 'Moderate'\n"
        "        WHEN bam_reference <= 120.0 THEN 'Poor'\n"
        "        WHEN bam_reference <= 250.0 THEN 'Very Poor'\n"
        "        ELSE 'Severe'\n"
        "    END as pollution_regime,\n"
        "    count(*) as count,\n"
        "    round(avg(bam_reference), 2) as avg_bam,\n"
        "    round(avg(pa_atm), 2) as avg_pa_atm\n"
        "FROM sensor_part_site\n"
        "GROUP BY 1\n"
        "ORDER BY count DESC;"
    ),
    "Q6: Daily 24h Aggregates (>= 18h Completeness)": (
        "SELECT site, CAST(timestamp AS DATE) as obs_date,\n"
        "       count(*) as valid_hours,\n"
        "       round(avg(bam_reference), 2) as daily_mean_bam\n"
        "FROM sensor_part_site\n"
        "GROUP BY site, obs_date\n"
        "HAVING count(*) >= 18\n"
        "ORDER BY site, obs_date\n"
        "LIMIT 20;"
    ),
    "Q7: Monitoring Temporal Extent & Gap Statistics": (
        "SELECT site, count(*) as total_obs,\n"
        "       min(timestamp) as start_time,\n"
        "       max(timestamp) as end_time\n"
        "FROM sensor_part_site\n"
        "GROUP BY site\n"
        "ORDER BY site;"
    ),
}


@st.cache_resource(show_spinner=False)
def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Initializes in-memory DuckDB connection, caches warehouse tables into RAM, and locks filesystem access."""
    con = duckdb.connect(":memory:")

    # Load warehouse tables into isolated in-memory tables
    table_configs = [
        ("sensor_part_site", f"read_parquet('{HIVE_DB_DIR}/sensor_part_site/*/*.parquet')"),
        ("sensor_part_none", f"read_parquet('{HIVE_DB_DIR}/sensor_part_none/*.parquet')"),
        ("sensor_format_parquet", f"read_parquet('{HIVE_DB_DIR}/sensor_format_parquet/*.parquet')"),
        ("sensor_format_csv", f"read_csv('{HIVE_DB_DIR}/sensor_format_csv/*.csv', header=True)"),
        ("sensor_part_site_year", f"read_parquet('{HIVE_DB_DIR}/sensor_part_site_year/*/*/*.parquet')"),
        ("sensor_part_site_year_month", f"read_parquet('{HIVE_DB_DIR}/sensor_part_site_year_month/*/*/*/*.parquet')"),
    ]

    for table_name, read_expr in table_configs:
        try:
            con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM {read_expr};")
        except Exception:
            pass

    # Engine-level security lockdown: disable all filesystem operations and external connections
    try:
        con.execute("SET enable_external_access = false;")
    except Exception:
        pass

    return con


def execute_safe_query(sql_query: str) -> Tuple[Optional[pd.DataFrame], float, Optional[str]]:
    """
    Executes a read-only SQL query against the warehouse catalog.
    Strictly enforces:
      1. SELECT or WITH (CTE) queries only.
      2. Single-statement execution (rejects multiple statements).
      3. Complete blocking of mutation keywords (DROP, INSERT, DELETE, UPDATE, etc.).
      4. Complete blocking of filesystem reading functions (read_parquet, read_csv, glob, etc.).
      5. Strips comments and checks case-insensitively.
    Returns:
        df: Query results
        elapsed_sec: Query execution latency
        error: Error message if query fails or is rejected
    """
    # 1. Strip comments
    q = re.sub(r"--.*?(\n|$)", " ", sql_query)
    q = re.sub(r"/\*.*?\*/", " ", q, flags=re.DOTALL).strip()

    if not q:
        return None, 0.0, "Query Rejected: SQL query is empty."

    # 2. Reject multiple statements
    statements = [s.strip() for s in q.split(";") if s.strip()]
    if len(statements) > 1:
        return None, 0.0, "Query Rejected: Multiple statements are not permitted in read-only analysis."

    single_q = statements[0]

    # 3. Require SELECT or WITH (CTE) entry point
    if not re.match(r"^(SELECT|WITH)\b", single_q, re.IGNORECASE):
        return None, 0.0, "Query Rejected: Only SELECT or WITH (CTE) queries are permitted."

    # 4. Check for forbidden mutation keywords
    forbidden_pattern = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|COPY|EXPORT|IMPORT|LOAD|INSTALL|CALL|EXEC|EXECUTE|WRITE|SET|RESET)\b"
    forbidden_match = re.search(forbidden_pattern, single_q, re.IGNORECASE)
    if forbidden_match:
        token = forbidden_match.group(1).upper()
        return None, 0.0, f"Query Rejected: Disallowed modification keyword '{token}'. Read-only queries permitted."

    # 5. Check for direct filesystem reading functions
    fs_pattern = r"\b(READ_PARQUET|READ_CSV|READ_JSON|SCAN_PARQUET|SCAN_CSV|READ_TEXT|READ_BLOB|GLOB)\b"
    if re.search(fs_pattern, single_q, re.IGNORECASE):
        return None, 0.0, "Query Rejected: Direct filesystem access functions are blocked. Please query pre-registered warehouse views."

    con = get_duckdb_connection()
    t0 = time.perf_counter()
    try:
        res_df = con.execute(single_q).df()
        elapsed = time.perf_counter() - t0
        return res_df, elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return None, elapsed, str(e)


@st.cache_data(show_spinner=False)
def get_warehouse_catalog() -> pd.DataFrame:
    """Inspects pre-registered DuckDB in-memory Hive tables without touching the filesystem during queries."""
    con = get_duckdb_connection()
    tables_res = con.execute("SHOW TABLES;").df()
    rows = []
    if not tables_res.empty:
        col_name = tables_res.columns[0]
        for tname in tables_res[col_name].tolist():
            cnt = int(con.execute(f"SELECT COUNT(*) FROM {tname};").fetchone()[0])
            desc = con.execute(f"DESCRIBE {tname};").df()
            rows.append(
                {
                    "table_name": tname,
                    "row_count": cnt,
                    "column_count": len(desc),
                    "columns": ", ".join(desc["column_name"].tolist()),
                }
            )
    return pd.DataFrame(rows)
