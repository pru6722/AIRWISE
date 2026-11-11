"""
AIRWISE Dashboard: Big Data Warehouse, Scaling & Guarded SQL Explorer
Visualizes Checkpoint 6 Spark/Hive warehouse benchmarks:
  - Read-Only Guarded DuckDB SQL Explorer & Warehouse Catalog
  - 1x to 50x Multi-Scale Latency & Throughput (Pandas vs Local Spark `local[*]`)
  - Columnar Format Footprint & Latency (CSV vs Parquet vs ORC)
  - Hive Partitioning Pruning vs Small-File Overhead
  - In-Memory Caching Amortization (`checkpoint6_caching.json`)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
for p in [str(PROJECT_ROOT), str(DASHBOARD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.config import THEME
from dashboard.components.sidebar import render_benchmark_sidebar
from dashboard.components.metric_cards import render_kpi_row
from dashboard.components.charts import (
    PLOTLY_CHART_CONFIG,
    apply_dark_layout,
    create_storage_comparison_chart,
)
from dashboard.services.data_service import (
    load_metrics_table,
    load_caching_benchmark,
)
from dashboard.services.sql_service import (
    CANNED_QUERIES,
    execute_safe_query,
    get_warehouse_catalog,
)

# 1. Page Setup
st.set_page_config(
    page_title="AIRWISE — Big Data Warehouse & SQL Analytics",
    layout="wide",
    page_icon="⚡",
)

css_path = DASHBOARD_DIR / "styles" / "dark_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Shared Sidebar (Global Data Warehouse Benchmarks are Non-Site Filtered)
render_benchmark_sidebar(
    show_site_filter=False,
    show_date_filter=False,
    show_scenario_filter=False,
    show_pipeline_filter=False,
    show_population_filter=False,
    show_metric_filter=False,
    active_page="Big Data Analytics",
)

# 3. Load Checkpoint 6 Benchmark Artifacts Dynamically
df_scaling = load_metrics_table("checkpoint6_scaling")
df_storage = load_metrics_table("checkpoint6_storage_formats")
df_part = load_metrics_table("checkpoint6_partitioning")
df_sql_bench = load_metrics_table("checkpoint6_sql_queries")
caching_data = load_caching_benchmark()

# 4. Page Header & Execution Environment Disclosure
st.title("⚡ Big Data Storage, Scaling & Guarded SQL Explorer")
st.markdown(
    "Evaluates Checkpoint 6 Apache Spark (`local[*]`) Hive warehouse engineering (`pm25_research.db`) "
    "alongside a live **read-only DuckDB analytical SQL console** over registered Parquet and CSV warehouse views."
)

st.info(
    "**Single-Host Execution Scope (`local[*]` vs Multi-Node Cluster):**  \n"
    "All scaling (`1×–50×`), storage (`CSV / Parquet / ORC`), partitioning, and caching timings come "
    "from single-machine **PySpark `local[*]`** benchmarks on the research workstation rather than a distributed "
    "multi-node cluster. Because JVM task scheduling and IPC serialization add a fixed overhead floor, vectorized "
    "single-node **Pandas** remains faster across all tested scales up to **50× (`929,200` rows)**, even though "
    "Spark scales more favorably as data grows (`37,651 -> 961,275 rows/s`).  \n"
    "**Warehouse Relation Scope:** Live DuckDB SQL exposes the registered **Parquet and CSV** tables under "
    "`enable_external_access = false`; the Spark-generated **ORC** table (`0.338 MB`, `-84.1%`) is covered by the "
    "frozen Spark format benchmarks below."
)

# 5. Executive KPI Row Derived Strictly from Frozen Artifacts
max_scale_row = df_scaling.sort_values("rows").iloc[-1] if not df_scaling.empty else None
csv_row = (
    df_storage[df_storage["format"] == "CSV"].iloc[0]
    if not df_storage.empty and (df_storage["format"] == "CSV").any()
    else None
)
parquet_row = (
    df_storage[df_storage["format"] == "PARQUET"].iloc[0]
    if not df_storage.empty and (df_storage["format"] == "PARQUET").any()
    else None
)
orc_row = (
    df_storage[df_storage["format"] == "ORC"].iloc[0]
    if not df_storage.empty and (df_storage["format"] == "ORC").any()
    else None
)
unpart_row = (
    df_part[df_part["strategy"] == "Unpartitioned"].iloc[0]
    if not df_part.empty and (df_part["strategy"] == "Unpartitioned").any()
    else None
)
sym_row = (
    df_part[df_part["strategy"] == "Site + Year + Month Partitioned"].iloc[0]
    if not df_part.empty and (df_part["strategy"] == "Site + Year + Month Partitioned").any()
    else None
)

parquet_savings = (
    (1.0 - float(parquet_row["size_mb"]) / float(csv_row["size_mb"])) * 100.0
    if csv_row is not None and parquet_row is not None and float(csv_row["size_mb"]) > 0
    else 0.0
)
orc_savings = (
    (1.0 - float(orc_row["size_mb"]) / float(csv_row["size_mb"])) * 100.0
    if csv_row is not None and orc_row is not None and float(csv_row["size_mb"]) > 0
    else 0.0
)

render_kpi_row([
    {
        "title": "50x Peak Throughput (Spark local[*])",
        "value": (
            f"{float(max_scale_row['spark_throughput_rows_sec']):,.1f} rows/s"
            if max_scale_row is not None else "N/A"
        ),
        "subtitle": (
            f"Spark: {float(max_scale_row['spark_median_sec']):.3f}s | Pandas: {float(max_scale_row['pandas_median_sec']):.3f}s ({float(max_scale_row['pandas_throughput_rows_sec']):,.1f} rows/s)"
            if max_scale_row is not None else ""
        ),
        "delta": (
            f"{int(max_scale_row['rows']):,} rows | {float(max_scale_row['file_size_mb']):.2f} MB Parquet"
            if max_scale_row is not None else ""
        ),
        "color": THEME["accent_teal"],
    },
    {
        "title": "Columnar Footprint Reduction",
        "value": f"ORC -{orc_savings:.1f}% | Parquet -{parquet_savings:.1f}%",
        "subtitle": (
            f"CSV {float(csv_row['size_mb']):.3f} MB -> Parquet {float(parquet_row['size_mb']):.3f} MB -> ORC {float(orc_row['size_mb']):.3f} MB"
            if csv_row is not None and parquet_row is not None and orc_row is not None else ""
        ),
        "delta": "3.3x faster format scan vs CSV",
        "color": THEME["accent_cyan"],
    },
    {
        "title": "Partition Small-File Tax",
        "value": (
            f"{float(sym_row['full_scan_time_sec']):.3f}s vs {float(unpart_row['full_scan_time_sec']):.3f}s"
            if sym_row is not None and unpart_row is not None else "N/A"
        ),
        "subtitle": (
            f"site/year/month ({int(sym_row['file_count'])} files) pruned: {float(sym_row['pruned_query_time_sec']):.3f}s"
            if sym_row is not None else ""
        ),
        "delta": "Over-partitioning slows full scans",
        "color": THEME["accent_amber"],
    },
    {
        "title": "Warm Spark Cache Speedup",
        "value": f"{float(caching_data.get('cache_speedup', 0.0)):.2f}x" if caching_data else "N/A",
        "subtitle": (
            f"Median uncached {float(caching_data.get('median_uncached_sec', 0.0)):.3f}s -> cached {float(caching_data.get('median_cached_sec', 0.0)):.3f}s"
            if caching_data else ""
        ),
        "delta": (
            f"Materialization cost: {float(caching_data.get('cache_materialization_time_sec', 0.0)):.3f}s"
            if caching_data else ""
        ),
        "color": THEME["accent_purple"],
    },
])

st.markdown("---")

# 6. Five Interactive Analytical Tabs
tab_sql, tab_scale, tab_format, tab_part, tab_cache = st.tabs([
    "🗄️ Guarded SQL Explorer & Catalog",
    "📈 1x–50x Scaling Benchmark",
    "📦 CSV vs Parquet vs ORC",
    "🗂️ Partitioning & Small-File Overhead",
    "⚡ Spark Memory Caching",
])

# ---------------------------------------------------------------------------
# TAB 1: Guarded Read-Only DuckDB SQL Explorer & Warehouse Catalog
# ---------------------------------------------------------------------------
with tab_sql:
    st.subheader("Read-Only DuckDB SQL Console & Registered Warehouse Catalog")
    st.markdown(
        "Queries execute inside an isolated in-memory **DuckDB engine** with `enable_external_access = false`. "
        "Only single-statement `SELECT` or `WITH` queries against pre-registered tables are permitted; DDL/DML mutations, "
        "multi-statement chaining, and direct filesystem readers (`read_parquet`, `read_csv`) are blocked."
    )

    df_catalog = get_warehouse_catalog()
    if not df_catalog.empty:
        with st.expander("📚 Inspect Pre-Registered DuckDB Warehouse Tables & Schemas", expanded=False):
            st.dataframe(df_catalog, width="stretch", hide_index=True)

    c_q1, c_q2 = st.columns([2, 1])
    with c_q1:
        selected_q_name = st.selectbox(
            "Select Benchmark Preset Query (Q1–Q7)",
            options=list(CANNED_QUERIES.keys()),
            index=0,
            key="bd_preset_query_select",
        )
    with c_q2:
        security_test_choice = st.selectbox(
            "Or Load Security Guardrail Test",
            options=[
                "None (Use Preset / Custom SQL)",
                "Block Mutation: DROP TABLE sensor_format_parquet;",
                "Block Filesystem Access: SELECT * FROM read_parquet('/etc/passwd');",
                "Block Multi-Statement: SELECT COUNT(*) FROM sensor_format_parquet; DELETE FROM sensor_format_parquet;",
            ],
            index=0,
            key="bd_guardrail_test_select",
        )

    if security_test_choice != "None (Use Preset / Custom SQL)":
        default_sql = security_test_choice.split(": ", 1)[1]
    else:
        default_sql = CANNED_QUERIES[selected_q_name]

    sql_input = st.text_area(
        "SQL Query Editor (Read-Only `SELECT` / `WITH`)",
        value=default_sql,
        height=175,
        key="bd_sql_editor",
    )

    run_btn_col, info_col = st.columns([1, 3])
    with run_btn_col:
        execute_clicked = st.button(
            "▶️ Execute Safe SQL Query",
            type="primary",
            width="stretch",
            key="bd_run_sql_btn",
        )
    with info_col:
        st.caption(
            "Tip: Pre-registered tables include `sensor_format_parquet`, `sensor_format_csv`, "
            "`sensor_part_none`, `sensor_part_site`, `sensor_part_site_year`, and "
            "`sensor_part_site_year_month`."
        )

    # Always execute cleanly on initial view or button click so users see live output immediately
    if execute_clicked or sql_input.strip():
        res_df, elapsed_sec, err_msg = execute_safe_query(sql_input)
        if err_msg:
            st.error(f"🛡️ {err_msg}")
        elif res_df is not None:
            st.success(
                f"✅ Query executed safely in **{elapsed_sec * 1000:.2f} ms** "
                f"({len(res_df):,} rows × {len(res_df.columns)} columns)."
            )
            st.dataframe(res_df, width="stretch", hide_index=True)

            sql_csv_bytes = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download SQL Result CSV",
                data=sql_csv_bytes,
                file_name="airwise_sql_result.csv",
                mime="text/csv",
                key="dl_sql_result",
            )

    st.markdown("---")
    st.markdown("**Historical Checkpoint 6 Spark SQL Execution Benchmarks (`checkpoint6_sql_queries.csv`)**")
    if not df_sql_bench.empty:
        c_sql_l, c_sql_r = st.columns([1.2, 1.0])
        with c_sql_l:
            st.dataframe(df_sql_bench, width="stretch", hide_index=True)
        with c_sql_r:
            fig_sql = go.Figure()
            fig_sql.add_trace(go.Bar(
                x=df_sql_bench["query_id"],
                y=df_sql_bench["median_runtime_sec"],
                text=df_sql_bench["median_runtime_sec"].map(lambda v: f"{v:.4f}s"),
                textposition="outside",
                marker_color=THEME["accent_cyan"],
                name="Spark SQL Median (s)",
            ))
            apply_dark_layout(
                fig_sql,
                "Spark SQL Median Runtimes (Q1–Q7)",
                x_title="Benchmark Query ID",
                y_title="Median Latency (Seconds)",
                height=340,
                hovermode="x",
            )
            st.plotly_chart(fig_sql, width="stretch", config=PLOTLY_CHART_CONFIG)

# ---------------------------------------------------------------------------
# TAB 2: 1x to 50x Multi-Scale Scaling Benchmark
# ---------------------------------------------------------------------------
with tab_scale:
    st.subheader("Pandas In-Memory vs Local Spark (`local[*]`) Scaling (1× to 50×)")
    st.markdown(
        "Measures group-by hourly aggregation latency and row throughput from `18,584` rows (`1×`) "
        "up to `929,200` rows (`50×`, `38.85 MB` Parquet)."
    )

    if not df_scaling.empty:
        scale_labels = [f"{int(s)}× ({int(r):,})" for s, r in zip(df_scaling["scale"], df_scaling["rows"])]

        c_sc1, c_sc2 = st.columns(2)
        with c_sc1:
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Scatter(
                x=scale_labels,
                y=df_scaling["pandas_median_sec"],
                mode="lines+markers+text",
                name="Pandas Vectorized (s)",
                text=df_scaling["pandas_median_sec"].map(lambda v: f"{v:.3f}s"),
                textposition="bottom right",
                line=dict(color=THEME["accent_teal"], width=2.5),
                marker=dict(size=9),
            ))
            fig_lat.add_trace(go.Scatter(
                x=scale_labels,
                y=df_scaling["spark_median_sec"],
                mode="lines+markers+text",
                name="Apache Spark local[*] (s)",
                text=df_scaling["spark_median_sec"].map(lambda v: f"{v:.3f}s"),
                textposition="top left",
                line=dict(color=THEME["accent_amber"], width=2.5),
                marker=dict(size=9),
            ))
            apply_dark_layout(
                fig_lat,
                "Median Aggregation Latency vs Dataset Scale",
                x_title="Scale Factor (Row Count)",
                y_title="Median Runtime (Seconds)",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig_lat, width="stretch", config=PLOTLY_CHART_CONFIG)

        with c_sc2:
            fig_tp = go.Figure()
            fig_tp.add_trace(go.Bar(
                name="Pandas Throughput (rows/s)",
                x=scale_labels,
                y=df_scaling["pandas_throughput_rows_sec"],
                marker_color=THEME["accent_teal"],
            ))
            fig_tp.add_trace(go.Bar(
                name="Spark local[*] Throughput (rows/s)",
                x=scale_labels,
                y=df_scaling["spark_throughput_rows_sec"],
                marker_color=THEME["accent_purple"],
            ))
            fig_tp.update_layout(barmode="group")
            apply_dark_layout(
                fig_tp,
                "Effective Processing Throughput (rows/sec)",
                x_title="Scale Factor (Row Count)",
                y_title="Rows / Second",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig_tp, width="stretch", config=PLOTLY_CHART_CONFIG)

        st.dataframe(df_scaling, width="stretch", hide_index=True)
    else:
        st.info("Scaling benchmark CSV unavailable.")

# ---------------------------------------------------------------------------
# TAB 3: Storage Format Trade-offs (CSV vs Parquet vs ORC)
# ---------------------------------------------------------------------------
with tab_format:
    st.subheader("Storage Serialization Footprint & Query Latency (CSV vs Parquet vs ORC)")

    if not df_storage.empty:
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            fig_sz = create_storage_comparison_chart(
                df_storage,
                title="Serialized Disk Footprint (18,584 Rows)",
            )
            st.plotly_chart(fig_sz, width="stretch", config=PLOTLY_CHART_CONFIG)

        with c_f2:
            fig_ft = go.Figure()
            fig_ft.add_trace(go.Bar(
                name="Table Creation Time (s)",
                x=df_storage["format"],
                y=df_storage["create_time_sec"],
                text=df_storage["create_time_sec"].map(lambda v: f"{v:.3f}s"),
                textposition="outside",
                marker_color=THEME["accent_amber"],
            ))
            fig_ft.add_trace(go.Bar(
                name="Analytical Scan Latency (s)",
                x=df_storage["format"],
                y=df_storage["query_time_sec"],
                text=df_storage["query_time_sec"].map(lambda v: f"{v:.3f}s"),
                textposition="outside",
                marker_color=THEME["accent_teal"],
            ))
            fig_ft.update_layout(barmode="group")
            apply_dark_layout(
                fig_ft,
                "Serialization Write Cost vs Analytical Read Latency",
                x_title="Warehouse Format",
                y_title="Duration (Seconds)",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig_ft, width="stretch", config=PLOTLY_CHART_CONFIG)

        st.dataframe(df_storage, width="stretch", hide_index=True)
    else:
        st.info("Storage format benchmark CSV unavailable.")

# ---------------------------------------------------------------------------
# TAB 4: Hive Partitioning & Small-File Penalty
# ---------------------------------------------------------------------------
with tab_part:
    st.subheader("Hive Partition Pruning Benefits vs. Small-File Scan Overhead")
    st.markdown(
        "Fine-grained partitioning by `site, year, month` creates **44 small Parquet files**, making "
        "unfiltered full-table scans **2.7× slower** (`0.242s` vs `0.089s`) due to file-open & metadata "
        "overhead—whereas selective single-site/month predicates still execute cleanly via partition pruning (`0.099s`)."
    )

    if not df_part.empty:
        c_p1, c_p2 = st.columns([1.4, 1.0])
        with c_p1:
            fig_pt = go.Figure()
            fig_pt.add_trace(go.Bar(
                name="Full Table Scan (s)",
                x=df_part["strategy"],
                y=df_part["full_scan_time_sec"],
                text=df_part["full_scan_time_sec"].map(lambda v: f"{v:.3f}s"),
                textposition="outside",
                marker_color=THEME["accent_rose"],
            ))
            fig_pt.add_trace(go.Bar(
                name="Pruned Predicate Query (s)",
                x=df_part["strategy"],
                y=df_part["pruned_query_time_sec"],
                text=df_part["pruned_query_time_sec"].map(lambda v: f"{v:.3f}s"),
                textposition="outside",
                marker_color=THEME["accent_teal"],
            ))
            fig_pt.update_layout(barmode="group")
            apply_dark_layout(
                fig_pt,
                "Full Scan vs Pruned Query Latency Across Partition Granularities",
                x_title="Partitioning Strategy",
                y_title="Latency (Seconds)",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig_pt, width="stretch", config=PLOTLY_CHART_CONFIG)

        with c_p2:
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Bar(
                name="Part File Count",
                x=df_part["strategy"],
                y=df_part["file_count"],
                text=df_part["file_count"].map(lambda v: f"{int(v)} files"),
                textposition="outside",
                marker_color=THEME["accent_amber"],
            ))
            apply_dark_layout(
                fig_fc,
                "Physical Output File Multiplication (`file_count`)",
                x_title="Partitioning Strategy",
                y_title="Number of Parquet Files",
                height=420,
                hovermode="x",
            )
            st.plotly_chart(fig_fc, width="stretch", config=PLOTLY_CHART_CONFIG)

        st.dataframe(df_part, width="stretch", hide_index=True)
    else:
        st.info("Partitioning benchmark CSV unavailable.")

# ---------------------------------------------------------------------------
# TAB 5: In-Memory Spark Caching Benchmark (`checkpoint6_caching.json`)
# ---------------------------------------------------------------------------
with tab_cache:
    st.subheader("In-Memory Spark DataFrame Caching (`checkpoint6_caching.json`)")

    if caching_data:
        uncached_runs = caching_data.get("uncached_times", [])
        cached_runs = caching_data.get("cached_times", [])
        med_u = float(caching_data.get("median_uncached_sec", 0.0))
        med_c = float(caching_data.get("median_cached_sec", 0.0))
        mat_t = float(caching_data.get("cache_materialization_time_sec", 0.0))
        speedup = float(caching_data.get("cache_speedup", 0.0))
        saving_per_iter = max(0.0, med_u - med_c)
        breakeven_iters = (mat_t / saving_per_iter) if saving_per_iter > 0 else float("inf")

        st.markdown(
            f"Tested on `sensor_format_parquet` (`18,584` rows): uncached query median is **{med_u:.3f}s** vs "
            f"warm cached median **{med_c:.3f}s** (**{speedup:.2f}×** speedup), after an initial cache "
            f"materialization overhead of **{mat_t:.3f}s** (~**{breakeven_iters:.1f}** repeated queries required "
            f"to amortize materialization)."
        )

        n_trials = max(len(uncached_runs), len(cached_runs))
        trials = [f"Run {i + 1}" for i in range(n_trials)]

        c_c1, c_c2 = st.columns([1.4, 1.0])
        with c_c1:
            fig_cache = go.Figure()
            fig_cache.add_trace(go.Bar(
                name="Uncached Parquet Scan (s)",
                x=trials[:len(uncached_runs)],
                y=uncached_runs,
                text=[f"{t:.3f}s" for t in uncached_runs],
                textposition="outside",
                marker_color=THEME["accent_amber"],
            ))
            fig_cache.add_trace(go.Bar(
                name="In-Memory Cached Scan (s)",
                x=trials[:len(cached_runs)],
                y=cached_runs,
                text=[f"{t:.3f}s" for t in cached_runs],
                textposition="outside",
                marker_color=THEME["accent_teal"],
            ))
            fig_cache.update_layout(barmode="group")
            apply_dark_layout(
                fig_cache,
                f"Repeated Trial Latency (Median Speedup: {speedup:.2f}×)",
                x_title="Repeated Query Execution",
                y_title="Latency (Seconds)",
                height=390,
                hovermode="x unified",
            )
            st.plotly_chart(fig_cache, width="stretch", config=PLOTLY_CHART_CONFIG)

        with c_c2:
            df_cache_tbl = pd.DataFrame([
                {"Metric": "Target Table", "Value": str(caching_data.get("table_name", "sensor_format_parquet"))},
                {"Metric": "Cache Materialization Time (s)", "Value": f"{mat_t:.3f}"},
                {"Metric": "Uncached Trials (s)", "Value": ", ".join(f"{x:.3f}" for x in uncached_runs)},
                {"Metric": "Median Uncached Time (s)", "Value": f"{med_u:.3f}"},
                {"Metric": "Cached Trials (s)", "Value": ", ".join(f"{x:.3f}" for x in cached_runs)},
                {"Metric": "Median Cached Time (s)", "Value": f"{med_c:.3f}"},
                {"Metric": "Cache Speedup", "Value": f"{speedup:.2f}×"},
                {"Metric": "Break-Even Query Count", "Value": f"{breakeven_iters:.1f} iterations"},
            ])
            st.dataframe(df_cache_tbl, width="stretch", hide_index=True)
    else:
        st.info("Caching benchmark JSON unavailable.")
