"""
Comprehensive Benchmarking Suite for Checkpoint 6.
Executes:
1. Pandas vs Spark scaling benchmark across 1x, 5x, 10x, 25x, 50x.
2. Hive / Spark SQL query benchmark (Q1 through Q7).
3. Spark configuration experiments (cores, shuffle partitions, AQE).
Records execution times, throughput, memory, and validation checks.
"""

import time
import os
import psutil
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.bigdata.ingestion import ingest_with_pandas, ingest_with_spark
from src.bigdata.preprocessing import (
    calculate_missingness_spark,
    calculate_missingness_pandas,
    compute_descriptive_stats_spark,
    compute_descriptive_stats_pandas,
)
from src.bigdata.aggregation import aggregate_24h_spark, aggregate_24h_pandas
from src.bigdata.validation import BigDataValidator


class BigDataBenchmarkRunner:
    """Orchestrates all performance benchmarks and metrics collection."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.validator = BigDataValidator(float_tolerance=1e-3)

    # -----------------------------------------------------
    # PHASE 5: PANDAS VS SPARK SCALING BENCHMARK
    # -----------------------------------------------------
    def run_pandas_pipeline(self, file_path: str, format: str = "parquet") -> Dict[str, Any]:
        """Executes full Pandas data preparation pipeline and records time."""
        t0 = time.perf_counter()
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        # 1. Ingestion
        df = ingest_with_pandas(file_path, format=format)
        n_rows = len(df)

        # 2. Missingness
        missingness = calculate_missingness_pandas(df)

        # 3. Descriptive Stats
        desc_stats = compute_descriptive_stats_pandas(df)

        # 4. 24h Aggregation
        daily_agg = aggregate_24h_pandas(df, min_hours=18)

        total_time = time.perf_counter() - t0
        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_mem_mb = max(0.0, mem_after - mem_before)

        return {
            "runtime_sec": total_time,
            "rows": n_rows,
            "throughput_rows_sec": n_rows / total_time if total_time > 0 else 0,
            "peak_mem_mb": round(peak_mem_mb, 2),
            "desc_stats": desc_stats,
            "daily_agg": daily_agg,
        }

    def run_spark_pipeline(self, file_path: str, format: str = "parquet") -> Dict[str, Any]:
        """Executes full Spark distributed data preparation pipeline and records time."""
        t0 = time.perf_counter()

        # 1. Ingestion
        df = ingest_with_spark(self.spark, file_path, format=format)
        n_rows = df.count()

        # 2. Missingness
        missingness = calculate_missingness_spark(df)

        # 3. Descriptive Stats
        desc_stats = compute_descriptive_stats_spark(df)
        desc_stats.collect()  # force execution

        # 4. 24h Aggregation
        daily_agg = aggregate_24h_spark(df, min_hours=18)
        daily_agg.collect()  # force execution

        total_time = time.perf_counter() - t0

        return {
            "runtime_sec": total_time,
            "rows": n_rows,
            "throughput_rows_sec": n_rows / total_time if total_time > 0 else 0,
            "desc_stats": desc_stats,
            "daily_agg": daily_agg,
        }

    def benchmark_scaling_comparison(
        self,
        scale_catalog: Dict[int, Any],
        repetitions: int = 3,
        format: str = "parquet",
    ) -> List[Dict[str, Any]]:
        """Runs comparative scaling benchmarks across 1x, 5x, 10x, 25x, 50x."""
        results = []

        for scale, meta in sorted(scale_catalog.items()):
            file_path = meta[f"{format}_path"]
            rows = meta["total_rows"]
            print(f"\n==========================================")
            print(f"BENCHMARKING SCALE {scale}× ({rows:,} rows, Format: {format.upper()})")
            print(f"==========================================")

            # Warmup
            print("Running warmup pass...")
            _ = self.run_pandas_pipeline(file_path, format=format)
            _ = self.run_spark_pipeline(file_path, format=format)

            # Pandas benchmark
            pandas_times = []
            pandas_res = None
            for r in range(repetitions):
                res = self.run_pandas_pipeline(file_path, format=format)
                pandas_times.append(res["runtime_sec"])
                pandas_res = res
            med_pandas = sorted(pandas_times)[len(pandas_times) // 2]

            # Spark benchmark
            spark_times = []
            spark_res = None
            for r in range(repetitions):
                res = self.run_spark_pipeline(file_path, format=format)
                spark_times.append(res["runtime_sec"])
                spark_res = res
            med_spark = sorted(spark_times)[len(spark_times) // 2]

            # Validation
            stat_ok, stat_errs = self.validator.validate_descriptive_stats(
                spark_res["desc_stats"], pandas_res["desc_stats"]
            )
            daily_ok, daily_errs = self.validator.validate_daily_aggregations(
                spark_res["daily_agg"], pandas_res["daily_agg"]
            )
            is_valid = stat_ok and daily_ok

            speedup = med_pandas / med_spark if med_spark > 0 else 1.0
            throughput_pandas = rows / med_pandas
            throughput_spark = rows / med_spark

            row_summary = {
                "scale": scale,
                "rows": rows,
                "format": format.upper(),
                "file_size_mb": meta[f"{format}_size_mb"],
                "pandas_median_sec": round(med_pandas, 3),
                "pandas_throughput_rows_sec": round(throughput_pandas, 1),
                "spark_median_sec": round(med_spark, 3),
                "spark_throughput_rows_sec": round(throughput_spark, 1),
                "spark_speedup_vs_pandas": round(speedup, 2),
                "faster_engine": "Spark" if speedup >= 1.0 else "Pandas",
                "validation_passed": is_valid,
            }
            results.append(row_summary)

            print(f"Scale {scale}× ({rows:,} rows):")
            print(f"  Pandas: {med_pandas:.3f}s ({throughput_pandas:,.0f} rows/s)")
            print(f"  Spark:  {med_spark:.3f}s ({throughput_spark:,.0f} rows/s)")
            print(f"  Speedup (Pandas / Spark): {speedup:.2f}× | Valid: {is_valid}")

        return results

    # -----------------------------------------------------
    # PHASE 6: HIVE / SPARK SQL QUERY BENCHMARK
    # -----------------------------------------------------
    def benchmark_sql_queries(self, table_name: str = "sensor_format_parquet") -> List[Dict[str, Any]]:
        """
        Runs 7 authoritative analytical queries through Spark SQL / Hive Warehouse.
        """
        queries = [
            (
                "Q1_Total_Obs_Per_Site",
                f"SELECT site, count(*) as count FROM {table_name} GROUP BY site ORDER BY site",
            ),
            (
                "Q2_Avg_BAM_Per_Site",
                f"SELECT site, round(avg(bam_reference), 2) as mean_bam, round(stddev(bam_reference), 2) as std_bam FROM {table_name} GROUP BY site ORDER BY site",
            ),
            (
                "Q3_Sensor_Readings_Site_Month",
                f"""SELECT site, year(timestamp) as yr, month(timestamp) as mo,
                           round(avg(pa_atm), 2) as avg_atm, round(avg(pa_cf1), 2) as avg_cf1, round(avg(pa_alt), 2) as avg_alt
                    FROM {table_name}
                    GROUP BY site, year(timestamp), month(timestamp)
                    ORDER BY site, yr, mo""",
            ),
            (
                "Q4_Humidity_Regimes",
                f"""SELECT 
                        CASE 
                            WHEN relative_humidity < 30 THEN 'Low (<30%)'
                            WHEN relative_humidity < 60 THEN 'Moderate (30-60%)'
                            WHEN relative_humidity < 80 THEN 'High (60-80%)'
                            ELSE 'Extreme (>80%)'
                        END as humidity_regime,
                        count(*) as count,
                        round(avg(bam_reference), 2) as avg_bam,
                        round(avg(pa_atm), 2) as avg_atm
                    FROM {table_name}
                    GROUP BY 1
                    ORDER BY count DESC""",
            ),
            (
                "Q5_Pollution_Regimes",
                f"""SELECT 
                        CASE 
                            WHEN bam_reference <= 30.0 THEN 'Good'
                            WHEN bam_reference <= 60.0 THEN 'Satisfactory'
                            WHEN bam_reference <= 90.0 THEN 'Moderate'
                            WHEN bam_reference <= 120.0 THEN 'Poor'
                            WHEN bam_reference <= 250.0 THEN 'Very Poor'
                            ELSE 'Severe'
                        END as pollution_regime,
                        count(*) as count,
                        round(avg(bam_reference), 2) as avg_bam,
                        round(avg(pa_atm), 2) as avg_pa
                    FROM {table_name}
                    GROUP BY 1
                    ORDER BY count DESC""",
            ),
            (
                "Q6_Daily_Aggregates_Completeness",
                f"""SELECT site, to_date(timestamp) as dt, count(*) as hours, round(avg(bam_reference), 2) as bam_mean
                    FROM {table_name}
                    GROUP BY site, to_date(timestamp)
                    HAVING count(*) >= 18
                    ORDER BY site, dt""",
            ),
            (
                "Q7_Gap_Statistics",
                f"""SELECT site, count(*) as total_obs,
                           min(timestamp) as min_ts, max(timestamp) as max_ts
                    FROM {table_name}
                    GROUP BY site""",
            ),
        ]

        results = []
        print(f"\n==========================================")
        print(f"BENCHMARKING SPARK SQL / HIVE QUERIES (Table: {table_name})")
        print(f"==========================================")

        for q_id, q_sql in queries:
            times = []
            out_rows = 0
            for r in range(3):
                t0 = time.perf_counter()
                res = self.spark.sql(q_sql).collect()
                times.append(time.perf_counter() - t0)
                out_rows = len(res)

            med_time = sorted(times)[1]
            results.append({
                "query_id": q_id,
                "output_row_count": out_rows,
                "median_runtime_sec": round(med_time, 4),
                "run1_sec": round(times[0], 4),
                "run2_sec": round(times[1], 4),
                "run3_sec": round(times[2], 4),
            })
            print(f"Query {q_id}: {med_time:.4f}s | Output Rows: {out_rows}")

        return results

    # -----------------------------------------------------
    # PHASE 8: SPARK CONFIGURATION EXPERIMENTS
    # -----------------------------------------------------
    def benchmark_spark_configurations(
        self,
        benchmark_file: str,
        format: str = "parquet",
    ) -> List[Dict[str, Any]]:
        """
        Evaluates Spark execution under varying shuffle partitions and AQE settings.
        """
        configs = [
            {"shuffle_partitions": 2, "aqe": False, "label": "Partitions=2, AQE=Off"},
            {"shuffle_partitions": 4, "aqe": False, "label": "Partitions=4, AQE=Off"},
            {"shuffle_partitions": 8, "aqe": False, "label": "Partitions=8, AQE=Off"},
            {"shuffle_partitions": 16, "aqe": False, "label": "Partitions=16, AQE=Off"},
            {"shuffle_partitions": 32, "aqe": False, "label": "Partitions=32, AQE=Off"},
            {"shuffle_partitions": 8, "aqe": True, "label": "Partitions=8, AQE=On (Adaptive)"},
            {"shuffle_partitions": 16, "aqe": True, "label": "Partitions=16, AQE=On (Adaptive)"},
        ]

        results = []
        print(f"\n==========================================")
        print(f"BENCHMARKING SPARK CONFIGURATIONS")
        print(f"==========================================")

        for cfg in configs:
            self.spark.conf.set("spark.sql.shuffle.partitions", str(cfg["shuffle_partitions"]))
            self.spark.conf.set("spark.sql.adaptive.enabled", "true" if cfg["aqe"] else "false")

            times = []
            for r in range(3):
                t0 = time.perf_counter()
                df = ingest_with_spark(self.spark, benchmark_file, format=format)
                daily = aggregate_24h_spark(df, min_hours=18)
                daily.collect()
                times.append(time.perf_counter() - t0)

            med_time = sorted(times)[1]
            results.append({
                "label": cfg["label"],
                "shuffle_partitions": cfg["shuffle_partitions"],
                "aqe_enabled": cfg["aqe"],
                "median_runtime_sec": round(med_time, 3),
            })
            print(f"Config: {cfg['label']} -> {med_time:.3f}s")

        return results
