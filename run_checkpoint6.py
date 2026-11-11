"""
Master Pipeline Script for Checkpoint 6: Hive + Apache Spark Big Data Scalability.
Executes all phases deterministically, measures real performance,
validates numerical consistency, generates figures, and writes comprehensive reports.
"""

import os
import sys
import time
import json
from pathlib import Path
import pandas as pd

from src.bigdata.environment_audit import run_environment_audit
from src.bigdata.data_generator import generate_benchmark_datasets
from src.bigdata.spark_session import get_spark_session, stop_spark_session
from src.bigdata.hive_manager import HiveWarehouseManager
from src.bigdata.benchmarking import BigDataBenchmarkRunner
from src.visualization.bigdata_plots import (
    plot_execution_time_vs_dataset_size,
    plot_throughput_vs_dataset_size,
    plot_pandas_vs_spark_comparison,
    plot_storage_format_comparison,
    plot_sql_query_benchmark,
    plot_partitioning_performance,
    plot_caching_performance,
    plot_spark_config_comparison,
)


def run_checkpoint6_pipeline():
    start_time = time.perf_counter()
    print("=" * 80)
    print("STARTING CHECKPOINT 6: HIVE + APACHE SPARK BIG DATA SCALABILITY PIPELINE")
    print("=" * 80)

    # -----------------------------------------------------
    # PHASE 0: ENVIRONMENT AND PROJECT AUDIT
    # -----------------------------------------------------
    print("\n[PHASE 0] Inspecting Hardware, Java, Hadoop, Hive, and Spark Infrastructure...")
    audit_results = run_environment_audit()

    # -----------------------------------------------------
    # PHASE 4: DATASET EXPANSION / SCALING GENERATOR
    # -----------------------------------------------------
    print("\n[PHASE 4] Generating Multi-Scale Benchmark Datasets (1x, 5x, 10x, 25x, 50x)...")
    scale_catalog = generate_benchmark_datasets(
        input_path="data/processed/cleaned_data.parquet",
        output_dir="data/benchmark_scales",
        scales=[1, 5, 10, 25, 50],
        seed=42,
        save_csv=True,
        save_parquet=True,
    )

    # -----------------------------------------------------
    # START SPARK SESSION
    # -----------------------------------------------------
    print("\n[SPARK INIT] Launching Apache Spark Session in Local Multi-Core Mode...")
    spark = get_spark_session(
        app_name="PM25_Research_Scalability",
        master="local[*]",
        driver_memory="4g",
        shuffle_partitions=8,
        enable_aqe=True,
        warehouse_dir="outputs/spark-warehouse",
        enable_hive=True,
    )
    print(f"Spark Version: {spark.version} | Master: {spark.sparkContext.master}")

    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("outputs/figures/checkpoint6")
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path("outputs/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    try:
        # -----------------------------------------------------
        # PHASE 2 & 7: HIVE DATA WAREHOUSE & STORAGE BENCHMARKS
        # -----------------------------------------------------
        print("\n[PHASE 2 & 7] Hive Data Warehouse & Storage Optimization Experiments...")
        hive_mgr = HiveWarehouseManager(spark, warehouse_dir="outputs/spark-warehouse")
        base_df = hive_mgr.load_base_dataframe(scale_catalog[1]["parquet_path"])

        # 1. Storage format comparison (CSV vs Parquet vs ORC)
        format_results = hive_mgr.benchmark_storage_formats(base_df)
        df_format = pd.DataFrame(format_results)
        df_format.to_csv(metrics_dir / "checkpoint6_storage_formats.csv", index=False)

        # 2. Partitioning strategies
        part_results = hive_mgr.benchmark_partitioning_strategies(base_df)
        df_part = pd.DataFrame(part_results)
        df_part.to_csv(metrics_dir / "checkpoint6_partitioning.csv", index=False)

        # 3. Caching vs Non-Caching
        caching_results = hive_mgr.benchmark_caching("sensor_format_parquet")
        with open(metrics_dir / "checkpoint6_caching.json", "w") as f:
            json.dump(caching_results, f, indent=2)

        # -----------------------------------------------------
        # PHASE 5: PANDAS VS SPARK SCALING BENCHMARK
        # -----------------------------------------------------
        print("\n[PHASE 5] Executing Comparative Scaling Benchmark (Pandas vs. Spark)...")
        bench_runner = BigDataBenchmarkRunner(spark)
        scaling_results = bench_runner.benchmark_scaling_comparison(
            scale_catalog=scale_catalog,
            repetitions=3,
            format="parquet",
        )
        df_scaling = pd.DataFrame(scaling_results)
        df_scaling.to_csv(metrics_dir / "checkpoint6_scaling_benchmark.csv", index=False)

        # -----------------------------------------------------
        # PHASE 6: HIVE / SPARK SQL QUERY BENCHMARK (Q1 - Q7)
        # -----------------------------------------------------
        print("\n[PHASE 6] Benchmarking Authoritative SQL Queries (Q1 – Q7)...")
        query_results = bench_runner.benchmark_sql_queries("sensor_format_parquet")
        df_queries = pd.DataFrame(query_results)
        df_queries.to_csv(metrics_dir / "checkpoint6_sql_queries.csv", index=False)

        # -----------------------------------------------------
        # PHASE 8: SPARK CONFIGURATION EXPERIMENTS
        # -----------------------------------------------------
        print("\n[PHASE 8] Evaluating Spark Configuration Tuning...")
        # Benchmark on 10x scale for meaningful shuffle impact
        config_results = bench_runner.benchmark_spark_configurations(
            benchmark_file=scale_catalog[10]["parquet_path"],
            format="parquet",
        )
        df_configs = pd.DataFrame(config_results)
        df_configs.to_csv(metrics_dir / "checkpoint6_spark_configs.csv", index=False)

        # -----------------------------------------------------
        # EXPORT 24-HOUR ANALYTICAL TABLES
        # -----------------------------------------------------
        print("\n[EXPORT] Exporting Canonical 24-Hour Processed Analytical Tables...")
        df_1x = spark.read.parquet("data/benchmark_scales/scale_1x.parquet")
        from src.bigdata.aggregation import aggregate_24h_spark
        daily_1x = aggregate_24h_spark(df_1x, min_hours=18)
        daily_1x.write.mode("overwrite").parquet("outputs/predictions/checkpoint6_spark_24h_aggregates_1x.parquet")
        daily_1x.toPandas().to_csv("outputs/predictions/checkpoint6_spark_24h_aggregates_1x.csv", index=False)
        print(f"Exported {daily_1x.count()} aggregated 24h days to outputs/predictions/")

        # -----------------------------------------------------
        # PHASE 10: GENERATE ALL 8 PUBLICATION FIGURES
        # -----------------------------------------------------
        print("\n[PHASE 10] Generating 8 Publication-Grade Performance Figures...")
        artifact_dir = "/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f"

        plot_execution_time_vs_dataset_size(scaling_results, out_dir=str(fig_dir), artifact_dir=artifact_dir)
        plot_throughput_vs_dataset_size(scaling_results, out_dir=str(fig_dir), artifact_dir=artifact_dir)
        plot_pandas_vs_spark_comparison(scaling_results, out_dir=str(fig_dir), artifact_dir=artifact_dir)
        plot_storage_format_comparison(format_results, out_dir=str(fig_dir), artifact_dir=artifact_dir)
        plot_sql_query_benchmark(query_results, out_dir=str(fig_dir), artifact_dir=artifact_dir)
        plot_partitioning_performance(part_results, out_dir=str(fig_dir), artifact_dir=artifact_dir)
        plot_caching_performance(caching_results, out_dir=str(fig_dir), artifact_dir=artifact_dir)
        plot_spark_config_comparison(config_results, out_dir=str(fig_dir), artifact_dir=artifact_dir)

    finally:
        print("\nStopping SparkSession...")
        stop_spark_session(spark)

    total_pipeline_time = time.perf_counter() - start_time
    print(f"\n================================================================================")
    print(f"CHECKPOINT 6 PIPELINE COMPLETE IN {total_pipeline_time:.2f} SECONDS")
    print(f"================================================================================")


if __name__ == "__main__":
    run_checkpoint6_pipeline()
