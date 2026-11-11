"""
Visualization Module for Checkpoint 6 Big Data Scalability.
Generates publication-quality figures for:
1. Execution time vs dataset size
2. Throughput vs dataset size
3. Pandas vs Spark comparison and speedup
4. CSV vs Parquet vs ORC storage size & query scan time
5. SQL query execution times (Q1 - Q7)
6. Partitioning performance and small-file trade-offs
7. Caching vs non-caching acceleration
8. Spark configuration sensitivity (shuffle partitions, AQE)
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Publication styling palette
STYLE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "coral": "#E05A47",
    "amber": "#D99B26",
    "purple": "#5C3A82",
    "gray": "#555555",
    "light_gray": "#EAEAEA",
}


def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
    })


def copy_figure_to_artifacts(fig_path: Path, artifact_dir: str):
    art_path = Path(artifact_dir) / "figures"
    art_path.mkdir(parents=True, exist_ok=True)
    target = art_path / fig_path.name
    shutil.copy2(fig_path, target)
    print(f"Copied figure to artifact: {target}")


def plot_execution_time_vs_dataset_size(
    scaling_results: List[Dict[str, Any]],
    out_dir: str = "outputs/figures/checkpoint6",
    artifact_dir: str = "/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f",
):
    setup_matplotlib()
    df = pd.DataFrame(scaling_results)
    fig, ax = plt.subplots(figsize=(8, 5))

    rows = df["rows"]
    ax.plot(rows, df["pandas_median_sec"], marker="o", linewidth=2.2, color=STYLE["coral"], label="Pandas (Single-Core In-Memory)")
    ax.plot(rows, df["spark_median_sec"], marker="s", linewidth=2.2, color=STYLE["navy"], label="Apache Spark (Local Multi-Core)")

    ax.set_title("Data Processing Execution Time vs. Dataset Volume", weight="bold")
    ax.set_xlabel("Dataset Volume (Number of Observations)")
    ax.set_ylabel("Median Execution Time (Seconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=True)

    # Annotate scales
    for _, r in df.iterrows():
        ax.annotate(
            f"{r['scale']}× ({r['rows']:,})",
            (r["rows"], r["spark_median_sec"]),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8,
            color=STYLE["navy"],
        )

    out_p = Path(out_dir) / "fig_execution_time_vs_dataset_size.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p)
    plt.close()
    copy_figure_to_artifacts(out_p, artifact_dir)


def plot_throughput_vs_dataset_size(
    scaling_results: List[Dict[str, Any]],
    out_dir: str = "outputs/figures/checkpoint6",
    artifact_dir: str = "/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f",
):
    setup_matplotlib()
    df = pd.DataFrame(scaling_results)
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df["pandas_throughput_rows_sec"], width, label="Pandas Throughput", color=STYLE["coral"], alpha=0.9)
    ax.bar(x + width/2, df["spark_throughput_rows_sec"], width, label="Spark Throughput", color=STYLE["navy"], alpha=0.9)

    ax.set_title("Processing Throughput (Rows/Second) Across Dataset Scales", weight="bold")
    ax.set_xlabel("Benchmark Scale Multiplier")
    ax.set_ylabel("Throughput (Observations Processed per Second)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}×\n({r:,})" for s, r in zip(df["scale"], df["rows"])])
    ax.legend(frameon=True)

    out_p = Path(out_dir) / "fig_throughput_vs_dataset_size.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p)
    plt.close()
    copy_figure_to_artifacts(out_p, artifact_dir)


def plot_pandas_vs_spark_comparison(
    scaling_results: List[Dict[str, Any]],
    out_dir: str = "outputs/figures/checkpoint6",
    artifact_dir: str = "/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f",
):
    setup_matplotlib()
    df = pd.DataFrame(scaling_results)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax2 = ax1.twinx()
    x = np.arange(len(df))
    width = 0.35

    ax1.bar(x - width/2, df["pandas_median_sec"], width, label="Pandas Runtime (s)", color=STYLE["coral"])
    ax1.bar(x + width/2, df["spark_median_sec"], width, label="Spark Runtime (s)", color=STYLE["navy"])
    ax2.plot(x, df["spark_speedup_vs_pandas"], color=STYLE["amber"], marker="D", linewidth=2.5, label="Spark Speedup (×)")

    ax1.set_xlabel("Benchmark Scale Multiplier")
    ax1.set_ylabel("Execution Time (Seconds)")
    ax2.set_ylabel("Speedup Multiplier (Pandas Time / Spark Time)")
    ax1.set_title("Pandas vs. Apache Spark Execution Time and Speedup", weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}×\n({r:,})" for s, r in zip(df["scale"], df["rows"])])
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Parity (1.0×)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    out_p = Path(out_dir) / "fig_pandas_vs_spark_comparison.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p)
    plt.close()
    copy_figure_to_artifacts(out_p, artifact_dir)


def plot_storage_format_comparison(
    format_results: List[Dict[str, Any]],
    out_dir: str = "outputs/figures/checkpoint6",
    artifact_dir: str = "/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f",
):
    setup_matplotlib()
    df = pd.DataFrame(format_results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Bar 1: Disk storage size
    colors = [STYLE["coral"], STYLE["teal"], STYLE["navy"]]
    bars1 = ax1.bar(df["format"], df["size_mb"], color=colors, alpha=0.9, width=0.5)
    ax1.set_title("Storage Footprint by Format (MB)", weight="bold")
    ax1.set_ylabel("Disk Size (MB)")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} MB", ha="center", va="bottom", fontsize=9)

    # Bar 2: Query scan time
    bars2 = ax2.bar(df["format"], df["query_time_sec"], color=colors, alpha=0.9, width=0.5)
    ax2.set_title("Aggregation Query Scan Time (Seconds)", weight="bold")
    ax2.set_ylabel("Execution Time (Seconds)")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f} s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_p = Path(out_dir) / "fig_storage_format_comparison.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p)
    plt.close()
    copy_figure_to_artifacts(out_p, artifact_dir)


def plot_sql_query_benchmark(
    query_results: List[Dict[str, Any]],
    out_dir: str = "outputs/figures/checkpoint6",
    artifact_dir: str = "/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f",
):
    setup_matplotlib()
    df = pd.DataFrame(query_results)
    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df["median_runtime_sec"], color=STYLE["navy"], alpha=0.85, height=0.55)

    ax.set_yticks(y_pos)
    labels = [q.replace("_", " ") for q in df["query_id"]]
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel("Median Execution Time (Seconds)")
    ax.set_title("Spark SQL / Hive Analytical Query Performance (Q1 – Q7)", weight="bold")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2.0, f"{w:.3f}s", ha="left", va="center", fontsize=9)

    out_p = Path(out_dir) / "fig_sql_query_benchmark.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p)
    plt.close()
    copy_figure_to_artifacts(out_p, artifact_dir)


def plot_partitioning_performance(
    part_results: List[Dict[str, Any]],
    out_dir: str = "outputs/figures/checkpoint6",
    artifact_dir: str = "/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f",
):
    setup_matplotlib()
    df = pd.DataFrame(part_results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Subplot 1: File count (small-file problem)
    bars1 = ax1.bar(df["strategy"], df["file_count"], color=STYLE["purple"], alpha=0.85, width=0.5)
    ax1.set_title("Partitioning Granularity vs. File Count", weight="bold")
    ax1.set_ylabel("Total Number of Part Files")
    ax1.tick_params(axis="x", rotation=25)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.5, f"{int(b.get_height())}", ha="center", va="bottom", fontsize=9)

    # Subplot 2: Pruned query time
    bars2 = ax2.bar(df["strategy"], df["pruned_query_time_sec"], color=STYLE["teal"], alpha=0.85, width=0.5)
    ax2.set_title("Targeted Query Execution Time (Seconds)", weight="bold")
    ax2.set_ylabel("Query Time (Seconds)")
    ax2.tick_params(axis="x", rotation=25)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.005, f"{b.get_height():.3f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_p = Path(out_dir) / "fig_partitioning_performance.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p)
    plt.close()
    copy_figure_to_artifacts(out_p, artifact_dir)


def plot_caching_performance(
    caching_res: Dict[str, Any],
    out_dir: str = "outputs/figures/checkpoint6",
    artifact_dir: str = "/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f",
):
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    categories = ["Uncached (Cold/Disk)", "Cached (In-Memory Hot)"]
    times = [caching_res["median_uncached_sec"], caching_res["median_cached_sec"]]
    colors = [STYLE["coral"], STYLE["teal"]]

    bars = ax.bar(categories, times, color=colors, width=0.45, alpha=0.9)
    ax.set_title(f"Spark In-Memory Caching Effect (Speedup: {caching_res['cache_speedup']}×)", weight="bold")
    ax.set_ylabel("Median Query Runtime (Seconds)")

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2.0, h + 0.005, f"{h:.3f} s", ha="center", va="bottom", fontsize=10)

    out_p = Path(out_dir) / "fig_caching_performance.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p)
    plt.close()
    copy_figure_to_artifacts(out_p, artifact_dir)


def plot_spark_config_comparison(
    config_results: List[Dict[str, Any]],
    out_dir: str = "outputs/figures/checkpoint6",
    artifact_dir: str = "/Users/prudhvisailingineni/.gemini/antigravity/brain/af0ce7d5-1831-49dd-a6a6-7c3f0fc1be6f",
):
    setup_matplotlib()
    df = pd.DataFrame(config_results)
    fig, ax = plt.subplots(figsize=(9, 4.8))

    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df["median_runtime_sec"], color=STYLE["navy"], alpha=0.85, height=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Median 24h Aggregation Runtime (Seconds)")
    ax.set_title("Spark Shuffle Partitions & Adaptive Query Execution (AQE)", weight="bold")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2.0, f"{w:.3f}s", ha="left", va="center", fontsize=9)

    out_p = Path(out_dir) / "fig_spark_config_comparison.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p)
    plt.close()
    copy_figure_to_artifacts(out_p, artifact_dir)
