#!/usr/bin/env python3
"""
Research Pipeline: Checkpoint 1 Execution Script
Performs end-to-end dataset audit, data quality checks, unit physics investigation,
cleaned Parquet persistence, and report generation for the Campmier air quality dataset.
"""

from pathlib import Path
import sys
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import load_from_config
from src.data.validation import run_full_dataset_audit
from src.data.preprocessing import preprocess_and_flag_dataset, save_cleaned_dataset, load_cleaned_dataset
from src.visualization.plots import (
    plot_raw_vs_bam_timeseries,
    plot_sensor_vs_bam_scatter,
    plot_dew_point_physics_investigation,
    plot_temporal_coverage_and_gaps,
    plot_site_distributions,
)
from src.visualization.report import generate_csv_reports, generate_html_report


def main():
    print("=" * 70)
    print("STARTING RESEARCH AUDIT & DATA ENGINEERING — CHECKPOINT 1")
    print("=" * 70)

    # 1. Load configuration
    config_path = PROJECT_ROOT / "configs" / "data_config.yaml"
    print(f"[1/6] Loading data configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Ingest raw workbook
    print("[2/6] Ingesting raw Excel workbook (all 3 sheets)...")
    combined_df, site_dfs = load_from_config(config_path)
    print(f"      Total pooled records loaded: {len(combined_df):,}")
    for site, df_site in site_dfs.items():
        print(f"      - {site:10s}: {len(df_site):,} records, {len(df_site.columns)} columns")

    # 3. Comprehensive Data Quality and Physics Audit
    print("[3/6] Running data quality, temporal continuity, and thermodynamics audit...")
    audit_report = run_full_dataset_audit(combined_df, site_dfs)
    
    # 4. Preprocess, flag anomalies, and save Parquet
    print("[4/6] Preprocessing and saving canonical cleaned Parquet...")
    cleaned_df = preprocess_and_flag_dataset(combined_df)
    parquet_path = save_cleaned_dataset(cleaned_df, cfg.get("processed_data_path", "data/processed/cleaned_data.parquet"))
    print(f"      Cleaned dataset saved to: {parquet_path.resolve()}")
    print(f"      Shape: {cleaned_df.shape}, File size: {parquet_path.stat().st_size / 1024:.1f} KB")

    # Verify loading parquet works cleanly
    verified_df = load_cleaned_dataset(parquet_path)
    assert len(verified_df) == len(cleaned_df), "Parquet record count mismatch!"
    print("      Verification check passed: Parquet loaded and verified successfully.")

    # 5. Generate high-resolution figures
    figures_dir = PROJECT_ROOT / cfg.get("output_figures_dir", "outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"[5/6] Generating publication-quality figures in: {figures_dir}")

    fig_delhi = plot_raw_vs_bam_timeseries(cleaned_df, "Delhi", figures_dir / "fig1_raw_vs_bam_delhi.png")
    print(f"      - Generated: {fig_delhi.name}")

    fig_hamirpur = plot_raw_vs_bam_timeseries(cleaned_df, "Hamirpur", figures_dir / "fig1_raw_vs_bam_hamirpur.png")
    print(f"      - Generated: {fig_hamirpur.name}")

    fig_bangalore = plot_raw_vs_bam_timeseries(cleaned_df, "Bangalore", figures_dir / "fig1_raw_vs_bam_bangalore.png")
    print(f"      - Generated: {fig_bangalore.name}")

    fig_scatter = plot_sensor_vs_bam_scatter(site_dfs, figures_dir / "fig1_raw_vs_bam_scatter.png")
    print(f"      - Generated: {fig_scatter.name}")

    fig_dp = plot_dew_point_physics_investigation(site_dfs, figures_dir / "fig_dew_point_inconsistency.png")
    print(f"      - Generated: {fig_dp.name}")

    fig_gaps = plot_temporal_coverage_and_gaps(site_dfs, figures_dir / "fig_temporal_coverage_and_gaps.png")
    print(f"      - Generated: {fig_gaps.name}")

    fig_dist = plot_site_distributions(cleaned_df, figures_dir / "fig_sensor_distribution_by_site.png")
    print(f"      - Generated: {fig_dist.name}")

    # 6. Generate CSV and HTML reports
    reports_dir = PROJECT_ROOT / cfg.get("output_reports_dir", "outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"[6/6] Compiling dataset audit reports in: {reports_dir}")

    csv_paths = generate_csv_reports(audit_report, reports_dir)
    for key, p in csv_paths.items():
        print(f"      - CSV Export: {p.name}")

    html_path = generate_html_report(audit_report, reports_dir / "dataset_audit_report.html")
    print(f"      - HTML Report: {html_path.resolve()}")

    print("=" * 70)
    print("CHECKPOINT 1 SUCCESSFULLY EXECUTED AND VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    main()
