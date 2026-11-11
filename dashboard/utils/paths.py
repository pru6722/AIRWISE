"""
AIRWISE Dashboard: Centralized Path Management
Resolves all paths to existing research datasets, model checkpoints, metric tables, and reports.
"""

import sys
from pathlib import Path

# Project root resolution (dashboard/utils/paths.py -> dashboard -> Project)
DASHBOARD_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DASHBOARD_DIR.parent

# Guarantee project root and dashboard dir are in sys.path
for p in [str(PROJECT_ROOT), str(DASHBOARD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Centralized Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_EXCEL_PATH = DATA_DIR / "raw" / "Campmier_Manuscript_Data.xlsx"
CLEANED_PARQUET_PATH = DATA_DIR / "processed" / "cleaned_data.parquet"
BENCHMARK_SCALES_DIR = DATA_DIR / "benchmark_scales"
SYNTHETIC_MASKS_DIR = DATA_DIR / "synthetic_masks"
CANONICAL_MASK_PATH = SYNTHETIC_MASKS_DIR / "benchmark_test_mask_mcar_20.npy"

# Configuration Paths
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_CONFIG_PATH = CONFIGS_DIR / "data_config.yaml"
MODEL_CONFIG_PATH = CONFIGS_DIR / "model_config.yaml"
EXPERIMENT_CONFIG_PATH = CONFIGS_DIR / "experiment_config.yaml"

# Outputs & Model Checkpoint Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
CALIBRATION_MODELS_DIR = MODELS_DIR / "calibration"
JOINT_TCN_MODELS_DIR = MODELS_DIR / "joint_tcn"
CONFORMAL_MODELS_DIR = MODELS_DIR / "conformal"
RECONSTRUCTION_MODELS_DIR = MODELS_DIR / "reconstruction"

# Specific Critical Model Artifacts
JOINT_TCN_BEST_PATH = JOINT_TCN_MODELS_DIR / "best_joint_tcn_checkpoint.pt"
JOINT_TCN_FROZEN_PATH = CONFORMAL_MODELS_DIR / "frozen_joint_tcn_checkpoint.pt"
JOINT_TCN_NORMALIZER_PATH = JOINT_TCN_MODELS_DIR / "normalizer.joblib"
CONFORMAL_NORMALIZER_PATH = CONFORMAL_MODELS_DIR / "normalizer.joblib"
CONFORMAL_QUANTILES_PATH = CONFORMAL_MODELS_DIR / "conformal_quantiles.json"
CONFORMAL_SPLIT_SUMMARY_PATH = CONFORMAL_MODELS_DIR / "split_summary.json"

# Predictions & Metrics Paths
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

# Phase 3A Experiment Paths
PHASE3A_RECON_DIR = EXPERIMENTS_DIR / "phase3A_reconstruction_corrected"
CORRECTED_RECONSTRUCTION_METRICS_CSV = PHASE3A_RECON_DIR / "corrected_reconstruction_metrics.csv"
CORRECTED_RECONSTRUCTION_PREDICTIONS_CSV = PHASE3A_RECON_DIR / "corrected_reconstruction_predictions.csv"

PHASE3A_CONFORMAL_DIR = EXPERIMENTS_DIR / "phase3A_conformal_reconciliation"
CONFORMAL_COVERAGE_RECONCILIATION_CSV = PHASE3A_CONFORMAL_DIR / "conformal_coverage_reconciliation.csv"
CONFORMAL_SIMULTANEOUS_BLOCK_COVERAGE_CSV = PHASE3A_CONFORMAL_DIR / "conformal_simultaneous_block_coverage.csv"

PHASE3A_CPCB_DIR = EXPERIMENTS_DIR / "phase3A_cpcb_controlled"
CPCB_CONTROLLED_COMPARISON_CSV = PHASE3A_CPCB_DIR / "cpcb_controlled_comparison.csv"
CPCB_STABILITY_ASSESSMENT_CSV = PHASE3A_CPCB_DIR / "cpcb_stability_assessment.csv"
CALIBRATION_BY_POLLUTION_RANGE_CSV = REPORTS_DIR / "calibration_by_pollution_range.csv"
HUMIDITY_METRICS_CSV = REPORTS_DIR / "humidity_metrics.csv"

# Specific Prediction Files
CALIBRATION_PREDICTIONS_CSV = PREDICTIONS_DIR / "calibration_predictions.csv"
RECONSTRUCTION_PREDICTIONS_CSV = PREDICTIONS_DIR / "reconstruction_predictions.csv"
JOINT_TCN_PREDICTIONS_CSV = PREDICTIONS_DIR / "joint_tcn_test_predictions.csv"
CONFORMAL_PREDICTIONS_CSV = PREDICTIONS_DIR / "conformal_test_predictions.csv"
CHECKPOINT5_24H_PREDICTIONS_CSV = PREDICTIONS_DIR / "checkpoint5_24h_predictions.csv"
CHECKPOINT5_24H_INTERVALS_CSV = PREDICTIONS_DIR / "checkpoint5_24h_intervals.csv"
CHECKPOINT5_CATEGORY_ASSESSMENT_CSV = PREDICTIONS_DIR / "checkpoint5_category_assessment.csv"
CHECKPOINT6_SPARK_AGGREGATES_CSV = PREDICTIONS_DIR / "checkpoint6_spark_24h_aggregates_1x.csv"
CHECKPOINT6_SPARK_AGGREGATES_PARQUET = PREDICTIONS_DIR / "checkpoint6_spark_24h_aggregates_1x.parquet"

# Specific Report & Metric Files
DATASET_AUDIT_CSV = REPORTS_DIR / "dataset_audit_report.csv"
SENSOR_BIAS_SUMMARY_CSV = REPORTS_DIR / "sensor_bias_summary.csv"
BASELINE_SUMMARY_CSV = REPORTS_DIR / "baseline_summary.csv"
PROPAGATION_GAP_METRICS_CSV = REPORTS_DIR / "propagation_gap_metrics.csv"
JOINT_TCN_METRICS_CSV = REPORTS_DIR / "joint_tcn_metrics.csv"
JOINT_TCN_ABLATION_CSV = REPORTS_DIR / "joint_tcn_ablation.csv"
CONFORMAL_COVERAGE_CSV = REPORTS_DIR / "conformal_coverage.csv"
CONFORMAL_SITE_METRICS_CSV = REPORTS_DIR / "conformal_site_metrics.csv"
CHECKPOINT5_24H_POINT_METRICS_CSV = METRICS_DIR / "checkpoint5_24h_point_metrics.csv"
CHECKPOINT5_24H_COVERAGE_METRICS_CSV = METRICS_DIR / "checkpoint5_24h_coverage_metrics.csv"
CHECKPOINT5_CATEGORY_METRICS_CSV = METRICS_DIR / "checkpoint5_category_metrics.csv"
CHECKPOINT5_CATEGORY_STABILITY_CSV = METRICS_DIR / "checkpoint5_category_stability.csv"
CHECKPOINT5_BOUNDARY_ANALYSIS_CSV = METRICS_DIR / "checkpoint5_boundary_analysis.csv"
CHECKPOINT5_SITE_RELIABILITY_CSV = METRICS_DIR / "checkpoint5_site_reliability.csv"
CHECKPOINT6_SCALING_BENCHMARK_CSV = METRICS_DIR / "checkpoint6_scaling_benchmark.csv"
CHECKPOINT6_STORAGE_FORMATS_CSV = METRICS_DIR / "checkpoint6_storage_formats.csv"
CHECKPOINT6_PARTITIONING_CSV = METRICS_DIR / "checkpoint6_partitioning.csv"
CHECKPOINT6_CACHING_JSON = METRICS_DIR / "checkpoint6_caching.json"
CHECKPOINT6_SQL_QUERIES_CSV = METRICS_DIR / "checkpoint6_sql_queries.csv"

# Phase 3C Benchmark Paths
PHASE3C_DIR = EXPERIMENTS_DIR / "phase3C"
BENCHMARK_PREDICTIONS_PARQUET = PHASE3C_DIR / "benchmark_predictions.parquet"
SCENARIO_METRICS_OVERALL_CSV = PHASE3C_DIR / "scenario_metrics_overall.csv"
SCENARIO_METRICS_MISSING_PM_CSV = PHASE3C_DIR / "scenario_metrics_missing_pm.csv"
SCENARIO_METRICS_COMMON_SUPPORT_CSV = PHASE3C_DIR / "scenario_metrics_common_support.csv"
BOOTSTRAP_SIGNIFICANCE_CSV = PHASE3C_DIR / "bootstrap_significance_results.csv"
BOOTSTRAP_SENSITIVITY_CSV = PHASE3C_DIR / "bootstrap_sensitivity_sweep.csv"
SITE_STRATIFIED_METRICS_CSV = PHASE3C_DIR / "site_stratified_metrics.csv"
PHASE3C_FIGURES_DIR = PHASE3C_DIR / "figures"

# Specific Model Checkpoint & Metadata Paths for Phase 3C
JOINT_TCN_V2_CHECKPOINT_PATH = JOINT_TCN_MODELS_DIR / "joint_tcn_v2_causal_checkpoint.pt"
JOINT_TCN_V2_METADATA_PATH = JOINT_TCN_MODELS_DIR / "joint_tcn_v2_metadata.json"

# Big Data & Hive Warehouse Paths
SPARK_WAREHOUSE_DIR = OUTPUTS_DIR / "spark-warehouse"
HIVE_DB_DIR = SPARK_WAREHOUSE_DIR / "pm25_research.db"
