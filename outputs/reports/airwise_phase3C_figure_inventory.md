# AIRWISE — Phase 3C.3 Publication Figure and Table Inventory

**Document ID:** `outputs/reports/airwise_phase3C_figure_inventory.md`  
**Phase:** 3C.3 (Manuscript Synthesis and Publication-Ready Reporting)  
**Author:** Antigravity Autonomous Research Assistant  
**Date:** November 2026  
**Status:** Complete Visual and Tabular Artifact Catalog  

---

## 1. Publication Figure Catalog

This catalog details the complete set of publication-quality figures prepared for the AIRWISE manuscript, including source data provenance, dimensions, and scientific interpretation.

### Figure 1: AIRWISE System Architecture
- **File Path:** [fig_joint_tcn_architecture.png](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/figures/fig_joint_tcn_architecture.png)
- **Source Module:** `src/visualization/tcn_plots.py`
- **Underlying Model:** Causal Joint TCN v2 architecture (`src/models/joint_tcn.py`)
- **Description:** Complete architectural schematic showing input multivariate channels (low-cost $\text{PM}_{2.5}$, meteorology, cyclical time encodings, binary mask), stacked dilated causal residual convolution blocks ($d \in \{1, 2, 4, 8\}$, $k=3$, $RF=61$h), `ChannelLayerNorm1d` normalization, and dual heads (calibration and reconstruction) with composite output synthesis.
- **Role in Manuscript:** Section 9 (Proposed AIRWISE Framework).

---

### Figure 2: Dataset and Preprocessing Summary
- **File Paths:**
  1. [fig_temporal_coverage_and_gaps.png](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/figures/fig_temporal_coverage_and_gaps.png)
  2. [fig_sensor_distribution_by_site.png](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/figures/fig_sensor_distribution_by_site.png)
- **Source Module:** `src/visualization/plots.py`
- **Underlying Data:** `data/processed/cleaned_data.parquet` ($N = 18,584$)
- **Description:** Two-panel publication visualization illustrating:
  - Panel A: Multi-month temporal coverage, sampling continuity, and natural sequence segment gap boundaries across Bangalore, Delhi, and Hamirpur.
  - Panel B: Empirical probability density distributions of raw sensor readings ($\text{PA\_CF1}, \text{PA\_ATM}, \text{PA\_ALT}$) vs. regulatory BAM ground truth, highlighting severe baseline differences between Bangalore ($28.4$ $\mu$g/m³) and Delhi ($98.7$ $\mu$g/m³).
- **Role in Manuscript:** Section 7 (Dataset and Data Provenance).

---

### Figure 3: Four-Pipeline Methodology Comparison
- **File Path:** [four_pipeline_methodology_comparison.png](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3C/figures/four_pipeline_methodology_comparison.png)
- **Source Script:** Phase 3C publication synthesis suite
- **Underlying Protocol:** `outputs/reports/airwise_phase3C_experiment_protocol.md`
- **Description:** Comprehensive architectural comparison diagram contrasting:
  - Pipeline A: $\text{Impute} \to \text{Calibrate}$ (Feature-level imputation followed by post-imputation calibration).
  - Pipeline B: $\text{Calibrate} \to \text{Impute}$ (Direct calibration on observed records followed by target-level reconstruction).
  - Pipeline C: Causal Joint TCN v2 (End-to-end multi-task deep sequence modeling).
  - Pipeline D: Complete-Case Reference (Calibration on complete instances only, outputting NaNs during outages).
- **Role in Manuscript:** Section 10 (Methodology) and Section 12 (Baseline Methods).

---

### Figure 4: Overall RMSE Comparison Across Missingness Scenarios
- **File Path:** [benchmark_rmse_by_scenario.png](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3C/figures/benchmark_rmse_by_scenario.png)
- **Source Script:** `src/evaluation/run_phase3C_benchmark.py`
- **Underlying Artifact:** `outputs/experiments/phase3C/scenario_metrics_overall.csv`
- **Description:** Grouped bar chart comparing overall full-population RMSE ($N = 3,720$) across all nine missingness regimes (20% MCAR, 1h, 3h, 6h, 12h, 24h block outages, MAR-RH, MAR-Lag, MNAR-PM). Visually demonstrates that Pipeline B consistently achieves the lowest overall error ($36.26$ to $41.64$ $\mu$g/m³) across all scenarios.
- **Role in Manuscript:** Section 13 (Results and Comparative Analysis).

---

### Figure 5: Low-Cost Sensor Reconstruction Comparison on Missing Subgroup
- **File Path:** [missing_pm_reconstruction_comparison.png](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3C/figures/missing_pm_reconstruction_comparison.png)
- **Source Script:** Phase 3C publication synthesis suite
- **Underlying Artifact:** `outputs/experiments/phase3C/scenario_metrics_missing_pm.csv`
- **Description:** Performance bar chart strictly isolating the missing PM subgroup ($m_t = 0$), comparing reconstruction fidelity between Pipelines A, B, and C during actual sensor dropouts. Highlights Joint TCN v2's advantage on short dropouts ($52.44$ vs $53.39$ on 1h blocks; $43.05$ vs $51.77$ on MAR-RH) and its severe breakdown on extended 24-hour outages ($81.86$ $\mu$g/m³).
- **Role in Manuscript:** Section 14 (Missingness Regime Analysis).

---

### Figure 6: Outage-Duration Error Degradation
- **File Path:** [block_outage_degradation.png](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3C/figures/block_outage_degradation.png)
- **Source Script:** `src/evaluation/run_phase3C_benchmark.py`
- **Underlying Artifact:** `outputs/experiments/phase3C/scenario_metrics_overall.csv` and `scenario_metrics_missing_pm.csv`
- **Description:** Continuous line plot illustrating RMSE degradation curves as block outage duration extends from 1 hour to 3, 6, 12, and 24 hours. Demonstrates the compounding error trajectory in deep sequence modeling (Joint TCN v2 error increases by $+29.4$ $\mu$g/m³ on the missing subgroup) compared to the stable behavior of Pipeline B.
- **Role in Manuscript:** Section 14 (Missingness Regime Analysis).

---

### Figure 7: Site-Stratified Performance Comparison
- **File Path:** [site_stratified_performance.png](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3C/figures/site_stratified_performance.png)
- **Source Script:** Phase 3C publication synthesis suite
- **Underlying Artifact:** `outputs/experiments/phase3C/site_stratified_metrics.csv`
- **Description:** Multi-panel plot stratifying model RMSE across Bangalore ($N=1,489$), Delhi ($N=1,502$), and Hamirpur ($N=729$) across representative scenarios (20% MCAR, Block 6h, MNAR-PM). Visually proves environmental performance heterogeneity: Pipeline B dominates in Bangalore and Delhi, while Joint TCN v2 dominates in Hamirpur.
- **Role in Manuscript:** Section 16 (Site-Level Analysis).

---

### Figure 8: Common-Support Calibration Comparison
- **File Path:** [common_support_comparison.png](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3C/figures/common_support_comparison.png)
- **Source Script:** `src/evaluation/run_phase3C_benchmark.py`
- **Underlying Artifact:** `outputs/experiments/phase3C/scenario_metrics_common_support.csv`
- **Description:** Bar chart comparing calibration accuracy on matched, clean observations ($m_t = 1$, $N = 2,951$). Illustrates that Pipelines A, B, and D achieve identical calibration performance ($29.82$ $\mu$g/m³), while Joint TCN v2 suffers an $8.19$ $\mu$g/m³ penalty due to multi-task gradient interference.
- **Role in Manuscript:** Section 11 (Common-Support Evaluation).

---

### Figure 9: Bootstrap Confidence-Interval Sensitivity Sweep
- **File Path:** [bootstrap_ci_sensitivity.png](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3C/figures/bootstrap_ci_sensitivity.png)
- **Source Script:** Phase 3C publication synthesis suite
- **Underlying Artifact:** `outputs/experiments/phase3C/bootstrap_sensitivity_sweep.csv`
- **Description:** Error-band plot depicting 95% moving-block bootstrap confidence intervals for $\Delta_{\text{RMSE}}$ across block lengths $b \in \{12, 24, 48, 72\}$ hours for MCAR, Block 6h, and MNAR-PM. Shows monotonic interval widening and demonstrates why statistical significance under $b=12$h collapses once the diurnal cycle ($b=24$h) is preserved.
- **Role in Manuscript:** Section 15 (Statistical Uncertainty and Robustness).

---

## 2. Publication Table Catalog

| Table Identifier | Title in Manuscript | Source Artifact | Rows $\times$ Cols | Description |
| :--- | :--- | :--- | :---: | :--- |
| **Table 1** | Dataset Summary and Partition Breakdown | `src/uncertainty/split_manager.py` | 5 $\times$ 5 | Breakdown of observations across Bangalore, Delhi, and Hamirpur into Train, Val, Calib, and Test splits. |
| **Table 2** | Empirical Outage Duration Distributions | `benchmark_predictions.parquet` | 5 $\times$ 8 | Min, median, mean, and max continuous outage run lengths across 1h, 3h, 6h, 12h, and 24h block scenarios. |
| **Table 3** | Overall Performance Across Nine Missingness Scenarios | `scenario_metrics_overall.csv` | 36 $\times$ 10 | Complete full-population ($N=3,720$) RMSE, MAE, $R^2$, and MBE for Pipelines A, B, C, and D. |
| **Table 4** | Reconstruction Error on Missing PM Subgroup | `scenario_metrics_missing_pm.csv` | 9 $\times$ 7 | Low-cost sensor reconstruction RMSE strictly on outage timesteps ($m_t = 0$), including $\Delta(\text{C} - \text{B})$. |
| **Table 5** | Common-Support Calibration Metrics | `scenario_metrics_common_support.csv` | 4 $\times$ 8 | Matched calibration accuracy on observed points ($m_t = 1$, $N = 2,951$), confirming floating-point identity of A, B, and D. |
| **Table 6** | Paired Moving-Block Bootstrap Results | `bootstrap_significance_results.csv` | 19 $\times$ 7 | Point $\Delta_{\text{RMSE}}$, 95% CIs, and empirical $p$-values at canonical $b=24$h, $B=1,000$ iterations. |
| **Table 7** | Sensitivity of Bootstrap Confidence Intervals to Block Length | `bootstrap_sensitivity_sweep.csv` | 12 $\times$ 8 | Bootstrap CI lower, upper, width, and $p$-values across $b \in \{12, 24, 48, 72\}$ hours for MCAR, Block 6h, and MNAR-PM. |
| **Table 8** | Site-Stratified Performance Breakdown | `site_stratified_metrics.csv` | 15 $\times$ 8 | Per-site RMSE, MAE, and $R^2$ across Bangalore, Delhi, and Hamirpur for representative missingness regimes. |

---

## 3. Provenance and Artifact Traceability Matrix

Every figure and table is strictly generated from frozen, verified experimental data. No value, line, bar, or interval has been fabricated or manually altered.

```
data/processed/cleaned_data.parquet (SHA-256: 9a8cb2c7...)
   │
   ├──> outputs/figures/fig_sensor_distribution_by_site.png
   └──> outputs/figures/fig_temporal_coverage_and_gaps.png

outputs/models/joint_tcn/joint_tcn_v2_causal_checkpoint.pt (SHA-256: 4a0c5704...)
   │
   └──> outputs/figures/fig_joint_tcn_architecture.png

outputs/experiments/phase3C/benchmark_predictions.parquet (SHA-256: 87e52637...)
   │
   ├──> Table 2 (Outage Durations)
   ├──> scenario_metrics_overall.csv (SHA-256: d0b0da45...) ─────────> Table 3 & Figure 4, 6
   ├──> scenario_metrics_missing_pm.csv (SHA-256: 0eb1f6cad...) ─────> Table 4 & Figure 5, 6
   ├──> scenario_metrics_common_support.csv (SHA-256: 1fa28543...) ─> Table 5 & Figure 8
   ├──> site_stratified_metrics.csv (SHA-256: 29396a9a...) ─────────> Table 8 & Figure 7
   ├──> bootstrap_significance_results.csv (SHA-256: 824c18a2...) ──> Table 6
   └──> bootstrap_sensitivity_sweep.csv (SHA-256: d7ef5ef2...) ─────> Table 7 & Figure 9
```
