# AIRWISE — Phase 3C.3 Limitations, Scientific Scope, and Cryptographic Reproducibility

**Document ID:** `outputs/reports/airwise_phase3C_limitations_and_reproducibility.md`  
**Phase:** 3C.3 (Manuscript Synthesis and Publication-Ready Reporting)  
**Author:** Antigravity Autonomous Research Assistant  
**Date:** November 2026  
**Status:** Methodological Qualifications & Cryptographic Register  

---

## 1. Scientific Scope and Practical Limitations

Transparent reporting of scientific boundaries is critical for environmental policy and reproducible artificial intelligence. The findings in AIRWISE must be interpreted within the following qualified boundaries:

### 1.1 Geographic Scope and Regional Generalization
- **Three Representative Environments:** The empirical dataset comprises three locations in India: Bangalore (tropical savanna plateau), Delhi (Indo-Gangetic plain with extreme winter inversions), and Hamirpur (sub-Himalayan mountainous valley).
- **No National Generalization:** While these sites represent distinct climatic and emission regimes, they do not establish universal generalization across all geographical microclimates of the Indian subcontinent (e.g., coastal marine aerosol regimes such as Mumbai or Chennai, arid desert dust regimes such as Rajasthan, or industrial coal-belt regions).
- **Spatial Transferability Caution:** Calibrations fitted in one geographical zone cannot be directly transferred to another without local recalibration or domain adaptation.

### 1.2 Extended Outage Reconstruction Limits
- **Degradation Beyond 12–24 Hours:** Reconstructing fine particulate matter concentrations during extended multi-day sensor outages ($> 24$ hours) remains a fundamental challenge for all four pipelines.
- **Compounding State Divergence:** Sequence models (Joint TCN v2) suffer severe error accumulation when low-cost sensor features remain unobserved for consecutive cycles ($81.86$ $\mu$g/m³ on 24-hour outages). In the absence of co-located physical proxies or spatial neighboring sensors, statistical imputers fall back to static diurnal averages.

### 1.3 Synthetic Missingness Mechanisms vs. Physical Hardware Failure
- **Controlled Simulation:** The missingness scenarios (20% MCAR, 1–24h block outages, MAR-RH, MAR-Lag, and MNAR-PM) provide mathematically controlled environments to evaluate algorithmic robustness.
- **Physical Failure Cascades:** Synthetic dropout models cannot perfectly capture the full complexity of in-situ physical degradation, such as progressive optical lens fouling, spiderweb or insect intrusion into optical chambers, fan bearing degradation, intermittent power surging, or analog-to-digital converter quantization freeze.

### 1.4 Oracle-Generated Synthetic MNAR-PM
- **Algorithmic Stress-Test:** The MNAR-PM scenario evaluates algorithmic behavior when dropouts correlate with true particulate intensity. An oracle function computes dropout propensity from true particulate levels to generate the mask $m_t$.
- **Not Real-World Detection:** In real-world operational deployments, the true particulate concentration is unobservable during an outage, and sensors do not emit an explicit MNAR diagnostic flag. This benchmark tests conditional imputation performance given known outage boundaries; it does **not** demonstrate that the models can autonomously detect or flag MNAR missingness in real-time.

### 1.5 Air Quality Metric Boundaries
- **$\text{PM}_{2.5}$ Focus:** All evaluations in this study reflect hourly dry mass concentration of fine particulate matter ($\text{PM}_{2.5}$) in $\mu$g/m³.
- **Not the Full National AQI:** While $\text{PM}_{2.5}$ is frequently the driving sub-index pollutant in India, the statutory National Air Quality Index (NAQI) established by the Central Pollution Control Board (CPCB) mandates the integration of up to eight criteria pollutants ($\text{PM}_{10}, \text{PM}_{2.5}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3, \text{NH}_3, \text{Pb}$). Our findings should not be interpreted as a full-spectrum NAQI assessment.

### 1.6 Sensor Drift and Aging Claims
- **Empirical Relationships:** Observations of temporal error variation in our multi-month field campaign reflect combined changes in ambient meteorology, seasonal source emission profiles, and potential sensor aging.
- **No Independent Physical Drift Validation:** We do not claim independently isolated physical sensor aging or component drift. Isolate physical drift from environmental confounding requires multi-year zero-air and span-gas laboratory chamber calibration.

### 1.7 Computational Infrastructure Context
- **Local Big Data Execution:** Big data scalability benchmarks were conducted using Apache Spark running in local multi-core mode within an isolated runtime environment.
- **Distributed Cluster Qualification:** Execution timing, shuffle overhead, and partition throughput under local execution do not directly establish distributed-scale network performance across multi-node physical clusters.

### 1.8 Non-Causal Nature of Meteorological Associations
- **Empirical Correlation:** Associations observed between relative humidity, temperature, and calibration error are empirical correlations identified by regression trees and convolutional filters.
- **No Direct Causal Claims:** While consistent with known aerosol physics (hygroscopic water uptake), these models do not establish direct microphysical causality in the absence of in-situ optical hygroscopicity measurements (e.g., $f(\text{RH})$ humidified nephelometry).

---

## 2. Statistical Uncertainty Reporting Standards

### 2.1 Paired Moving-Block Bootstrap Protocol
- **Autocorrelation Preservation:** Standard i.i.d. bootstrap resampling and conventional paired $t$-tests severely underestimate variance in environmental time series. All statistical inference in AIRWISE utilizes segment-constrained Moving-Block Bootstrap (MBB).
- **Primary Uncertainty Evidence:** The 95% bootstrap percentile confidence intervals of paired differences ($\Delta_{\text{RMSE}} = \text{RMSE}_{\text{Joint}} - \text{RMSE}_{\text{Baseline}}$) serve as the primary statistical evidence.

### 2.2 Rejection of Spurious Significance Under Short Blocks
- **Block-Length Sensitivity:** Under a 12-hour block specification ($b=12\text{h}$), Joint TCN v2 appeared statistically superior on 20% MCAR ($p = 0.010$). However, because 12 hours bisects the 24-hour diurnal atmospheric cycle, adjacent blocks remain strongly correlated.
- **Diurnal Robustness:** When the full diurnal cycle is preserved ($b=24\text{h}$), the confidence interval expands from $[-5.27, -0.37]$ to $[-4.31, +0.70]$ and the empirical tail proportion shifts to $p = 0.084$, confirming non-significance at $\alpha = 0.05$. At $b=48\text{h}$ ($p = 0.384$) and $b=72\text{h}$ ($p = 0.652$), the null hypothesis is unequivocally retained.

### 2.3 Characterization of Multi-Site Differences
- **Performance Heterogeneity:** Differences across Bangalore, Delhi, and Hamirpur are formally reported as geographic and environmental performance heterogeneity. In MAR-RH, this heterogeneity explains why the pooled point estimate ($\Delta = -0.325$) diverges from the bootstrap median ($+2.248$), as Hamirpur's high squared errors pull the sample estimate, while Delhi and Bangalore (representing 80.4% of samples) pull the bootstrap resamples.

---

## 3. Cryptographic Provenance & Reproducibility Register

Every experimental artifact, model checkpoint, and data partition used in this benchmark is permanently cataloged with its cryptographic SHA-256 checksum:

### 3.1 Model Checkpoint Checksums

| Checkpoint Path | Architecture | Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `outputs/models/conformal/frozen_joint_tcn_checkpoint.pt` | Joint TCN v1 (Frozen) | 122,165 bytes | `3eae6ea2b0f6c7abdbca72df672c61f068d78b4e49c9006e0eb54ac8a2fa7350` |
| `outputs/models/joint_tcn/best_joint_tcn_checkpoint.pt` | Joint TCN v1 Best (Frozen) | 122,165 bytes | `ab94386fe408a4d580f3c01b1926c0022c3843b0ef26d864297382c81d37e37c` |
| `outputs/models/joint_tcn/joint_tcn_v2_causal_checkpoint.pt` | Joint TCN v2 Causal (Phase 3C) | 122,165 bytes | `4a0c5704257fd556e53cfc23b1c26439881009f8aabb423b3f6f2752e7c5217c` |

### 3.2 Experiment Data and Prediction Artifacts

| Artifact File Path | Description | Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `data/processed/cleaned_data.parquet` | Cleaned paired dataset ($N=18,584$) | 1,489,120 bytes | `9a8cb2c7e0fef318859942a926f2bb283dd98101684c3c6f85ff294e6378e9b6` |
| `outputs/experiments/phase3C/benchmark_predictions.parquet` | Predictions for all 4 pipelines | 914,278 bytes | `87e5263742403644d96605f9bc742de6e5009a29d8eb5509c149b8b154a4f662` |
| `outputs/experiments/phase3C/scenario_metrics_overall.csv` | Overall metrics ($N=3,720$) | 5,756 bytes | `d0b0da45438d4604c9f79b95fe568a355ced9edba221ec82d838f3906941e4c4` |
| `outputs/experiments/phase3C/scenario_metrics_missing_pm.csv` | Outage metrics ($m_t=0$) | 4,885 bytes | `0eb1f6cadc1462f181e3ef757c675dc5964774e50ac641c61c6fa91575b9ac7e` |
| `outputs/experiments/phase3C/scenario_metrics_common_support.csv`| Common support ($m_t=1$) | 5,709 bytes | `1fa28543a629f2a413aa38e752e0dedd16abb500d34dbb97644ad6236a357cf7` |
| `outputs/experiments/phase3C/site_stratified_metrics.csv` | Site-stratified breakdown | 3,491 bytes | `29396a9a0dc8b21b2c6edf0ec636038b2094b3dba64e6c0e4e04e626916e47f6` |
| `outputs/experiments/phase3C/bootstrap_significance_results.csv`| Bootstrap comparisons | 3,495 bytes | `824c18a2706f1f8206d7b840d1789a4b0a6c20fe4264f21e3f171c817be44b25` |
| `outputs/experiments/phase3C/bootstrap_sensitivity_sweep.csv` | Block sensitivity sweep | 1,071 bytes | `d7ef5ef2a2b4b98cecd0990f20e8a3b212bad6f26512f53f247542801b892891` |

### 3.3 Unit and Regression Test Verification
The complete test suite verifies mathematical constraints, causality invariance, shape consistency, and pipeline separation:
```bash
python3 -m unittest discover -s tests -v
```
- **Total Test Cases:** 81
- **Passed:** 80
- **Skipped:** 1 (`test_spark_and_pandas_numerical_consistency`, safely bypassed due to local sandbox loopback restriction)
- **Failed:** 0
- **Errors:** 0

All experimental records, model checkpoints, and analytical reports are fully archived and permanently auditable.
