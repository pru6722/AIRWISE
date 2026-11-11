# Checkpoint 5 Phase 0 Data Audit Report

**Project:** Uncertainty-Aware Joint Calibration and Reconstruction of Low-Cost $\text{PM}_{2.5}$ Sensor Networks  
**Component:** Checkpoint 5 — $\text{PM}_{2.5}$-Based CPCB Category Reliability Assessment  
**Audit Date:** November 2026  
**Auditor:** Machine Learning Research & Verification Team  

---

## 1. Objective of Phase 0 Audit

Before implementing 24-hour windowing, CPCB category classification, and uncertainty-aware category stability analysis, this audit verifies the integrity of:
1. The canonical cleaned dataset and chronological split boundaries.
2. The frozen Joint TCN model and normalization pipeline.
3. The exact test partition row counts and site breakdown.
4. The numerical reconciliation of the reported $\text{BAM} > 250\,\mu\text{g}/\text{m}^3$ observations.
5. The column schemas and prediction alignments across calibration and test partitions.

---

## 2. Dataset and Partition Integrity Verification

### 2.1 Cleaned Parquet Dataset
- **File Path:** `data/processed/cleaned_data.parquet`
- **Total Validated Observations:** 18,584 hourly records across Delhi (7,504), Hamirpur (3,639), and Bangalore (7,441).
- **Features Preserved:** `timestamp`, `site`, `bam_reference`, `pa_cf1`, `pa_atm`, `pa_alt`, `relative_humidity`, `temperature`, `dew_point_raw`, `is_dew_point_suspicious`, `time_diff_hours`, `is_contiguous_step`, `segment_id`.
- **Integrity Status:** MD5/SHA256 verified against Checkpoint 1 baseline; no rows deleted, modified, or reordered.

### 2.2 Canonical Four-Way Chronological Split
The dataset was partitioned using `src.uncertainty.split_manager.create_conformal_splits`:
- **Training (60%):** 11,149 observations (Delhi: 4,502; Bangalore: 4,464; Hamirpur: 2,183)
- **Model-Selection Validation (10%):** 1,857 observations (Delhi: 750; Bangalore: 744; Hamirpur: 363)
- **Conformal Calibration (10%):** 1,858 observations (Delhi: 750; Bangalore: 744; Hamirpur: 364)
- **Final Evaluation Test (20%):** 3,720 observations (Delhi: 1,502; Bangalore: 1,489; Hamirpur: 729)
- **Total Sum:** $11,149 + 1,857 + 1,858 + 3,720 = 18,584$ observations. Exact 100% population preservation.

---

## 3. Reconciliation of BAM > 250 µg/m³ Count Discrepancy

A discrepancy was identified between previous research reports:
- *Checkpoint 3 Revised Diagnostics:* Reported **257** observations with $\text{BAM} > 250\,\mu\text{g}/\text{m}^3$.
- *Checkpoint 4 Overall Report:* Reported **263** observations with $\text{BAM} > 250\,\mu\text{g}/\text{m}^3$.

### Empirical Reconciliation
Querying `bam_reference > 250.0` on the canonical test partition ($N=3,720$) yields:
- **Delhi Test Partition ($N=1,502$):** Exactly **257** observations ($17.11\%$ of Delhi test data).
- **Hamirpur Test Partition ($N=729$):** Exactly **6** observations ($0.82\%$ of Hamirpur test data).
- **Bangalore Test Partition ($N=1,489$):** Exactly **0** observations ($0.00\%$ of Bangalore test data).
- **Pooled Test Partition ($N=3,720$):** Exactly $257 + 6 + 0 = \mathbf{263}$ observations ($7.07\%$ of pooled test data).

### Cause and Resolution
- The Checkpoint 3 diagnostic report focused strictly on **Delhi high-concentration diagnostics** ($N = 257$).
- The Checkpoint 4 overall table aggregated across **all three study sites** ($N = 263$).
- **Resolution:** There is zero data inconsistency. Both numbers are mathematically correct: 257 is the site-specific count for Delhi; 263 is the pooled count across Delhi and Hamirpur. Going forward, all reports will explicitly state whether a statistic applies to Delhi only ($N = 257$) or the pooled test partition ($N = 263$).

---

## 4. Frozen Model and Feature Pipeline Verification

- **Model Checkpoint:** `outputs/models/conformal/frozen_joint_tcn_checkpoint.pt`
- **Feature Normalizer:** `outputs/models/conformal/normalizer.joblib`
- **Test Predictions File:** `outputs/predictions/conformal_test_predictions.csv` ($N = 3,720$)
- **Reproducibility Test:** Re-running `predict_joint_tcn_full_sequence` using the frozen weights and normalizer on the test partition produced a maximum absolute difference of $3.73 \times 10^{-5}\,\mu\text{g}/\text{m}^3$ compared to `conformal_test_predictions.csv` (attributable to standard floating-point precision).
- **Frozen Status:** The model weights and normalizer remain strictly frozen. Calibration and test observations will NOT be used for retraining, tuning, or feature adaptation.

---

## 5. Column Identification for Checkpoint 5

The exact schemas for the 24-hour aggregation and CPCB category assessment pipeline:

| Column Name in Test DataFrame / Predictions | Physical Meaning | Role in Checkpoint 5 |
| :--- | :--- | :--- |
| `timestamp` | UTC Datetime of observation | Chronological sorting, natural gap detection, 24h window framing |
| `site` | Monitoring site (Delhi, Hamirpur, Bangalore) | Site stratification, preventing cross-site window construction |
| `bam_reference` / `true_bam` | Reference $\text{PM}_{2.5}$ concentration ($\mu\text{g}/\text{m}^3$) | Target for 24h reference mean and true category classification |
| `point_prediction` | Frozen Joint TCN point estimate ($\mu\text{g}/\text{m}^3$) | Basis for 24h predicted mean and predicted category classification |
| `lower_90`, `upper_90` | 90% Conformal prediction interval bounds | Diagnostic comparison against 24h direct conformal bounds |
| `is_contiguous_step` / `time_diff_hours` | Contiguity indicator ($\Delta t = 1.0\text{h}$) | Defining unbroken segments to prevent spanning natural data gaps |
| `segment_id` | Contiguous temporal run identifier | Strict boundary enforcement for non-overlapping 24h windows |

---

## 6. Audit Conclusion and Approval

All data files, split boundaries, frozen model weights, and sample counts have been verified with complete consistency. Phase 0 is complete and approved for proceeding to Phase 1 (CPCB Category Breakpoints Verification).
