# Checkpoint 3 Final Consistency Audit

**Document Purpose:** Formal resolution of numerical discrepancies and population ambiguities identified during post-evaluation peer review.  
**Primary Auditor:** Lead Research Data Scientist & Machine Learning Engineer  
**Date:** November 2026  

---

## 1. Resolution of Identified Audit Items

### Item 1: Delhi Test Partition Count Discrepancy (1,501 vs. 1,502)
- **Observation:** Earlier report summaries cited "Delhi ($N=1,501$)", whereas the three stratified pollution bands summed to $725 + 520 + 257 = 1,502$.
- **Audit Finding:** The true, verified row count of `test_df[test_df['site'] == 'Delhi']` is **exactly 1,502 rows**.
- **Root Cause:** Total rows in Delhi = 7,504. Under a 60/20/20 split:
  - Train (60%): $\lfloor 7,504 \times 0.60 \rfloor = 4,502$ rows.
  - Val (20%): $\lfloor 7,504 \times 0.20 \rfloor = 1,500$ rows.
  - Test (Remainder 20%): $7,504 - 4,502 - 1,500 = \mathbf{1,502}$ rows.
  An early textual draft mistakenly printed 1,501 due to rounding $7,504 \times 0.20 = 1,500.8 \approx 1,501$ instead of taking the actual dataframe slice length.
- **Resolution:** The correct population count is **1,502**, perfectly matching the sum of its three mutually exclusive pollution bands ($725 + 520 + 257 = 1,502$).

### Item 2: Raw PA_ATM RMSE Discrepancy (25.80 vs. 31.9 µg/m³)
- **Observation:** Checkpoint 2 reported pooled raw PA_ATM test RMSE as 25.80, while a later Delhi diagnostic cited 31.9.
- **Audit Finding:** Both numbers are mathematically correct, but represent **different geographical populations**:
  - **25.80 $\mu\text{g}/\text{m}^3$** is the **Pooled Full Test RMSE** across all three cities combined ($N = 3,720$; Delhi, Hamirpur, Bangalore).
  - **31.95 $\mu\text{g}/\text{m}^3$** (rounded to 31.9) is the **Delhi-Specific Full Test RMSE** ($N = 1,502$).
  - **26.30 $\mu\text{g}/\text{m}^3$** is the **Pooled Hidden Test RMSE** across all three cities ($N = 769$).
  - **34.52 $\mu\text{g}/\text{m}^3$** is the **Delhi Hidden Test RMSE** ($N = 311$).
- **Resolution:** Table labels must explicitly state "Pooled (3 Sites)" versus "Delhi Only" to prevent reader confusion.

### Item 3: Verification of Exact Timestamp Populations
- Audited against `data/processed/cleaned_data.parquet` and `create_chronological_splits()`:
  - **Train Partition (60%):** $N = 11,149$ (Delhi: 4,502; Hamirpur: 2,183; Bangalore: 4,464).
  - **Validation Partition (20%):** $N = 3,715$ (Delhi: 1,500; Hamirpur: 727; Bangalore: 1,488).
  - **Test Partition (20%):** $N = 3,720$ (Delhi: 1,502; Hamirpur: 729; Bangalore: 1,489).
  - **Sum Verification:** $11,149 + 3,715 + 3,720 = 18,584$ observations (100.0% of dataset).

### Item 4: Verification of Missingness Masks & Target Alignment
- Confirmed that `benchmark_test_mask_mcar_20.npy` was applied identically to all models:
  - `mask_rec == mask_tcn == mask_saved` across all 3,720 rows (`True` for 100% of entries).
  - Exactly 769 hidden points and 2,951 observed points.
  - Reference targets: Ground-truth BAM values are 100% identical (`np.allclose = True`).
  - Block outage masks ($g \in \{1, 3, 6, 12, 24\}$): Deterministically generated with seed 42.

### Item 5: Information Setting Parity
- Documented in `checkpoint3_information_setting_audit.md`. Causal TCN uses filtration $\mathcal{F}_t$ (past only), while two-stage interpolation uses filtration $\mathcal{F}_{\mathrm{all}}$ (past and future boundary anchors).

### Item 6: Test Contamination Audit
- Confirmed: Zero test features or test BAM targets were used during feature normalization, gradient updates, or early stopping in Checkpoint 3.
- Contamination Alert for Checkpoint 4: The Checkpoint 3 model checkpoint was tuned on the entire 20% validation split ($N=3,715$). For Checkpoint 4, Phase 1 requires splitting this into 10% model-selection validation and 10% conformal calibration. To avoid contaminating the conformal calibration set, **the TCN must be retrained from scratch on Train (60%) and the new 10% Validation set**, strictly freezing weights before calculating calibration residuals.
