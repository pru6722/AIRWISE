# AIRWISE — Phase 3A.4: Controlled CPCB Category Evaluation Report (Final Verification Patch)
**Document ID:** `AIRWISE-P3A4-CPCB-003`  
**Date:** November 9, 2026  
**Status:** Complete — Audited & Verified (Documentation-Only Update)  
**Scope:** Controlled benchmark comparing Raw PurpleAir sensor channels (CF=1 and ATM) against Calibrated Joint TCN predictions across identical 24-hour evaluation windows under official CPCB PM2.5 breakpoints and completeness policies. Verification of prediction provenance, historical BatchNorm checkpoint limitation disclosure, qualification of physical mechanisms, resolution of the historical 13.43% ambiguity vs genuine calibration gains, and mathematical analysis of quadratic weighted kappa.

---

## Executive Summary

Phase 3A.4 establishes the first **statistically controlled, identical-window comparison** between raw low-cost optical sensor telemetry and calibrated deep learning predictions under the Central Pollution Control Board (CPCB) regulatory framework. This verification patch establishes exact data provenance, discloses the historical checkpoint's architectural limitation, and qualifies all causal assertions:

1. **Controlled Experimental Design:**
   - Both raw sensor channels (`pa_cf1` and `pa_atm`) and calibrated model predictions were aggregated into 24-hour daily means over **identical non-overlapping calendar windows** strictly evaluated against true Federal Equivalent Method (FEM) Beta Attenuation Monitor (BAM-1020) reference measurements.
   - Enforced the official CPCB **$\ge 18/24\text{ hours}$ (75%) valid-hour completeness rule**, yielding exactly **127 valid 24-hour test days** across Bangalore ($N=55$), Delhi ($N=56$), and Hamirpur ($N=16$) from 403 total evaluated windows (276 rejected due to sensor/reference downtime).
2. **Empirical Improvement Over Raw Sensors:**
   - **Calibrated Model (Joint TCN):** Achieves **72.44% category accuracy**, with a Macro F1 of **58.96%**, Weighted F1 of **71.54%**, and Quadratic Cohen's Kappa of **0.9271**.
   - **Raw PurpleAir (CF=1):** Achieves only **39.37% accuracy** (Macro F1 28.03%, Weighted F1 43.00%). Calibration delivers an improvement of **+33.07 percentage points** in regulatory accuracy.
   - **Raw PurpleAir (ATM):** Achieves **66.14% accuracy** (Macro F1 55.76%, Weighted F1 66.21%). Calibration delivers an improvement of **+6.30 percentage points** in accuracy and **+3.20 percentage points** in Macro F1.
3. **Historical Checkpoint Architectural Limitation:**
   - The underlying model checkpoint (`outputs/models/conformal/frozen_joint_tcn_checkpoint.pt`) uses **`nn.BatchNorm1d`** (with running buffers `running_mean`, `running_var`, and `num_batches_tracked`).
   - Its training-time temporal sequence statistics leakage was formally identified in Phase 3A.2, and it was not retrained with `ChannelLayerNorm1d`.
   - The Phase 3A.4 evaluation reflects the performance of this **historical checkpoint with this documented limitation**. These results do not establish causality-safe training performance. All historical weights, quantiles, and predictions remain preserved without retraining.
4. **Quadratic Weighted Cohen's Kappa Nuance:**
   - Quadratic weighted kappa depends on the complete confusion matrix and category marginals (the distribution of row and column totals and the chance-expected agreement matrix $E_{ij}$, in addition to quadratic distance weights $w_{ij} = (i - j)^2$).
   - We retain the observed comparison: **Raw ATM achieved 0.9392** versus **0.9271 for the calibrated Joint TCN**. While the calibrated model achieved higher exact diagonal agreement (92 vs 84 days), differences in marginal distributions and off-diagonal configurations across categories yielded a higher quadratic kappa for raw ATM.
5. **Severe Category Recall (0/4):**
   - The calibrated model failed to identify any of the 4 Severe test days ($>250\,\mu\text{g}/\text{m}^3$), misclassifying all 4 as Very Poor due to extreme-range model shrinkage and physical sensor non-linearities.
6. **Forensic Resolution of the Historical 13.43% Metric:**
   - The historical figure of **13.43 percentage points** ($83.33\% - 69.90\%$) was **NOT** an improvement of calibrated models over raw sensors.
   - It represents the **accuracy spread between Stable prediction intervals (83.33%) and Ambiguous prediction intervals (69.90%) within the calibrated model itself**.
   - Conflating interval stability with calibration gain is scientifically invalid. The true calibration gain is **+33.07%** vs CF=1 and **+6.30%** vs ATM.
7. **Prediction Provenance Disclosure:**
   - Calibrated predictions were computed directly from the frozen historical artifact [`outputs/predictions/checkpoint5_category_assessment.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/predictions/checkpoint5_category_assessment.csv), which inherits hourly predictions from [`outputs/predictions/conformal_test_predictions.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/predictions/conformal_test_predictions.csv).
   - Raw PurpleAir channels (`pa_cf1`, `pa_atm`) were averaged from [`data/processed/cleaned_data.parquet`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/data/processed/cleaned_data.parquet) across the exact same window timestamps. Zero models were retrained.
8. **Regulatory Scope Qualification:**
   - All assessments in this project are strictly **PM2.5-based CPCB category classifications**, not multi-pollutant National Air Quality Index (NAQI) scores.

---

## 1. Methodology and Evaluation Design

### 1.1 The CPCB PM2.5 Regulatory Framework
The Central Pollution Control Board (CPCB) of India specifies six discrete health risk categories based on 24-hour average PM2.5 mass concentration ($\mu\text{g}/\text{m}^3$):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CPCB 24-HOUR PM2.5 REGULATORY BREAKPOINTS                       │
├───────────────┬────────────────────┬───────────┬───────────────────────────────────────┤
│ Category Code │ Category Name      │ PM2.5 Band│ Health Warning / Advisory             │
├───────────────┼────────────────────┼───────────┼───────────────────────────────────────┤
│ 0             │ Good               │ 0 – 30    │ Minimal impact                        │
│ 1             │ Satisfactory       │ 31 – 60   │ Minor breathing discomfort to sens.   │
│ 2             │ Moderate           │ 61 – 90   │ Breathing discomfort to asthma/heart  │
│ 3             │ Poor               │ 91 – 120  │ Breathing discomfort to most people   │
│ 4             │ Very Poor          │ 121 – 250 │ Respiratory illness on prol. exposure │
│ 5             │ Severe             │ > 250     │ Affects healthy people, serious resp. │
└───────────────┴────────────────────┴───────────┴───────────────────────────────────────┘
```

### 1.2 Aggregation and Window Completeness Rules
Under CPCB protocol, sub-daily air quality readings cannot be directly assigned to an AQI category; they must be aggregated into a 24-hour mean:
$$\bar{y}_{24h} = \frac{1}{K} \sum_{k=1}^K y_k$$
where $K$ is the number of valid hourly observations in the 24-hour window ($[t_{\text{start}}, t_{\text{start}} + 24\text{h})$).
* **The 18/24 Completeness Policy:** A 24-hour window is valid if and only if $K \ge 18$ valid hourly readings ($\ge 75\%$ temporal completeness). Windows with $K < 18$ are rejected.
* **Non-Overlapping Partitions:** Windows are constructed strictly within site boundaries and never cross chronological partitions (Train, Val, Calib, Test).
* **Window Population Statistics:**
  - Evaluated windows: **403 windows**
  - Rejected windows (<18h valid data): **276 windows**
  - **Eligible valid 24h test windows:** **127 windows**
    - Delhi: **56 windows** (Valid rate: 30.27%)
    - Bangalore: **55 windows** (Valid rate: 48.67%)
    - Hamirpur: **16 windows** (Valid rate: 15.24%)

---

## 2. Controlled Empirical Comparison: Raw vs Calibrated

Evaluated across the exact same 127 test windows against true 24-hour BAM-1020 measurements:
Persisted in [`outputs/experiments/phase3A_cpcb_controlled/cpcb_controlled_comparison.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3A_cpcb_controlled/cpcb_controlled_comparison.csv).

| Model / Sensor Channel | Eligible Windows | Category Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | Quadratic Cohen's Kappa | Exact Matches ($N=127$) | Adjacent Errors ($|\Delta| = 1$) | Large Errors ($|\Delta| \ge 2$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw PurpleAir (CF=1)** | 127 | **39.37%** | 28.03% | 43.00% | 0.8792 | 50 (39.4%) | 68 (53.5%) | **9 (7.1%)** |
| **Raw PurpleAir (ATM)** | 127 | **66.14%** | 55.76% | 66.21% | **0.9392** | 84 (66.1%) | 43 (33.9%) | **0 (0.0%)** |
| **Calibrated Model (Joint TCN)** | 127 | **72.44%** | **58.96%** | **71.54%** | 0.9271 | **92 (72.4%)** | 33 (26.0%) | **2 (1.6%)** |

### Empirical Gain Breakdown:
1. **Calibrated Model vs Raw CF=1:**
   - Accuracy Improvement: **+33.07 percentage points** ($72.44\%$ vs $39.37\%$).
   - Macro F1 Improvement: **+30.93 percentage points** ($58.96\%$ vs $28.03\%$).
   - Large Errors ($|\Delta| \ge 2$ categories): Reduced from 9 days down to 2 days (78% reduction).
2. **Calibrated Model vs Raw ATM:**
   - Accuracy Improvement: **+6.30 percentage points** ($72.44\%$ vs $66.14\%$).
   - Macro F1 Improvement: **+3.20 percentage points** ($58.96\%$ vs $55.76\%$).
   - Weighted F1 Improvement: **+5.33 percentage points** ($71.54\%$ vs $66.21\%$).
3. **Mathematical Analysis of Quadratic Weighted Kappa:**
   - Quadratic weighted kappa is defined as:
     $$\kappa_{\text{quad}} = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$
     where $O_{ij}$ is the observed confusion matrix, $E_{ij} = \frac{R_i C_j}{N}$ is the expected matrix under chance agreement based on row and column marginal totals, and $C=6$ categories.
   - Because $\kappa_{\text{quad}}$ depends on the complete confusion matrix and category marginals, the chance-expected disagreement term $\sum w_{ij} E_{ij}$ differs between the two classifiers.
   - We retain the observed comparison: **Raw ATM achieved 0.9392** versus **0.9271 for the calibrated Joint TCN**. While the calibrated model achieved higher exact diagonal agreement (92 vs 84 days) and fewer total errors (35 vs 43), differences across the entire confusion matrix and category marginal distributions resulted in a slightly higher quadratic score for raw ATM.

---

## 3. Confusion Matrix Analysis (6×6 Regulatory Grid)

Rows represent True BAM-1020 Categories; columns represent Predicted Categories.  
Persisted in [`outputs/experiments/phase3A_cpcb_controlled/cpcb_confusion_matrices.json`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3A_cpcb_controlled/cpcb_confusion_matrices.json).

```
========================================================================================================
1. RAW PURPLEAIR (CF=1) CONFUSION MATRIX (Overall Accuracy: 39.37%)
========================================================================================================
True \ Pred         Good    Satisfactory    Moderate        Poor    Very Poor      Severe      Total
Good                  25              22           0           0            0           0         47
Satisfactory           0              13          13           5            0           0         31
Moderate               0               0           0           1            4           0          5
Poor                   0               0           0           0           12           0         12
Very Poor              0               0           0           0            8          20         28
Severe                 0               0           0           0            0           4          4
--------------------------------------------------------------------------------------------------------
Diagnosis: CF=1 exhibits systematic positive bias, pushing Satisfactory into Moderate/Poor,
Poor into Very Poor, and Very Poor into Severe. 9 days suffered large errors (|Delta| >= 2).

========================================================================================================
2. RAW PURPLEAIR (ATM) CONFUSION MATRIX (Overall Accuracy: 66.14%)
========================================================================================================
True \ Pred         Good    Satisfactory    Moderate        Poor    Very Poor      Severe      Total
Good                  26              21           0           0            0           0         47
Satisfactory           0              24           7           0            0           0         31
Moderate               0               0           4           1            0           0          5
Poor                   0               0           0           4            8           0         12
Very Poor              0               0           0           0           25           3         28
Severe                 0               0           0           0            3           1          4
--------------------------------------------------------------------------------------------------------
Diagnosis: ATM applies factory calibration scaling. All 43 errors were adjacent (|Delta| = 1),
but it underperformed on Poor (only 4/12 correct; 8 misclassified as Very Poor).

========================================================================================================
3. CALIBRATED JOINT TCN CONFUSION MATRIX (Overall Accuracy: 72.44%)
========================================================================================================
True \ Pred         Good    Satisfactory    Moderate        Poor    Very Poor      Severe      Total
Good                  28              18           0           1            0           0         47
Satisfactory           1              30           0           0            0           0         31
Moderate               0               1           4           0            0           0          5
Poor                   0               0           3           9            0           0         12
Very Poor              0               0           1           6           21           0         28
Severe                 0               0           0           0            4           0          4
--------------------------------------------------------------------------------------------------------
Diagnosis: Calibrated model achieves superior Satisfactory (96.8%) and Poor (75.0%) recall.
However, during severe winter episodes in Delhi, it under-predicts all 4 Severe days into Very Poor.
========================================================================================================
```

---

## 4. Category-Wise Performance and Recall Audit

Persisted in [`outputs/experiments/phase3A_cpcb_controlled/cpcb_category_recalls.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3A_cpcb_controlled/cpcb_category_recalls.csv).

| CPCB Category | Ground Truth Support ($N$) | Raw PurpleAir (CF=1) Recall | Raw PurpleAir (ATM) Recall | Calibrated Model Recall | Calibrated Model Precision | Calibrated Model F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Good** | 47 | 53.2% | 55.3% | **59.6%** | 96.6% | **73.7%** |
| **Satisfactory** | 31 | 41.9% | 77.4% | **96.8%** | 61.2% | **75.0%** |
| **Moderate** | 5 | 0.0% | 80.0% | **80.0%** | 50.0% | **61.5%** |
| **Poor** | 12 | 0.0% | 33.3% | **75.0%** | 56.2% | **64.3%** |
| **Very Poor** | 28 | 28.6% | **89.3%** | **75.0%** | 100.0% | **85.7%** |
| **Severe** | 4 | **100.0%** | 25.0% | **0.0%** | 0.0% | **0.0%** |

### Qualified Findings:
1. **Satisfactory Category Improvement:** Calibrated predictions substantially reduce the systematic over-prediction seen in raw CF=1, which is consistent with mitigating humidity-associated positive bias, lifting Satisfactory category recall from 41.9% to **96.8%** (30/31 correct).
2. **Poor Category Recovery:** In the Poor category ($91\text{--}120\,\mu\text{g}/\text{m}^3$), Raw CF=1 achieved 0.0% recall, and Raw ATM achieved 33.3%. Calibrated TCN achieves **75.0% recall**, enabling actionable public health warnings before pollution crosses into Very Poor.
3. **Severe Category Limitation:** The calibrated model achieved **0.0% recall (0/4 days)** on the Severe category ($>250\,\mu\text{g}/\text{m}^3$). This severe under-prediction is consistent with documented optical nephelometer coincidence loss and photodiode saturation at high particle loads, combined with conservative regression shrinkage under Huber loss on rare extreme values.

---

## 5. Site-Wise Performance Breakdown

Persisted in [`outputs/experiments/phase3A_cpcb_controlled/cpcb_site_wise_comparison.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3A_cpcb_controlled/cpcb_site_wise_comparison.csv).

| Monitoring Station | Valid 24h Windows | Raw PurpleAir (CF=1) Acc (%) | Raw PurpleAir (ATM) Acc (%) | Calibrated Model Acc (%) | Calibrated Macro F1 (%) | Calibrated Quad Kappa |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bangalore** | 55 | 60.00% | 65.45% | **69.09%** | **66.95%** | 0.4003 |
| **Delhi** | 56 | 30.36% | 69.64% | **69.64%** | **56.99%** | 0.8721 |
| **Hamirpur** | 16 | **0.00%** | 56.25% | **93.75%** | **90.20%** | **0.9810** |

---

## 6. Forensic Audit of the 13.43% Difference

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          THE 13.43 PERCENTAGE-POINT AUDIT                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Reported Figure: +13.43 percentage points                                              │
│ True Calculation: Accuracy(Stable Windows) - Accuracy(Ambiguous Windows)               │
│                   = 83.33% - 69.90% = +13.43%                                          │
│ Scientific Meaning: The internal confidence spread of conformal stability flagging     │
│ INVALID Interpretation: Claiming this as the calibrated-vs-raw accuracy improvement   │
│ VALID Improvement: +33.07% vs Raw CF=1 (72.44% vs 39.37%)                              │
│                    +6.30% vs Raw ATM (72.44% vs 66.14%)                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

In Checkpoint 5, the model partitioned predictions into **Stable** and **Ambiguous** subsets based on conformal prediction intervals:
* **Stable Windows ($N=24$, 18.9%):** Point classification accuracy was **83.33%**.
* **Ambiguous Windows ($N=103$, 81.1%):** Point classification accuracy was **69.90%**.
* **The Accuracy Spread:** $83.33\% - 69.90\% = \mathbf{13.43\%}$.

This 13.43 percentage-point difference is an **uncertainty-awareness quality metric**, demonstrating that when conformal intervals do not cross CPCB boundaries, the user can place substantially higher trust in the point classification. It must **never** be cited as the accuracy improvement of calibration over raw PurpleAir sensors.

---

## 7. Contrast: 24h Block Conformal vs 24h Daily-Mean CPCB Calibration

A common source of confusion is equating 24-hour block conformal with 24-hour CPCB daily-mean conformal intervals:

| Attribute | 24-Hour Block Conformal (Phase 3A.3) | 24-Hour Daily-Mean CPCB Calibration (Phase 3A.4) |
|---|---|---|
| **Data Structure** | Strictly contiguous 24-hour sequences ($n_{\text{seg}} // 24$) | Wall-clock 24-hour days meeting $\ge 18/24\text{h}$ completeness |
| **Calibration Sample Size** | **32 blocks** | **78 valid windows** |
| **Test Sample Size** | 59 blocks (or 3,720 hourly points) | **127 valid windows** |
| **Nonconformity Score** | Block maximum hourly residual: $\max_{t} \|y_t - \hat{y}_t\|$ | Scalar daily-mean residual: $\|\bar{y}_{24h} - \hat{\bar{y}}_{24h}\|$ |
| **Quantile ($90\%$)** | **$100.42\,\mu\text{g}/\text{m}^3$** | **$15.30\,\mu\text{g}/\text{m}^3$** |
| **Interval Width ($90\%$)** | **$200.84\,\mu\text{g}/\text{m}^3$** | **$30.59\,\mu\text{g}/\text{m}^3$** |
| **Primary Purpose** | Simultaneous joint protection across individual hours | Category boundary stability around daily average |

These two methods operate on different sample units and different mathematical targets; their quantiles and metrics are separate.

---

## 8. Uncertainty and Category Boundaries

Persisted in [`outputs/experiments/phase3A_cpcb_controlled/cpcb_stability_assessment.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3A_cpcb_controlled/cpcb_stability_assessment.csv).

| Metric | All Windows ($N=127$) | Stable Windows ($N=24$, 18.9%) | Ambiguous Windows ($N=103$, 81.1%) |
|---|:---:|:---:|:---:|
| **Point Category Accuracy** | **72.44%** | **83.33%** | **69.90%** |
| **Category Compatibility (%)** | **92.13%** | **83.33%** | **94.17%** |
| **Mean Distance to Nearest Boundary** | 12.83 $\mu\text{g}/\text{m}^3$ | **32.80 $\mu\text{g}/\text{m}^3$** | **8.17 $\mu\text{g}/\text{m}^3$** |
| **Median Distance to Nearest Boundary** | 8.59 $\mu\text{g}/\text{m}^3$ | **41.95 $\mu\text{g}/\text{m}^3$** | **7.69 $\mu\text{g}/\text{m}^3$** |
| **Mean 24h Conformal Interval Width** | 30.59 $\mu\text{g}/\text{m}^3$ | 30.59 $\mu\text{g}/\text{m}^3$ | 30.59 $\mu\text{g}/\text{m}^3$ |

### Interpretation Rules:
1. **Distance to Boundary Governs Stability:** Stable windows lie deep inside regulatory categories (mean distance $32.80\,\mu\text{g}/\text{m}^3$). Ambiguous windows hover immediately along regulatory thresholds (mean distance $8.17\,\mu\text{g}/\text{m}^3$).
2. **Category Compatibility is NOT Probability:**
   - Overall, **92.13% of all 24-hour windows** have an uncertainty interval that encompasses the true regulatory category.
   - However, **compatibility is not a probability, nor is it conformal coverage**. A wide interval that spans three categories is trivial to make "compatible" but provides low operational utility.
3. **PM2.5-Only Regulatory Scope:**
   - Official Indian NAQI requires sub-indices across $\text{PM}_{10}$, $\text{PM}_{2.5}$, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$, $\text{NH}_3$, and $\text{Pb}$.
   - Because the AIRWISE sensor network monitors only particulate matter, this evaluation must always be qualified as a **PM2.5-based CPCB Category Classification**.

---

## 9. Historical Checkpoint & Prediction Provenance Audit

```
========================================================================================================
HISTORICAL CHECKPOINT ARCHITECTURAL LIMITATION & PROVENANCE AUDIT
========================================================================================================
Model Checkpoint: outputs/models/conformal/frozen_joint_tcn_checkpoint.pt
Architecture: Dilated Causal Conv1D with nn.BatchNorm1d Normalization Layers
Running Buffers Detected: running_mean, running_var, num_batches_tracked
Status: HISTORICAL BATCHNORM CHECKPOINT (Training-time temporal leakage identified in Phase 3A.2).
        The model was NOT retrained with ChannelLayerNorm1d.
Limitation: The Phase 3A.4 evaluation reflects the performance of this historical checkpoint.
            It DOES NOT establish causality-safe training performance.
Historical Assessment File: outputs/predictions/checkpoint5_category_assessment.csv
Underlying Hourly File: outputs/predictions/conformal_test_predictions.csv
Raw Sensor Source: data/processed/cleaned_data.parquet (pa_cf1, pa_atm)
Eligible Windows: 127 non-overlapping 24h windows (Delhi: 56, Bangalore: 55, Hamirpur: 16)
Completeness Enforced: >= 18 valid hours (75%) per window
Inference Execution: NO models were retrained. Persisted hourly predictions from the frozen
                     Joint TCN were averaged into 24-hour daily means. Raw PurpleAir channels
                     were averaged across the exact same window timestamps.
========================================================================================================
```
