# AIRWISE — Phase 3A.3: Conformal Coverage Reconciliation Report (Final Verification Patch)
**Document ID:** `AIRWISE-P3A3-CONF-003`  
**Date:** November 9, 2026  
**Status:** Complete — Audited & Verified (Documentation-Only Update)  
**Scope:** Rigorous empirical performance audit of Ordinary Split Conformal and Temporal Block Conformal (6h, 12h, 24h), reconciliation of reported coverage anomalies (the 95.70% metric), forensic investigation of population definitions, gap-length sample count accounting, prediction provenance verification, historical BatchNorm checkpoint limitation disclosure, evaluation of temporal dependence and distribution shift, and analysis of catastrophic undercoverage during Delhi winter pollution spikes.

---

## Executive Summary

Checkpoint 3A.3 establishes the exact mathematical and empirical performance of conformal uncertainty estimation methods across the held-out test partition ($N_{\text{test}} = 3,720$). This verification patch formally resolves all sample-count questions, clarifies model provenance, discloses the historical checkpoint's architectural limitation, and qualifies causal mechanisms:

1. **Reconciliation of the Reported 95.70% Coverage:**
   - **Pointwise Coverage Qualification:** The observed pointwise coverage was 95.70%, obtained with a mean interval width of 200.84 µg/m³. This wide interval has limited operational usefulness, and pointwise coverage must not be confused with simultaneous 24-hour block coverage.
   - **Origin of the Metric:** In historical Checkpoint 4, the exact same quantile $q = 100.4218\,\mu\text{g}/\text{m}^3$ was produced by both **Block Conformal (24h) at nominal 90%** ($N_{\text{calib}} = 32$ blocks, $k = \lceil 33 \times 0.90 \rceil = 30$) and **Temporal Block Conformal (12h) at nominal 95%** ($N_{\text{calib}} = 105$ blocks, $k = \lceil 106 \times 0.95 \rceil = 101$).
   - **Simultaneous Block Coverage:** When evaluated at the true simultaneous block level (requiring all observations in a block to fall within intervals), 24-hour block coverage is **only 74.58%** (44/59 test blocks covered), remaining **15.42 percentage points below nominal 90%**.
2. **Historical Checkpoint Architectural Limitation:**
   - The underlying model checkpoint (`outputs/models/conformal/frozen_joint_tcn_checkpoint.pt`) uses **`nn.BatchNorm1d`** (containing parameter running buffers `running_mean`, `running_var`, and `num_batches_tracked`).
   - Its training-time temporal sequence statistics leakage was formally identified in Phase 3A.2, and it was not retrained with `ChannelLayerNorm1d`.
   - The Phase 3A.3 and 3A.4 evaluations are strictly **historical-checkpoint evaluations with this documented limitation**. These results do not establish causality-safe training performance. All historical weights, quantiles, and predictions remain preserved without retraining.
3. **Reconciliation of Gap-Length Denominators (3,735 vs 3,720):**
   - The reported hidden-point counts ($744 + 744 + 746 + 751 + 750 = 3,735$) do **NOT** represent a partition of the 3,720 test observations.
   - They represent **five independent synthetic masking trials**, each masking approximately 20% of the 3,720 test observations ($N_{\text{hidden}} \approx 744\text{--}751$) with gap durations of 1h, 3h, 6h, 12h, and 24h.
   - Groups overlap across trials (2,322 unique test observations were masked in at least one trial). Each trial's coverage is calculated strictly against its own denominator.
4. **Prediction Provenance Disclosure:**
   - All Phase 3A.3 evaluations were performed using persisted, frozen historical prediction artifacts: [`outputs/predictions/conformal_test_predictions.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/predictions/conformal_test_predictions.csv) and [`outputs/models/conformal/conformal_quantiles.json`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/models/conformal/conformal_quantiles.json).
   - Scratch attempts in the terminal session to instantiate `JointTCN` with incorrect argument signatures were purely exploratory; no checkpoint was reloaded or retrained, and no predictions were regenerated.
5. **Failure of Temporal Exchangeability Under Out-of-Distribution Shift:**
   - In Bangalore (stationary, low-pollution regime), ordinary split conformal achieves **97.58%** coverage at nominal 90%.
   - In Delhi (severe non-stationary distribution shift), ordinary split conformal collapses to **51.07%** coverage at nominal 90%, suffering a **-38.93% coverage deficit**.
   - During severe episodic Delhi winter pollution (true BAM $> 250\,\mu\text{g}/\text{m}^3$, $N=257$), coverage catastrophically collapses to **1.95%** with a systematic mean under-prediction bias of **$-111.16\,\mu\text{g}/\text{m}^3$**.
6. **Separation of Conformal Calibration Experiments:**
   - Temporal Block Conformal (24h) yields **32 calibration blocks** because it requires strictly uninterrupted 24-hour contiguous runs ($n_{\text{seg}} // 24$).
   - Direct 24-hour CPCB Daily-Mean Conformal Calibration yields **78 calibration windows** because it aggregates wall-clock calendar days meeting the regulatory $\ge 18/24\text{h}$ completeness rule.
   - These are mathematically distinct experiments with different targets, sample counts, and quantiles ($100.42\,\mu\text{g}/\text{m}^3$ vs $15.30\,\mu\text{g}/\text{m}^3$) that must never be conflated.

---

## 1. Conformal Uncertainty Formulation

### 1.1 Ordinary Inductive Split Conformal Prediction
Given calibration nonconformity scores $s_i = |y_i - \hat{y}_i|$ on held-out calibration set $\mathcal{D}_{\text{calib}}$ ($n = 1,858$), the finite-sample conformal quantile for significance level $\alpha \in (0, 1)$ is:
$$k = \left\lceil (n + 1)(1 - \alpha) \right\rceil, \quad \hat{q}_\alpha = s_{(k)}$$
Under exchangeability, the prediction interval $\mathcal{C}(x_{n+1}) = [\hat{y}_{n+1} - \hat{q}_\alpha, \hat{y}_{n+1} + \hat{q}_\alpha]$ satisfies marginal coverage:
$$\mathbb{P}\left(Y_{n+1} \in \mathcal{C}(X_{n+1})\right) \ge 1 - \alpha$$

### 1.2 Temporal Block Conformal Prediction
Standard split conformal assumes independent and identically distributed (i.i.d.) or exchangeable residuals. In environmental time series, residuals exhibit strong serial autocorrelation.  
Temporal block conformal groups continuous observations into non-overlapping blocks of duration $B \in \{6, 12, 24\}$ hours within continuous segment and site boundaries:
$$S_b = \max_{t \in \text{Block}_b} |y_t - \hat{y}_t|$$
$$\hat{q}_{B, \alpha} = S_{(\lceil (M + 1)(1 - \alpha) \rceil)}$$
where $M$ is the number of valid calibration blocks.

---

## 2. Experimental Verification: Master Method Comparison

We evaluated Ordinary Split Conformal and Temporal Block Conformal (6h, 12h, 24h) across both **90% ($\alpha = 0.10$)** and **95% ($\alpha = 0.05$)** nominal confidence levels on the untouched held-out test partition ($N_{\text{test}} = 3,720$ observations).

Results are persisted in [`outputs/experiments/phase3A_conformal_reconciliation/conformal_coverage_reconciliation.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3A_conformal_reconciliation/conformal_coverage_reconciliation.csv).

| Conformal Method | Nominal Level ($1-\alpha$) | Calib Sample Count ($N_{\text{calib}}$) | Test Sample Count ($N_{\text{test}}$) | Empirical Coverage (%) | Coverage Error (%) | Mean Interval Width ($\mu\text{g}/\text{m}^3$) | Median Interval Width ($\mu\text{g}/\text{m}^3$) | Winkler Interval Score ($IS_\alpha$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ordinary Split Conformal** | **90%** | 1,858 hours | 3,720 hours | **77.02%** | -12.98% | **51.40** | 49.00 | **263.94** |
| **Ordinary Split Conformal** | **95%** | 1,858 hours | 3,720 hours | **80.27%** | -14.73% | **61.45** | 58.32 | **443.47** |
| **Block Conformal (6h)** | **90%** | 265 blocks | 3,720 hours | **83.55%** | -6.45% | **72.42** | 72.42 | **244.96** |
| **Block Conformal (6h)** | **95%** | 265 blocks | 3,720 hours | **86.75%** | -8.25% | **90.62** | 90.62 | **382.09** |
| **Block Conformal (12h)** | **90%** | 105 blocks | 3,720 hours | **86.40%** | -3.60% | **88.67** | 88.67 | **237.03** |
| **Block Conformal (12h)** | **95%** | 105 blocks | 3,720 hours | **95.70%** | +0.70% | **200.84** | 200.84 | **318.54** |
| **Block Conformal (24h)** | **90%** | 32 blocks | 3,720 hours | **95.70%** | +5.70% | **200.84** | 200.84 | **259.69** |
| **Block Conformal (24h)** | **95%** | 32 blocks | 3,720 hours | **97.80%** | +2.80% | **283.37** | 283.37 | **349.99** |

---

## 3. Forensic Investigation of the Reported 95.70% Metric

### 3.1 Mathematical Root of the 95.70% Metric
Why does the exact number **95.70%** appear under both **Block Conformal (24h) at 90%** and **Block Conformal (12h) at 95%**?

1. **Finite-Sample Quantile Calculation on Calibration Blocks:**
   - In `calib_conformal_df` ($N=1,858$), continuous segment slicing yields:
     - 12h blocks: $M = 105$ blocks. For $\alpha = 0.05$ (95% nominal):
       $$k_{12h, 95\%} = \lceil (105 + 1) \times 0.95 \rceil = \lceil 100.7 \rceil = 101 \implies S_{(101)} = \mathbf{100.4218\,\mu\text{g}/\text{m}^3}$$
     - 24h blocks: $M = 32$ blocks (Bangalore: 10, Delhi: 20, Hamirpur: 2). For $\alpha = 0.10$ (90% nominal):
       $$k_{24h, 90\%} = \lceil (32 + 1) \times 0.90 \rceil = \lceil 29.7 \rceil = 30 \implies S_{(30)} = \mathbf{100.4218\,\mu\text{g}/\text{m}^3}$$
2. **Identical Prediction Intervals:**
   Because the calculated conformal quantile is identically $100.4218\,\mu\text{g}/\text{m}^3$, both models construct the exact same interval:
   $$\mathcal{C}(x_t) = [\hat{y}_t - 100.42, \hat{y}_t + 100.42]$$
   with width $2 \times 100.4218 = 200.84\,\mu\text{g}/\text{m}^3$.
3. When evaluated against the 3,720 hourly BAM measurements in the test set, this interval covers exactly 3,560 out of 3,720 points:
   $$\text{Coverage} = \frac{3,560}{3,720} \times 100.0\% = \mathbf{95.70\%}$$

### 3.2 Pointwise versus Simultaneous Block-Level Coverage
The true theoretical goal of temporal block conformal prediction is **simultaneous joint coverage**: guaranteeing that all $B$ hours within a future block fall simultaneously within the prediction band:
$$\mathbb{P}\left(\forall t \in \text{Block}_b, Y_t \in \mathcal{C}(X_t)\right) \ge 1 - \alpha$$

We evaluated simultaneous block coverage across all valid blocks in the test partition:
Persisted in [`outputs/experiments/phase3A_conformal_reconciliation/conformal_simultaneous_block_coverage.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3A_conformal_reconciliation/conformal_simultaneous_block_coverage.csv).

| Block Duration ($B$) | Nominal Target | Quantile $\hat{q}$ ($\mu\text{g}/\text{m}^3$) | Interval Width ($\mu\text{g}/\text{m}^3$) | Test Blocks ($M_{\text{test}}$) | Blocks Fully Covered | Simultaneous Block Coverage (%) | Pointwise Hourly Coverage (%) | Simultaneous Gap vs Nominal |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **6 Hours** | **90%** | 36.21 | 72.42 | 489 | 333 | **68.10%** | 83.55% | **-21.90%** |
| **6 Hours** | **95%** | 45.31 | 90.62 | 489 | 363 | **74.23%** | 86.75% | **-20.77%** |
| **12 Hours** | **90%** | 44.34 | 88.67 | 188 | 120 | **63.83%** | 86.40% | **-26.17%** |
| **12 Hours** | **95%** | 100.42 | 200.84 | 188 | 161 | **85.64%** | 95.70% | **-9.36%** |
| **24 Hours** | **90%** | 100.42 | 200.84 | 59 | 44 | **74.58%** | 95.70% | **-15.42%** |
| **24 Hours** | **95%** | 141.68 | 283.37 | 59 | 50 | **84.75%** | 97.80% | **-10.25%** |

### 3.3 Scientific Conclusion on 95.70%
1. **Pointwise vs Simultaneous Block Coverage:** The observed pointwise coverage was 95.70%, obtained with a mean interval width of 200.84 µg/m³. This wide interval has limited operational usefulness, and pointwise coverage must not be confused with simultaneous 24-hour block coverage.
2. **Simultaneous Undercoverage:** True simultaneous 24-hour block coverage is **only 74.58%**, missing the nominal 90% target by **15.42 percentage points**.
3. **Small-Sample Calibration Instability:** The 24-hour calibration quantile was derived from only **32 calibration blocks**. The 30th order statistic is highly susceptible to small-sample noise.

---

## 4. Disentangling the Four Separate Conformal Experiments

To prevent confusion across reports and dashboard pages, we document the four distinct conformal paradigms evaluated in AIRWISE:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   AIRWISE CONFORMAL TAXONOMY                                           │
├──────────────────────┬──────────────────────┬──────────┬───────────┬──────────────┬────────────────────┤
│ Experiment Name      │ Target Variable      │ Unit     │ Calib $N$ │ Test $N$     │ 90% Quantile Width │
├──────────────────────┼──────────────────────┼──────────┼───────────┼──────────────┼────────────────────┤
│ 1. Ordinary Split    │ Hourly BAM PM2.5     │ 1 Hour   │ 1,858 pts │ 3,720 pts    │ 51.40 ug/m3        │
│ 2. Block Conformal   │ Block-Max Residual   │ 6h / 12h │ 265 / 105 │ 489 / 188 blk│ 72.42 / 88.67 ug/m3│
│ 3. 24h Block Conf.   │ Block-Max Residual   │ 24 Hours │ 32 blocks │ 59 blocks    │ 200.84 ug/m3       │
│ 4. 24h CPCB Daily    │ 24h Mean BAM PM2.5   │ 24h Day  │ 78 days   │ 127 days     │ 30.59 ug/m3        │
└──────────────────────┴──────────────────────┴──────────┴───────────┴──────────────┴────────────────────┘
```

### Why 24h Block Conformal has 32 Blocks while 24h CPCB Assessment has 78 Windows:
* **Temporal Block Conformal (32 Blocks):** Requires **strictly contiguous 24-hour uninterrupted sequences** ($n_{\text{seg}} // 24$) with 0 missing hours. Any single missing hour terminates the segment and discards the remainder. In `calib_df`, this strict filter retains only 32 blocks (Bangalore: 10, Delhi: 20, Hamirpur: 2). The nonconformity score is the **maximum absolute hourly error** in the block ($S_b = \max_{t} |y_t - \hat{y}_t|$), yielding quantile $q = 100.42\,\mu\text{g}/\text{m}^3$ and width $200.84\,\mu\text{g}/\text{m}^3$.
* **CPCB 24h Daily-Mean Windowing (78 Windows):** Enforces the regulatory completeness rule: **$\ge 18/24$ valid hours** ($\ge 75\%$). Up to 6 missing hours are permitted in a wall-clock day. In `calib_df`, 117 days were evaluated and 78 met completeness (Bangalore: 33, Delhi: 31, Hamirpur: 14). The nonconformity score is computed on the **24-hour mean scalar concentration** ($s_w = |\bar{y}_{24h} - \hat{\bar{y}}_{24h}|$). Because daily averaging smooths zero-mean hourly noise, the quantile is **$q = 15.30\,\mu\text{g}/\text{m}^3$** (width $30.59\,\mu\text{g}/\text{m}^3$).
* **Conclusion:** These two methods measure completely different statistical phenomena. They must never be combined or equated.

---

## 5. Site-Wise Heterogeneity and Geographic Breakdown

Conformal intervals evaluated at nominal 90% display extreme geographic disparity:
Persisted in [`outputs/experiments/phase3A_conformal_reconciliation/conformal_site_breakdown.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3A_conformal_reconciliation/conformal_site_breakdown.csv).

| Monitoring Station | Test Sample Size | Ordinary Split Cov (%) | Block 6h Cov (%) | Block 12h Cov (%) | Block 24h Cov (%) | Mean Width (Ord / Blk 12h / Blk 24h) | Winkler Score (Ord / Blk 12h / Blk 24h) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bangalore** | 1,489 | **97.58%** | 98.99% | 99.19% | 99.46% | 51.50 / 88.67 / 200.84 | 73.36 / 106.44 / 212.16 |
| **Hamirpur** | 729 | **88.48%** | 94.38% | 96.57% | 99.04% | 51.45 / 88.67 / 200.84 | 100.73 / 116.40 / 212.19 |
| **Delhi** | 1,502 | **51.07%** | 62.98% | 68.77% | **90.35%** | 51.27 / 88.67 / 200.84 | 532.07 / 425.03 / 329.87 |

### Analysis of Geographic Divergence
* **Bangalore (Stationary Distribution):** Ambient concentrations follow a consistent diurnal profile. Ordinary split conformal easily satisfies nominal coverage (**97.58%**). Block conformal severely overcovers (**99.46%**) with bloated intervals.
* **Hamirpur (Mild Seasonality):** Ordinary split achieves **88.48%** (close to 90%). Block 12h achieves **96.57%**.
* **Delhi (Severe Distribution Shift):** Ordinary split collapses to **51.07%**, meaning nearly half of all predictions violate their 90% confidence bands. Even 12h block conformal achieves only **68.77%**. Only 24h block conformal crosses nominal 90% (**90.35%**), and it does so only by inflating interval width to $200.84\,\mu\text{g}/\text{m}^3$.

---

## 6. Pollution Regime Breakdown and Delhi Winter Failure

Persisted in [`outputs/experiments/phase3A_conformal_reconciliation/conformal_regime_breakdown.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3A_conformal_reconciliation/conformal_regime_breakdown.csv).

| Operational Pollution Regime | Test Sample Size | Ordinary Split Cov (%) | Block 6h Cov (%) | Block 12h Cov (%) | Block 24h Cov (%) | Mean Interval Width ($\mu\text{g}/\text{m}^3$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Low ($\le 60\,\mu\text{g}/\text{m}^3$)** | 2,089 | **96.74%** | 98.95% | 99.43% | 99.86% | 51.46 / 72.42 / 88.67 / 200.84 |
| **Moderate ($60\text{--}120\,\mu\text{g}/\text{m}^3$)** | 815 | **68.59%** | 80.98% | 86.50% | 98.16% | 51.25 / 72.42 / 88.67 / 200.84 |
| **Severe ($> 120\,\mu\text{g}/\text{m}^3$)** | 816 | **34.93%** | 46.69% | 52.94% | **82.60%** | 51.37 / 72.42 / 88.67 / 200.84 |

### Breakdown During Delhi Winter Smog Episodes
When evaluated strictly on Delhi ground-truth BAM bands (`outputs/reports/conformal_extreme_pollution.csv`):

| Delhi BAM Band | Sample Count | Point RMSE | Point MBE ($\mu\text{g}/\text{m}^3$) | Ordinary Split Cov 90% | Ordinary Split Cov 95% | Winkler Score 90 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| $\le 60\,\mu\text{g}/\text{m}^3$ | 367 | 83.19 | +19.46 | 85.56% | 88.56% | 317.88 |
| $60\text{--}120\,\mu\text{g}/\text{m}^3$ | 358 | 21.06 | -9.23 | 78.77% | 85.20% | 97.10 |
| $120\text{--}250\,\mu\text{g}/\text{m}^3$ | 520 | 44.89 | -34.04 | 31.92% | 40.77% | 375.26 |
| **$> 250\,\mu\text{g}/\text{m}^3$** | **257** | **124.68** | **-111.16** | **1.95%** | **3.11%** | **1,761.15** |

### Qualified Mechanistic Explanation
1. **Seasonal Distribution Shift:** The Delhi calibration partition spanned May 25, 2019 to July 2, 2019 ($N=750$ hours; summer pre-monsoon), during which **only 1 single hour exceeded $250\,\mu\text{g}/\text{m}^3$**. In contrast, the test partition spanned July 2, 2019 to January 3, 2020 ($N=1,502$ hours), capturing the winter peak crisis where **257 hours exceeded $250\,\mu\text{g}/\text{m}^3$**. This sharp seasonal divergence provides clear empirical evidence of distribution shift, violating the exchangeability required for conformal coverage guarantees.
2. **Under-Prediction at High Concentrations:** During extreme episodes, the model exhibits severe negative bias (MBE = $-111.16\,\mu\text{g}/\text{m}^3$). This is consistent with known optical sensor coincidence loss and scattering non-linearities at heavy particulate loads, coupled with conservative regression shrinkage under Huber loss on rare extreme values. Fixed intervals of half-width $\pm 25\,\mu\text{g}/\text{m}^3$ cannot cover an under-prediction of over $100\,\mu\text{g}/\text{m}^3$.

---

## 7. Reconciled Gap-Length Accounting (3,735 vs 3,720)

Persisted in [`outputs/experiments/phase3A_conformal_reconciliation/conformal_gap_length_breakdown.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/experiments/phase3A_conformal_reconciliation/conformal_gap_length_breakdown.csv).

| Experimental Scenario | Target Missing Rate | Hidden Test Denominator | Total Test Population | Unique Points Masked | Ordinary Split Cov (%) | Block Conformal (12h) Cov (%) | Point MBE ($\mu\text{g}/\text{m}^3$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gap 1 Hour** | 20.0% | **744** | 3,720 (20.00%) | 744 | **74.06%** | **80.51%** | -11.59 |
| **Gap 3 Hours** | 20.0% | **744** | 3,720 (20.00%) | 744 | **72.98%** | **79.97%** | -11.52 |
| **Gap 6 Hours** | 20.0% | **746** | 3,720 (20.05%) | 746 | **62.33%** | **70.11%** | -18.36 |
| **Gap 12 Hours** | 20.0% | **751** | 3,720 (20.19%) | 751 | **59.12%** | **68.84%** | -33.15 |
| **Gap 24 Hours** | 20.0% | **750** | 3,720 (20.16%) | 750 | **49.73%** | **59.20%** | **-44.96** |
| **Sum Across Scenarios**| — | **3,735** | — | **2,322 unique** | — | — | — |

### Clarification of the Sum:
* **Separate Experiments:** The five rows represent five **independent synthetic masking scenarios**, each applying a randomized block outage with target rate $20\%$ across the 3,720 test points.
* **Why the Sum is 3,735:** Summing $744 + 744 + 746 + 751 + 750 = 3,735$ is simply adding sample sizes across 5 separate experimental runs. It is not an accounting discrepancy against the 3,720 test set size.
* **Overlap:** Across all 5 experiments, 2,322 unique observations were masked in at least one trial.
* **Exact Coverage Calculation:** In each scenario $g$, empirical coverage is strictly computed as the fraction of hidden points covered within that scenario:
  $$\text{Coverage}_g = \frac{\text{Number of Hidden Observations Covered in Scenario } g}{N_{\text{hidden}, g}}$$
* **Monotonic Degradation:** Coverage degrades monotonically from **74.06% down to 49.73%** as gap length increases from 1h to 24h, driven by a fourfold increase in negative point bias (from $-11.59$ to $-44.96\,\mu\text{g}/\text{m}^3$).

---

## 8. Historical Checkpoint & Prediction Provenance Audit

```
========================================================================================================
HISTORICAL CHECKPOINT ARCHITECTURAL LIMITATION & PROVENANCE AUDIT
========================================================================================================
Model Checkpoint: outputs/models/conformal/frozen_joint_tcn_checkpoint.pt
Architecture: Dilated Causal Conv1D with nn.BatchNorm1d Normalization Layers
Running Buffers Detected: running_mean, running_var, num_batches_tracked
Status: HISTORICAL BATCHNORM CHECKPOINT (Training-time temporal leakage identified in Phase 3A.2).
        The model was NOT retrained with ChannelLayerNorm1d.
Limitation: The Phase 3A.3 and 3A.4 evaluations reflect the performance of this historical checkpoint.
            They DO NOT establish causality-safe training performance.
Prediction Artifact: outputs/predictions/conformal_test_predictions.csv (N=3,720 hourly points)
Normalizer Artifact: outputs/models/conformal/normalizer.joblib
Quantiles Artifact: outputs/models/conformal/conformal_quantiles.json
Inference Execution: NO predictions were regenerated. All evaluations audited the frozen historical
                     predictions artifact directly. All historical checkpoints remain intact.
========================================================================================================
```

---

## 9. Synthesis and Verified Scientific Conclusions

1. **Pointwise vs Simultaneous Coverage:** The observed pointwise coverage was 95.70%, obtained with a mean interval width of 200.84 µg/m³. This wide interval has limited operational usefulness, and pointwise coverage must not be confused with simultaneous 24-hour block coverage (which reached only 74.58%).
2. **Historical Checkpoint Limitation:** The evaluated model uses BatchNorm with documented training-time sequence statistics leakage; it must not be represented as a fully causality-safe trained model.
3. **Delhi Winter Breakdown:** Conformal coverage collapses to **1.95%** during severe winter smog episodes ($>250\,\mu\text{g}/\text{m}^3$), reflecting empirical distribution shift and sensor/model saturation.
4. **Transparent Governance:** AIRWISE reports site-specific and regime-conditioned coverage rather than making unsupportable universal claims.
