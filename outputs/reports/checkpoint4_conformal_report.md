# Checkpoint 4 Research Report: Regime-Conditioned Conformal Uncertainty Estimation (Post-Audit Revision)

**Project:** Uncertainty-Aware Joint Calibration and Reconstruction of Low-Cost $\text{PM}_{2.5}$ Sensor Networks  
**Dataset:** Campmier et al. (Delhi, Hamirpur, Bangalore — 18,584 hourly observations)  
**Evaluation Protocol:** Strict 4-way chronological partition (60% Train, 10% Model Val, 10% Conformal Calib, 20% Test), frozen TCN weights, identical 20% benchmark mask, zero reference target leakage into inputs.  
**Execution Date:** November 2026 (Audited and Revised)  

---

## Executive Summary

This report establishes the statistical implementation, empirical validation, and failure-mode analysis of **Inductive Split Conformal Prediction, Regime-Conditioned Conformal Prediction, and Temporal Block Conformal Prediction** for the Joint Calibration-and-Reconstruction TCN.

All experiments were executed under strict chronological separation:
1. **Model Freezing:** The Joint TCN was retrained strictly on 60% Train ($N=11,149$) and 10% Model-Selection Validation ($N=1,857$) with early stopping, completely isolating the 10% Conformal Calibration partition ($N=1,858$) from architecture, loss-weight, or gradient influence.
2. **Untouched Test Partition:** Final evaluation was executed on the untouched 20% test partition ($N=3,720$) using exact finite-sample conformal quantiles $k = \lceil (n + 1)(1 - \alpha) \rceil$ for 90% ($\alpha = 0.10$) and 95% ($\alpha = 0.05$) nominal coverage.
3. **Primary Findings (Audit Corrected):**
   - **Marginal Under-Coverage:** On the pooled test population, Ordinary Split Conformal achieved **77.02% empirical coverage** (nominal 90%, error $-12.98\%$) and **80.27%** (nominal 95%, error $-14.73\%$).
   - **Block Conformal Under-Coverage:** Temporal Block Conformal (12h) elevated empirical coverage to **86.40% at 90% nominal**, which is an improvement over ordinary split conformal, but **remains strictly under the nominal target** ($\Delta \mathrm{Cov} = -3.60\%$). Block conformal accounts for temporal grouping within blocks under stationarity; it does **not** solve or protect against out-of-distribution shifts.
   - **Geographical Breakdown:** Ordinary conformal achieved **97.58% coverage in Bangalore** and **88.48% in Hamirpur**, but collapsed to **51.07% in Delhi**.
   - **Extreme Pollution Collapse:** In Delhi observations with true BAM $>250\,\mu\text{g}/\text{m}^3$ ($N=257$), 90% coverage collapsed to **1.95%** due to physical optical sensor saturation and seasonal distribution shift.
   - **Winkler Score Parity:** At the exact same nominal coverage ($\alpha = 0.10$), test rows ($N = 3,720$), and scoring convention, Block Conformal (12h) achieved a Winkler score of **237.03** compared to **263.94** for Ordinary Split. Despite wider intervals ($88.67$ vs. $51.40\,\mu\text{g}/\text{m}^3$), block calibration sharply curtails severe tail boundary violations in Delhi that carry heavy miscoverage penalties ($\times 20$).
   - **Calibration Sample Depletion in Extreme Regimes:** In the calibration partition, only **7 observations** had model predictions $>120\,\mu\text{g}/\text{m}^3$ ($0.38\%$), triggering automatic fallback to pooled quantiles ($N < 30$). Furthermore, true BAM $>250\,\mu\text{g}/\text{m}^3$ had **zero calibration observations ($N = 0$)**, whereas it accounted for 263 test hours (257 in Delhi).
   - **Marginal vs. Conditional Coverage:** Pooled 90% marginal metrics conceal severe local failures across sites (Delhi 51.07%), pollution regimes ($>120\,\mu\text{g}/\text{m}^3$: 34.31%), and prolonged outages (24h: 49.73%).

---

## 1. Conformal Methodology and Finite-Sample Quantiles

### 1.1 Non-Conformity Scores and Quantile Formulation
For calibration pairs $(X_i, Y_i)_{i=1}^n$, non-conformity is quantified via absolute prediction residuals:
$$s_i = |Y_i - \widehat{Y}_i|$$
For a given nominal miscoverage rate $\alpha \in (0, 1)$, the finite-sample adjusted conformal quantile $q_\alpha$ is computed as the $k$-th order statistic:
$$k = \left\lceil (n + 1)(1 - \alpha) \right\rceil, \quad q_\alpha = s_{(k)}$$
Prediction intervals for test points $\widehat{Y}_{n+1}$ are constructed symmetrically:
$$C_\alpha(X_{n+1}) = \left[ \widehat{Y}_{n+1} - q_\alpha, \; \widehat{Y}_{n+1} + q_\alpha \right]$$

### 1.2 Dual-Head Calibrator Separation
To prevent cross-task residual contamination:
- **Calibration Head ($M_t = 1$):** Fitted on observed calibration points ($N_{\mathrm{obs}} = 1,475$). Quantiles: $q_{90} = 24.50\,\mu\text{g}/\text{m}^3$, $q_{95} = 29.16\,\mu\text{g}/\text{m}^3$.
- **Reconstruction Head ($M_t = 0$):** Fitted on hidden calibration points ($N_{\mathrm{hid}} = 383$) under a 20% synthetic MCAR outage. Quantiles: $q_{90} = 30.30\,\mu\text{g}/\text{m}^3$, $q_{95} = 36.73\,\mu\text{g}/\text{m}^3$ (+23.7% wider than calibration head).

---

## 2. Overall Benchmark Results on Untouched Test Partition ($N = 3,720$)

The table below summarizes empirical performance across conformal methods at nominal 90% ($\alpha = 0.10$) and 95% ($\alpha = 0.05$) coverage evaluated on the exact same 3,720 test rows:

| Conformal Method | $N$ | 90% Cov (%) | Cov Error (%) | Mean Width 90 ($\mu\text{g}/\text{m}^3$) | Winkler Score 90 | 95% Cov (%) | Cov Error (%) | Mean Width 95 ($\mu\text{g}/\text{m}^3$) | Winkler Score 95 | Point RMSE | Point MBE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ordinary Split (Dual Head)** | 3,720 | 77.02 | -12.98 | 51.40 | 263.94 | 80.27 | -14.73 | 61.45 | 443.47 | 48.30 | -8.84 |
| **Humidity-Conditioned** | 3,720 | 73.49 | -16.51 | 44.66 | 274.54 | 78.58 | -16.42 | 56.76 | 462.26 | 48.30 | -8.84 |
| **Predicted-Pollution-Conditioned** | 3,720 | 78.79 | -11.21 | 53.75 | 256.26 | 83.58 | -11.42 | 66.35 | 423.69 | 48.30 | -8.84 |
| **Site-Conditioned** | 3,720 | 75.91 | -14.09 | 49.42 | 251.93 | 80.89 | -14.11 | 59.40 | 411.03 | 48.30 | -8.84 |
| **Temporal Block (12h)** | 3,720 | **86.40** | **-3.60** | 88.67 | **237.03** | **95.70** | **+0.70** | 200.84 | **318.54** | 48.30 | -8.84 |

---

## 3. Site-Wise Heterogeneity and Geographic Breakdown

| Study Site | Conformal Method | $N$ | Empirical 90% Cov (%) | Coverage Error (%) | Mean Width 90 ($\mu\text{g}/\text{m}^3$) | Winkler Score 90 | Point RMSE ($\mu\text{g}/\text{m}^3$) | Point MBE ($\mu\text{g}/\text{m}^3$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delhi** | Ordinary Split (Dual Head) | 1,502 | **51.07** | -38.93 | 51.27 | 532.07 | 71.79 | -28.25 |
| | Humidity-Conditioned | 1,502 | 48.74 | -41.26 | 47.25 | 554.88 | 71.79 | -28.25 |
| | Predicted-Pollution | 1,502 | 57.06 | -32.94 | 59.04 | 515.27 | 71.79 | -28.25 |
| | Site-Conditioned | 1,502 | 56.72 | -33.28 | 60.62 | 498.83 | 71.79 | -28.25 |
| | Temporal Block (12h) | 1,502 | 68.77 | -21.23 | 88.67 | 425.03 | 71.79 | -28.25 |
| **Hamirpur** | Ordinary Split (Dual Head) | 729 | **88.48** | -1.52 | 51.45 | 100.73 | 23.27 | -3.80 |
| | Humidity-Conditioned | 729 | 78.60 | -11.40 | 42.47 | 120.65 | 23.27 | -3.80 |
| | Predicted-Pollution | 729 | 88.75 | -1.25 | 58.28 | 106.64 | 23.27 | -3.80 |
| | Site-Conditioned | 729 | 76.27 | -13.73 | 34.51 | 114.56 | 23.27 | -3.80 |
| | Temporal Block (12h) | 729 | 96.57 | +6.57 | 88.67 | 116.40 | 23.27 | -3.80 |
| **Bangalore** | Ordinary Split (Dual Head) | 1,489 | **97.58** | +7.58 | 51.50 | 73.36 | 19.10 | +8.28 |
| | Humidity-Conditioned | 1,489 | 95.97 | +5.97 | 43.13 | 67.09 | 19.10 | +8.28 |
| | Predicted-Pollution | 1,489 | 95.84 | +5.84 | 46.19 | 68.24 | 19.10 | +8.28 |
| | Site-Conditioned | 1,489 | 95.10 | +5.10 | 45.43 | 70.12 | 19.10 | +8.28 |
| | Temporal Block (12h) | 1,489 | 99.19 | +9.19 | 88.67 | 106.44 | 19.10 | +8.28 |

---

## 4. Operational Predicted-Pollution Regime Conditioning

Regimes are assigned strictly at prediction time based on model point predictions $\widehat{Y}_{\mathrm{final}}$:

| Predicted Pollution Regime | Conformal Method | $N$ | Empirical 90% Cov (%) | Mean Width 90 ($\mu\text{g}/\text{m}^3$) | Winkler Score 90 | Point RMSE | Point MBE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low ($\le 60\,\mu\text{g}/\text{m}^3$)** | Ordinary Split | 2,089 | 96.74 | 51.46 | 60.96 | 13.42 | +3.16 |
| | Predicted-Pollution-Conditioned | 2,089 | **94.40** | **45.43** | **58.12** | 13.42 | +3.16 |
| | Temporal Block (12h) | 2,089 | 99.43 | 88.67 | 93.61 | 13.42 | +3.16 |
| **Moderate ($60\text{--}120\,\mu\text{g}/\text{m}^3$)** | Ordinary Split | 815 | 68.59 | 51.25 | 217.52 | 39.84 | -12.26 |
| | Predicted-Pollution-Conditioned | 815 | **83.31** | **78.47** | **183.41** | 39.84 | -12.26 |
| | Temporal Block (12h) | 815 | 86.50 | 88.67 | 178.43 | 39.84 | -12.26 |
| **Severe ($>120\,\mu\text{g}/\text{m}^3$)** | Ordinary Split | 816 | 34.93 | 51.37 | 829.93 | 92.68 | -36.15 |
| | Predicted-Pollution-Conditioned | 816 | 34.31 | 50.36 | 836.25 | 92.68 | -36.15 |
| | Temporal Block (12h) | 816 | 52.94 | 88.67 | 662.72 | 92.68 | -36.15 |

---

## 5. Temporal Block Conformal Evaluation (Serial Dependence vs. Distribution Shift)

To account for temporal autocorrelation, block calibrators construct non-overlapping blocks of duration $B$ within contiguous segments, evaluating block maximum residuals:

| Conformal Method | Block Duration ($B$) | Calibration Blocks | Test Empirical 90% Cov (%) | Cov Error (%) | Mean Width 90 ($\mu\text{g}/\text{m}^3$) | Winkler Score 90 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ordinary Split Conformal** | 1 Hour | 1,858 | 77.02 | -12.98 | 51.40 | 263.94 |
| **Block Conformal (6h)** | 6 Hours | 265 | 83.55 | -6.45 | 72.42 | 244.96 |
| **Block Conformal (12h)** | 12 Hours | 105 | **86.40** | **-3.60** | **88.67** | **237.03** |
| **Block Conformal (24h)** | 24 Hours | 32 | 95.70 | +5.70 | 200.84 | 259.69 |

### Critical Clarifications:
1. **86.40% is NOT Satisfied 90% Coverage:** While 12h block conformal closes 9.38 percentage points of the coverage gap compared to ordinary split ($86.40\%$ vs. $77.02\%$), it **remains under the nominal 90% target by 3.60%**.
2. **Block Calibration Does NOT Solve Distribution Shift:** Block conformal models within-block error dependency (temporal grouping) under exchangeability of blocks. It cannot protect against out-of-distribution shifts (such as the 5.3-fold increase in extreme winter pollution in Delhi). For true BAM $>250\,\mu\text{g}/\text{m}^3$, 12h block conformal achieves only **3.50% coverage** ($N = 9/257$).

---

## 6. Investigation of Calibration vs. Test Group Counts

| Stratification Category | Subgroup / Regime | Calibration $N$ | Calibration % | Test $N$ | Test % | Ratio Test / Calib | Fallback Triggered? ($N < 30$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Predicted Pollution** | Low ($\le 60\,\mu\text{g}/\text{m}^3$) | 1,521 | 81.86% | 2,089 | 56.16% | 1.37x | No |
| | Moderate ($60\text{--}120\,\mu\text{g}/\text{m}^3$) | 330 | 17.76% | 815 | 21.91% | 2.47x | No |
| | **Severe ($> 120\,\mu\text{g}/\text{m}^3$)** | **7** | **0.38%** | **816** | **21.94%** | **116.57x** | **YES ($N=7 < 30$)** |
| **Humidity** | Low ($\le 40\%$ RH) | 1,065 | 57.32% | 1,155 | 31.05% | 1.08x | No |
| | Medium ($40\text{--}65\%$ RH) | 499 | 26.86% | 1,993 | 53.58% | 3.99x | No |
| | High ($> 65\%$ RH) | 294 | 15.82% | 572 | 15.38% | 1.95x | No |
| **Site** | Delhi | 750 | 40.37% | 1,502 | 40.38% | 2.00x | No |
| | Hamirpur | 364 | 19.59% | 729 | 19.60% | 2.00x | No |
| | Bangalore | 744 | 40.04% | 1,489 | 40.03% | 2.00x | No |
| **True Reference BAM** | Low ($\le 60\,\mu\text{g}/\text{m}^3$) | 1,511 | 81.32% | 2,067 | 55.56% | 1.37x | — |
| | Moderate ($60\text{--}120\,\mu\text{g}/\text{m}^3$) | 320 | 17.22% | 688 | 18.49% | 2.15x | — |
| | Very Poor ($120\text{--}250\,\mu\text{g}/\text{m}^3$) | 27 | 1.45% | 702 | 18.87% | 26.00x | — |
| | **Severe ($> 250\,\mu\text{g}/\text{m}^3$)** | **0** | **0.00%** | **263** | **7.07%** | **$\infty$** | — |

**Finding on Calibration Sample Depletion:** In the calibration partition, model predictions exceeded $120\,\mu\text{g}/\text{m}^3$ in only 7 hours. The calibrator enforced the pre-specified $N_{\min}=30$ fallback protocol and defaulted to the pooled quantile ($q_{90} = 25.18\,\mu\text{g}/\text{m}^3$). True BAM $>250\,\mu\text{g}/\text{m}^3$ was entirely absent from the calibration set ($N = 0$).

---

## 7. Impact of Consecutive Outage Gap Duration on Coverage

| Outage Gap Duration | Conformal Method | Hidden $N$ | Empirical 90% Cov (%) | Mean Width 90 ($\mu\text{g}/\text{m}^3$) | Point RMSE ($\mu\text{g}/\text{m}^3$) | Point MBE ($\mu\text{g}/\text{m}^3$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1 Hour** | Ordinary Split (Rec Head) | 744 | 74.06 | 60.59 | 67.10 | -11.59 |
| | Block Conformal (12h) | 744 | 80.51 | 88.67 | 67.10 | -11.59 |
| **3 Hours** | Ordinary Split (Rec Head) | 744 | 72.98 | 60.59 | 53.84 | -11.52 |
| | Block Conformal (12h) | 744 | 79.97 | 88.67 | 53.84 | -11.52 |
| **6 Hours** | Ordinary Split (Rec Head) | 746 | 62.33 | 60.59 | 105.47 | -18.36 |
| | Block Conformal (12h) | 746 | 70.11 | 88.67 | 105.47 | -18.36 |
| **12 Hours** | Ordinary Split (Rec Head) | 751 | 59.12 | 60.59 | 73.73 | -33.15 |
| | Block Conformal (12h) | 751 | 68.84 | 88.67 | 73.73 | -33.15 |
| **24 Hours** | Ordinary Split (Rec Head) | 750 | **49.73** | 60.59 | 84.86 | -44.96 |
| | Block Conformal (12h) | 750 | **59.20** | 88.67 | 84.86 | -44.96 |

---

## 8. Retrospective Extreme Pollution Failure Analysis (Delhi True BAM Strata)

| True Ground-Truth BAM Stratum | $N$ | Point RMSE ($\mu\text{g}/\text{m}^3$) | Point MBE ($\mu\text{g}/\text{m}^3$) | 90% Cov (%) | 95% Cov (%) | Mean Width 90 ($\mu\text{g}/\text{m}^3$) | Winkler Score 90 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\le 60\,\mu\text{g}/\text{m}^3$** | 367 | 83.19 | +19.46 | 85.56 | 88.56 | 51.12 | 317.88 |
| **$60\text{--}120\,\mu\text{g}/\text{m}^3$** | 358 | 21.06 | -9.23 | 78.77 | 85.20 | 51.20 | 97.10 |
| **$120\text{--}250\,\mu\text{g}/\text{m}^3$** | 520 | 44.89 | -34.04 | 31.92 | 40.77 | 51.25 | 375.26 |
| **$> 250\,\mu\text{g}/\text{m}^3$** | 257 | 124.68 | -111.16 | **1.95** | **3.11** | 51.62 | **1,761.15** |

---

## 9. Answers to the 10 Mandatory Scientific Questions (Revised Post-Audit)

1. **Does ordinary split conformal achieve approximately nominal marginal coverage on the test population?**  
   *No.* On the pooled test partition across all three cities, ordinary split conformal achieved **77.02% empirical coverage** at 90% nominal ($\alpha=0.10$, coverage error $-12.98\%$) and **80.27%** at 95% nominal ($\alpha=0.05$, coverage error $-14.73\%$). Site-stratified analysis reveals that ordinary split conformal was nominal in Bangalore (97.58%) and Hamirpur (88.48%), but under-covered in Delhi (51.07%) due to severe temporal distribution shift.

2. **Does humidity conditioning improve interval informativeness?**  
   *Yes, selectively.* In high humidity (>65% RH), humidity conditioning tightens mean interval width to 27.15 $\mu\text{g}/\text{m}^3$ (vs. 51.55 $\mu\text{g}/\text{m}^3$ for ordinary split). In medium humidity (40–65% RH), coverage drops to 66.78% because Delhi's winter observations dominate that stratum.

3. **Does predicted-pollution conditioning improve interval informativeness?**  
   *Yes, significantly.* In clean conditions ($\widehat{Y} \le 60\,\mu\text{g}/\text{m}^3$), conditioning tightens interval width to 45.43 $\mu\text{g}/\text{m}^3$ while achieving 94.40% coverage. In moderate pollution ($60 < \widehat{Y} \le 120\,\mu\text{g}/\text{m}^3$), conditioning dynamically expands interval width to 78.47 $\mu\text{g}/\text{m}^3$, lifting coverage from 68.59% to 83.31% (+14.7% improvement).

4. **How does coverage differ across sites?**  
   Coverage is geographically disparate: **Bangalore (97.58%)**, **Hamirpur (88.48%)**, and **Delhi (51.07%)**. Stationary distributions produce valid conformal intervals; unmodeled episodic crises cause under-coverage.

5. **Does block-based conformal behave differently from ordinary split conformal?**  
   *Yes.* By calibrating over block-maximum residuals, Block Conformal (12h) accounts for temporal autocorrelation, widening intervals to 88.67 $\mu\text{g}/\text{m}^3$ and lifting empirical coverage to **86.40% at 90% nominal** and **95.70% at 95% nominal**. However, **86.40% remains an under-coverage of 3.60 percentage points below the nominal 90% target**, and block conformal does not protect against seasonal distribution shift.

6. **Does coverage deteriorate during long outages?**  
   *Yes, monotonically.* Coverage drops from 74.06% (1h gap) to 49.73% (24h gap) for Ordinary Split, and from 80.51% to 59.20% for Block Conformal, as negative bias quadruples from $-11.59$ to $-44.96\,\mu\text{g}/\text{m}^3$.

7. **Are Delhi extreme-pollution observations adequately covered?**  
   *No.* Coverage drops to 1.95% for BAM $>250\,\mu\text{g}/\text{m}^3$. Conformal prediction cannot fix severe physical sensor saturation or structural model underfitting.

8. **Are intervals excessively wide in ordinary conditions?**  
   Under ordinary split conformal, a 51.4 $\mu\text{g}/\text{m}^3$ interval is somewhat wide in clean Bangalore air ($\text{BAM} \sim 20\,\mu\text{g}/\text{m}^3$). Predicted-pollution conditioning resolves this, narrowing clean-air intervals to 45.43 $\mu\text{g}/\text{m}^3$, and humidity conditioning narrows high-humidity intervals to 27.15 $\mu\text{g}/\text{m}^3$.

9. **Which method offers the best balance between coverage and interval width?**  
   **Temporal Block Conformal (12h)** yields the best empirical performance against serial autocorrelation (Winkler score **237.03** at $\alpha=0.10$), while **Predicted-Pollution-Conditioned Conformal** provides the best adaptive efficiency across operational pollution levels. However, neither method achieves the nominal 90% target on the pooled test set.

10. **What limitations remain due to temporal dependence and distribution shift?**  
    Exchangeability violations remain the primary barrier: temporal non-stationarity breaks finite-sample coverage guarantees in severe winter inversions, persistent point prediction bias undermines symmetric intervals, and severe calibration sample depletion prevents empirical estimation of extreme non-conformity quantiles.
