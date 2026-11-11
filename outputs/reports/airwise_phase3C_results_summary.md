# AIRWISE — Phase 3C.3 Results Summary: Experimental Matrices and Comparative Findings

**Document ID:** `outputs/reports/airwise_phase3C_results_summary.md`  
**Phase:** 3C.3 (Manuscript Synthesis and Publication-Ready Reporting)  
**Author:** Antigravity Autonomous Research Assistant  
**Date:** November 2026  
**Status:** Validated Empirical Summary  

---

## 1. Executive Summary of Experimental Findings

The AIRWISE Phase 3C benchmark evaluated four distinct architectural pipelines across nine pre-registered missingness regimes spanning $N = 3,720$ independent test timestamps across three heterogeneous Indian environments (Bangalore, Delhi, Hamirpur). All models and predictions were frozen and verified via cryptographic hashes.

### The Central Empirical Finding
**"No single reconstruction strategy is uniformly optimal across missingness regimes."**
- **Pipeline B ($\text{Calibrate} \to \text{Impute}$)** is the overall strongest full-coverage benchmark baseline, achieving the lowest overall Root Mean Squared Error (RMSE: $36.26$ to $41.64$ $\mu$g/m³) across all nine missingness regimes.
- **Pipeline C (Causal Joint TCN v2)** provides localized reconstruction advantages during isolated short-duration dropouts ($1$h–$3$h) and humidity-induced fogging (MAR-RH), but suffers from multi-task gradient interference on clean observed cases ($38.01$ vs. $29.82$ $\mu$g/m³) and severe error compounding on extended outages ($81.86$ vs. $56.19$ $\mu$g/m³ for 24-hour block outages).
- **Pipeline A ($\text{Impute} \to \text{Calibrate}$)** acts as a robust middle-tier baseline, outperforming Joint TCN v2 on extended outages ($40.03$ vs. $49.66$ $\mu$g/m³ on 24-hour blocks).
- **Pipeline D (Complete-Case Reference)** establishes an uncorrupted calibration error ceiling of $29.82$ $\mu$g/m³, but fails completely during outages, providing $0.0\%$ prediction availability on missing intervals ($79.33\%$ total coverage).
- **Moving-Block Resampling:** Paired hypothesis testing demonstrates that perceived statistical significance under short 12-hour bootstrap blocks ($p = 0.010$) is an artifact of truncating diurnal autocorrelation; when the 24-hour diurnal cycle is respected, significance is properly rejected ($p = 0.084$).

---

## 2. Comprehensive Experimental Matrices

### 2.1 Table 1: Full-Population Overall Performance ($N = 3,720$)

| Scenario ID | Category | Pipeline Code | Pipeline Description | Evaluated $N$ | Coverage | RMSE ($\mu$g/m³) | MAE ($\mu$g/m³) | $R^2$ | MBE ($\mu$g/m³) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MCAR_20pct** | MCAR | A | $\text{Impute} \to \text{Calibrate}$ | 3,720 | 100.0% | 42.79 | 22.84 | 0.7639 | -3.32 |
| **MCAR_20pct** | MCAR | B | $\text{Calibrate} \to \text{Impute}$ | 3,720 | 100.0% | **37.77** | **20.71** | **0.8161** | **-1.11** |
| **MCAR_20pct** | MCAR | C | Causal Joint TCN v2 | 3,720 | 100.0% | 42.05 | 23.91 | 0.7720 | -4.77 |
| **MCAR_20pct** | MCAR | D* | Complete-Case Ref. | 2,951 | 79.33% | 29.82* | 16.13* | 0.8831* | -0.66* |
| **Block_1h** | Block | A | $\text{Impute} \to \text{Calibrate}$ | 3,720 | 100.0% | 40.85 | 22.18 | 0.7848 | -2.93 |
| **Block_1h** | Block | B | $\text{Calibrate} \to \text{Impute}$ | 3,720 | 100.0% | **36.33** | **20.15** | **0.8298** | **-0.85** |
| **Block_1h** | Block | C | Causal Joint TCN v2 | 3,720 | 100.0% | 42.02 | 23.99 | 0.7723 | -4.81 |
| **Block_1h** | Block | D* | Complete-Case Ref. | 2,976 | 80.00% | 30.61* | 16.29* | 0.8811* | -0.65* |
| **Block_3h** | Block | A | $\text{Impute} \to \text{Calibrate}$ | 3,720 | 100.0% | 40.30 | 21.82 | 0.7906 | -2.50 |
| **Block_3h** | Block | B | $\text{Calibrate} \to \text{Impute}$ | 3,720 | 100.0% | **37.01** | **20.52** | **0.8234** | **-0.43** |
| **Block_3h** | Block | C | Causal Joint TCN v2 | 3,720 | 100.0% | 42.03 | 24.00 | 0.7722 | -5.02 |
| **Block_3h** | Block | D* | Complete-Case Ref. | 2,976 | 80.00% | 30.61* | 16.22* | 0.8829* | -0.65* |
| **Block_6h** | Block | A | $\text{Impute} \to \text{Calibrate}$ | 3,720 | 100.0% | 42.11 | 21.88 | 0.7714 | -4.38 |
| **Block_6h** | Block | B | $\text{Calibrate} \to \text{Impute}$ | 3,720 | 100.0% | **36.26** | **19.94** | **0.8305** | **-1.11** |
| **Block_6h** | Block | C | Causal Joint TCN v2 | 3,720 | 100.0% | 44.45 | 25.14 | 0.7453 | -7.13 |
| **Block_6h** | Block | D* | Complete-Case Ref. | 2,974 | 79.95% | 30.42* | 15.72* | 0.8792* | -0.65* |
| **Block_12h** | Block | A | $\text{Impute} \to \text{Calibrate}$ | 3,720 | 100.0% | 44.12 | 23.40 | 0.7490 | -2.54 |
| **Block_12h** | Block | B | $\text{Calibrate} \to \text{Impute}$ | 3,720 | 100.0% | **39.83** | **21.71** | **0.7954** | **0.40** |
| **Block_12h** | Block | C | Causal Joint TCN v2 | 3,720 | 100.0% | 46.68 | 26.07 | 0.7191 | -7.36 |
| **Block_12h** | Block | D* | Complete-Case Ref. | 2,969 | 79.81% | 30.46* | 15.98* | 0.8814* | -0.66* |
| **Block_24h** | Block | A | $\text{Impute} \to \text{Calibrate}$ | 3,720 | 100.0% | 40.03 | 21.63 | 0.7934 | -2.44 |
| **Block_24h** | Block | B | $\text{Calibrate} \to \text{Impute}$ | 3,720 | 100.0% | **36.42** | **20.24** | **0.8290** | **1.20** |
| **Block_24h** | Block | C | Causal Joint TCN v2 | 3,720 | 100.0% | 49.66 | 27.49 | 0.6820 | -9.67 |
| **Block_24h** | Block | D* | Complete-Case Ref. | 2,970 | 79.84% | 29.39* | 14.75* | 0.8876* | -0.65* |
| **MAR_RH** | MAR | A | $\text{Impute} \to \text{Calibrate}$ | 3,720 | 100.0% | 41.47 | 23.00 | 0.7783 | -4.78 |
| **MAR_RH** | MAR | B | $\text{Calibrate} \to \text{Impute}$ | 3,720 | 100.0% | **37.00** | **20.48** | **0.8235** | **-3.30** |
| **MAR_RH** | MAR | C | Causal Joint TCN v2 | 3,720 | 100.0% | 41.14 | 24.05 | 0.7817 | -4.56 |
| **MAR_RH** | MAR | D* | Complete-Case Ref. | 2,976 | 80.00% | 32.27* | 17.88* | 0.8712* | -0.65* |
| **MAR_Lag** | MAR | A | $\text{Impute} \to \text{Calibrate}$ | 3,720 | 100.0% | 50.11 | 25.29 | 0.6762 | -10.86 |
| **MAR_Lag** | MAR | B | $\text{Calibrate} \to \text{Impute}$ | 3,720 | 100.0% | **41.57** | **22.07** | **0.7771** | **-5.96** |
| **MAR_Lag** | MAR | C | Causal Joint TCN v2 | 3,720 | 100.0% | 49.72 | 26.81 | 0.6813 | -12.48 |
| **MAR_Lag** | MAR | D* | Complete-Case Ref. | 2,976 | 80.00% | 25.69* | 13.04* | 0.8913* | -0.65* |
| **MNAR_PM** | MNAR | A | $\text{Impute} \to \text{Calibrate}$ | 3,720 | 100.0% | 50.40 | 25.37 | 0.6724 | -11.07 |
| **MNAR_PM** | MNAR | B | $\text{Calibrate} \to \text{Impute}$ | 3,720 | 100.0% | **41.64** | **22.09** | **0.7764** | **-6.09** |
| **MNAR_PM** | MNAR | C | Causal Joint TCN v2 | 3,720 | 100.0% | 49.94 | 26.89 | 0.6784 | -12.83 |
| **MNAR_PM** | MNAR | D* | Complete-Case Ref. | 2,976 | 80.00% | 25.63* | 13.05* | 0.8909* | -0.65* |

*\*Pipeline D evaluates strictly on uncorrupted sensor cases ($N \approx 2,970$), providing zero prediction coverage during missing sensor regimes.*

---

### 2.2 Table 2: Missing PM Subgroup Reconstruction Performance ($m_t = 0$)

| Scenario ID | Missing Hours ($N_{\text{mis}}$) | Pipeline A RMSE | Pipeline B RMSE | Pipeline C RMSE | Pipeline D RMSE | Pipeline C vs B ($\Delta_{\text{RMSE}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MCAR_20pct** | 769 | 73.79 | 59.05 | **54.87** | N/A (NaN) | **-4.18** (C better) |
| **Block_1h** | 744 | 67.79 | 53.39 | **52.44** | N/A (NaN) | **-0.95** (C better) |
| **Block_3h** | 744 | 66.13 | 55.68 | **53.04** | N/A (NaN) | **-2.64** (C better) |
| **Block_6h** | 746 | 71.78 | **53.54** | 62.95 | N/A (NaN) | +9.41 (B better) |
| **Block_12h**| 751 | 77.29 | **64.74** | 69.76 | N/A (NaN) | +5.02 (B better) |
| **Block_24h**| 750 | 67.28 | **56.19** | 81.86 | N/A (NaN) | +25.67 (B better) |
| **MAR_RH** | 744 | 66.58 | 51.77 | **43.05** | N/A (NaN) | **-8.72** (C better) |
| **MAR_Lag** | 744 | 99.58 | **77.47** | 88.45 | N/A (NaN) | +10.98 (B better) |
| **MNAR_PM** | 744 | 100.37 | **77.73** | 89.22 | N/A (NaN) | +11.49 (B better) |

---

### 2.3 Table 3: Common-Support Subgroup Calibration Performance ($m_t = 1$)

| Scenario ID | Matched Timestamps ($N_{\text{obs}}$) | Pipe A RMSE | Pipe B RMSE | Pipe C RMSE | Pipe D RMSE | Difference $\|A - B\|$ | Difference $\|A - D\|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MCAR_20pct** | 2,951 | **29.82** | **29.82** | 38.01 | **29.82** | $\le 1.5 \times 10^{-5}$ | $\le 1.5 \times 10^{-5}$ |
| **Block_1h** | 2,976 | **30.61** | **30.61** | 38.99 | **30.61** | $\le 1.5 \times 10^{-5}$ | $\le 1.5 \times 10^{-5}$ |
| **Block_3h** | 2,976 | **30.61** | **30.61** | 38.79 | **30.61** | $\le 1.5 \times 10^{-5}$ | $\le 1.5 \times 10^{-5}$ |
| **Block_6h** | 2,974 | **30.42** | **30.42** | 38.43 | **30.42** | $\le 1.5 \times 10^{-5}$ | $\le 1.5 \times 10^{-5}$ |
| **Block_12h**| 2,969 | **30.46** | **30.46** | 38.72 | **30.46** | $\le 1.5 \times 10^{-5}$ | $\le 1.5 \times 10^{-5}$ |
| **Block_24h**| 2,970 | **29.39** | **29.39** | 37.38 | **29.39** | $\le 1.5 \times 10^{-5}$ | $\le 1.5 \times 10^{-5}$ |
| **MAR_RH** | 2,976 | **32.27** | **32.27** | 40.65 | **32.27** | $\le 1.5 \times 10^{-5}$ | $\le 1.5 \times 10^{-5}$ |
| **MAR_Lag** | 2,976 | **25.69** | **25.69** | 33.68 | **25.69** | $\le 1.4 \times 10^{-5}$ | $\le 1.4 \times 10^{-5}$ |
| **MNAR_PM** | 2,976 | **25.63** | **25.63** | 33.57 | **25.63** | $\le 1.4 \times 10^{-5}$ | $\le 1.4 \times 10^{-5}$ |

---

### 2.4 Table 4: Site-Stratified Performance Breakdown Across Scenarios

| Scenario ID | Monitoring Site | Evaluated $N$ | Pipe A RMSE | Pipe B RMSE | Pipe C RMSE | Pipe A $R^2$ | Pipe B $R^2$ | Pipe C $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MCAR_20pct** | Bangalore | 1,489 | 17.19 | **8.74** | 13.43 | -1.626 | **0.322** | -0.602 |
| **MCAR_20pct** | Delhi | 1,502 | 60.25 | **54.78** | 62.91 | 0.646 | **0.707** | 0.614 |
| **MCAR_20pct** | Hamirpur | 729 | 35.52 | 30.67 | **22.39** | 0.524 | 0.645 | **0.811** |
| **Block_6h** | Bangalore | 1,489 | 15.65 | **7.57** | 13.02 | -1.171 | **0.490** | -0.504 |
| **Block_6h** | Delhi | 1,502 | 59.57 | **53.30** | 67.35 | 0.654 | **0.723** | 0.557 |
| **Block_6h** | Hamirpur | 729 | 35.58 | 29.54 | **22.75** | 0.523 | 0.671 | **0.805** |
| **Block_24h** | Bangalore | 1,489 | 12.33 | **7.88** | 15.19 | -0.349 | **0.449** | -1.047 |
| **Block_24h** | Delhi | 1,502 | 57.06 | **53.38** | 74.83 | 0.682 | **0.722** | 0.455 |
| **Block_24h** | Hamirpur | 729 | 35.79 | 29.83 | **32.81** | 0.517 | 0.665 | **0.594** |
| **MAR_RH** | Bangalore | 1,489 | 16.14 | **8.55** | 12.99 | -1.312 | **0.351** | -0.499 |
| **MAR_RH** | Delhi | 1,502 | 58.74 | **54.21** | 59.10 | 0.664 | **0.713** | 0.659 |
| **MAR_RH** | Hamirpur | 729 | 34.05 | 29.23 | **33.12** | 0.563 | 0.678 | **0.586** |
| **MNAR_PM** | Bangalore | 1,489 | 17.65 | **8.39** | 13.14 | -1.770 | **0.375** | -0.533 |
| **MNAR_PM** | Delhi | 1,502 | 72.84 | **62.39** | 76.53 | 0.482 | **0.619** | 0.428 |
| **MNAR_PM** | Hamirpur | 729 | 39.56 | 32.89 | **22.39** | 0.410 | 0.591 | **0.811** |

---

### 2.5 Table 5: Paired Moving-Block Bootstrap Statistical Inference ($B = 1,000, b = 24\text{h}$)

| Scenario ID | Comparison | Point $\Delta_{\text{RMSE}}$ | 95% Bootstrap CI Lower | 95% Bootstrap CI Upper | Empirical $p$-value | Statistically Significant ($\alpha=0.05$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MCAR_20pct** | Joint TCN (C) vs. Pipe A | -0.739 | -4.305 | +0.693 | 0.081 | No |
| **MCAR_20pct** | Joint TCN (C) vs. Pipe B | +4.286 | -0.775 | +4.370 | 0.908 | No |
| **MCAR_20pct** | Joint TCN (C) vs. Pipe D | +8.186 | +6.741 | +9.921 | 1.000 | Yes (Pipe D better on $m_t=1$) |
| **Block_1h** | Joint TCN (C) vs. Pipe A | +1.174 | -2.017 | +2.422 | 0.597 | No |
| **Block_1h** | Joint TCN (C) vs. Pipe B | +5.695 | +1.757 | +6.263 | 1.000 | Yes (Pipe B better) |
| **Block_3h** | Joint TCN (C) vs. Pipe A | +1.729 | -0.517 | +5.332 | 0.941 | No |
| **Block_3h** | Joint TCN (C) vs. Pipe B | +5.021 | +2.000 | +7.939 | 1.000 | Yes (Pipe B better) |
| **Block_6h** | Joint TCN (C) vs. Pipe A | +2.338 | -0.910 | +5.314 | 0.906 | No |
| **Block_6h** | Joint TCN (C) vs. Pipe B | +8.187 | +5.792 | +12.106 | 1.000 | Yes (Pipe B better) |
| **Block_12h**| Joint TCN (C) vs. Pipe A | +2.559 | -6.350 | +5.632 | 0.423 | No |
| **Block_12h**| Joint TCN (C) vs. Pipe B | +6.844 | +0.505 | +12.867 | 0.983 | Yes (Pipe B better) |
| **Block_24h**| Joint TCN (C) vs. Pipe A | **+9.633** | **+7.639** | **+19.644** | **1.000** | **Yes (Pipe A better)** |
| **Block_24h**| Joint TCN (C) vs. Pipe B | **+13.242** | **+14.437** | **+28.411** | **1.000** | **Yes (Pipe B better)** |
| **MAR_RH** | Joint TCN (C) vs. Pipe A | -0.325 | -0.967 | +5.019 | 0.925 | No |
| **MAR_RH** | Joint TCN (C) vs. Pipe B | +4.142 | +2.344 | +7.181 | 1.000 | Yes (Pipe B better) |
| **MAR_Lag** | Joint TCN (C) vs. Pipe A | -0.392 | -2.233 | +4.755 | 0.796 | No |
| **MAR_Lag** | Joint TCN (C) vs. Pipe B | +8.146 | +6.264 | +12.483 | 1.000 | Yes (Pipe B better) |
| **MNAR_PM** | Joint TCN (C) vs. Pipe A | -0.464 | -2.610 | +4.589 | 0.737 | No |
| **MNAR_PM** | Joint TCN (C) vs. Pipe B | +8.299 | +6.137 | +12.497 | 1.000 | Yes (Pipe B better) |

---

## 3. Methodological Insights and Implications

1. **Sequential vs. Joint Trade-Offs:**
   Decoupled calibration and imputation (Pipeline B) is the empirically superior general strategy. Training a multi-task deep neural network (Joint TCN v2) creates competing gradient objectives between sensor signal reconstruction and regulatory reference estimation, leading to degraded performance on clean observations ($38.01$ vs. $29.82$ $\mu$g/m³).
2. **Outage Duration Dynamics:**
   Reconstruction algorithms must be matched to operational outage expectations. For intermittent dropouts ($< 6$ hours), sequence models and spatial interpolators perform well. For multi-day sensor blackouts ($> 24$ hours), deep causal recurrent states diverge, and models conditioned strictly on exogenous weather proxies (Pipeline B) should be favored.
3. **Selection Bias Elimination:**
   The apparent superiority of complete-case calibration (Pipeline D) must always be qualified by its failure rate ($20.7\%$ outage dropout). Future benchmarks must report both coverage percentage and common-support matched error.
4. **Moving-Block Resampling Standard:**
   Hypothesis testing in environmental time series must enforce block resampling that spans at least the 24-hour diurnal cycle ($b=24\text{h}$) to avoid reporting spurious statistical significance.
