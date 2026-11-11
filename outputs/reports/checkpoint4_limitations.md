# Checkpoint 4 Limitations and Methodological Boundaries (Post-Audit Revision)

This document formalizes the unresolved scientific, statistical, and engineering limitations identified during Checkpoint 4 (Regime-Conditioned and Temporal Block Conformal Uncertainty Estimation), incorporating detailed post-audit diagnostics.

---

## 1. 86.40% Empirical Coverage is NOT Successful 90% Coverage

- **Persistent Under-Coverage:** While Temporal Block Conformal (12h) improves empirical coverage substantially over Ordinary Split Conformal ($86.40\%$ vs. $77.02\%$), it **strictly fails to achieve nominal 90% coverage**.
- **Coverage Deficit:** At significance level $\alpha = 0.10$, the empirical coverage error is $\Delta \mathrm{Cov} = -3.60$ percentage points. In scientific literature, an under-coverage of 3.6% cannot be claimed as a satisfied guarantee.
- **Root Cause:** Even with intervals widened to $88.67\,\mu\text{g}/\text{m}^3$, 13.60% of test observations (506 out of 3,720 hours) fall outside the prediction bounds, clustered predominantly in Delhi's winter inversion period.

---

## 2. Block Conformal Accounts for Temporal Grouping, NOT Distribution Shift

- **Theoretical Scope of Block Conformal:** Temporal block conformal constructs quantiles from block-maximum non-conformity scores ($s_{\mathrm{block}, b} = \max_{t \in \mathcal{B}_b} |Y_t - \widehat{Y}_t|$). This provides validity under the assumption that blocks are stationary and exchangeable. It accounts for serial autocorrelation (temporal grouping of errors within blocks of length $B$).
- **Distribution Shift Invariance:** Block calibration **does not and cannot protect against temporal distribution shift** where future environmental dynamics depart from historical observations:
  - In Delhi, observations with true reference $\text{BAM} > 250\,\mu\text{g}/\text{m}^3$ surged from $0\%$ in the conformal calibration partition (summer/monsoon) to $17.11\%$ in the test partition (winter inversion).
  - Even under 12h Block Conformal, empirical coverage for Delhi observations with $\text{BAM} > 250\,\mu\text{g}/\text{m}^3$ reached only **3.50%** ($N = 9/257$).
  - Block conformal widened intervals universally, but uniform widening cannot track or compensate for non-stationary meteorological and emission regime changes.

---

## 3. Winkler Score Apples-to-Apples Verification ($\alpha = 0.10$, $N = 3,720$)

To eliminate ambiguity regarding the Winkler interval score comparison:
- **Exact Scoring Convention:** The Winkler interval score is defined identically across all methods as:
  $$S_\alpha(L, U, Y) = (U - L) + \frac{2}{\alpha}(L - Y)\mathbb{I}(Y < L) + \frac{2}{\alpha}(Y - U)\mathbb{I}(Y > U)$$
- **Identical Evaluation Parameters:**
  - Nominal Significance Level: $\alpha = 0.10$ ($90\%$ nominal coverage target), resulting in a boundary penalty multiplier of $\frac{2}{0.10} = 20.0$.
  - Evaluation Population: Exact same $N = 3,720$ test observations from the untouched test partition.
  - Scoring Metric:
    - **Ordinary Split Conformal (Dual Head):** Mean Width $= 51.40\,\mu\text{g}/\text{m}^3$, Empirical Coverage $= 77.02\%$, Winkler Score $= \mathbf{263.94}$.
    - **Temporal Block Conformal (12h):** Mean Width $= 88.67\,\mu\text{g}/\text{m}^3$, Empirical Coverage $= 86.40\%$, Winkler Score $= \mathbf{237.03}$.
- **Why Block Conformal Achieves a Lower Score Despite Wider Intervals:**
  - Although Block Conformal adds $+37.27\,\mu\text{g}/\text{m}^3$ to the baseline interval width penalty $(U - L)$, it reduces the empirical miscoverage rate from $22.98\%$ down to $13.60\%$.
  - In Ordinary Split, the $22.98\%$ boundary violations in Delhi often deviate by $50\text{--}150\,\mu\text{g}/\text{m}^3$. Multiplied by the penalty weight of $20$, each severe violation adds $1,000\text{--}3,000$ to the point score.
  - Block Conformal drastically curtails the frequency and magnitude of these extreme tail penalties, resulting in an overall lower mean Winkler score ($237.03$ vs. $263.94$).
- **At 95% Nominal Level ($\alpha = 0.05$, penalty weight $= 40.0$):**
  - Ordinary Split Conformal: Mean Width $= 61.45\,\mu\text{g}/\text{m}^3$, Coverage $= 80.27\%$, Winkler Score $= \mathbf{443.47}$.
  - Temporal Block Conformal (12h): Mean Width $= 200.84\,\mu\text{g}/\text{m}^3$, Coverage $= 95.70\%$, Winkler Score $= \mathbf{318.54}$.

---

## 4. High-Pollution Regime Sample Sizes and Calibration Depletion

Regime conditioning can become brittle or degenerate if the calibration partition lacks adequate representation of extreme observations:

### Group Sample Sizes: Calibration vs. Test

| Stratification Category | Subgroup / Regime | Calibration $N$ | Calibration % | Test $N$ | Test % | Test / Calib Ratio | Fallback Triggered? ($N < 30$) | Fitted $q_{90}$ ($\mu\text{g}/\text{m}^3$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Predicted Pollution** | Low ($\le 60\,\mu\text{g}/\text{m}^3$) | 1,521 | 81.86% | 2,089 | 56.16% | 1.37x | No | 22.71 |
| | Moderate ($60\text{--}120\,\mu\text{g}/\text{m}^3$) | 330 | 17.76% | 815 | 21.91% | 2.47x | No | 39.23 |
| | **Severe ($> 120\,\mu\text{g}/\text{m}^3$)** | **7** | **0.38%** | **816** | **21.94%** | **116.57x** | **YES ($N=7 < 30$)** | **25.18 (Pooled)** |
| **Humidity** | Low ($\le 40\%$ RH) | 1,065 | 57.32% | 1,155 | 31.05% | 1.08x | No | 27.98 |
| | Medium ($40\text{--}65\%$ RH) | 499 | 26.86% | 1,993 | 53.58% | 3.99x | No | 21.57 |
| | High ($> 65\%$ RH) | 294 | 15.82% | 572 | 15.38% | 1.95x | No | 13.58 |
| **Site** | Delhi | 750 | 40.37% | 1,502 | 40.38% | 2.00x | No | 30.31 |
| | Hamirpur | 364 | 19.59% | 729 | 19.60% | 2.00x | No | 17.26 |
| | Bangalore | 744 | 40.04% | 1,489 | 40.03% | 2.00x | No | 22.71 |
| **True Reference BAM** | Low ($\le 60\,\mu\text{g}/\text{m}^3$) | 1,511 | 81.32% | 2,067 | 55.56% | 1.37x | — | — |
| | Moderate ($60\text{--}120\,\mu\text{g}/\text{m}^3$) | 320 | 17.22% | 688 | 18.49% | 2.15x | — | — |
| | Very Poor ($120\text{--}250\,\mu\text{g}/\text{m}^3$) | 27 | 1.45% | 702 | 18.87% | 26.00x | — | — |
| | **Severe ($> 250\,\mu\text{g}/\text{m}^3$)** | **0** | **0.00%** | **263** | **7.07%** | **$\infty$** | — | — |

### Key Diagnostic Insights
1. **Severe Predicted-Pollution Calibration Depletion:** In the calibration partition, only **7 observations** had model predictions $> 120\,\mu\text{g}/\text{m}^3$. Because $7 < N_{\min} = 30$, the calibrator **fell back to the pooled quantile** ($q_{90} = 25.18\,\mu\text{g}/\text{m}^3$). This explains why the predicted-pollution-conditioned interval in the severe regime failed to widen beyond the pooled baseline ($50.36\,\mu\text{g}/\text{m}^3$ vs. $51.37\,\mu\text{g}/\text{m}^3$).
2. **Complete Absence of Extreme Ground Truth:** In the calibration partition, true BAM $> 250\,\mu\text{g}/\text{m}^3$ had **zero samples ($N=0$)**. In the test partition, this extreme regime occurred in **263 hours** (257 in Delhi, representing 17.11% of Delhi's test set). The calibration set had no empirical basis to learn the non-conformity distribution for hazardous pollution.

---

## 5. Marginal vs. Conditional Coverage: The Danger of Pooled Metrics

A central theoretical and practical takeaway for the paper is the strict distinction between **marginal coverage** and **conditional coverage**:

### Mathematical Definitions
- **Marginal Coverage:** The expectation of coverage over the entire joint distribution of features and targets:
  $$\mathbb{P}\left( Y \in C_\alpha(X) \right) = \mathbb{E}_{(X,Y) \sim P}\left[ \mathbb{I}\left(Y \in C_\alpha(X)\right) \right] \ge 1 - \alpha$$
- **Conditional Coverage:** Coverage conditioned on a specific subgroup, regime, or feature value $X \in \mathcal{R}_k$:
  $$\mathbb{P}\left( Y \in C_\alpha(X) \;\middle|\; X \in \mathcal{R}_k \right) \ge 1 - \alpha, \quad \forall k$$

### Why Pooled Marginal Coverage is Deceptive in Sensor Networks
Standard split conformal prediction only provides a guarantee on **marginal coverage under exchangeability**. It provides **zero finite-sample guarantee on conditional coverage**.

In our experiments, pooled metrics conceal catastrophic local failures:
1. **Geographic Confounding:**
   - Pooled Marginal 90% Coverage: **77.02%** (Ordinary Split)
   - Bangalore Conditional Coverage: **97.58%** (+7.58% over-coverage)
   - Hamirpur Conditional Coverage: **88.48%** (-1.52% near-nominal)
   - Delhi Conditional Coverage: **51.07%** (**-38.93% catastrophic failure**)
   High sample representation in clean Bangalore air ($N = 1,489$) artificially inflates the pooled coverage figure.
2. **Pollution Severity Confounding:**
   - Clean Air ($\le 60\,\mu\text{g}/\text{m}^3$): **96.74%** conditional coverage ($N = 2,089$).
   - Moderate Air ($60\text{--}120\,\mu\text{g}/\text{m}^3$): **68.59%** conditional coverage ($N = 815$).
   - Severe Air ($> 120\,\mu\text{g}/\text{m}^3$): **34.93%** conditional coverage ($N = 816$).
   - Extreme Air ($> 250\,\mu\text{g}/\text{m}^3$): **1.95%** conditional coverage ($N = 257$).
3. **Outage Duration Confounding:**
   - 1-Hour Missingness Outage: **74.06%** conditional coverage.
   - 24-Hour Missingness Outage: **49.73%** conditional coverage.

**Paper Recommendation:** In the paper, we must explicitly state that achieving acceptable or near-nominal pooled marginal coverage does not imply reliability during hazardous air quality episodes or extended sensor dropouts. Public health interventions require conditional reliability in high-exposure regimes, where inductive conformal prediction currently fails due to physical sensor saturation, structural underfitting, and seasonal exchangeability violations.
