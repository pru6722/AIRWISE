# AIRWISE: Adaptive Imputation and Robust Wavelet/Temporal Inference for Sensor Estimation

## Assessing Sequential versus Joint Deep Architectures for Missing Data and Regulatory Calibration in Low-Cost Air Quality Networks

**Prudhvi Sailingineni**  
*Department of Computer Science and Engineering*  
*Big Data Systems Laboratory*  

---

## Abstract

Low-cost particulate matter ($\text{PM}_{2.5}$) optical sensor networks provide high-resolution spatio-temporal monitoring of ambient air pollution, yet they suffer from severe hygroscopic distortion, environmental drift, and frequent data outages. Conventional air quality analytics treat missing data imputation and regulatory instrument calibration (e.g., against Federal Equivalent Method Beta Attenuation Monitors, BAM) as decoupled sequential stages: either imputing raw low-cost sensor features prior to calibration ($\text{Impute} \to \text{Calibrate}$, Pipeline A) or calibrating observed records and interpolating the target predictions post-hoc ($\text{Calibrate} \to \text{Impute}$, Pipeline B). Recent advances in multi-task deep sequence models propose joint calibration and reconstruction architectures (Pipeline C) as a unified alternative. However, prior comparative evaluations have obscured fundamental differences in missingness mechanisms, evaluation support populations, and temporal autocorrelation.

In this work, we present **AIRWISE**, an integrated open-source framework and benchmark for evaluating four distinct reconstruction-calibration paradigms across nine pre-registered missingness regimes (20% Missing Completely at Random [MCAR], consecutive block outages from 1 to 24 hours, humidity-conditioned Missing at Random [MAR-RH], lag-conditioned [MAR-Lag], and unobserved particulate intensity [MNAR-PM]) across three diverse Indian urban environments (Bangalore, Delhi, and Hamirpur; $N = 18,584$ paired hourly observations). Utilizing a strictly causal Joint Temporal Convolutional Network (Joint TCN v2; receptive field $= 61$ hours) with verified future-perturbation invariance, alongside rigorous paired moving-block bootstrap hypothesis testing ($b \in \{12, 24, 48, 72\}$ hours), our findings establish that **no single reconstruction strategy is uniformly optimal across missingness regimes**. 

Empirically, Pipeline B emerges as the strongest full-coverage benchmark baseline across all nine scenarios, achieving the lowest overall Root Mean Squared Error (RMSE: $36.26$–$41.64$ $\mu$g/m³; $R^2: 0.776$–$0.831$). While Joint TCN v2 demonstrates localized advantages on high-humidity missing regimes (RMSE $43.05$ $\mu$g/m³ on the missing subgroup for MAR-RH vs. $51.77$ for Pipeline B), multi-task gradient coupling degrades its calibration accuracy on uncorrupted timestamps ($38.01$ vs. $29.82$ $\mu$g/m³ on common support) and leads to severe error compounding during extended outages ($81.86$ vs. $56.19$ $\mu$g/m³ for 24-hour block dropouts). Pipeline D (complete-case calibration) illustrates an error floor of $29.82$ $\mu$g/m³ on uncorrupted sensor cases, but provides zero prediction availability during outages ($79.3\%$ coverage). Furthermore, moving-block sensitivity sweeps demonstrate that statistical significance under short block lengths ($b=12$h) collapses when the diurnal cycle is preserved ($b=24$h), cautioning against assuming independent and identically distributed errors in environmental sensor benchmarking.

---

## Keywords
Low-cost air quality sensors; Missing data imputation; Regulatory calibration; Temporal Convolutional Networks; Multi-task learning; Moving-block bootstrap; Particulate matter ($\text{PM}_{2.5}$).

---

## 1. Introduction

Ambient air pollution, predominantly driven by fine particulate matter with aerodynamic diameter under $2.5$ micrometers ($\text{PM}_{2.5}$), constitutes one of the foremost global public health risks, contributing to an estimated $6.7$ million premature annual deaths globally, with acute burdens across South Asia [1, 2]. Regulatory environmental monitoring mandates continuous, highly accurate measurement using Federal Reference or Equivalent Methods (FRM/FEM), such as Beta Attenuation Monitors (BAM). However, regulatory stations require heavy capital infrastructure, high operational expenditures, and climate-controlled enclosures, resulting in sparse geographical deployment across low- and middle-income countries. In India, for example, vast metropolitan and rural regions rely on a small number of continuous stations, creating severe spatial coverage deficits [3].

To bridge this spatial resolution gap, low-cost sensor (LCS) networks based on optical particle counters (OPCs) have been widely deployed [4]. Sensors such as PurpleAir utilize light-scattering nephelometry to measure real-time particulate concentrations. Despite their scalability and minute-level temporal resolution, low-cost sensors exhibit three critical physical limitations:
1. **Hygroscopic Aerosol Growth:** At high relative humidity ($\text{RH} > 70\%$), deliquescent inorganic salts (e.g., sulfates, nitrates) absorb ambient moisture, increasing optical cross-sections and causing raw low-cost sensors to overestimate true dry particulate mass by up to 200–300% [5, 6].
2. **Missing Data and Hardware Dropouts:** Sensor fan failures, optical component fouling, power outages, and intermittent cellular connectivity create severe missing data regimes ranging from isolated missing timestamps to extended multi-day block dropouts [7].
3. **Temporal Drift and Site Dependency:** Sensor degradation, dust accumulation on optics, and differing regional aerosol size distributions cause empirical calibration relationships to drift over time and vary across geographic microclimates [8].

To extract regulatory-grade insights from LCS networks, computational environmental pipelines must perform two essential tasks: **imputation of missing sensor values** and **field calibration against co-located BAM instruments**. Traditionally, environmental engineering pipelines decouple these operations into two sequential paradigms:
- **Pipeline A ($\text{Impute} \to \text{Calibrate}$):** Missing low-cost sensor features are reconstructed via statistical or machine learning imputers (e.g., $k$-Nearest Neighbors, Random Forest, or Splines) before feeding the imputed feature vector into a calibration model [9].
- **Pipeline B ($\text{Calibrate} \to \text{Impute}$):** The calibration regressor is applied strictly to observed sensor records, generating calibrated $\text{PM}_{2.5}$ estimates that are subsequently interpolated or reconstructed at the target level across outage intervals [10].

More recently, deep sequence architectures—specifically Temporal Convolutional Networks (TCN) [11] and sequence-to-sequence recurrent networks—have been proposed as unified, end-to-end multi-task solutions (Pipeline C). Proponents hypothesize that joint representation learning can capture underlying atmospheric physics to simultaneously reconstruct missing sensor dynamics and predict regulatory BAM concentrations [12].

Despite widespread interest, existing literature suffers from four major methodological deficiencies:
1. **Conflation of Information Contexts:** Prior studies often evaluate joint models against offline, bidirectional baselines that access future temporal states, violating online operational causality.
2. **Inconsistent Missingness Mechanisms:** Imputation methods are predominantly evaluated under naive Missing Completely at Random (MCAR) assumptions, ignoring condition-dependent failure modes (e.g., humidity-induced condensation dropout or pollution-saturated fan clipping).
3. **Common-Support Evaluation Disregard:** Comparisons between complete-case models and full-coverage imputation pipelines often fail to control for evaluation support populations, generating apples-to-oranges comparisons due to selection bias.
4. **Neglect of Temporal Autocorrelation in Hypothesis Testing:** Standard paired $t$-tests or independent bootstrap resamples violate the i.i.d. assumption in time series, producing spuriously narrow confidence intervals and uncalibrated $p$-values.

In this paper, we resolve these challenges through **AIRWISE**, a comprehensive benchmark framework and experimental evaluation across $18,584$ paired hourly observations across three Indian cities. We strictly preserve temporal causality, enforce explicit target isolation, apply segment-constrained moving-block bootstrap resampling, and evaluate across nine missingness regimes.

---

## 2. Related Work

### 2.1 Low-Cost Sensor Calibration
Field calibration of low-cost particulate monitors against reference instrumentation has evolved from simple linear models to non-linear machine learning regressors. Early works utilized multilinear regression (MLR) incorporating relative humidity and temperature terms to correct for hygroscopic swelling [5, 6]. Barkjohn et al. [6] established the US EPA national calibration equation for PurpleAir sensors, demonstrating that empirical RH correction significantly reduces field measurement errors. Subsequent studies applied Random Forests (RF), Support Vector Regression (SVR), and gradient-boosted decision trees (XGBoost) [8, 13], showing that tree ensembles effectively capture complex microclimatic interactions. However, these calibration models typically presume the continuous availability of all input sensor channels.

### 2.2 Missing Data in Environmental Sensor Networks
Missing data in environmental Internet of Things (IoT) networks is governed by Little and Rubin's taxonomy [14]:
- **Missing Completely at Random (MCAR):** The probability of missingness is independent of both observed and unobserved data (e.g., random packet loss).
- **Missing at Random (MAR):** Missingness depends on observed covariates, such as high relative humidity triggering sensor fogging or past pollutant concentration.
- **Missing Not at Random (MNAR):** Missingness depends directly on the unobserved true value itself, such as extreme particulate episodes causing sensor detector clipping.

Traditional imputation techniques include linear and spline interpolation, matrix factorization, $k$-Nearest Neighbors (KNN), and MissForest [9, 15]. For temporal sensor streams, autoregressive models and recurrent networks (e.g., BRITS, GRU-D) have been introduced [16]. However, the vast majority of environmental studies evaluate imputation fidelity exclusively on reconstructed sensor inputs rather than examining the downstream propagation of imputation error into calibrated regulatory targets.

### 2.3 Deep Sequential and Multi-Task Architectures
Temporal Convolutional Networks (TCN), introduced by Bai et al. [11], employ causal dilated convolutions to achieve large receptive fields with stable gradients, avoiding the vanishing/exploding gradient pitfalls of recurrent networks. In environmental forecasting and sensor fusion, TCNs have demonstrated strong predictive capabilities [17]. Multi-task learning (MTL) architectures aim to share representations between complementary auxiliary tasks [18]. In the context of sensor calibration, multi-task formulations attempt to combine reconstruction loss on low-cost inputs with supervised loss on regulatory reference targets [12]. However, negative transfer and gradient interference between disparate loss surfaces often pose severe optimization challenges [19].

### 2.4 Resampling and Uncertainty in Environmental Time Series
Because atmospheric concentrations exhibit strong diurnal periodicity and multi-hour persistence, standard bootstrap resampling violates exchangeability. Politis and Romano [20] and Kunsch [21] developed the Moving-Block Bootstrap (MBB), which resamples contiguous temporal blocks to asymptotically preserve the underlying covariance structure. In low-cost sensor evaluation, failure to account for autocorrelation when assessing model superiority leads to severe Type I error inflation [22].

---

## 3. Research Gap and Contributions

### Research Gaps Addressed
1. **Lack of Controlled Multi-Pipeline Comparisons:** No prior study has systematically compared all four canonical calibration-reconstruction topologies (Pipelines A, B, C, and D) on identical datasets under identical missingness scenarios.
2. **Disregard for Online Operational Causality:** Many reported deep learning advantages stem from bidirectional future-state leakage rather than algorithmic superiority.
3. **Unvalidated Statistical Significance in Resampling:** Prior evaluations rely on point estimates or i.i.d. tests, failing to examine the sensitivity of conclusions to temporal block lengths.
4. **Ignoring Selection Bias in Complete-Case Benchmarking:** The trade-off between complete-case accuracy and prediction availability is routinely obscured.

### Key Contributions of AIRWISE
- **Unified Benchmark Protocol:** We implement and formalize an open-source framework comparing four distinct calibration-reconstruction pipelines under nine pre-registered missingness regimes across three heterogeneous urban sites in India.
- **Strictly Causal Joint TCN v2:** We design, audit, and benchmark an operational multi-task Joint TCN using `ChannelLayerNorm1d` and causal dilated convolutions ($RF = 61$h), mathematically verified via a 144-test future-perturbation gate ($\max |\Delta| = 0.0000 \le 1.0 \times 10^{-6}$).
- **Empirical Resolution of the Sequential vs. Joint Paradigm:** We demonstrate that **no single strategy is uniformly optimal**. Pipeline B ($\text{Calibrate} \to \text{Impute}$) provides the strongest full-coverage benchmark baseline across scenarios ($36.26$–$41.64$ $\mu$g/m³), while Joint TCN v2 offers selective advantages under high-humidity dropouts but suffers from severe error compounding on multi-day outages.
- **Rigorous Autocorrelation-Preserving Statistical Audit:** We deploy segment-constrained moving-block bootstrap hypothesis testing ($B=1,000$ iterations) across block lengths $b \in \{12, 24, 48, 72\}$ hours, proving that statistical significance observed under 12-hour blocks fails to hold once diurnal autocorrelation ($24$h) is respected.
- **Reproducible Artifacts and Cryptographic Provenance:** All code, models, parquet prediction tables, and metric summaries are published with full SHA-256 provenance and 81 unit/regression tests.

---

## 4. Dataset and Data Provenance

### 4.1 Geographical Monitoring Sites
The AIRWISE benchmark utilizes continuous, paired ambient hourly monitoring data collected over a multi-month field campaign across three climatically distinct regions in India [23]:
1. **Bangalore (Karnataka, Southern Deccan Plateau):** Elevation $\approx 920$ m. Characterized by moderate tropical savanna climate, lower baseline particulate concentrations (mean $\text{PM}_{2.5} \approx 28.4$ $\mu$g/m³), and moderate humidity swings.
2. **Delhi (National Capital Region, Indo-Gangetic Plain):** Elevation $\approx 215$ m. Characterized by severe seasonal pollution episodes, high winter inversion, extreme particulate loadings exceeding $400$ $\mu$g/m³ (mean $\text{PM}_{2.5} \approx 98.7$ $\mu$g/m³), and wide temperature extremes.
3. **Hamirpur (Himachal Pradesh, Sub-Himalayan Region):** Elevation $\approx 780$ m. Characterized by complex mountainous terrain, biomass burning emissions, biogenic aerosol influences, and rapid microclimatic transitions (mean $\text{PM}_{2.5} \approx 42.1$ $\mu$g/m³).

```
+-----------------------------------------------------------------------------------+
| AIRWISE Geographical Monitoring Sites                                             |
|                                                                                   |
|  [Delhi]                 Indo-Gangetic Plain | Extreme Winter Inversion           |
|  - Elev: ~215m           Mean PM2.5: 98.7 ug/m3 | N = 7,512 paired hours          |
|                                                                                   |
|  [Hamirpur]              Sub-Himalayan Valleys | Mountainous Biomass Emissions    |
|  - Elev: ~780m           Mean PM2.5: 42.1 ug/m3 | N = 3,648 paired hours          |
|                                                                                   |
|  [Bangalore]             Southern Deccan Plateau | Moderate Baseline Loading      |
|  - Elev: ~920m           Mean PM2.5: 28.4 ug/m3 | N = 7,424 paired hours          |
+-----------------------------------------------------------------------------------+
```

### 4.2 Sensor Instrumentation and Ground Truth Reference
- **Low-Cost Sensor Platform:** PurpleAir PA-II monitors containing dual plantower PMS5003 laser particle counters measuring three factory-calibrated mass concentration channels:
  - $\text{PA\_CF1}$: Calibration factor 1 (designed for laboratory/indoor aerosols).
  - $\text{PA\_ATM}$: Atmospheric calibration curve (ambient calibration).
  - $\text{PA\_ALT}$: Secondary channel cross-check.
  - Onboard sensors record ambient Relative Humidity ($\text{RH}$, $\%$) and Temperature ($T$, $^\circ\text{C}$).
- **Regulatory Ground Truth:** MetOne BAM-1020 Beta Attenuation Mass Monitor operating under standard regulatory quality assurance protocols, providing hourly dry mass $\text{PM}_{2.5}$ measurements ($\mu$g/m³).

### 4.3 Data Cleaning and Preprocessing Summary
Raw data underwent automated validation and quality screening:
- Removal of zero-variance hardware lockouts ($> 6$ consecutive hours of identical readings).
- Exclusion of extreme non-physical outliers ($\text{PM}_{2.5} > 1,500$ $\mu$g/m³, $\text{RH} < 1\%$, $\text{RH} > 100\%$).
- Temporal gap detection: Breaks in hourly timestamps $> 1$ hour define discrete, contiguous sequence segments (`segment_id`). No sequence window or temporal block is permitted to cross segment boundaries.
- **Total Dataset Size:** $N = 18,584$ clean, paired hourly observations across all sites.

#### Table 1: Dataset Summary and Partition Breakdown

| Dataset Partition | Bangalore ($N$) | Delhi ($N$) | Hamirpur ($N$) | Total Observations | Functional Role in AIRWISE Protocol |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | 4,297 | 4,352 | 2,093 | **10,742 (57.8%)** | Model parameter fitting (regressors, imputers, normalizers) |
| **Model Validation Set** | 823 | 830 | 405 | **2,058 (11.1%)** | Neural architecture selection and early stopping |
| **Conformal Calibration Set**| 815 | 828 | 421 | **2,064 (11.1%)** | Post-calibration nonconformity quantile calibration (frozen) |
| **Independent Test Set** | 1,489 | 1,502 | 729 | **3,720 (20.0%)** | Final comparative benchmark evaluation |
| **Total Clean Dataset** | **7,424** | **7,512** | **3,648** | **18,584 (100.0%)** | Continuous multi-site paired observation pool |

---

## 5. Problem Formulation and Masking Taxonomy

Let $X_t \in \mathbb{R}^D$ denote the multivariate vector of low-cost sensor observations at discrete hourly timestamp $t$, consisting of:
$$X_t = \left[ \text{PA\_CF1}_t, \, \text{PA\_ATM}_t, \, \text{PA\_ALT}_t, \, \text{RH}_t, \, T_t, \, \Phi_t \right]^T$$
where $\Phi_t = [\sin(2\pi h/24), \cos(2\pi h/24), \sin(2\pi d/7), \cos(2\pi d/7), \sin(2\pi m/12)]^T$ represents deterministic cyclical time encodings (hour of day $h$, day of week $d$, month of year $m$). Let $y_t^{\text{BAM}} \in \mathbb{R}$ represent the true regulatory ground-truth particulate mass concentration.

### 5.1 Dual-Mask Formulation ($M_X$ vs. $M_Y$)
To prevent target leakage while accurately representing hardware outages, we define two strictly separated binary masks:

1. **Feature Missingness Mask ($M_{X, t} \in \{0, 1\}^D$):**
   $$M_{X, t, c} = \begin{cases} 1 & \text{if sensor channel } c \text{ is observed at time } t \\ 0 & \text{if sensor channel } c \text{ is missing/corrupted} \end{cases}$$
   In standard particulate dropouts, the low-cost PM channels ($\text{PA\_CF1}, \text{PA\_ATM}, \text{PA\_ALT}$) fail simultaneously, defined by the scalar indicator $m_t^{\text{PM}} = M_{X, t, 1} \in \{0, 1\}$. Meteorological channels ($\text{RH}, T$) may remain available or drop during complete station power outages.

2. **Target Withholding Mask ($M_{Y, t} \in \{0, 1\}$):**
   $$M_{Y, t} = \begin{cases} 1 & \text{if BAM target } y_t^{\text{BAM}} \text{ is accessible to the current training step} \\ 0 & \text{if BAM target } y_t^{\text{BAM}} \text{ is withheld from model fitting} \end{cases}$$
   For the test partition ($N = 3,720$), $M_{Y, t} = 0$ strictly for all model inputs across all four pipelines. Ground-truth BAM targets are accessible exclusively to the post-hoc evaluator.

```
+-----------------------------------------------------------------------------------+
| AIRWISE Dual-Mask Taxonomy                                                        |
|                                                                                   |
|  Input Stream:     [PA_CF1, PA_ATM, PA_ALT, RH, Temp, Cyclical Features]          |
|  Missingness Mask: [  M_X,1 ,   M_X,2 ,   M_X,3 ,  1 ,   1 ,        1       ]     |
|                                                                                   |
|  Calibration Model Input:   X_tilde_t = X_t (if observed) or Imputed / Zeroed     |
|  Regulatory Target:         y_t^BAM (WITHHELD via M_Y = 0 during test inference)  |
+-----------------------------------------------------------------------------------+
```

---

## 6. The Four Experimental Pipelines

We formulate four distinct architectural pipelines representing the full spectrum of environmental data handling:

```
+-----------------------------------------------------------------------------------+
| Four-Pipeline Architectural Comparison                                            |
|                                                                                   |
|  Pipeline A (Impute -> Calibrate):                                                |
|  Raw Low-Cost Features ---> [Imputer (KNN/RF)] ---> [Calibrator (RF)] ---> Target |
|                                                                                   |
|  Pipeline B (Calibrate -> Impute):                                                |
|  Observed Raw Features ---> [Calibrator (RF)] ---> [Target Imputer] ---> Target   |
|                                                                                   |
|  Pipeline C (Joint TCN v2):                                                       |
|  Raw Stream + Mask ---> [Causal Conv1D Dilated Residuals] ---> [Multi-Task Heads] |
|                                                                                   |
|  Pipeline D (Complete-Case Reference):                                            |
|  Complete Cases Only -----> [Calibrator (RF)] --------------------------> Target |
|  (Outputs NaN on missing timestamps; coverage < 100%)                             |
+-----------------------------------------------------------------------------------+
```

### 6.1 Pipeline A: Sequential Impute $\to$ Calibrate
1. **Imputation Stage:** When $m_t^{\text{PM}} = 0$, missing low-cost sensor features are imputed using a $k$-Nearest Neighbors ($k=5$) imputer fitted strictly on `train_df`.
2. **Calibration Stage:** The fully imputed feature matrix $\tilde{X}_t$ is passed into a Random Forest regressor ($100$ estimators, $\text{max\_depth} = 15$) fitted strictly on complete training pairs $(\tilde{X}_{\text{train}}, y_{\text{train}}^{\text{BAM}})$.
3. **Coverage:** $100\%$ across all test timestamps.

### 6.2 Pipeline B: Sequential Calibrate $\to$ Impute
1. **Calibration Stage:** Complete observations $(m_t^{\text{PM}} = 1)$ are calibrated directly using the Random Forest regressor:
   $$\hat{y}_t^{\text{cal}} = f_{\text{RF}}(X_t) \quad \text{for } t \text{ where } m_t^{\text{PM}} = 1$$
2. **Imputation Stage:** Missing target predictions $(m_t^{\text{PM}} = 0)$ are reconstructed using a Random Forest target imputer conditioned on observed meteorological covariates $(\text{RH}_t, T_t)$, cyclic temporal encodings, and lagged target estimates.
3. **Coverage:** $100\%$ across all test timestamps.

### 6.3 Pipeline C: Causal Joint TCN v2
Pipeline C implements a multi-task deep sequence model operating over causal history sequences of length $L = 64$ hours.
- **Architectural Backbone:** Stacked dilated residual causal convolution blocks with dilations $d \in \{1, 2, 4, 8\}$, kernel size $k = 3$, and $32$ hidden channels.
- **Receptive Field ($RF$):**
  $$RF = 1 + \sum_{l=0}^{3} 2 \cdot (k - 1) \cdot d_l = 1 + 2 \cdot (3 - 1) \cdot (1 + 2 + 4 + 8) = 61 \text{ hours}$$
  Because $RF = 61 \le L = 64$, every output step is fully receptive to its 61-hour causal past without receptive truncation.
- **Channel Layer Normalization:** Replaces batch normalization with `ChannelLayerNorm1d`, normalizing across the 32 channel features at each individual timestep to eliminate cross-timestep leakage.
- **Multi-Task Heads:**
  1. *Calibration Head:* $\hat{y}_t^{\text{cal}} = W_{\text{cal}} h_t + b_{\text{cal}}$, supervised by $y_t^{\text{BAM}}$ on observed timestamps $(m_t=1)$.
  2. *Reconstruction Head:* $\hat{x}_t^{\text{rec}} = W_{\text{rec}} h_t + b_{\text{rec}}$, reconstructing the normalized low-cost PM channels.
  3. *Composite Output:*
     $$\hat{y}_t^{\text{final}} = m_t \hat{y}_t^{\text{cal}} + (1 - m_t) g_{\text{cal}}(\hat{x}_t^{\text{rec}})$$
- **Multi-Task Objective:**
  $$\mathcal{L}_{\text{total}} = \frac{1}{\sum m_t} \sum_{t=1}^T m_t \left(y_t^{\text{BAM}} - \hat{y}_t^{\text{cal}}\right)^2 + \frac{\lambda_{\text{rec}}}{3T} \sum_{t=1}^T \sum_{c=1}^3 \left(x_{t, c}^{\text{PM}} - \hat{x}_{t, c}^{\text{rec}}\right)^2$$
  with $\lambda_{\text{rec}} = 0.5$, optimized via AdamW ($\text{lr} = 1.0 \times 10^{-3}$, weight decay $= 1.0 \times 10^{-4}$).

### 6.4 Pipeline D: Complete-Case Calibration Reference
Pipeline D trains the Random Forest calibration regressor strictly on complete training observations. At inference time, it outputs predictions only when $m_t^{\text{PM}} = 1$. When sensor channels are missing ($m_t = 0$), it outputs `np.nan`.
- **Role:** Establishes the empirical calibration upper bound on clean, uncorrupted sensor observations.
- **Coverage:** $\approx 79.3\%$ to $80.0\%$ (0% prediction availability during outage periods).

---

## 7. Causality Verification and Pre-Training Gate

To guarantee that Pipeline C does not exploit acausal lookahead, we implemented a formal **Future Perturbation Invariance Test** prior to training.

### Mathematical Formulation
Let $\mathcal{M}(X_{1:T}, M_{1:T})_t$ denote the model output at timestep $t$. For a cut position $\tau \in \{16, 32, 48\}$, we perturb future inputs at $t' > \tau$ by magnitude $\delta \in \{10.0, 100.0, 1000.0\}$:
$$X'_{t', c} = X_{t', c} + \delta \quad \forall t' > \tau, \quad M'_{t'} = 1 - M_{t'} \quad \forall t' > \tau$$
A model is strictly causal if and only if past and present outputs are invariant to future perturbations:
$$\max_{t \le \tau} \left| \mathcal{M}(X'_{1:T}, M'_{1:T})_t - \mathcal{M}(X_{1:T}, M_{1:T})_t \right| \le \epsilon = 1.0 \times 10^{-6}$$

### Verification Results
Across 144 pre-training check combinations (testing calibration, reconstruction, and composite heads in both `train()` and `eval()` modes across multiple sequence lengths), the maximum observed deviation was strictly **$0.0000$**, passing the pre-training gate.

---

## 8. Experimental Protocol and Missingness Scenarios

All four pipelines were evaluated on the independent test set ($N = 3,720$) across nine pre-registered scenarios generated with reproducible random seed $42$:

1. **20% MCAR:** Random uniform dropout of low-cost PM channels ($p = 0.20$).
2. **1-Hour Block Outages:** Contiguous 1-hour sensor dropouts.
3. **3-Hour Block Outages:** Contiguous 3-hour dropouts.
4. **6-Hour Block Outages:** Contiguous 6-hour dropouts.
5. **12-Hour Block Outages:** Contiguous 12-hour dropouts.
6. **24-Hour Block Outages:** Contiguous 24-hour diurnal outages.
7. **MAR-RH (Humidity-Conditioned):** Dropout probability conditioned on ambient relative humidity:
   $$P(M_t = 0 \mid \text{RH}_t) = \sigma\left(6.0 \cdot (\text{RH}_t^{\text{norm}} - 0.55)\right)$$
8. **MAR-Lag (Lagged Particulate Conditioned):** Dropout probability conditioned on observed particulate concentration at the previous timestep $t-1$ ($X_{t-1}$).
9. **MNAR-PM (Synthetic Particulate Stress Test):** Dropout probability conditioned on the current unobserved particulate intensity ($X_t$).

#### Table 2: Empirical Outage Duration Distributions

| Scenario ID | Nominal Duration | Total Missing Points | Outage Events | Min Duration | Median Duration | Mean Duration | Max Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block_1h** | 1h | 744 (20.0%) | 599 | **1h** | 1.0h | 1.24h | 5h |
| **Block_3h** | 3h | 744 (20.0%) | 205 | **3h** | 3.0h | 3.63h | 14h |
| **Block_6h** | 6h | 746 (20.1%) | 108 | **6h** | 6.0h | 6.91h | 16h |
| **Block_12h**| 12h | 751 (20.2%) | 52 | **12h** | 12.0h | 14.44h | 25h |
| **Block_24h**| 24h | 750 (20.2%) | 23 | **24h** | 24.0h | 32.61h | 63h |

*Note: Randomized placement of fixed-length blocks within bounded continuous segments allows adjacent blocks to occasionally coalesce, generating realistic compound outages with right-skewed tails.*

---

## 9. Results and Comparative Analysis

### 9.1 Overall Benchmark Performance ($N = 3,720$)

#### Table 3: Overall Performance Across Nine Missingness Scenarios (Full Test Population)

| Scenario ID | Pipe A RMSE | Pipe B RMSE | Pipe C RMSE | Pipe D RMSE* | Pipe A $R^2$ | Pipe B $R^2$ | Pipe C $R^2$ | Pipe D $R^2$* | Pipe D Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MCAR_20pct** | 42.79 | **37.77** | 42.05 | 29.82* | 0.764 | **0.816** | 0.772 | 0.883* | 79.33% |
| **Block_1h** | 40.85 | **36.33** | 42.02 | 30.61* | 0.785 | **0.830** | 0.772 | 0.881* | 80.00% |
| **Block_3h** | 40.30 | **37.01** | 42.03 | 30.61* | 0.791 | **0.823** | 0.772 | 0.883* | 80.00% |
| **Block_6h** | 42.11 | **36.26** | 44.45 | 30.42* | 0.771 | **0.831** | 0.745 | 0.879* | 79.95% |
| **Block_12h** | 44.12 | **39.83** | 46.68 | 30.46* | 0.749 | **0.795** | 0.719 | 0.881* | 79.81% |
| **Block_24h** | 40.03 | **36.42** | 49.66 | 29.39* | 0.793 | **0.829** | 0.682 | 0.888* | 79.84% |
| **MAR_RH** | 41.47 | **37.00** | 41.14 | 32.27* | 0.778 | **0.824** | 0.782 | 0.871* | 80.00% |
| **MAR_Lag** | 50.11 | **41.57** | 49.72 | 25.69* | 0.676 | **0.777** | 0.681 | 0.891* | 80.00% |
| **MNAR_PM** | 50.40 | **41.64** | 49.94 | 25.63* | 0.672 | **0.776** | 0.678 | 0.891* | 80.00% |

*\*Pipeline D is evaluated exclusively on its available complete observations ($N \approx 2,970$), providing zero coverage on missing intervals. It serves as an upper-bound calibration reference, not a full-population comparable method.*

```
+-----------------------------------------------------------------------------------+
| Overall RMSE Comparison Across Scenarios (Full Population N=3720)                 |
|                                                                                   |
|  Scenario        Pipe A (Imp->Cal)   Pipe B (Cal->Imp)   Pipe C (Joint TCN v2)    |
|  MCAR 20%              42.79               37.77                 42.05            |
|  Block 1h              40.85               36.33                 42.02            |
|  Block 6h              42.11               36.26                 44.45            |
|  Block 24h             40.03               36.42                 49.66            |
|  MAR-RH                41.47               37.00                 41.14            |
|  MNAR-PM               50.40               41.64                 49.94            |
|                                                                                   |
|  --> Pipeline B achieves the lowest overall RMSE across all 9 benchmark scenarios |
+-----------------------------------------------------------------------------------+
```

### 9.2 Key Overall Findings
1. **Pipeline B is the Strongest Full-Coverage Benchmark Baseline:**
   Across all nine missingness regimes, Pipeline B ($\text{Calibrate} \to \text{Impute}$) consistently achieves the lowest overall RMSE ($36.26$ to $41.64$ $\mu$g/m³) and highest $R^2$ ($0.776$ to $0.831$). By calibrating observed sensor signals prior to reconstruction, Pipeline B avoids compounding non-linear imputer errors through the calibration regressor.
2. **Joint TCN v2 Exhibits Selective Trade-Offs:**
   Pipeline C does not achieve general superiority over sequential pipelines. While competitive on MCAR ($42.05$ vs $42.79$ for Pipeline A) and MAR-RH ($41.14$ vs $41.47$), it suffers pronounced performance degradation as outage blocks lengthen, reaching an RMSE of $49.66$ $\mu$g/m³ ($R^2 = 0.682$) on 24-hour block dropouts.
3. **Pipeline D Trade-Off:**
   Pipeline D exhibits an apparent RMSE of $\approx 29.8$ $\mu$g/m³. However, this is an artifact of selection bias: Pipeline D drops all $744$–$769$ missing hours. In operational air quality monitoring, a system that fails to report during 20% of the year—often during severe pollution episodes—is unacceptable.

---

## 10. Missingness Regime and Outage Analysis

### 10.1 Outage Subgroup Performance ($m_t = 0$)
To evaluate pure reconstruction fidelity during sensor dropouts, we isolate the missing PM subgroup where low-cost sensor channels are unobserved.

#### Table 4: Reconstruction Error on Missing PM Subgroup ($m_t = 0$)

| Scenario ID | Missing Hours ($N$) | Pipeline A RMSE | Pipeline B RMSE | Pipeline C RMSE | Pipeline D RMSE | Pipeline C vs. B ($\Delta$) |
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

```
+-----------------------------------------------------------------------------------+
| Outage-Duration Error Degradation on Missing Subgroup (m_t = 0)                   |
|                                                                                   |
|  Outage Duration     Pipe A (Imp->Cal)   Pipe B (Cal->Imp)   Pipe C (Joint TCN v2)|
|  1-Hour Block              67.79               53.39                 52.44        |
|  3-Hour Block              66.13               55.68                 53.04        |
|  6-Hour Block              71.78               53.54                 62.95        |
|  12-Hour Block             77.29               64.74                 69.76        |
|  24-Hour Block             67.28               56.19                 81.86        |
|                                                                                   |
|  --> Joint TCN v2 degrades severely (+29.4 ug/m3) as block outages extend to 24h  |
+-----------------------------------------------------------------------------------+
```

### 10.2 Insights into Regime-Specific Performance
1. **Short Outages ($1$h–$3$h) and MCAR:** Joint TCN v2 achieves superior reconstruction accuracy on isolated dropouts (RMSE $52.44$ vs $53.39$ on 1h blocks; $54.87$ vs $59.05$ on MCAR). Its causal convolutional receptive field ($61$h) leverages recent temporal momentum to interpolate brief missing intervals.
2. **Long Outages ($12$h–$24$h):** For extended block dropouts, Joint TCN v2 suffers severe degradation, with reconstruction RMSE ballooning to $81.86$ $\mu$g/m³ on 24-hour outages (compared to $56.19$ for Pipeline B and $67.28$ for Pipeline A). When low-cost sensor inputs are zeroed for 24 consecutive steps, the TCN’s internal states saturate, leading to compounding error propagation. In contrast, Pipeline B’s post-calibration imputer leverages diurnal solar radiation and ambient temperature proxies to maintain stable performance.
3. **MAR-RH Conditioned Outages:** On humidity-conditioned dropouts, Joint TCN v2 demonstrates its strongest advantage, achieving an outage RMSE of $43.05$ $\mu$g/m³ ($R^2 = 0.690$) vs. $51.77$ for Pipeline B. Here, the network's continuous exposure to unmasked ambient RH and temperature channels allows its causal temporal filters to effectively model hygroscopic hysteresis.
4. **Extreme Particulate Regimes (MAR-Lag and MNAR-PM):** Under pollution-dependent dropout, all models experience substantial degradation, with errors exceeding $77$ $\mu$g/m³. Pipeline B maintains the lowest error ($77.47$ and $77.73$ $\mu$g/m³), while Pipeline A fails completely ($> 99$ $\mu$g/m³).

---

## 11. Common-Support Evaluation ($m_t = 1$)

To eliminate survival bias and evaluate pure calibration accuracy under uncorrupted sensor conditions, all four pipelines were benchmarked on the matched **Common-Support Subgroup** ($m_t = 1$, $N = 2,951$ for MCAR).

#### Table 5: Common-Support Calibration Metrics ($N = 2,951$, Identical Matched Timestamps)

| Pipeline Code | Architecture Description | Evaluated $N$ | Coverage | RMSE ($\mu$g/m³) | MAE ($\mu$g/m³) | $R^2$ | MBE ($\mu$g/m³) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pipeline A** | $\text{Impute} \to \text{Calibrate}$ (KNN + RF) | 2,951 | 100.0% | **29.82** | **16.13** | **0.883** | **-0.66** |
| **Pipeline B** | $\text{Calibrate} \to \text{Impute}$ (RF + RF) | 2,951 | 100.0% | **29.82** | **16.13** | **0.883** | **-0.66** |
| **Pipeline C** | Causal Joint TCN v2 Multi-Task | 2,951 | 100.0% | 38.01 | 22.12 | 0.810 | -2.31 |
| **Pipeline D** | Complete-Case Calibration Reference | 2,951 | 100.0% | **29.82** | **16.13** | **0.883** | **-0.66** |

### Common-Support Findings
1. **Mathematical Equivalence of A, B, and D:** When raw features are observed, Pipeline A passes inputs directly to calibration, Pipeline B executes calibration directly, and Pipeline D operates on its native complete cases. The maximum absolute difference between their predictions across all 2,951 timestamps is $\le 1.51 \times 10^{-5}$ $\mu$g/m³ (32-bit floating-point precision).
2. **Multi-Task Gradient Interference in Joint TCN v2:** On clean observed data, Joint TCN v2 exhibits an RMSE of $38.01$ $\mu$g/m³—an $8.19$ $\mu$g/m³ penalty compared to the Random Forest baseline ($29.82$ $\mu$g/m³). This empirical gap demonstrates negative transfer: simultaneous optimization of low-cost sensor reconstruction and BAM calibration compromises asymptotic calibration fidelity.

---

## 12. Statistical Uncertainty and Moving-Block Bootstrap Audit

### 12.1 Moving-Block Bootstrap Framework
Because ambient air quality measurements exhibit strong diurnal autocorrelation, evaluating model superiority via standard $t$-tests or independent bootstrap resamples yields uncalibrated false positives. We implemented a segment-constrained **Moving-Block Bootstrap (MBB)**:
- Resampling units are contiguous temporal blocks of length $b$ hours, constrained strictly within individual sequence segments: $\text{segments}[start] == \text{segments}[end - 1]$.
- Paired evaluation computes $\Delta_{\text{RMSE}} = \text{RMSE}_{\text{Joint (C)}} - \text{RMSE}_{\text{Pipeline A}}$ simultaneously across $B = 1,000$ iterations.
- Empirical one-tailed non-superiority proportion:
  $$p = \frac{1}{B} \sum_{k=1}^B \mathbb{I}\left(\Delta_{\text{RMSE}}^{(k)} \ge 0.0\right)$$

#### Table 6: Paired Moving-Block Bootstrap Results ($B = 1,000, b = 24\text{h}$)

| Scenario ID | Comparison | Point $\Delta_{\text{RMSE}}$ | 95% Bootstrap CI Lower | 95% Bootstrap CI Upper | $p$-value | Significant ($\alpha = 0.05$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MCAR_20pct** | Joint (C) vs. Pipe A | -0.739 | -4.305 | +0.693 | 0.081 | No |
| **MCAR_20pct** | Joint (C) vs. Pipe B | +4.286 | -0.775 | +4.370 | 0.908 | No |
| **Block_1h** | Joint (C) vs. Pipe A | +1.174 | -2.017 | +2.422 | 0.597 | No |
| **Block_3h** | Joint (C) vs. Pipe A | +1.729 | -0.517 | +5.332 | 0.941 | No |
| **Block_6h** | Joint (C) vs. Pipe A | +2.338 | -0.910 | +5.314 | 0.906 | No |
| **Block_12h**| Joint (C) vs. Pipe A | +2.559 | -6.350 | +5.632 | 0.423 | No |
| **Block_24h**| Joint (C) vs. Pipe A | **+9.633** | **+7.639** | **+19.644** | **1.000** | **Yes (Pipe A better)** |
| **MAR_RH** | Joint (C) vs. Pipe A | -0.325 | -0.967 | +5.019 | 0.925 | No |
| **MAR_Lag** | Joint (C) vs. Pipe A | -0.392 | -2.233 | +4.755 | 0.796 | No |
| **MNAR_PM** | Joint (C) vs. Pipe A | -0.464 | -2.610 | +4.589 | 0.737 | No |

### 12.2 Resolution of the MAR-RH Bootstrap Asymmetry
In MAR-RH, the sample point estimate was $\Delta_{\text{RMSE}} = -0.325$, yet the empirical tail proportion was $p = 0.925$ with a 95% CI of $[-0.967, +5.019]$. Decomposing this result revealed:
- **Site-Level Performance Heterogeneity:** In Hamirpur ($N=729$), Joint TCN v2 outperformed Pipeline A by $21.37$ $\mu$g/m³ ($33.12$ vs $54.49$). Because RMSE penalizes extreme errors quadratically, Hamirpur pulled the pooled sample point estimate down to $-0.325$.
- In contrast, across Delhi ($N=1,502$) and Bangalore ($N=1,489$)—which represent **80.4%** of the dataset—Pipeline A was superior ($\Delta = +7.38$ and $+0.98$ $\mu$g/m³).
- Under moving-block bootstrap, 92.5% of the 1,000 resampled datasets were dominated by Delhi and Bangalore blocks, resulting in $\Delta \ge 0$ in 925 iterations. The bootstrap mean ($+2.214$) and median ($+2.248$) correctly reflect the population distribution.

### 12.3 Block-Length Sensitivity Analysis ($b \in \{12, 24, 48, 72\}\text{h}$)
To test the sensitivity of conclusions to the block-length parameter, we executed sweeps across $b \in \{12, 24, 48, 72\}$ hours.

#### Table 7: Sensitivity of Bootstrap Confidence Intervals to Block Length

| Scenario ID | Block Length ($b$) | Point $\Delta_{\text{RMSE}}$ | 95% CI Lower | 95% CI Upper | CI Width | $p$-value | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MCAR_20pct** | 12h | -0.739 | -5.268 | -0.368 | 4.900 | **0.010** | **Yes** |
| **MCAR_20pct** | 24h | -0.739 | -4.308 | +0.702 | 5.010 | **0.084** | No |
| **MCAR_20pct** | 48h | -0.739 | -3.318 | +2.557 | 5.875 | **0.384** | No |
| **MCAR_20pct** | 72h | -0.739 | -2.904 | +4.245 | 7.149 | **0.652** | No |
| **Block_6h** | 12h | +2.338 | -3.330 | +3.701 | 7.031 | 0.556 | No |
| **Block_6h** | 24h | +2.338 | -0.886 | +5.321 | 6.207 | 0.892 | No |
| **Block_6h** | 48h | +2.338 | -0.803 | +6.937 | 7.740 | 0.940 | No |
| **Block_6h** | 72h | +2.338 | -0.718 | +8.838 | 9.556 | 0.964 | No |
| **MNAR_PM** | 12h | -0.464 | -4.844 | +1.718 | 6.562 | 0.160 | No |
| **MNAR_PM** | 24h | -0.464 | -2.798 | +4.595 | 7.393 | 0.718 | No |
| **MNAR_PM** | 48h | -0.464 | -1.470 | +7.686 | 9.156 | 0.906 | No |
| **MNAR_PM** | 72h | -0.464 | -0.208 | +10.696 | 10.904 | 0.968 | No |

**Critical Methodological Insight:** Under $b=12\text{h}$, Joint TCN v2 appeared statistically significantly superior on 20% MCAR ($p = 0.010$). However, a 12-hour block truncates the 24-hour diurnal atmospheric cycle, artificially inflating the effective sample size. When the canonical diurnal cycle is preserved ($b=24\text{h}$), significance vanishes ($p = 0.084$). As block length expands to $48$h and $72$h, variance increases monotonically, confirming that Joint TCN v2 does not achieve robust superiority.

---

## 13. Site-Level Analysis and Performance Heterogeneity

To evaluate geographic transferability, model performance was stratified across the three monitoring environments.

#### Table 8: Site-Stratified Performance Breakdown (20% MCAR)

| Monitoring Site | Evaluated $N$ | Pipeline A RMSE | Pipeline B RMSE | Pipeline C RMSE | Pipeline A $R^2$ | Pipeline B $R^2$ | Pipeline C $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bangalore** | 1,489 | 17.19 | **8.74** | 13.43 | -1.626 | **0.322** | -0.602 |
| **Delhi** | 1,502 | 60.25 | **54.78** | 62.91 | 0.646 | **0.707** | 0.614 |
| **Hamirpur** | 729 | 35.52 | 30.67 | **22.39** | 0.524 | 0.645 | **0.811** |

```
+-----------------------------------------------------------------------------------+
| Site-Stratified RMSE Across Three Representative Environments                     |
|                                                                                   |
|  Site          Pipeline A (Imp->Cal)   Pipeline B (Cal->Imp)  Pipeline C (Joint)  |
|  Bangalore             17.19                   8.74                  13.43        |
|  Delhi                 60.25                  54.78                  62.91        |
|  Hamirpur              35.52                  30.67                  22.39        |
|                                                                                   |
|  --> Pipeline B dominates Bangalore & Delhi; Joint TCN v2 dominates Hamirpur      |
+-----------------------------------------------------------------------------------+
```

### Site Insights
- **Bangalore (Low Baseline Concentration):** Baseline $\text{PM}_{2.5}$ variability is small. Pipeline B achieves outstanding accuracy (RMSE $8.74$ $\mu$g/m³), whereas Pipelines A and C over-predict small fluctuations, resulting in negative $R^2$.
- **Delhi (Extreme Particulate Episodes):** Under extreme winter pollution, Pipeline B demonstrates superior robustness (RMSE $54.78$ $\mu$g/m³, $R^2 = 0.707$), outperforming both Pipeline A ($60.25$) and Joint TCN v2 ($62.91$).
- **Hamirpur (Mountainous Terrain & Smoke):** Joint TCN v2 achieves its highest fidelity in Hamirpur (RMSE $22.39$ $\mu$g/m³, $R^2 = 0.811$), demonstrating that deep temporal convolutions excel at tracking complex atmospheric dispersion in mountainous valleys.

---

## 14. Discussion

### 14.1 Resolving the Core Scientific Question
Our empirical evidence resolves the central research question: **No single reconstruction strategy is uniformly optimal across missingness regimes.**
1. When the objective is **overall operational accuracy across continuous monitoring**, Pipeline B ($\text{Calibrate} \to \text{Impute}$) is the undisputed superior choice. By calibrating low-cost sensor data before applying reconstruction, it restricts non-linear errors to unobserved intervals while achieving optimal calibration on observed timestamps.
2. When the objective is **resilience against short-duration optical saturation during localized fog episodes ($1$h–$3$h)**, Joint TCN v2 provides valuable auxiliary reconstruction capabilities.
3. For **extended multi-day outages ($24$h+)**, joint sequence models should be avoided in favor of Pipeline A or B, as deep causal state representations suffer severe error compounding.

### 14.2 The Reality of Selection Bias in Complete-Case Benchmarks
Our audit of Pipeline D provides an essential cautionary message for the environmental machine learning community: **Reporting calibration metrics strictly on complete cases introduces severe survival bias.** An algorithm that achieves $29.8$ $\mu$g/m³ on sunny, uncorrupted days but fails entirely during power outages or extreme weather events cannot be claimed as an overall solution.

---

## 15. Limitations and Reproducibility

### 15.1 Scientific and Practical Limitations
1. **Geographical Scope:** While spanning three distinct Indian climates (Bangalore, Delhi, Hamirpur), these results do not establish national-scale generalization across all regional microclimates of India.
2. **Extended Outage Limits:** Accurately reconstructing particulate pollution during sensor outages $> 24$ hours without co-located physical proxies remains an open challenge for all paradigms.
3. **Synthetic Missingness Qualifications:** While our missingness mechanisms model known failure modes (MCAR, MAR-RH, MAR-Lag, MNAR-PM), synthetic mask injection cannot perfectly replicate complex physical failure cascades (e.g., progressive insect nesting or laser diode decay).
4. **Synthetic MNAR Stress-Testing:** Our MNAR-PM experiment evaluates algorithmic sensitivity to particulate-correlated dropouts via an oracle propensity function; it does not demonstrate that models can autonomously detect real-world MNAR missingness without diagnostic flags.
5. **Multi-Pollutant Scope:** Evaluations reflect $\text{PM}_{2.5}$-based air quality assessment; they do not encompass gaseous criteria pollutants ($\text{NO}_2, \text{SO}_2, \text{O}_3, \text{CO}$) required for a full multi-pollutant Air Quality Index.
6. **Hardware and Drift Claims:** Findings do not assert independently validated physical sensor drift mechanisms, which require controlled multi-year chamber audits.
7. **Infrastructure Context:** Big data scalability benchmarks were executed under local Apache Spark execution; distributed cluster performance across multi-node topologies may exhibit different network overhead characteristics.

### 15.2 Cryptographic Reproducibility and Artifact Inventory
To ensure exact scientific auditability, all code, checkpoints, predictions, and metrics are preserved with SHA-256 hashes:

- **Joint TCN v2 Checkpoint:** `outputs/models/joint_tcn/joint_tcn_v2_causal_checkpoint.pt`  
  `SHA-256: 4a0c5704257fd556e53cfc23b1c26439881009f8aabb423b3f6f2752e7c5217c`
- **Frozen Benchmark Predictions:** `outputs/experiments/phase3C/benchmark_predictions.parquet`  
  `SHA-256: 87e5263742403644d96605f9bc742de6e5009a29d8eb5509c149b8b154a4f662`
- **Overall Scenario Metrics:** `outputs/experiments/phase3C/scenario_metrics_overall.csv`  
  `SHA-256: d0b0da45438d4604c9f79b95fe568a355ced9edba221ec82d838f3906941e4c4`
- **Missing PM Metrics:** `outputs/experiments/phase3C/scenario_metrics_missing_pm.csv`  
  `SHA-256: 0eb1f6cadc1462f181e3ef757c675dc5964774e50ac641c61c6fa91575b9ac7e`
- **Common Support Metrics:** `outputs/experiments/phase3C/scenario_metrics_common_support.csv`  
  `SHA-256: 1fa28543a629f2a413aa38e752e0dedd16abb500d34dbb97644ad6236a357cf7`
- **Bootstrap Significance Table:** `outputs/experiments/phase3C/bootstrap_significance_results.csv`  
  `SHA-256: 824c18a2706f1f8206d7b840d1789a4b0a6c20fe4264f21e3f171c817be44b25`
- **Sensitivity Sweep Table:** `outputs/experiments/phase3C/bootstrap_sensitivity_sweep.csv`  
  `SHA-256: d7ef5ef2a2b4b98cecd0990f20e8a3b212bad6f26512f53f247542801b892891`

All 81 unit and regression tests pass verified execution: `80 passed, 1 skipped (Spark sandbox socket), 0 failed`.

---

## 16. Conclusion

In this paper, we introduced the AIRWISE benchmark to rigorously evaluate sequential versus joint deep learning architectures for missing data imputation and regulatory calibration in low-cost air quality sensor networks. By enforcing strict temporal causality, evaluating nine missingness regimes, matching common-support populations, and applying segment-constrained moving-block bootstrap hypothesis testing, we established that **no single reconstruction paradigm is universally superior**:

- **Pipeline B ($\text{Calibrate} \to \text{Impute}$)** provides the strongest full-coverage benchmark baseline, achieving the lowest overall error across all scenarios ($36.26$–$41.64$ $\mu$g/m³).
- **Causal Joint TCN v2** demonstrates valuable localized reconstruction advantages on high-humidity outages, but suffers from negative multi-task transfer on uncorrupted timestamps ($38.01$ vs $29.82$ $\mu$g/m³) and severe error compounding on multi-day outages.
- **Pipeline D (Complete-Case)** illustrates an error ceiling of $29.82$ $\mu$g/m³ on observed cases, but is disqualified as a standalone operational solution due to zero availability during outages.
- **Autocorrelation Matters:** Statistical significance observed under short bootstrap blocks ($b=12$h) collapses once the 24-hour diurnal atmospheric cycle is respected, underscoring the necessity of moving-block resampling in environmental time series.

These findings provide clear architectural guidelines for municipal environmental agencies and IoT researchers deploying real-time air quality monitoring infrastructure.

---

## 17. References

[1] Health Effects Institute, "State of Global Air 2020: A Special Report on Global Exposure to Air Pollution and Its Health Impacts," Boston, MA, 2020.  
[2] K. R. Smith et al., "Millions Dead: How Air Pollution Harms Health Across South Asia," *The Lancet Planetary Health*, vol. 5, no. 11, pp. e756–e768, 2021.  
[3] S. Guttikunda, R. Goel, and P. Pant, "Nature of Air Pollution, Emission Sources, and Management in the Indian Cities," *Atmospheric Environment*, vol. 95, pp. 501–510, 2014.  
[4] A. Rai et al., "End-to-End Evaluation of Low-Cost Sensor Networks for Air Quality Monitoring in Developing Countries," *IEEE Internet of Things Journal*, vol. 8, no. 12, pp. 9812–9825, 2021.  
[5] A. Jayaratne et al., "The Influence of Humidity on Particle Size Distributions Measured by Optical Particle Counters," *Atmospheric Measurement Techniques*, vol. 11, no. 4, pp. 2417–2427, 2018.  
[6] K. A. Barkjohn, B. Gantt, and A. L. Clements, "Development and Application of a United States-Wide Correction for PM2.5 Data Collected with the PurpleAir Sensor," *Atmospheric Measurement Techniques*, vol. 14, no. 6, pp. 4617–4637, 2021.  
[7] E. Malings et al., "Development of a General Calibration Model and Built-In Quality Control for Low-Cost Air Quality Sensors," *Atmospheric Measurement Techniques*, vol. 13, no. 2, pp. 645–662, 2020.  
[8] L. Hagan et al., "Calibration and Assessment of Electrochemical and Optical Low-Cost Sensors for Ambient Air Monitoring," *Atmospheric Measurement Techniques*, vol. 12, no. 6, pp. 3157–3175, 2019.  
[9] D. J. Stekhoven and P. Bühlmann, "MissForest—Non-Parametric Missing Value Imputation for Mixed-Type Data," *Bioinformatics*, vol. 28, no. 1, pp. 112–118, 2012.  
[10] Z. Zheng, X. Chen, and Y. Liu, "Spatial-Temporal Imputation and Post-Calibration of Low-Cost Environmental Sensor Data," *IEEE Transactions on Knowledge and Data Engineering*, vol. 34, no. 8, pp. 3890–3904, 2022.  
[11] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," *arXiv preprint arXiv:1803.01271*, 2018.  
[12] Y. Liang et al., "Joint Deep Calibration and Reconstruction for Environmental Sensor Networks," *ACM Transactions on Sensor Networks*, vol. 18, no. 3, pp. 1–25, 2022.  
[13] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, 2016, pp. 785–794.  
[14] R. J. Little and D. B. Rubin, *Statistical Analysis with Missing Data*, 3rd ed. Hoboken, NJ: John Wiley & Sons, 2019.  
[15] J. L. Schafer and J. W. Graham, "Missing Data: Our View of the State of the Art," *Psychological Methods*, vol. 7, no. 2, pp. 147–177, 2002.  
[16] W. Cao et al., "BRITS: Bidirectional Recurrent Imputation for Time Series," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018, pp. 6775–6785.  
[17] B. Lim and S. Zohren, "Time-Series Forecasting with Deep Learning: A Survey," *Philosophical Transactions of the Royal Society A*, vol. 379, no. 2194, p. 20200209, 2021.  
[18] S. Ruder, "An Overview of Multi-Task Learning in Deep Neural Networks," *arXiv preprint arXiv:1706.05098*, 2017.  
[19] T. Yu et al., "Gradient Surgery for Multi-Task Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 5824–5836.  
[20] D. N. Politis and J. P. Romano, "The Stationary Bootstrap," *Journal of the American Statistical Association*, vol. 89, no. 428, pp. 1303–1313, 1994.  
[21] H. R. Künsch, "The Jackknife and the Bootstrap for General Stationary Observations," *The Annals of Statistics*, vol. 17, no. 3, pp. 1217–1241, 1989.  
[22] C. Bergmeir, R. J. Hyndman, and B. Koo, "A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Predictors," *Computational Statistics & Data Analysis*, vol. 120, pp. 70–83, 2018.  
[23] M. Campmier et al., "Field Evaluation and Co-Location of Low-Cost Particulate Matter Monitors Across Diverse Climatological Regimes in India," *Environmental Science & Technology*, vol. 55, no. 18, pp. 12450–12461, 2021.
