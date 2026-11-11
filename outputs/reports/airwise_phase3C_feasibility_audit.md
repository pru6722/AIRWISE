# AIRWISE — Phase 3C.0: Research Strengthening Feasibility & Component Reuse Audit (Refined v5)

**Document ID:** `AIRWISE-P3C-FEASIBILITY-AUDIT-005`  
**Date:** November 9, 2026  
**Status:** Audit & Architecture Design Complete — Ready for Review (Pre-Execution Gate)  
**Execution Restriction:** AUDIT AND DESIGN ONLY. No experimental runs, model retraining, checkpoint modifications, or prediction file overwrites have been executed.

---

## 1. Executive Summary & Purpose

Phase 3A and Phase 3B established absolute empirical evidence reconciliation and dashboard transparency for AIRWISE. With all historical anomalies forensically reconciled, receptive field formulas mathematically corrected ($RF = 61\text{h}$), and scientific boundaries openly disclosed, the research platform is now positioned for **Phase 3C: Research Strengthening**.

The purpose of Phase 3C is to elevate the scientific rigor, methodological depth, and empirical contribution of AIRWISE to the standard required for top-tier publication (e.g., *ACM Transactions on Sensor Networks (TOSN)*, *IEEE Transactions on Big Data*, or *Atmospheric Measurement Techniques*).

This refined feasibility audit inspects the current codebase, dataset attributes, computational environment, and architectural components to:
1. Formally delineate the **four distinct roles of BAM reference data** across project partitions.
2. Define input missingness as a **feature-level mask ($M_X \in \{0, 1\}^{N \times C}$)** and derive the **timestep-level PM indicator ($m_t^{\text{PM}} \in \{0, 1\}$)**.
3. Formulate the exact training and supervision mechanics for the four outage scenarios across the calibration and reconstruction heads.
4. Verify that the reconstruction head cannot receive unmasked calibration estimates at missing timesteps while excluding BAM targets from network inputs.
5. Strengthen the full-model causality verification suite to evaluate multiple cut positions, multiple perturbation magnitudes, both output heads, and both train/eval modes with an explicit Float32 numerical tolerance bound ($\epsilon = 1.0 \times 10^{-6}$).
6. Ground the sequence length decision ($L = 64\text{h}$) strictly on theoretical receptive field realization ($RF = 61\text{h}$) and empirical sample density (1,906 continuous windows), treating hardware alignment as an unverified hypothesis.
7. Correctly classify missingness mechanisms under Rubin's formal taxonomy (**MCAR**, **MAR**, **MNAR**).
8. Establish strictly isolated Leave-One-Site-Out (LOSO) spatial generalization protocols.
9. Formalize segment-constrained, site-aware moving-block bootstrap and exact McNemar testing with $\Delta\text{RMSE} = \text{RMSE}_{\text{Joint}} - \text{RMSE}_{\text{Baseline}}$.
10. Formulate external literature baselines (US EPA Barkjohn 2021 correction) and prequential adaptive conformal inference (ACI) without unverified transfer claims.
11. Categorize every project component into **Already Implemented**, **Can Be Reused**, **Requires Modification**, **Requires New Implementation**, and **Not Feasible**.

---

## 2. Objective-by-Objective Feasibility & Architecture Audit

### 2.1 Objective 1: Joint versus Sequential Imputation/Calibration Pipeline Comparison

#### Four Explicit BAM Reference Data Roles
To resolve any ambiguity regarding target "withholding":
1. **Training BAM ($Y_{\text{train}}$ on `train_df`, $N=11,149$):** Used for model parameter fitting via gradient descent (Huber loss minimization). It provides supervision for calibration and reconstruction heads. **It is never passed as an input feature to the network.**
2. **Validation BAM ($Y_{\text{val}}$ on `val_model_df`, $N=1,857$):** Used strictly for model selection, hyperparameter tuning, and early stopping. It evaluates generalization error on out-of-sample sequences. **It is never accessible to the model as an input feature and never used for backpropagation parameter updates.**
3. **Conformal Calibration BAM ($Y_{\text{calib}}$ on `calib_conformal_df`, $N=1,858$):** Used *strictly post-prediction* by the conformal calibration engine to calculate nonconformity scores $s_i = |y_i - \hat{y}_i|$. **It is never used for parameter fitting, architecture selection, or early stopping, and is never fed as an input feature.**
4. **Test BAM ($Y_{\text{test}}$ on `test_df`, $N=3,720$):** Completely inaccessible during all phases of model development, training, validation, tuning, and calibration. Used exclusively by the independent evaluation harness to compute unbiased benchmark metrics (RMSE, MAE, $R^2$, Winkler score, CPCB accuracy).

> [!IMPORTANT]
> **Ground Truth Withholding Principle:** Withholding a target from model inputs means strictly that the true ground-truth value is excluded from the model's feature set ($X_t$). It does *not* mean the evaluator cannot use its ground truth for the designated evaluation purpose (computing validation loss, conformal quantiles, or test metrics).

#### Feature-Level Mask ($M_X$) and Timestep-Level Indicator ($m_t^{\text{PM}}$)
Input missingness is defined at the feature level across $C = 10$ input channels (or $C = 11$ when the mask indicator is appended):
* **PM2.5 Sensor Channels ($\mathcal{C}_{\text{PM}}$):** `pa_cf1`, `pa_atm`, `pa_alt`.
* **Meteorological Channels ($\mathcal{C}_{\text{meteo}}$):** `relative_humidity`, `temperature`.
* **Cyclical Calendar Harmonics ($\mathcal{C}_{\text{time}}$):** `hour_sin`, `hour_cos`, `day_of_year_sin`, `day_of_year_cos`, `time_diff_hours`.

The **timestep-level PM availability indicator** is defined as the product across all three PM channels:
$$m_t^{\text{PM}} = \prod_{c \in \mathcal{C}_{\text{PM}}} M_{X, t, c} = \mathbb{I}\left(M_{X, t, \text{cf1}} = 1 \land M_{X, t, \text{atm}} = 1 \land M_{X, t, \text{alt}} = 1\right)$$

The **complete station availability indicator** is defined as:
$$m_t^{\text{station}} = \prod_{c \in \mathcal{C}_{\text{PM}} \cup \mathcal{C}_{\text{meteo}}} M_{X, t, c}$$

#### Explicit Treatment across Outage Scenarios
1. **Scenario 1: All three PM channels observed ($m_t^{\text{PM}} = 1$, meteorology observed):**
   - *Calibration Head:* Supervised. Huber loss $\mathcal{L}_{\text{cal}}$ computed against $Y_{\text{BAM}, t}$.
   - *Reconstruction Head:* Inactive/masked out. Huber loss $\mathcal{L}_{\text{rec}}$ has weight 0.
   - *Composite Output:* $\hat{Y}_{\text{final}, t} = \hat{Y}_{\text{cal}, t}$.
2. **Scenario 2: One or two PM channels missing ($m_t^{\text{PM}} = 0$, partial PM corruption):**
   - *Treatment:* Standard calibration models require the full uncorrupted sensor triplet (`pa_cf1`, `pa_atm`, `pa_alt`) to predict calibrated PM2.5. Partial PM channel absence invalidates direct unassisted calibration.
   - *Pipeline A (Impute $\to$ Calibrate):* Multi-channel imputer $f_{\text{imp}}$ imputes the 1 or 2 missing PM channels first from the observed PM channel(s) and meteorology. Once completed, the calibration model predicts $\hat{Y}_{A, t}$.
   - *Pipeline C (Joint TCN v2):* $m_t^{\text{PM}} = 0$. The missing PM channels are zeroed in the input tensor ($X_t \odot M_{X, t}$), the mask indicator channel reflects the dropout, and this timestep is assigned to the **Reconstruction Head** (supervising $\mathcal{L}_{\text{rec}}$ against $Y_{\text{BAM}, t}$). The calibration head output is masked out to prevent corrupted calibration estimates.
3. **Scenario 3: All three PM channels missing ($m_t^{\text{PM}} = 0$, meteorology available):**
   - Low-cost PM channels zeroed: $X_{\text{PM}, t} = 0$. Meteorology (`relative_humidity`, `temperature`) and cyclical harmonics remain available.
   - *Calibration Head:* Inactive/masked out ($\mathcal{L}_{\text{cal}}$ weight = 0).
   - *Reconstruction Head:* Supervised. Huber loss $\mathcal{L}_{\text{rec}}$ computed against $Y_{\text{BAM}, t}$ using temporal context and available meteorology.
   - *Composite Output:* $\hat{Y}_{\text{final}, t} = \hat{Y}_{\text{rec}, t}$.
4. **Scenario 4: Complete station outage ($m_t^{\text{station}} = 0$, total telemetry blackout):**
   - All physical features zeroed: $X_{\text{PM}, t} = 0, X_{\text{meteo}, t} = 0$. Only deterministic calendar harmonics (`hour_sin`, `hour_cos`, `day_of_year_sin`, `day_of_year_cos`, `time_diff_hours`) remain available.
   - *Calibration Head:* Inactive/masked out ($\mathcal{L}_{\text{cal}}$ weight = 0).
   - *Reconstruction Head:* Supervised. Huber loss $\mathcal{L}_{\text{rec}}$ computed against $Y_{\text{BAM}, t}$ using purely historical temporal representations and cyclical harmonics.
   - *Composite Output:* $\hat{Y}_{\text{final}, t} = \hat{Y}_{\text{rec}, t}$.

#### Prevention of Unmasked Calibration Leakage
In `ReconstructionHead.forward(h, y_cal, mask)`:
$$\hat{Y}_{\text{cal\_observed}, t} = \hat{Y}_{\text{cal}, t} \times m_t^{\text{PM}}$$
When $m_t^{\text{PM}} = 0$, $\hat{Y}_{\text{cal\_observed}, t} = 0$ identically. This mathematically guarantees that the reconstruction head **never receives an unmasked calibration estimate at a missing timestep**, preventing corrupt/zeroed calibration artifacts from leaking into reconstruction.

* **Historical 20% MCAR Preservation:** The historical Checkpoint 3/4 benchmark (which zeroed `pa_cf1`, `pa_atm`, `pa_alt` while evaluating BAM error on those timestamps) is preserved as an explicit reference scenario: **"Historical Joint Masking (Simulated Sensor Dropout with Concurrent BAM Evaluation)"**.

---

### 2.2 Detailed Pipeline Architectures & Information Accounting

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PIPELINE ARCHITECTURE COMPARISON                               │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Pipeline A (Impute ──> Calibrate):                                                               │
│   X_t [Masked Channels, m_t^PM=0] ──> Multi-Channel Imputer ──> X_completed,t ──> Calib Head ──> BAM│
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Pipeline B (Calibrate ──> Impute):                                                               │
│   X_t [Observed, m_t^PM=1] ──> Calib Head ──> Y_cal,t [Masked] ──> Imputation Baseline ──> BAM     │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Pipeline C (Joint Multi-Task TCN v2):                                                            │
│   [X_t ⊙ M_X, M_X, Meteo] ──> Shared Causal Temporal Encoder ──> [Cal Head, Rec Head] ──> BAM   │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Pipeline D (Complete-Case Calibration Reference):                                                │
│   X_t [m_t^PM = 1 Only] ──> Calibration Head ──> Evaluated on observed points; missing dropped  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Pipeline Specification | Pipeline A (Impute $\to$ Calibrate) | Pipeline B (Calibrate $\to$ Impute) | Pipeline C (Joint TCN v2) | Pipeline D (Complete-Case Reference) |
|---|---|---|---|---|
| **Input Channels Available (Partial Outage)** | Meteorology (RH, Temp) + calendar harmonics when $m_{t}^{\text{PM}}=0$; all channels when $m_{t}^{\text{PM}}=1$. | All channels when $m_{t}^{\text{PM}}=1$; none for direct calibration when $m_{t}^{\text{PM}}=0$. | Full feature tensor $[X_t \odot M_{X,t}, M_{X,t}, \text{Meteo}_t]$. | Full channels strictly when $m_{t}^{\text{PM}}=1$. |
| **Input Channels Available (Complete Blackout)** | Only deterministic calendar harmonics when $m_t^{\text{station}}=0$. | None for direct calibration when $m_t^{\text{station}}=0$. | Feature tensor $[0, M_{X,t}, \text{Harmonics}_t]$. | Full channels strictly when $m_t^{\text{station}}=1$. |
| **Artificially Missing Channels** | Channels where $M_{X,t,c} = 0$. | Channels where $M_{X,t,c} = 0$. | Channels where $M_{X,t,c} = 0$ (zeroed out). | All channels where $M_{X,t,c} = 0$. |
| **Reconstruction Mechanism** | Multi-channel imputer $f_{\text{imp}}$ (fitted strictly on `train_df`) reconstructs raw channels $\tilde{X}_t$. | Calibrates observed points first ($\hat{Y}_{\text{cal},t}$); then time-series imputer $h_{\text{recon}}$ reconstructs missing predictions. | Dedicated **Reconstruction Head** conditioned on shared temporal representation $H_t$, masked calibration estimate, and $M_{X,t}$. | **None.** Missing inputs are **not silently imputed**. Missing periods produce no prediction. |
| **BAM Reference Access** | Used strictly on `train_df` to fit calibration regressor $g_{\text{cal}}$. Never accessible at test inference. | Used strictly on `train_df` to fit $g_{\text{cal}}$. Never used to fit $h_{\text{recon}}$ on test data. | Accessible on `train_df` during training to compute multi-task Huber loss. Never accessible at test inference. | Used strictly on `train_df` complete cases to fit $g_{\text{cal}}$. Never accessible at test inference. |
| **Evaluation Timestamps** | Evaluated on all test BAM timestamps ($N=3,720$). Separate metrics on $m_t^{\text{PM}}=0$ and $m_t^{\text{PM}}=1$. | Evaluated on all test BAM timestamps ($N=3,720$). Separate metrics on $m_t^{\text{PM}}=0$ and $m_t^{\text{PM}}=1$. | Evaluated on all test BAM timestamps ($N=3,720$). Separate metrics on $m_t^{\text{PM}}=0$ (reconstruction) and $m_t^{\text{PM}}=1$ (calibration). | Evaluated **strictly on its eligible observed subset** ($m_t^{\text{PM}}=1$, $N \approx 2,976$). |
| **Prediction Availability / Coverage** | **100% Coverage** (predicts at all $N=3,720$ test timestamps). | **100% Coverage** (predicts at all $N=3,720$ test timestamps). | **100% Coverage** (predicts at all $N=3,720$ test timestamps). | **Partial Coverage:** $(1 - \text{missing\_rate}) \times 100\%$ (e.g. 80.0% coverage under 20% missingness). **0% on $m_t^{\text{PM}}=0$**. |

#### Fair Comparison Protocol & Selection Bias Accounting
* **Pipeline D Classification:** Pipeline D is designated as a **Complete-Case Calibration Reference**, *not* an "upper bound".
* **Selection Bias Disclosure:** Complete-case observations ($m_t^{\text{PM}} = 1$) may exhibit a different underlying distribution from missing periods ($m_t^{\text{PM}} = 0$). If dropouts occur during extreme pollution or high humidity, evaluating solely on complete cases introduces selection bias. Pipeline D is formally presented as a **Complete-Case Calibration Reference**, not an "upper bound".
* **Unequal Population Separation:** Pipeline D is evaluated **only on its eligible subset ($m_t^{\text{PM}} = 1$)**. It is NEVER directly ranked against Pipelines A, B, C on missing timestamps ($m_t^{\text{PM}} = 0$) where it produces no predictions. When computing 24-hour CPCB daily aggregations, days with missing hours are dropped if $< 18/24\text{h}$ complete, quantifying the regulatory non-compliance penalty.

---

### 2.3 Objective 2: Causality-Safe Joint TCN v2 Retraining

#### Sequence Length Decision & Justification (`seq_len = 64\text{h}` vs `72\text{h}`)
1. **Receptive Field Formula:** $RF = 1 + 2(K - 1)\sum_{i=0}^3 d_i = 1 + 2(3 - 1)(1 + 2 + 4 + 8) = \mathbf{61\text{ hours}}$.
2. **Primary Choice (`seq_len = 64\text{h}`):**
   - *Justification 1 (Full RF Realization):* Setting $L = 64 > 61$ strictly satisfies $L \ge RF$, ensuring all 4 residual blocks operate without truncation.
   - *Justification 2 (Sample Density):* Audit of `cleaned_data.parquet` reveals `train_df` contains **1,906 contiguous 64-hour windows** (compared to 1,714 for 72h), providing ~11% more valid training sequences and lower fragmentation near natural segment boundaries.
   - *Hardware Alignment Hypothesis:* While power-of-two tensor dimensions are often hypothesized to offer memory alignment advantages on modern accelerator hardware, this remains an **unverified hypothesis** in our workstation environment. The definitive justification for $L = 64$ is that $64 > 61$, strictly exposing the full theoretical receptive field while maximizing available training sequences.
3. **Sensitivity Sweep (`seq_len = 72\text{h}`):** Evaluated as a secondary sensitivity comparison to determine whether capturing exactly three 24-hour diurnal cycles ($3 \times 24\text{h}$) yields measurable predictive advantage over 64h.

---

### 2.4 Specification of TCN v2 Reconstruction Training & Multi-Task Supervision

During Joint TCN v2 training, the dual prediction heads are trained simultaneously under artificial missingness:
1. **Dynamic Training Mask Generation:**
   - In `TCNWindowDataset`, each training window of length $L = 64$ receives an artificial binary mask $M_{X, t} \in \{0, 1\}^L$ drawn with missing rates uniformly distributed between 5% and 30%.
   - At masked steps ($m_t^{\text{PM}} = 0$), the low-cost sensor channels (`pa_cf1`, `pa_atm`, `pa_alt`) are zeroed: $X_{\text{sensor}, t} \odot M_{X, t}$.
   - Exogenous meteorology and calendar harmonics remain unmasked unless simulating total station outage.
   - The binary mask indicator vector $M_{X, t}$ is concatenated as an explicit feature channel (channel dimension $D = 11$).
2. **Calibration Head Supervision:**
   $$\mathcal{L}_{\text{cal}} = \frac{1}{\sum_t m_t^{\text{PM}}} \sum_{t: m_t^{\text{PM}} = 1} \ell_{\text{Huber}}(\hat{Y}_{\text{cal}, t}, Y_{\text{BAM}, t})$$
   Evaluated **exclusively on observed positions**, supervising the calibration head on valid sensor readings.
3. **Reconstruction Head Supervision:**
   $$\mathcal{L}_{\text{rec}} = \frac{1}{\sum_t (1 - m_t^{\text{PM}})} \sum_{t: m_t^{\text{PM}} = 0} \ell_{\text{Huber}}(\hat{Y}_{\text{rec}, t}, Y_{\text{BAM}, t})$$
   Conditioned on shared representation $H_t$, masked calibration estimate $\hat{Y}_{\text{cal}, t} \times m_t^{\text{PM}}$, and $M_{X, t}$. Evaluated **exclusively on artificially hidden positions**, supervising the reconstruction head to predict reference BAM PM2.5 using exclusively temporal context and available covariates.
4. **Multi-Task Loss Combination:**
   $$\mathcal{L}_{\text{joint}} = \lambda_{\text{cal}} \mathcal{L}_{\text{cal}} + \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} \quad (\lambda_{\text{cal}} = 1.0, \lambda_{\text{rec}} = 1.0)$$
5. **Target Exclusion from Inputs:**
   - $Y_{\text{BAM}}$ is never passed into the network. It enters solely through the Huber loss function at the output heads during backpropagation.

---

### 2.5 Strengthened Causality Verification Acceptance Criteria

To rigorously verify that Joint TCN v2 is free of temporal sequence leakage, the model must pass a comprehensive causality verification suite across multiple perturbation magnitudes, multiple cut points, both output heads, and both execution modes:
* **Test A (Value Perturbation with Fixed Observed Mask):** Evaluates $\delta \in \{0.1, 10.0, 100.0, 500.0, -50.0\}\,\mu\text{g/m}^3$ across cut positions $t_0 \in \{16, 32, 48, 60\}$. Verifies that $\max_{b, t < t_0} |\hat{Y}_{b, t} - \hat{Y}_{\text{pert}, b, t}| \le \epsilon_{\text{causal}} = 1.0 \times 10^{-6}$ for both output heads and composite prediction.
* **Test B (Mask Perturbation with Fixed Input Values):** Flips future mask states ($M_{X, t} = 0$) across cut positions $t_0 \in \{16, 32, 48, 60\}$. Verifies that $\max_{b, t < t_0} |\hat{Y}_{b, t} - \hat{Y}_{\text{pert}, b, t}| \le \epsilon_{\text{causal}} = 1.0 \times 10^{-6}$ for both output heads and composite prediction.
* **Execution Conditions:** Evaluated in both training mode (`model.train()`) and evaluation mode (`model.eval()`), with stochastic dropout disabled (`p=0.0`).
* **Full-Model Normalization Verification:** Confirms that `ChannelLayerNorm1d` normalizes independently across channels $C$ at each timestep $t$ without pooling across the sequence dimension $L$, and confirms no global temporal pooling exists in the architecture.
* **Scope Qualification:** This test verifies **architectural temporal causality only** (verifying that the mathematical structure of the neural network prevents information from time step $t+k$ from influencing output at time step $t$). It does *not* replace separate audits of chronological dataset splitting, feature lag engineering, or out-of-fold preprocessing leakage, which must continue to be independently audited and enforced.

---

### 2.6 Objective 3: Missingness Mechanism Taxonomy & Robustness Testing

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                               MISSINGNESS STRESS TEST TAXONOMY                           │
├──────────────┬──────────────┬──────────────┬─────────────────────────────────────────────┤
│ Scenario     │ Rubin Type   │ Target Rate  │ Conditioning Variable / Equation            │
├──────────────┼──────────────┼──────────────┼─────────────────────────────────────────────┤
│ 1. MCAR      │ MCAR         │ 20%          │ Bernoulli dropout: u_t ~ Uniform(0,1) >= 0.20│
│ 2. Block 1h  │ MCAR (Block) │ 20%          │ 1-hour isolated outages placed in segments  │
│ 3. Block 3h  │ MCAR (Block) │ 20%          │ 3-hour contiguous outages in segments       │
│ 4. Block 6h  │ MCAR (Block) │ 20%          │ 6-hour contiguous outages in segments       │
│ 5. Block 12h │ MCAR (Block) │ 20%          │ 12-hour contiguous outages in segments      │
│ 6. Block 24h │ MCAR (Block) │ 20%          │ 24-hour full diurnal outages in segments    │
│ 7. MAR-RH    │ MAR          │ 20%          │ P(M_X=0) = Sigmoid(6.0 * (RH_norm - 0.55))  │
│ 8. MAR-Lag   │ MAR          │ 20%          │ P(M_X=0) conditioned on observed PM_{t-1}   │
│ 9. MNAR-PM   │ MNAR         │ 20%          │ P(M_X=0) = Sigmoid(5.0 * (PM_curr - 0.50))  │
└──────────────┴──────────────┴──────────────┴─────────────────────────────────────────────┘
```

* **MAR (Scenarios 7 & 8):** Dropout probability depends strictly on *observed contemporaneous covariates* (relative humidity) or *past observed particulate matter* ($X_{t-1}$).
* **MNAR (Scenario 9):** Dropout probability depends directly on the *unobserved current particulate concentration ($Y_t$)* itself. Strictly designated as **MNAR**.
* **5 Fixed Random Seeds:** $\mathcal{S} = \{42, 101, 2024, 777, 9999\}$; Mean $\pm$ Standard Error reported.

---

### 2.7 Objective 4: Geographic Generalization (Leave-One-Site-Out / LOSO)

Three non-overlapping spatial transfer experiments:
* **LOSO-G1:** Train Delhi ($N=7,504$) + Hamirpur ($N=3,639$) = $11,143$ $\to$ Test Bangalore ($N=7,441$).
* **LOSO-G2:** Train Delhi ($N=7,504$) + Bangalore ($N=7,441$) = $14,945$ $\to$ Test Hamirpur ($N=3,639$).
* **LOSO-G3:** Train Bangalore ($N=7,441$) + Hamirpur ($N=3,639$) = $11,080$ $\to$ Test Delhi ($N=7,504$).

#### Strict Preprocessing Isolation Rules
1. Preprocessing standardizers (`StandardScaler`, MinMax) and imputers are fitted **strictly on the two training sites**.
2. 15% of the chronological tail of the training sites is reserved for validation/early stopping.
3. The selected model is frozen prior to evaluating on the held-out test site.
4. Reference BAM measurements on the held-out site are **never accessed** during model fitting, threshold tuning, or hyperparameter selection.
5. Pooled and site-disaggregated results are reported.

---

### 2.8 Objective 5: Statistical Validation & Hypothesis Testing

1. **Moving-Block Bootstrap (MBB):**
   - Block length justified by empirical autocorrelation: $b = 24\text{ hours}$ (matching the diurnal cycle).
   - Sensitivity sweeps: $b \in \{12\text{h}, 24\text{h}, 48\text{h}, 72\text{h}\}$.
   - Constraint: Resampled blocks are drawn strictly **within their original site and contiguous time segment**.
   - Resamples: $B = 1,000$ iterations computing 95% bootstrap percentile confidence intervals.
2. **Paired Model Comparisons:**
   - Metric difference definition:
     $$\Delta \text{RMSE}^{(k)} = \text{RMSE}_{\text{Joint}}^{(k)} - \text{RMSE}_{\text{Baseline}}^{(k)}$$
     *(A negative $\Delta\text{RMSE}$ directly denotes that Joint TCN achieves lower error / superior performance).*
   - Empirical one-tailed $p$-value: $p = \frac{1}{B}\sum_{k=1}^B \mathbb{I}(\Delta \text{RMSE}^{(k)} \ge 0)$.
3. **Exact McNemar Test:**
   - Evaluates daily 24-hour CPCB regulatory category discordant pairs with two-tailed exact binomial $p$-values, supplemented by site-stratified tables and cluster-robust bootstrap intervals.

---

### 2.9 Objective 6: Literature & Advanced Baselines

1. **US EPA Barkjohn (2021) Correction Formulation:**
   $$\text{PM}_{2.5} = 0.524 \times \text{PA\_CF1} - 0.0862 \times \text{RH} + 5.75$$
   - Documented US development population (~50 sites across continental US) and 24-hour aggregation assumptions ($\ge 75\%$ completeness).
   - Primary comparison on identical Indian held-out test observations. Performance divergence is documented without unverified chemical claims, citing relevant literature (e.g., Campmier et al. 2023).
2. **Derived Thermodynamic Dew Point:**
   - Original `dew_point_raw` preserved intact.
   - Separate `derived_dewpoint` calculated via Magnus-Tetens formula with input validation ($T \in [-40, 60]^\circ\text{C}$, $\text{RH} \in (0, 100]\%$, $T_d \le T$).
3. **Synthetic Sensor Drift Scenarios:**
   - Explicit physical units: Additive baseline drift ($+10\,\mu\text{g/m}^3$), Sudden offset ($+15\,\mu\text{g/m}^3$), Multiplicative decay ($-20\%$), and Noise-increase ($\sigma_t$ up to $10\,\mu\text{g/m}^3$).
   - All labeled as synthetic; detected via CUSUM change-point tracking.
4. **Advanced Uncertainty (ACI & CQR):**
   - Conducted as separate experiments.
   - Prequential ACI update: Reference BAM measurement updates conformal state only after becoming chronologically available ($\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$).
   - Framed as empirical investigations; does not promise elimination of Delhi undercoverage.
   - CPCB prediction sets report set coverage, average set size, singleton frequency, and Severe-category inclusion.

---

### 2.10 Objective 7: Big Data & PySpark Scalability

1. **Spark SQL Windowing:** Re-implements temporal rolling means (1h, 6h, 24h), rolling standard deviations, and lag differences natively in PySpark Window operations.
2. **Workload Benchmarks:** Systematically measures ingestion, window calculation, and aggregation runtimes across 1x ($18.5\text{k}$), 10x ($185.8\text{k}$), and 50x ($929.2\text{k}$) scales across CSV, Parquet Snappy, and ORC ZSTD.
3. **Spark MLlib Baseline:** Fits `pyspark.ml.regression.LinearRegression` and `RandomForestRegressor` directly on the Spark DataFrame.
4. **Mandatory Disclosure:** Local workstation execution (`local[*]`) explicitly documented throughout.

---

## 3. Component Reusability & Implementation Matrix

```
┌──────────────────────────────────┬─────────────────────────────┬─────────────────────────────────┬─────────────────────┐
│ Component / Capability           │ Existing Source File        │ Classification                  │ Modification Scope  │
├──────────────────────────────────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────┤
│ Cleaned Gold Dataset (18,584)    │ data/processed/cleaned_data │ ALREADY IMPLEMENTED (Read-only) │ None                │
│ 4-Way Chronological Split Manager│ src/uncertainty/split_manage│ CAN BE REUSED                   │ Canonical partitions│
│ ChannelLayerNorm1d Implementation│ src/models/tcn_blocks.py    │ ALREADY IMPLEMENTED & VERIFIED  │ Tested in P3A.2     │
│ TemporalEncoder (RF=61h)         │ src/models/tcn_blocks.py    │ CAN BE REUSED                   │ Receptive field 61h │
│ JointTCN Multi-Task Architecture │ src/models/joint_tcn.py     │ CAN BE REUSED                   │ Pass norm="layer"   │
│ JointMultiTaskLoss               │ src/models/joint_tcn.py     │ CAN BE REUSED                   │ Multi-task Huber    │
│ TCNWindowDataset (Sliding Window)│ src/training/train_joint_tcn│ REQUIRES MODIFICATION           │ Expose seq_len=64h  │
│ TCN Early-Stopping Trainer       │ src/training/train_joint_tcn│ REQUIRES MODIFICATION           │ Save v2 checkpoint  │
│ MCAR Mask Generator              │ src/missingness/random_mask │ CAN BE REUSED                   │ Multi-seed support  │
│ Block Outage Mask Generator      │ src/missingness/block_mask  │ CAN BE REUSED                   │ Multi-seed support  │
│ MAR Mask Generator (Obs Covar)   │ src/missingness/regime_mask │ REQUIRES MODIFICATION           │ Rename from MAR/MNAR│
│ MNAR Mask Generator (Target-Dep) │ src/missingness/regime_mask │ REQUIRES MODIFICATION           │ Explicit MNAR label │
│ Corrected Recon Evaluator (P-B)  │ src/models/reconstruction.py│ CAN BE REUSED                   │ Pipeline B baseline │
│ Multi-Channel Imputer (P-A)      │ N/A                         │ REQUIRES NEW IMPLEMENTATION     │ Pipeline A wrapper  │
│ Complete-Case Reference (P-D)    │ src/models/calibration.py   │ CAN BE REUSED                   │ Reference baseline  │
│ Leave-One-Site-Out (LOSO) Runner │ N/A                         │ REQUIRES NEW IMPLEMENTATION     │ LOSO script         │
│ Moving-Block Bootstrap Engine    │ N/A                         │ REQUIRES NEW IMPLEMENTATION     │ Stats module        │
│ Paired Bootstrap & McNemar Tests │ N/A                         │ REQUIRES NEW IMPLEMENTATION     │ Stats module        │
│ US EPA Correction Baseline       │ N/A                         │ REQUIRES NEW IMPLEMENTATION     │ Baseline script     │
│ Derived Dew Point (Magnus)       │ N/A                         │ REQUIRES NEW IMPLEMENTATION     │ Feature utility     │
│ Adaptive Conformal (ACI)         │ N/A                         │ REQUIRES NEW IMPLEMENTATION     │ Prequential ACI     │
│ Conformalized Quantile Reg (CQR) │ N/A                         │ REQUIRES NEW IMPLEMENTATION     │ CQR script          │
│ Synthetic Drift & CUSUM Detector │ N/A                         │ REQUIRES NEW IMPLEMENTATION     │ Drift module        │
│ Spark Window Feature Engineering │ src/bigdata/preprocessing.py│ REQUIRES MODIFICATION           │ Window functions    │
│ Spark MLlib Baseline Regressor   │ N/A                         │ REQUIRES NEW IMPLEMENTATION     │ MLlib script        │
│ Streamlit Research Dashboard     │ dashboard/                  │ ALREADY ALIGNED (PHASE 3B)      │ Preserve intact     │
│ Historical TCN Checkpoint        │ outputs/models/joint_tcn/   │ FROZEN & PRESERVED              │ NEVER OVERWRITE     │
└──────────────────────────────────┴─────────────────────────────┴─────────────────────────────────┴─────────────────────┘
```

---

## 4. Test Suite Status & Final Stop Condition

* **Test Suite Verification:** 65 tests total — **64 passed, 1 skipped, 0 failed, 0 errors**. (1 test skipped due to local sandboxed loopback socket constraints in `test_bigdata.py`; not every test executed, but all runnable tests passed with zero failures).
* **Strict Boundary:** This document is protocol specification only. No model weights, checkpoints, predictions, or conformal quantiles have been altered.

**Phase 3C.0 Protocol Clarification is complete.** Stopping per instruction to await explicit authorization before implementing Phase 3C.1.
