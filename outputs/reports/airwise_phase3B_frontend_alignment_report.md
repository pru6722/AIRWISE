# AIRWISE — Phase 3B Completion Report: Research Dashboard Frontend Alignment

**Document ID:** `AIRWISE-PHASE3B-FRONTEND-ALIGNMENT-001`  
**Date:** November 9, 2026  
**Status:** Complete & Fully Verified  
**Scope:** Exhaustive review, code alignment, empirical verification, and automated browser testing of the AIRWISE Streamlit Research Dashboard across all seven pages against the authoritative Phase 3A evidence reconciliation (`AIRWISE-EVIDENCE-RECONCILIATION-001` and `AIRWISE-COMPLIANCE-MATRIX-001`).

---

## 1. Executive Summary & Verification Scorecard

In Phase 3B, the AIRWISE frontend was brought into strict, scientifically rigorous alignment with the approved Phase 3A evidence reconciliation. No model weights, checkpoints, predictions, or conformal quantiles were altered. Every displayed metric now possesses explicit provenance, historical versus corrected reconstruction benchmarks are unambiguously distinguished, and all research limitations are prominently displayed.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       PHASE 3B DASHBOARD ALIGNMENT SCORECARD                                           │
├───────────────────────────────────┬──────────────┬────────────┬─────────────┬──────────────┬───────────────────────────┤
│ Dashboard Page / Module           │ HTTP Status  │ Charts     │ Dataframes  │ Exceptions   │ Verification Status       │
├───────────────────────────────────┼──────────────┼────────────┼─────────────┼──────────────┼───────────────────────────┤
│ 00 • Main Portal (app.py)         │ 200 OK       │ Layout     │ 0           │ 0 Errors     │ VERIFIED & ALIGNED        │
│ 01 • Research Overview            │ 200 OK       │ Layout     │ 0           │ 0 Errors     │ VERIFIED & ALIGNED        │
│ 02 • Dataset Explorer             │ 200 OK       │ 6 Charts   │ 2 Tables    │ 0 Errors     │ VERIFIED & ALIGNED        │
│ 03 • Calibration & Reconstruction │ 200 OK       │ 3 Charts   │ 3 Tables    │ 0 Errors     │ VERIFIED & ALIGNED        │
│ 04 • Joint TCN Deep Learning      │ 200 OK       │ 1 Chart    │ 1 Table     │ 0 Errors     │ VERIFIED & ALIGNED        │
│ 05 • Conformal Uncertainty        │ 200 OK       │ 2 Charts   │ 2 Tables    │ 0 Errors     │ VERIFIED & ALIGNED        │
│ 06 • CPCB Category Reliability    │ 200 OK       │ 2 Charts   │ 4 Tables    │ 0 Errors     │ VERIFIED & ALIGNED        │
│ 07 • Big Data Warehousing & Spark │ 200 OK       │ 3 Charts   │ 5 Tables    │ 0 Errors     │ VERIFIED & ALIGNED        │
├───────────────────────────────────┴──────────────┴────────────┴─────────────┴──────────────┴───────────────────────────┤
│ Unit Test Suite: 65 Tests Executed — 64 Passed, 1 Skipped (Sandbox Loopback), 0 Failures, 0 Errors                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Page-by-Page Audit & Implementation Details

### 2.1 Main Portal & Faculty Demonstration Gateway (`dashboard/app.py` & `dashboard/config.py`)
* **Verified Academic Citation:** Updated to the exact peer-reviewed journal citation:
  > Campmier et al. (2023), *Seasonally optimized calibrations improve low-cost sensor performance: long-term field evaluation of PurpleAir sensors in urban and rural India*, **Atmospheric Measurement Techniques**, 16, 4357–4374. DOI: `10.5194/amt-16-4357-2023` | Dryad: `10.6078/D1RQ70` (CC0 1.0 Universal).
* **Audited Metric Card Alignment:**
  - *Total Curated Records:* Standardized to **18,584 Predominantly Hourly Observations** (0 nulls, 0 duplicate timestamps).
  - *Sensor Calibration:* Labeled as **Pooled RF Test RMSE: 30.22 µg/m³** (MAE 16.25 µg/m³, R² 0.8823).
  - *Deep Neural Network:* Labeled as **Joint TCN (RF: 61h, Context: 24h)**.
  - *CPCB Category Accuracy:* Qualified as **PM2.5-Based Assessment | Quadratic κ = 0.9271**.
  - *Spark Big Data Throughput:* Labeled as **983k rows/s (50x Scale, Local Workstation Mode)**.
* **Homepage Research Disclosures Box:** Injected a prominent warning card detailing TCN historical BatchNorm training leakage, the 61h receptive field correction, confounding in humidity associations, and single-machine local Spark constraints.

---

### 2.2 Module 01: Research Overview (`dashboard/pages/01_Research_Overview.py`)
* **Checkpoint 1 (Data Audit):** Documented 18,584 predominantly hourly records, Dryad CC0 repository provenance, thermodynamic dew point exclusion ($T_d > T$ in 100% of records), and the humidity association ($15.15\,\mu\text{g}/\text{m}^3$ on $N=81$ high-RH points confounded by Bangalore monsoon). Removed unverified row-level checksum claim.
* **Checkpoint 2 (Calibration & Reconstruction Baselines):**
  - Stated overall test performance: Pooled RF $\text{RMSE} = 30.22\,\mu\text{g}/\text{m}^3$ ($N=3,720$). Clarified that $17.64\,\mu\text{g}/\text{m}^3$ was the Moderate/Poor sub-band ($N=688$).
  - Displayed the **Phase 3A.2 Corrected Reconstruction Benchmarks** on reference BAM target:
    - *Offline / 2-Sided Lookahead:* Bidir GRU ($36.17$), Bidir LSTM ($36.83$), Linear Interp ($36.93$).
    - *Strictly Causal / Past Only:* Causal GRU ($44.89$), Causal LSTM ($45.40$).
    - *Tabular Imputers:* Corrected RF ($59.03$), Corrected KNN ($68.87$).
  - Compounding error designated as **partially addressed**.
* **Checkpoint 3 (Joint TCN Deep Learning):** Corrected receptive field to **61 Hours** (effective training context: **24 Hours**); documented historical BatchNorm training-time temporal leakage; clarified that Joint TCN achieves $56.76\,\mu\text{g}/\text{m}^3$ on hidden BAM points, while historical $31.84$ evaluated raw sensor signals, is insufficiently documented, and cannot be ranked against BAM baselines.
* **Checkpoint 4 (Conformal Uncertainty):** Formally distinguished 24h pointwise coverage ($95.70\%$, width $200.84\,\mu\text{g}/\text{m}^3$) from simultaneous 24h block coverage ($74.58\%$).
* **Checkpoint 5 (CPCB Regulatory Reliability):** Qualified as PM2.5-based CPCB assessment; noted that Raw ATM had higher Quadratic Kappa ($0.9392$ vs $0.9271$) due to marginal distributions and 0 large errors; labeled +13.43% stability spread as an internal heuristic, not a Bayesian probability; noted Severe recall was 0/4.
* **Checkpoint 6 (Big Data Scalability):** Exact warehouse footprints documented (CSV: 2.126 MB, Parquet: 0.573 MB, ORC: 0.338 MB); single-host local execution disclosed.
* **Prominent Research Limitations Section:** Expanded with all 8 mandatory scientific disclosures.

---

### 2.3 Module 02: Dataset Explorer (`dashboard/pages/02_Dataset_Explorer.py`)
* **Dataset Attribution:** Features full AMT 2023 publication citation and Dryad DOI `10.6078/D1RQ70` (CC0 1.0).
* **Predominantly Hourly Observations:** Explicitly noted natural sensor outages, variable transmission intervals, and site-isolated segmentations.
* **Meteorological Confounding Disclosure:** Tab 1 includes an explicit scientific notice that high humidity (>65%, $N=81$) occurs in low-pollution Bangalore monsoon conditions ($\sim 20\,\mu\text{g}/\text{m}^3$), whereas low humidity occurs during severe Delhi winter inversions ($>200\,\mu\text{g}/\text{m}^3$), confounding error distributions.
* **Raw Excel Inspection:** Displays unaltered observations straight from the Dryad deposit with CC0 licensing details.

---

### 2.4 Module 03: Calibration & Reconstruction Baselines (`dashboard/pages/03_Calibration_Reconstruction.py`)
* **Tab 2 (Calibration Benchmarks):**
  - Displays overall test split summary ($N=3,720$) showing Pooled RF RMSE of **$30.22\,\mu\text{g}/\text{m}^3$** ($R^2 = 0.8823$, $\text{MAE} = 16.25\,\mu\text{g}/\text{m}^3$).
  - Added the **Concentration Sub-Band Breakdown** (`calibration_by_pollution_range.csv`), transparently displaying that the $17.64\,\mu\text{g}/\text{m}^3$ metric belongs strictly to the Moderate/Poor sub-band ($N=688$).
* **Tab 3 (Reconstruction Baselines):**
  - Fully integrated the **Phase 3A.2 Corrected Reconstruction Benchmarks** (`corrected_reconstruction_metrics.csv`):
    - Bidirectional GRU: $36.17\,\mu\text{g}/\text{m}^3$
    - Bidirectional LSTM: $36.83\,\mu\text{g}/\text{m}^3$
    - Linear Interpolation: $36.93\,\mu\text{g}/\text{m}^3$
    - Causal GRU: $44.89\,\mu\text{g}/\text{m}^3$
    - Causal LSTM: $45.40\,\mu\text{g}/\text{m}^3$
    - Corrected RF Imputer: $59.03\,\mu\text{g}/\text{m}^3$
    - Corrected KNN Imputer: $68.87\,\mu\text{g}/\text{m}^3$
  - Distinctly categorized Offline (2-sided lookahead) vs Strictly Causal (past context only).
  - Explicit warning prohibiting ranking the historical $31.84\,\mu\text{g}/\text{m}^3$ raw-channel metric against BAM-reference baselines.
* **Tab 4 (Error Propagation Across Gaps):**
  - Added an alert disclosing that compounding error remains **partially addressed** pending a controlled 3-way sequential pipeline ablation.

---

### 2.5 Module 04: Joint TCN Deep Learning (`dashboard/pages/04_Joint_TCN.py`)
* **Receptive Field Card & Proof:** Corrected from 31h to **61 Hours** ($RF = 1 + 2(K-1)\sum d = 1 + 2 \times 2 \times 15 = 61\text{ Hours}$); effective training sequence context confirmed as **24 Hours**.
* **Historical Normalization Notice:** Prominent warning explaining that `frozen_joint_tcn_checkpoint.pt` was trained with BatchNorm (inference causal, training sequence leakage identified in Phase 3A.2). Noted that `ChannelLayerNorm1d` is implemented in code for future retraining.
* **Target Variable Clarification:** Explicitly separated the Checkpoint 3 BAM-reference evaluation (Hidden RMSE = $56.76\,\mu\text{g}/\text{m}^3$) from the historical raw-channel imputation metric ($31.84\,\mu\text{g}/\text{m}^3$), labeling the latter as insufficiently documented.

---

### 2.6 Module 05: Conformal Uncertainty Quantification (`dashboard/pages/05_Uncertainty_Analysis.py`)
* **Tab 4 (Temporal Block Conformal):**
  - Updated block conformal table with exact reconciled numbers from `conformal_coverage_reconciliation.csv` and `conformal_simultaneous_block_coverage.csv`.
  - Added distinct columns for **Pointwise Coverage** and **Simultaneous Block Coverage**.
  - **Prominent Scientific Disclosure:**
    > *"The observed pointwise coverage for 24-hour block conformal was 95.70%, obtained with a mean interval width of 200.84 µg/m³. This wide interval has limited operational usefulness, and pointwise coverage must not be confused with simultaneous 24-hour block coverage (74.58%), which misses the nominal 90% target by 15.42%."*
* **Tab 3 (Regimes):** Added a confounding note explaining that relative humidity tiers reflect combined site and seasonal effects rather than an isolated causal humidity mechanism.

---

### 2.7 Module 06: CPCB Category Reliability (`dashboard/pages/06_CPCB_Reliability.py`)
* **Tab 2 (Controlled Benchmark Comparison):**
  - Loaded and displayed the controlled 127-window comparison from `cpcb_controlled_comparison.csv`:
    - Calibrated Joint TCN: **72.44% Accuracy**, Macro F1 58.96%, Quadratic $\kappa = 0.9271$
    - Raw PurpleAir (CF=1): **39.37% Accuracy** (+33.07 pp calibration gain)
    - Raw PurpleAir (ATM): **66.14% Accuracy** (+6.30 pp calibration gain), Quadratic $\kappa = \mathbf{0.9392}$
  - **Quadratic Kappa Nuance Explained:** Added text explaining why Raw ATM achieved higher Quadratic Kappa (0.9392 vs 0.9271) despite lower exact accuracy: Raw ATM had 0 large errors ($\ge 2$ categories off), whereas Calibrated TCN had 2 large errors.
* **Tab 3 (Uncertainty-Aware Stability):**
  - Labeled the +13.43% spread (Stable 83.33% vs Ambiguous 69.90%) as an **internal heuristic screening property**, not a formal Bayesian posterior probability or hardware trust score.
  - Documented that Severe category recall was 0/4 due to extreme sensor saturation and small test sample size ($N=4$).
* **Scope Clarification:** Prominently stated that this is strictly a **PM2.5-based CPCB Category Assessment**, not a full multi-pollutant NAQI.

---

### 2.8 Module 07: Big Data Warehousing & Spark Analytics (`dashboard/pages/07_Big_Data_Analytics.py`)
* **Reconciled Storage Footprints ($N=18,584$ Warehouse Tables):**
  - CSV (`sensor_format_csv`): **$2.126\,\text{MB}$** (Baseline text format)
  - Parquet Snappy (`sensor_format_parquet`): **$0.573\,\text{MB}$** (**$-73.0\%$** space savings)
  - ORC ZSTD (`sensor_format_orc`): **$0.338\,\text{MB}$** (**$-84.1\%$** space savings, maximum compression density)
  - Reconciled draft figure: Documented that the earlier 4.8 MB draft figure represented raw uncompressed source file exports.
* **Distributed Processing Disclosure:** Prominently stated that all processing executes on an 8-core Apple Silicon M2 workstation in local mode (`local[*]`). **Network-scale distributed processing is NOT DEMONSTRATED.**

---

## 3. Automated Browser Walkthrough & Visual Verification

An automated headless Chrome browser walkthrough was executed using `scratch/browser_walkthrough.js` against the live local Streamlit server (`http://127.0.0.1:8501`, health endpoint verified `ok`). The walkthrough exercised all eight views (the main gateway and all seven analytical dashboard pages), interacted with interactive elements, and captured fresh screenshots:

* **Execution Timestamp:** November 9, 2026, 14:31:45+05:30
* **Execution Command:** `node scratch/browser_walkthrough.js`
* **Process Exit Code:** `0` (Success)
* **Error / Exception Count:** `0` across all pages (0 `.stException`, 0 red tracebacks, 0 missing containers)

### Page-to-Screenshot Mapping

| # | Page Name | Route | HTTP Status | Visual Elements | Exceptions / Errors | Screenshot Filename & Disk Footprint |
|:---:|---|---|:---:|:---:|:---:|---|
| **0** | Main Portal | `/` | 200 OK | KPI Cards, Limitation Box | 0 Errors | `00_Home_Portal.png` (239.0 KB) |
| **1** | Research Overview | `/Research_Overview` | 200 OK | 6 Checkpoint Tabs, Disclosures Box | 0 Errors | `01_research_overview.png` (278.1 KB) |
| **2** | Dataset Explorer | `/Dataset_Explorer` | 200 OK | 6 Plotly Charts, 2 Dataframes | 0 Errors | `02_Dataset_Explorer.png` (219.4 KB) |
| **3** | Calibration & Recon | `/Calibration_Reconstruction` | 200 OK | 3 Plotly Charts, 3 Dataframes, Sliders | 0 Errors | `03_Calibration_Reconstruction.png` (228.2 KB) |
| **4** | Joint TCN Deep Learning | `/Joint_TCN` | 200 OK | 1 Plotly Chart, 1 Dataframe, Sequence Runner | 0 Errors | `04_Joint_TCN.png` (234.1 KB) |
| **5** | Uncertainty Analysis | `/Uncertainty_Analysis` | 200 OK | 2 Plotly Charts, 2 Dataframes, Ribbon View | 0 Errors | `05_Uncertainty_Analysis.png` (220.1 KB) |
| **6** | CPCB Reliability | `/CPCB_Reliability` | 200 OK | 2 Plotly Charts, 4 Dataframes, Confusion Matrix | 0 Errors | `06_CPCB_Reliability.png` (204.0 KB) |
| **7** | Big Data Analytics | `/Big_Data_Analytics` | 200 OK | 3 Plotly Charts, 5 Dataframes, Live SQL Runner | 0 Errors | `07_Big_Data_Analytics.png` (254.7 KB) |

**Walkthrough Outcome:** 100% of pages passed automated verification with **zero runtime exceptions, zero red alert banners, complete chart/table hydration, and fresh verified screenshots preserved in `outputs/reports/screenshots/`**.

---

## 4. Post-Change Test Suite Execution Status

The full unit test suite was executed after all Phase 3B dashboard source code modifications were in place to confirm complete absence of regressions:

* **Execution Command:** `python3 -m unittest discover -s tests -v`
* **Execution Timestamp:** November 9, 2026, 14:31:16+05:30
* **Execution Environment:** macOS Darwin 24.2.0 (ARM64 Apple Silicon M2), Python 3.13.0
* **Exit Code:** `0`
* **Test Counts:**
  - **Total Tests Executed:** `65`
  - **Passed Tests:** `64`
  - **Skipped Tests:** `1` (`test_spark_and_pandas_numerical_consistency` in `test_bigdata.py`, skipped due to sandboxed loopback socket permission constraint)
  - **Failed Tests:** `0`
  - **Errors:** `0`
* **Execution Duration:** `3.450 seconds`
* **Verification Timing:** Confirmed executed strictly **after** Phase 3B dashboard code edits were finalized.

---

## 5. Critical Metric Spot-Check Verification

Direct inspection of the rendered dashboard pages confirms the following mandatory scientific and evidence alignments:

1. **Historical Joint TCN BAM Hidden Reconstruction RMSE (56.76 µg/m³):**
   - *Rendered Location:* `04_Joint_TCN.py` (Research Disclosures box) and `01_Research_Overview.py` (Checkpoint 3 tab).
   - *Confirmation:* Explicitly labeled as evaluating $N=769$ hidden BAM points under 20% MCAR masking. Accompanied by the mandatory qualification that serialized checkpoint `frozen_joint_tcn_checkpoint.pt` was trained with historical BatchNorm, which exhibits training-mode temporal sequence leakage.
2. **Historical Raw-Channel RMSE (31.84 µg/m³):**
   - *Rendered Location:* `03_Calibration_Reconstruction.py` (Tab 3 alert), `04_Joint_TCN.py` (Disclosures list), and `01_Research_Overview.py` (Checkpoint 2 & 3 tabs).
   - *Confirmation:* Explicitly decoupled from BAM-reference reconstruction baselines. Text warns that 31.84 evaluated raw PurpleAir sensor channels, lacks serialized prediction files, is classified as insufficiently documented, and must **never** be ranked against reference BAM baselines.
3. **Theoretical Receptive Field vs. Effective Context:**
   - *Rendered Location:* `04_Joint_TCN.py` (KPI card & Disclosures box), `01_Research_Overview.py` (Checkpoint 3 tab), and `app.py` (Homepage KPI card).
   - *Confirmation:* Analytical receptive field of **61 Hours** ($RF = 1 + 2(K-1)\sum d = 1 + 2 \times 2 \times 15 = 61\text{h}$) is rigorously distinguished from the historical training window length of **24 Hours** (`seq_len = 24`), which effectively truncated the available historical context during training.
4. **Pointwise vs. Simultaneous Block Coverage:**
   - *Rendered Location:* `05_Uncertainty_Analysis.py` (Tab 4 alert & comparative table) and `01_Research_Overview.py` (Checkpoint 4 tab).
   - *Confirmation:* Dedicated table columns and a prominent scientific alert explicitly decouple the observed **95.70% pointwise coverage** (with its wide, operationally limited mean interval width of $200.84\,\mu\text{g}/\text{m}^3$) from the simultaneous 24-hour block coverage (**74.58%**), which falls 15.42 pp short of the nominal 90% confidence target.
5. **CPCB Regulatory Assessment Scope:**
   - *Rendered Location:* `06_CPCB_Reliability.py` (Regulatory alert banner), `01_Research_Overview.py` (Checkpoint 5 tab), and `app.py` (KPI card subtitle).
   - *Confirmation:* Strictly described as a **PM2.5-based category assessment** under CPCB standards. Explicitly clarifies that it is *not* an official multi-pollutant National Air Quality Index (NAQI), which mandates concurrent monitoring of PM10, NO2, SO2, CO, and O3.
6. **Big Data Distributed Processing Scope:**
   - *Rendered Location:* `07_Big_Data_Analytics.py` (Infrastructure Architecture disclosure) and `01_Research_Overview.py` (Checkpoint 6 tab).
   - *Confirmation:* Prominently states: **"Network-scale distributed processing is NOT DEMONSTRATED"**. Discloses that Spark operates in single-workstation local mode (`local[*]`) on Apple Silicon M2 with embedded Derby metastore.

---

## 6. Phase 3B Acceptance Sign-off & Stop Condition

All Phase 3B final verification requirements have been satisfied:
- Streamlit server verified active and responsive.
- Browser walkthrough passed all eight views with exit code 0 and zero exceptions.
- All fresh screenshots verified and mapped in `outputs/reports/screenshots/`.
- Full unit test suite executed post-change (65 tests: 64 passed, 1 skipped, 0 failed, 0 errors, exit code 0).
- All six critical metric spot-checks confirmed on the rendered dashboard.
- No model weights, checkpoints, predictions, or conformal quantiles were altered.

**Phase 3B is fully verified and formally closed.** Stopping per instruction to await user review and sign-off.
