# AIRWISE — Phase 3C.3B: Submission Readiness and Final Scientific Verification Checklist

**Document ID:** `outputs/reports/airwise_phase3C_submission_readiness_checklist.md`  
**Phase:** 3C.3B (Final Statistical Corrections and Manuscript Freeze)  
**Evaluated Artifact:** `outputs/reports/airwise_phase3C_final_manuscript_v3.md`  
**Author:** Antigravity Autonomous Research Assistant  
**Date:** November 2026  
**Status:** All Substantive Audits PASSED — Manuscript Frozen  

---

## 1. Executive Summary & Freeze Declaration

This checklist serves as the definitive quality, consistency, and scientific integrity audit for the **AIRWISE Research Manuscript (Version 3.0 Frozen Release)**.

### Strict Preservation Verification
- **Zero Retraining:** No models were retrained.
- **Zero Reruns:** No benchmark experiments were re-executed.
- **Zero Checkpoint Modifications:** Checkpoint hashes remain identical to pre-benchmark baselines.
- **Zero Prediction / Metric Overwriting:** `outputs/experiments/phase3C/benchmark_predictions.parquet` and all scenario metric CSVs remain frozen read-only artifacts.
- **Preservation of Version 2.0:** `outputs/reports/airwise_phase3C_final_manuscript_v2.md` is strictly preserved on disk without modification.

---

## 2. Comprehensive Verification Checklist

### Part 1: Dataset Partition Reconciliation & Sample Counts

| Check Item | Requirement | Verification Finding | Status |
| :--- | :--- | :--- | :---: |
| **Total Clean Dataset** | Exactly $18,584$ paired hourly observations | Verified in `data/processed/cleaned_data.parquet` and Table 1. | **PASS** |
| **Training Partition (60.0%)** | Exactly $11,149$ observations (BLR: 4464, DEL: 4502, HAM: 2183) | Verified in `split_summary.json` and Table 1; identical to code split. | **PASS** |
| **Validation Partition (10.0%)**| Exactly $1,857$ observations (BLR: 744, DEL: 750, HAM: 363) | Verified in `split_summary.json` and Table 1; used for early stopping. | **PASS** |
| **Conformal Calibration (10.0%)**| Exactly $1,858$ observations (BLR: 744, DEL: 750, HAM: 364) | Verified in `split_summary.json` and Table 1; isolated from training. | **PASS** |
| **Independent Test Set (20.0%)**| Exactly $3,720$ observations (BLR: 1489, DEL: 1502, HAM: 729) | Verified in `benchmark_predictions.parquet` and Table 1. | **PASS** |
| **Partition Exclusivity** | Zero overlapping timestamps across splits | Chronological per-site slicing verified in `src/uncertainty/split_manager.py`. | **PASS** |

---

### Part 2: Statistical Uncertainty & Moving-Block Bootstrap Audit

| Check Item | Requirement | Verification Finding | Status |
| :--- | :--- | :--- | :---: |
| **Bootstrap Tail Area Labeling** | Do not present uncentered bootstrap tail proportions as formal $p$-values | Formula labeled as empirical uncentered bootstrap tail proportion $\hat{p}_{\text{tail}}$; term "p-value" removed from all tables and text. | **PASS** |
| **Primary Uncertainty Evidence** | Paired bootstrap percentile confidence intervals must serve as primary evidence | Table 6 and Table 7 emphasize 95% CIs and classify significance by whether CIs exclude zero. | **PASS** |
| **Block-Length Sensitivity Rigor**| Explicitly disclose sensitivity across $b \in \{12, 24, 48, 72\}$ hours | Table 7 and Figure 9 document that intervals excluding zero at $12$h expand across zero once the diurnal cycle ($24$h) is preserved. | **PASS** |
| **Superiority Claims Bounded** | Do not claim statistically significant superiority solely from tail proportions | Text explicitly clarifies that Joint TCN v2 superiority cannot be claimed on 20% MCAR or MAR-RH at $b=24$h. | **PASS** |
| **MAR-RH Deconstruction** | Reconcile negative point estimate ($-0.325$) with high tail proportion ($0.925$) | Fully explained by site heterogeneity: Hamirpur pulls point estimate down, while Delhi & Bangalore (80.4% of data) dominate resampled blocks. | **PASS** |

---

### Part 3: Outage Duration Precision & Degradation Language

| Check Item | Requirement | Verification Finding | Status |
| :--- | :--- | :--- | :---: |
| **Nominal vs. Merged Durations** | Distinguish insertion block size ($h$) from empirical continuous outage duration | Explicitly documented in Table 2 and Section 8; notes overlapping block coalescence (mean $=32.61$h, max $=63$h for $h=24$h). | **PASS** |
| **24-Hour Outage Clarification** | Correct any implication that all 24h scenario outages lasted exactly 24 hours | Revised to state: "Outages generated with a nominal insertion block length of 24 hours... empirical merged outages range from 24h to 63h". | **PASS** |
| **Quantification of Change** | Describe $+29.42$ $\mu$g/m³ change strictly between nominal 1h and 24h scenarios | Revised in Section 10.2: "Comparing the nominal 1-hour to the nominal 24-hour block-generation scenarios on the missing PM subgroup...". | **PASS** |
| **Non-Monotonicity Qualification**| Do not claim universal monotonic degradation across all pipelines and metrics | Section 10.2 explicitly notes that while Joint TCN increases on missing intervals, sequential baselines exhibit non-monotonic fluctuations. | **PASS** |

---

### Part 4: Scientific Claims & Theoretical Language

| Check Item | Requirement | Verification Finding | Status |
| :--- | :--- | :--- | :---: |
| **Title Accuracy** | Title must not invent a "wavelet" component | Title revised to: *"AIRWISE: Adaptive Imputation and Robust Windowed Inference for Sensor Estimation"*. | **PASS** |
| **Causal Inference Definition** | Clarify that causality refers to non-anticipative temporal filtering ($t \le \tau$) | Section 7 and Abstract explicitly define causality as causal dilated convolution with zero future lookahead. | **PASS** |
| **MNAR Disclosure** | Disclose oracle-generated nature of synthetic MNAR-PM | Section 8, 10.2, and 15.1 explicitly disclose that MNAR-PM is an oracle stress test, not operational real-world MNAR detection. | **PASS** |
| **Humidity Associations** | Do not claim physical microphysical causality without laboratory hygrometer | Section 15.1 explicitly qualifies humidity effects as empirical statistical correlations from ambient field co-locations. | **PASS** |
| **Geographic Generalization** | Do not claim national generalization across all of India | Section 15.1 bounds findings to the three monitored microclimates (Bangalore, Delhi, Hamirpur). | **PASS** |
| **Sensor Drift Claims** | Do not claim independently validated physical sensor drift | Section 15.1 clarifies that seasonal error shifts reflect atmospheric source changes, not verified component aging. | **PASS** |
| **Regulatory Scope** | Clarify that evaluations represent $\text{PM}_{2.5}$ sub-index, not full 8-pollutant NAQI | Section 15.1 explicitly bounds regulatory scope to $\text{PM}_{2.5}$ category reliability. | **PASS** |
| **Big Data Execution Context**| Disclose that Spark benchmarks reflect local multi-core execution | Section 15.1 states that Spark scaling benchmarks reflect local execution, qualifying distributed cluster network overhead. | **PASS** |

---

### Part 5: Metric Provenance and Tabular Audit

| Check Item | Requirement | Verification Finding | Status |
| :--- | :--- | :--- | :---: |
| **Table 1 (Partitions)** | Exact 60/10/10/20 partition counts | Verified against `split_summary.json` ($11,149 / 1,857 / 1,858 / 3,720$). | **PASS** |
| **Table 2 (Empirical Outages)** | Min, median, mean, max for 1h, 3h, 6h, 12h, 24h | Verified against `benchmark_predictions.parquet`. | **PASS** |
| **Table 3 (Overall Full-N)** | All 36 rows match `scenario_metrics_overall.csv` | Pipeline B lowest across all 9 scenarios ($36.26$–$41.64$ $\mu$g/m³). | **PASS** |
| **Table 4 (Missing Subgroup)** | All rows match `scenario_metrics_missing_pm.csv` | Verified ($52.44$ for C on 1h; $43.05$ for C on MAR-RH; $81.86$ for C on 24h). | **PASS** |
| **Table 5 (Common Support)** | Exact matched calibration on $m_t = 1$ ($N = 2,951$) | Confirmed identity of A, B, and D ($29.82$ $\mu$g/m³, max diff $\le 1.51 \times 10^{-5}$). | **PASS** |
| **Table 6 (Bootstrap Inference)**| Canonical $b=24$h, $B=1,000$ iterations | Verified against `bootstrap_significance_results.csv`. | **PASS** |
| **Table 7 (Sensitivity Sweep)**| Sweeps across $b \in \{12, 24, 48, 72\}$ hours | Verified against `bootstrap_sensitivity_sweep.csv`. | **PASS** |
| **Table 8 (Site Stratification)**| Site-stratified metrics across BLR, DEL, HAM | Verified against `site_stratified_metrics.csv`. | **PASS** |

---

### Part 6: Bibliography and In-Text Citation Audit

| Check Item | Requirement | Verification Finding | Status |
| :--- | :--- | :--- | :---: |
| **Source Dataset (Ref [23])** | Real, authentic paper for `Campmier_Manuscript_Data.xlsx` | Campmier et al. (*Atmospheric Measurement Techniques*, 2023, DOI: `10.5194/amt-16-4357-2023`). | **PASS** |
| **Lancet India Burden (Ref [2])**| Real landmark paper for air pollution deaths in India | Pandey et al. (*The Lancet Planetary Health*, 2021, DOI: `10.1016/S2542-5196(20)30298-9`). | **PASS** |
| **LCS Foundations (Ref [4])** | Foundational review for low-cost sensing in cities | Kumar et al. (*Environment International*, 2015, DOI: `10.1016/j.envint.2014.11.019`). | **PASS** |
| **Urban Air & Fusion (Refs [10, 12])**| Authentic spatial-temporal air quality references | Zheng et al. (*KDD* 2013, DOI: `10.1145/2487575.2487688`) and Yi et al. (*KDD* 2018, DOI: `10.1145/3219819.3219922`). | **PASS** |
| **Methodological Refs ([1, 3, 5-9, 11, 13-22])**| Authentic, widely cited literature with real DOIs | All 18 methodological papers verified (Barkjohn, Malings, Bai, Little & Rubin, Politis, Künsch, etc.). | **PASS** |
| **Citation Citation Coverage** | Every bibliography entry must be cited in text | All 23 references are cited at least once in manuscript body. | **PASS** |

---

### Part 7: Figure Integration & Visual Verification

| Check Item | Requirement | Verification Finding | Status |
| :--- | :--- | :--- | :---: |
| **Figure 1 (Architecture)** | Causal Joint TCN v2 schematic embedded | Embedded and cited in Section 6.3. | **PASS** |
| **Figure 2 (Dataset Summary)** | Temporal gaps and density plots embedded | Embedded and cited in Section 4.3. | **PASS** |
| **Figure 3 (Four Pipelines)** | Methodology comparison schematic embedded | Embedded and cited in Section 6. | **PASS** |
| **Figure 4 (Overall RMSE)** | Grouped bar chart across 9 scenarios embedded | Embedded and cited in Section 9.1. | **PASS** |
| **Figure 5 (Missing PM Subgroup)**| Subgroup reconstruction bar chart embedded | Embedded and cited in Section 10.1. | **PASS** |
| **Figure 6 (Outage Degradation)**| Trajectory curves across nominal blocks embedded | Embedded and cited in Section 10.1. | **PASS** |
| **Figure 7 (Site Stratification)**| Multi-panel plot across BLR, DEL, HAM embedded | Embedded and cited in Section 13. | **PASS** |
| **Figure 8 (Common Support)** | Matched calibration comparison embedded | Embedded and cited in Section 11. | **PASS** |
| **Figure 9 (Bootstrap Sweep)** | Error-band plot across $b \in \{12, 24, 48, 72\}$h embedded | Embedded and cited in Section 12.3. | **PASS** |

---

## 3. Exact Change Log: Manuscript v2 to Manuscript v3

| Manuscript Location | Manuscript v2 Text | Manuscript v3 Frozen Text | Methodological Justification |
| :--- | :--- | :--- | :--- |
| **Frontmatter** | Version 2.0 Change Log | Added Version 3.0 Change Log documenting statistical hardening, outage precision, and monotonicity qualifications. | Transparent version control. |
| **Abstract (Lines 26–28)** | *"...moving-block sensitivity sweeps demonstrate that statistical significance under short block lengths ($b=12$h) collapses when the diurnal cycle is preserved ($b=24$h)..."* | *"...moving-block sensitivity sweeps demonstrate that while the 95% bootstrap confidence interval excludes zero under a short 12-hour block specification ($b=12$h), it expands across zero once the 24-hour diurnal cycle is preserved ($b=24$h), cautioning against claiming statistically significant superiority without accounting for temporal autocorrelation."* | Removes claim of significance based on uncentered tail proportion; grounds finding in confidence intervals. |
| **Section 3 (Line 93)** | *"...proving that statistical significance observed under 12-hour blocks fails to hold once diurnal autocorrelation ($24$h) is respected."* | *"...proving that paired confidence intervals excluding zero under 12-hour blocks span zero once diurnal autocorrelation ($24$h) is preserved."* | Replaces "significance" terminology with rigorous interval-based statement. |
| **Section 8 (Line 220)** | *"6. 24-Hour Block Outages: Contiguous 24-hour diurnal outages."* | *"6. 24-Hour Block Outages (Nominal): Outages generated with a nominal insertion block length of 24 hours. Because contiguous segment bounds allow adjacent randomly placed blocks to overlap, empirical outages in this scenario range from a minimum of 24 hours to a maximum merged duration of 63 hours (mean $= 32.61$ hours, median $= 24.0$ hours; see Table 2)."* | Clearly distinguishes nominal block-generation parameter from merged continuous duration. |
| **Section 8 (Line 238)** | Caption of Table 2: *"Note: Randomized placement of fixed-length blocks within bounded continuous segments allows adjacent blocks to occasionally coalesce, generating realistic compound outages with right-skewed tails."* | Expanded Caption: *"As documented in Table 2, randomized placement of fixed-length blocks within continuous temporal segments allows adjacent blocks to occasionally coalesce. Consequently, nominal block-generation duration must be distinguished from the actual continuous merged outage durations observed in deployment."* | Explicitly instructs reader on nominal vs merged duration distinction. |
| **Section 10.1 (Figure 6 Caption)**| *"Figure 6: Error degradation trajectories as outage duration increases from 1 to 24 hours. Sequence models exhibit steep error compounding during multi-day dropouts."* | *"Figure 6: Reconstruction and overall RMSE trajectories across nominal block-generation scenarios ($1$h to $24$h). While Joint TCN v2 displays sharp error compounding on missing intervals between nominal 1h and 24h scenarios ($+29.42$ $\mu$g/m³), sequential baselines exhibit non-monotonic behavior across intermediate block durations."* | Discloses that sequential baselines are non-monotonic across intermediate blocks. |
| **Section 10.2 (Line 300)** | *"Figure 6 highlights that for extended block dropouts, Joint TCN v2 suffers severe degradation, with reconstruction RMSE ballooning to $81.86$ $\mu$g/m³ on 24-hour outages (compared to $56.19$ for Pipeline B and $67.28$ for Pipeline A)."* | *"Comparing the nominal 1-hour to the nominal 24-hour block-generation scenarios on the missing PM subgroup, Joint TCN v2 reconstruction RMSE increases by $+29.42$ $\mu$g/m³ (from $52.44$ to $81.86$ $\mu$g/m³), compared to a $+2.80$ $\mu$g/m³ change for Pipeline B ($53.39$ to $56.19$ $\mu$g/m³) and a $-0.51$ $\mu$g/m³ change for Pipeline A ($67.79$ to $67.28$ $\mu$g/m³). Importantly, while Joint TCN v2 exhibits steady error increases on the missing subgroup across nominal block lengths ($52.44 \to 53.04 \to 62.95 \to 69.76 \to 81.86$ $\mu$g/m³), error degradation is not universally monotonic across pipelines or metrics: for example, Pipeline A's missing subgroup RMSE fluctuates ($67.79 \to 66.13 \to 71.78 \to 77.29 \to 67.28$ $\mu$g/m³), and overall full-population RMSE exhibits non-monotonic variations (Table 3). Furthermore, in the nominal 24-hour scenario, overlapping block placement merged contiguous outages to an average duration of $32.61$ hours (with a maximum merged duration of $63$ hours), meaning that these error levels reflect compound multi-day dropouts rather than isolated 24-hour intervals."* | Accurately describes the $+29.42$ $\mu$g/m³ change between nominal 1h and 24h scenarios; removes claim of universal monotonicity. |
| **Section 12.1 (Line 334)** | Equation labeled: *"Empirical one-tailed non-superiority proportion: $$p = \frac{1}{B} \sum_{k=1}^B \mathbb{I}(\Delta_{\text{RMSE}}^{(k)} \ge 0.0)$$"* | Equation labeled: *"Empirical Uncentered Bootstrap Tail Proportion ($\hat{p}_{\text{tail}}$): $$\hat{p}_{\text{tail}} = \frac{1}{B} \sum_{k=1}^B \mathbb{I}(\Delta_{\text{RMSE}}^{(k)} \ge 0.0)$$"* Followed by explicit disclosure: *"We report paired bootstrap percentile confidence intervals as our primary uncertainty evidence. The uncentered tail proportion $\hat{p}_{\text{tail}}$ quantifies the fraction of bootstrap resamples where Joint TCN v2 error equals or exceeds the baseline ($\Delta_{\text{RMSE}} \ge 0$); it is an empirical distribution summary and should not be interpreted as a formal null-hypothesis test $p$-value."* | Eliminates misleading presentation of uncentered tail area as a formal hypothesis-test $p$-value. |
| **Section 12.1 (Table 6)** | Column headers: `Point Delta_RMSE`, `95% Bootstrap CI Lower`, `95% Bootstrap CI Upper`, `$p$-value`, `Significant (alpha = 0.05)?` | Column headers: `Point Delta_RMSE`, `95% Bootstrap CI Lower`, `95% Bootstrap CI Upper`, `Tail Proportion (hat{p}_{tail})`, `95% CI Excludes Zero?` | Replaces $p$-value header with $\hat{p}_{\text{tail}}$; bases decision strictly on whether 95% CI excludes zero. |
| **Section 12.3 (Table 7)** | Column headers: `Block Length`, `Point Delta_RMSE`, `95% CI Lower`, `95% CI Upper`, `CI Width`, `$p$-value`, `Significant?` | Column headers: `Block Length`, `Point Delta_RMSE`, `95% CI Lower`, `95% CI Upper`, `CI Width`, `Tail Proportion (hat{p}_{tail})`, `95% CI Excludes Zero?` | Replaces $p$-value header with $\hat{p}_{\text{tail}}$; bases decision strictly on whether 95% CI excludes zero. |
| **Section 12.3 (Line 380)** | *"Critical Methodological Insight: As visualised in Figure 9 and Table 7, under $b=12\text{h}$, Joint TCN v2 appeared statistically significantly superior on 20% MCAR ($p = 0.010$). However, a 12-hour block truncates the 24-hour diurnal atmospheric cycle, artificially inflating the effective sample size. When the canonical diurnal cycle is preserved ($b=24\text{h}$), significance vanishes ($p = 0.084$). As block length expands to $48$h and $72$h, variance increases monotonically, confirming that Joint TCN v2 does not achieve robust superiority."* | *"Critical Methodological Insight: As visualised in Figure 9 and Table 7, under a short 12-hour block specification ($b=12\text{h}$), the paired 95% bootstrap confidence interval strictly excluded zero ($[-5.27, -0.37]$) with a tail proportion of $\hat{p}_{\text{tail}} = 0.010$. However, a 12-hour block truncates the natural 24-hour diurnal atmospheric cycle, artificially underestimating temporal variance. When the canonical diurnal block length is enforced ($b=24\text{h}$), the confidence interval expands across zero ($[-4.31, +0.70]$) and the uncentered tail proportion shifts to $\hat{p}_{\text{tail}} = 0.084$, indicating that superiority cannot be claimed. As the block length is extended to $b=48\text{h}$ ($[-3.32, +2.56]$) and $b=72\text{h}$ ($[-2.90, +4.25]$), the confidence interval width increases monotonically ($4.90 \to 5.01 \to 5.88 \to 7.15$ $\mu$g/m³), confirming that any perceived advantage under short block specifications is not robust to diurnal and multi-day autocorrelation."* | Statistically hardened interpretation centered strictly on confidence interval expansion across zero. |
| **Section 16 (Conclusion)**| *"- Autocorrelation Matters: Statistical significance observed under short bootstrap blocks ($b=12$h) collapses once the 24-hour diurnal atmospheric cycle is respected, underscoring the necessity of moving-block resampling in environmental time series."* | *"- Autocorrelation Matters: Paired bootstrap confidence intervals that exclude zero under short 12-hour blocks span zero once the 24-hour diurnal atmospheric cycle is respected, demonstrating that empirical tail proportions under short blocks cannot be interpreted as robust evidence of superiority."* | Rigorous interval-grounded conclusion. |

---

## 4. Final Submission-Readiness Verdict

**VERDICT: SUBMISSION-READY (Subject to Target Venue Formatting)**

- **Substantive Scientific Blockers Remaining:** **ZERO.**
- **Empirical Discrepancies Remaining:** **ZERO.**
- **Unverified Literature Citations Remaining:** **ZERO.**
- **Unreferenced Figures or Tables Remaining:** **ZERO.**

The manuscript `outputs/reports/airwise_phase3C_final_manuscript_v3.md` represents an evidence-grounded, statistically audited, and methodologically conservative scientific contribution ready to be converted into venue-specific camera-ready formats (e.g., IEEEtran, ACM, or Copernicus LaTeX templates).
