# AIRWISE — Phase 3A.3 & Phase 3A.4 Final Verification Note (Scientific Qualification Patch)
**Document ID:** `AIRWISE-P3A-VERIF-002`  
**Date:** November 9, 2026  
**Status:** Complete — Audited & Verified (Documentation-Only Update)  

---

## 1. Historical Checkpoint Architectural Limitation

```
========================================================================================================
HISTORICAL CHECKPOINT CLASSIFICATION: frozen_joint_tcn_checkpoint.pt
========================================================================================================
Checkpoint Artifact: outputs/models/conformal/frozen_joint_tcn_checkpoint.pt
Architecture: Dilated Causal Conv1D Backbone with nn.BatchNorm1d Normalization Layers
Detected Parameters/Buffers:
  • Norm Weights & Biases: norm1.weight, norm1.bias, norm2.weight, norm2.bias
  • Running Statistics Buffers: running_mean, running_var, num_batches_tracked
Status:
  • Historical BatchNorm Architecture.
  • Training-mode temporal sequence statistics leakage was formally identified in Phase 3A.2.
  • The model was NOT retrained with ChannelLayerNorm1d.
Scientific Qualification:
  • The Phase 3A.3 and Phase 3A.4 evaluations reflect the performance of this historical checkpoint.
  • These evaluations DO NOT establish causality-safe training performance.
  • All historical weights, quantiles, predictions, and reports remain preserved without retraining.
========================================================================================================
```

---

## 2. Pointwise Coverage Formulation (95.70%)

The wording regarding the 95.70% metric across Phase 3A.3 reports has been formally qualified:

> **Scientific Definition:**  
> *"The observed pointwise coverage was 95.70%, obtained with a mean interval width of 200.84 µg/m³. This wide interval has limited operational usefulness, and pointwise coverage must not be confused with simultaneous 24-hour block coverage."*

* **Simultaneous Block Coverage:** Across the 59 contiguous 24-hour test blocks, true simultaneous joint coverage (where all 24 hours fall within bounds) is **only 74.58%**, remaining **15.42 percentage points below the nominal 90% target**.
* **Quantile Parity:** The value $95.70\%$ was produced because both Block Conformal (24h) at 90% and Temporal Block Conformal (12h) at 95% share the identical quantile $\hat{q} = 100.4218\,\mu\text{g}/\text{m}^3$ (width $200.84\,\mu\text{g}/\text{m}^3$).

---

## 3. Mathematical Basis of Quadratic Weighted Kappa

The wording evaluating Quadratic Weighted Cohen's Kappa has been qualified to reflect its complete mathematical formulation:

$$\kappa_{\text{quad}} = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$

* **Matrix and Marginal Dependence:** Quadratic weighted kappa depends on the complete confusion matrix and category marginals (the distribution of row and column totals and the chance-expected agreement matrix $E_{ij} = \frac{R_i C_j}{N}$, in addition to the quadratic distance penalty matrix $w_{ij}$).
* **Observed Comparison Retained:**
  - **Raw PurpleAir (ATM):** **$\kappa_{\text{quad}} = 0.9392$**
  - **Calibrated Joint TCN:** **$\kappa_{\text{quad}} = 0.9271$**
* **Analysis:** While the calibrated model achieved higher exact diagonal accuracy (92 vs 84 days) and fewer total misclassifications (35 vs 43), the differences in category marginal distributions and off-diagonal dispersion across the entire confusion matrix resulted in a higher quadratic kappa for raw ATM. This difference is not attributed solely to the number of large errors.

---

## 4. Reconciled Gap-Length Sample Counts (3,735 vs 3,720)

In [`outputs/reports/conformal_gap_metrics.csv`](file:///Users/prudhvisailingineni/Documents/7th%20Sem/Big%20Data/Project/outputs/reports/conformal_gap_metrics.csv), the hidden sample counts across gap durations are:
* **1-Hour Gap:** 744 samples (20.00% of test set)
* **3-Hour Gap:** 744 samples (20.00% of test set)
* **6-Hour Gap:** 746 samples (20.05% of test set)
* **12-Hour Gap:** 751 samples (20.19% of test set)
* **24-Hour Gap:** 750 samples (20.16% of test set)
* **Sum Across Rows:** **3,735 samples**

* **Experimental Design:** The five rows represent five **independent synthetic masking trials**, each masking approximately 20% of the 3,720 test observations ($N_{\text{hidden}} \approx 744\text{--}751$) with gap durations of 1h, 3h, 6h, 12h, and 24h.
* **Non-Mutually Exclusive:** The groups overlap across trials (2,322 unique test observations were masked in at least one trial). Summing to 3,735 is an artifact of adding across separate runs; it is not an accounting discrepancy against the 3,720 population.
* **Exact Coverage Calculation:** In each scenario $g$, empirical coverage is strictly computed as the fraction of hidden observations covered within that scenario's hidden points:
  $$\text{Coverage}_g = \frac{\text{Number of Hidden Observations Covered in Scenario } g}{N_{\text{hidden}, g}}$$

---

## 5. Disentangling the Conformal Calibration Experiments

* **Temporal Block Conformal (32 Blocks):** Requires **strictly contiguous 24-hour uninterrupted sequences** ($n_{\text{seg}} // 24$) with 0 missing hours. In `calib_df`, this strict filter retains 32 blocks. Nonconformity score is the block-maximum hourly error ($S_b = \max_{t} |y_t - \hat{y}_t|$), yielding quantile $q = 100.42\,\mu\text{g}/\text{m}^3$ and width $200.84\,\mu\text{g}/\text{m}^3$.
* **CPCB 24h Daily-Mean Windowing (78 Windows):** Enforces regulatory completeness: **$\ge 18/24$ valid hours** ($\ge 75\%$). In `calib_df`, 78 wall-clock days met completeness. Nonconformity score is the error on the 24-hour scalar mean ($s_w = |\bar{y}_{24h} - \hat{\bar{y}}_{24h}|$), yielding quantile $q = 15.30\,\mu\text{g}/\text{m}^3$ and width $30.59\,\mu\text{g}/\text{m}^3$.
* **Conclusion:** These two methods measure completely different statistical phenomena. They must never be combined or equated.

---

## 6. Summary of Preserved Scientific Conclusions

1. **Controlled CPCB Performance on 127 Windows:**
   - **Calibrated Model:** **72.44% accuracy** (Macro F1 58.96%, Weighted F1 71.54%, Quad Kappa 0.9271).
   - **Raw PurpleAir (CF=1):** **39.37% accuracy** (Calibration gain: **+33.07%**).
   - **Raw PurpleAir (ATM):** **66.14% accuracy** (Calibration gain: **+6.30%**).
2. **Severe Category Recall (0/4):** The calibrated model misclassified all 4 Severe test days into Very Poor due to model shrinkage and sensor saturation.
3. **The 13.43% Clarification:** 13.43% ($83.33\% - 69.90\%$) is the accuracy spread between Stable and Ambiguous prediction intervals within the calibrated model; it is **NOT** the raw-versus-calibrated gain.
4. **Scope Limitation:** The evaluation is strictly **PM2.5-based CPCB category classification**, not full multi-pollutant NAQI.

---

## 7. Change Control & Test Suite Status

* **Documentation-Only Update:** This verification patch modified only Markdown documentation files (`outputs/reports/`). No source code, model weights, scalers, prediction arrays, or test scripts were modified.
* **Test Suite Status:** Because no executable source code or tests were modified, no new test execution was required. The test suite status remains at its verified count: **65 tests run, 64 passed, 1 skipped, 0 failed, 0 errors**.
