# Checkpoint 3 Limitations and Methodological Boundaries (Post-Review Audit)

This document formalizes the unresolved scientific and engineering limitations identified during Checkpoint 3 (Joint TCN development and evaluation), updated following detailed post-review empirical diagnostics.

---

## 1. Deconstructed Extreme Pollution Failure (Concentrations Above 120 and 250 µg/m³)
- In Delhi test observations above $250\,\mu\text{g}/\text{m}^3$ ($N=257$), the Joint TCN exhibits an MBE of $-111.16\,\mu\text{g}/\text{m}^3$ (RMSE $124.68\,\mu\text{g}/\text{m}^3$), while Random Forest exhibits an MBE of $-78.66\,\mu\text{g}/\text{m}^3$ and LSTM exhibits $-106.6\,\mu\text{g}/\text{m}^3$.
- Quantitative diagnostics confirm this is the combined result of three distinct mechanisms:
  1. **Optical Sensor Saturation:** The raw PurpleAir response slope collapses by 50% from $1.007$ (for $\le 120\,\mu\text{g}/\text{m}^3$) to $0.506$ (for $>250\,\mu\text{g}/\text{m}^3$). Low-cost Plantower sensors undergo particle coincidence loss and severe beam extinction.
  2. **Extreme Distribution Shift:** Hours with $\text{BAM} > 250\,\mu\text{g}/\text{m}^3$ constituted only 3.2% of the Delhi training partition (144 hours) but surged 5.3-fold to 17.1% of the test partition (257 hours) during the late 2019 winter inversion crisis.
  3. **Conditional Mean Shrinkage:** Standard symmetric L2/Huber losses heavily penalize variance, shrinking predictions toward the global conditional training mean ($\sim 45\,\mu\text{g}/\text{m}^3$). The Joint TCN output ceiling was capped at $272.7\,\mu\text{g}/\text{m}^3$ despite true reference BAM peaking at $508.0\,\mu\text{g}/\text{m}^3$.

---

## 2. Multi-Task Pareto Tradeoff (Capacity vs. Inductive Regularization)
- Removing reconstruction loss ($\lambda_{\mathrm{rec}}=0$) obviously damages reconstruction ($102.94\,\mu\text{g}/\text{m}^3$), but comparing single-task specialists reveals a clear Pareto frontier:
  - **Calibration Specialist (Config E):** Achieves **38.76 RMSE** on observed points.
  - **Full Joint TCN (Config A):** Degrades calibration to **45.84 RMSE** (+18.3% error increase).
  - **Reconstruction Specialist (Config F):** Achieves **66.13 RMSE** on hidden points.
  - **Full Joint TCN (Config A):** Improves reconstruction to **56.76 RMSE** (-14.2% error reduction).
- Joint training is **not a cost-free win-win**: it acts as a valuable inductive regularizer for reconstruction, but directly taxes calibration accuracy by forcing the shared encoder to represent missingness dynamics.

---

## 3. Propagation Gap vs. Absolute Accuracy
- Evaluating error growth $G(g) = \text{RMSE}(g) - \text{RMSE}(1\text{h})$ in isolation is scientifically deceptive:
  - Joint TCN displays a flatter error growth slope from 1h to 12h ($G(12) = +6.63\,\mu\text{g}/\text{m}^3$) than Linear Interpolation ($+13.79\,\mu\text{g}/\text{m}^3$).
  - However, **Two-Stage Random Forest and Linear Interpolation achieve substantially lower absolute RMSE at every single outage duration** (RF: 36.57 at 1h, 48.43 at 24h vs. Joint TCN: 67.10 at 1h, 84.86 at 24h).
  - A lower rate of error increase does not compensate for an 83% higher initial error baseline.

---

## 4. Temporal Causality Asymmetry
- In streaming operational networks, causal convolution ($t \le t_{\mathrm{now}}$) is strictly required to prevent future leakage.
- However, in retrospective environmental epidemiology, missing data occurs between known historical boundaries ($t_{\mathrm{start}}$ and $t_{\mathrm{end}}$). Conventional two-stage interpolation utilizes bidirectional lookahead, granting it an inherent structural advantage on short gaps that a purely causal model cannot match.

---

## 5. Single Collocated Sensor Representation
- Each study location possesses only a single collocated sensor unit.
- Inter-sensor manufacturing variances, aging drift across identical production batches, and microclimatic spatial gradients across a city cannot be evaluated from this dataset.

---

## 6. Deterministic Point Prediction Deficit (Bridge to Checkpoint 4)
- All evaluated models produce single scalar outputs ($\widehat{Y}_t$) with zero quantification of confidence or prediction variance.
- When predictions in Delhi drop from $500\,\mu\text{g}/\text{m}^3$ to $270\,\mu\text{g}/\text{m}^3$, a point prediction provides false certainty that could jeopardize public health advisories.
- **Direct Motivation for Checkpoint 4:** We must develop **Regime-Conditioned Conformal Prediction** to wrap these models with distribution-free, finite-sample coverage guarantees:
$$P\left(Y_t \in \left[\widehat{Y}_t - \widehat{C}_t(\text{Regime}), \widehat{Y}_t + \widehat{C}_t(\text{Regime})\right]\right) \ge 1 - \alpha$$
