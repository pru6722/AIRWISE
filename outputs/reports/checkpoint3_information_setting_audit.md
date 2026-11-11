# Checkpoint 3 Information Setting Audit: Causal Streaming vs. Offline Retrospective Evaluation

**Audit Scope:** Distinction between causal temporal autoregression (Joint TCN) and offline bidirectional interpolation/imputation baselines.  
**Auditor:** Senior Statistical Machine Learning Researcher  
**Date:** November 2026  

---

## 1. Information Filtration Definitions

In comparative missing-data reconstruction, models cannot be fairly evaluated without formalizing their underlying filtration (information set available at prediction time $t$).

### A. Causal Online Streaming Setting (Joint TCN)
- **Filtration:** $\mathcal{F}_t = \sigma(\{\mathbf{X}_s, M_s\}_{s \le t})$.
- **Padding:** Strictly left-padded causal 1D temporal convolutions with pad length $(k - 1) \times d$.
- **Temporal Horizon:** Output $\widehat{Y}_t$ depends **exclusively on historical and current inputs** up to time $t$.
- **Real-World Deployment Scenario:** Real-time continuous sensor stream. When a sensor goes offline at $t_0$, the system must forecast reference-equivalent PM2.5 without knowing how long the outage will last or what value the sensor will read when it resumes service at $t_{\mathrm{end}}$.
- **Structural Constraint:** Zero lookahead; cannot use future observations $s > t$.

### B. Offline Retrospective Archive Setting (Two-Stage Baselines: Linear Interpolation, KNN, Random Forest Imputer)
- **Filtration:** $\mathcal{F}_{\mathrm{all}} = \sigma(\{\mathbf{X}_s, M_s\}_{s \in [0, T]} \setminus \{\mathbf{X}_{\mathrm{outage}}\})$.
- **Temporal Horizon:** Imputation is performed on historical data tables where both boundary endpoints—before the outage ($t_0 - 1$) and after service restoration ($t_{\mathrm{end}} + 1$)—are fully known.
- **Real-World Deployment Scenario:** Post-hoc environmental epidemiological studies, compliance auditing, or regulatory reporting where complete yearly batch archives are processed retrospectively.
- **Structural Advantage:** Two-Stage Linear Interpolation computes:
  $$\widehat{Y}(t) = Y(t_0 - 1) + \frac{t - (t_0 - 1)}{t_{\mathrm{end}} + 1 - (t_0 - 1)} \left[ Y(t_{\mathrm{end}} + 1) - Y(t_0 - 1) \right]$$
  This leverages **bidirectional boundary anchoring**, bypassing the physical autoregression problem during short random gaps.

---

## 2. Experimental Fairness & Reconciliation

1. **Why Two-Stage Linear Interpolation Achieves Lower Absolute Error:**
   - In short outages (1–6 hours), adjacent atmospheric PM2.5 concentrations exhibit high temporal autocorrelation. Bidirectional interpolation anchors the missing interval at both ends, yielding low RMSE (36.93–38.79 $\mu\text{g}/\text{m}^3$).
   - The causal Joint TCN operates with one arm tied behind its back: it has no knowledge of $Y(t_{\mathrm{end}} + 1)$ and must reconstruct the trajectory blindly via historical dilated convolutions and contemporaneous meteorology.
2. **Methodological Recommendation for Checkpoint 4:**
   - Causal streaming models and offline bidirectional interpolators belong to **fundamentally different information settings**.
   - In Checkpoint 4 conformal uncertainty estimation, conformal intervals for causal streaming predictions must be calibrated and reported under causal filtration $\mathcal{F}_t$.
   - Retrospective offline intervals must be clearly labeled as such, preventing deceptive claims of algorithmic superiority between non-equivalent information regimes.
