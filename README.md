# AIRWISE — Adaptive Imputation and Robust Windowed Inference for Sensor Estimation

**Assessing Sequential versus Joint Deep Architectures for Missing Data and Regulatory Calibration in Low-Cost Air Quality Networks**

AIRWISE is an end-to-end research framework, benchmark suite, and interactive Streamlit analytics platform comparing decoupled sequential calibration/imputation pipelines against strictly causal multi-task Temporal Convolutional Networks (TCNs), block conformal uncertainty quantification, CPCB $\text{PM}_{2.5}$ regulatory category reliability, and Spark/Hive columnar storage scaling across paired hourly PurpleAir and regulatory BAM-1020 observations in India.

---

## Core Research Findings

- **Central Finding:** **No single reconstruction strategy is uniformly optimal across missingness regimes** across the independent test set ($N = 3,720$) spanning Bangalore, Delhi, and Hamirpur.
- **Pipeline B ($\text{Calibrate} \to \text{Impute}$) leads overall:** Attains the lowest full-population RMSE (`36.26–41.64 µg/m³`, $R^2$: `0.776–0.831`) across all nine pre-registered missingness scenarios (`20% MCAR`, `Block 1h/3h/6h/12h/24h`, `MAR-RH`, `MAR-Lag`, `MNAR-PM`).
- **Pipeline C (Causal Joint TCN v2) vs. Multi-Task Coupling:** Strictly causal dilated convolutions (`ChannelLayerNorm1d`, receptive field $RF = 61\text{h} \le L = 64\text{h}$, `144/144` future-perturbation causality checks with $\max|\Delta| = 0.0000$) outperform Pipeline B on the humidity-conditioned missing subgroup (`MAR-RH`: `43.05` vs. `51.77 µg/m³`), but multi-task gradient interference degrades clean common-support accuracy (`38.01` vs. `29.82 µg/m³`) and compounds errors over long outages (`81.86` vs. `56.19 µg/m³` under nominal 24h block dropouts, whose merged continuous outages reach up to `63h`, mean `32.61h`).
- **Pipeline A ($\text{Impute} \to \text{Calibrate}$) & Pipeline D (Complete-Case Reference):** Pipeline A provides a consistent middle-tier full-coverage baseline (`40.03–50.40 µg/m³`), while Pipeline D demonstrates the uncorrupted sensor error floor (`29.82 µg/m³` on `MCAR_20pct`) at only `79.33%–80.00%` prediction availability.
- **Moving-Block Bootstrap Sensitivity ($B = 1,000$, $b \in \{12, 24, 48, 72\}\text{h}$):** Paired moving-block bootstrap confidence intervals that exclude zero under short `12h` blocks can cross zero once the `24h` diurnal cycle is preserved; empirical tail proportions ($\hat{p}_{\text{tail}}$) are treated as sensitivity diagnostics rather than formal null-hypothesis $p$-values.
- **Conformal Uncertainty & Extreme Haze Saturation (Historical Checkpoint 4, `frozen_joint_tcn_checkpoint.pt`):** Lag-1 residual autocorrelation drops `90%` ordinary split coverage to `77.02%` (`width 51.40 µg/m³`); at `90%` nominal, block conformal yields `86.40%` pointwise vs. `63.83%` (`120/188`) simultaneous full-block coverage at `12h`, and `95.70%` pointwise vs. `74.58%` (`44/59`) simultaneous coverage at `24h`. Winter haze (`BAM > 250 µg/m³`, $n = 257$) saturates optical scattering and collapses `90%` coverage to `1.95%`.
- **CPCB 24h Band Reliability & Big-Data Scaling:** Across `127` valid (`>= 18/24h`) 24-hour test windows (`PM2.5`-only band evaluation, not full 8-pollutant NAQI), calibrated accuracy reaches `72.44%` (`RMSE = 25.39`, $R^2 = 0.8798$), but `Severe (>250 µg/m³)` recall is `0 / 4`. PySpark/Hive storage benchmarks reduce footprint from `2.126 MB` (CSV) to `0.573 MB` (Parquet, `-73.0%`) and `0.338 MB` (ORC, `-84.1%`).

---

## Authoritative Dataset & Chronological Partition

Source dataset: paired PurpleAir PA-II (`PA_CF1`, `PA_ATM`, `PA_ALT`, `RH`, `T`) and regulatory MetOne BAM-1020 hourly measurements (`N = 18,584` cleaned hours, Dryad [DOI: 10.6078/D1RQ70](https://doi.org/10.6078/D1RQ70)).

| Partition | Bangalore | Delhi | Hamirpur | Total | Ratio | Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | 4,464 | 4,502 | 2,183 | **11,149** | 60.00% | Calibration, imputation & normalizer fitting |
| **Validation** | 744 | 750 | 363 | **1,857** | 9.99% | Model selection & early stopping |
| **Conformal Calib** | 744 | 750 | 364 | **1,858** | 10.00% | Split & temporal block conformal calibration |
| **Test** | 1,489 | 1,502 | 729 | **3,720** | 20.01% | Independent final benchmark evaluation |
| **Total Cleaned** | **7,441** | **7,504** | **3,639** | **18,584** | 100.00% | Preserved segment boundaries (`cleaned_data.parquet`) |

---

## Repository Structure

```text
AIRWISE/
├── configs/                        # YAML dataset, experiment, and model configurations
├── dashboard/                      # Interactive 8-page Streamlit analytics platform
│   ├── app.py                      # Home / Research landing page
│   ├── pages/                      # Pages 01–07 (Overview, Explorer, Lab, TCN, Uncertainty, CPCB, Big Data)
│   ├── components/                 # Sidebar filters, Plotly chart factory, architecture diagrams, KPI cards
│   ├── services/                   # Cached data, TCN inference, calibration, uncertainty, CPCB & DuckDB SQL services
│   ├── styles/                     # Custom dark-mode CSS styling
│   ├── utils/                      # Canonical paths, formatting helpers, input validation
│   └── requirements-dashboard.txt  # Dashboard Python dependencies
├── data/
│   ├── raw/                        # Campmier_Manuscript_Data.xlsx (Dryad DOI: 10.6078/D1RQ70)
│   ├── processed/                  # cleaned_data.parquet (18,584 hourly records)
│   ├── synthetic_masks/            # Pre-registered MCAR benchmark mask
│   └── benchmark_scales/           # 1x Parquet/CSV included (5x–50x omitted; see below)
├── outputs/
│   ├── experiments/                # Phase 3A & Phase 3C frozen parquet, CSVs, bootstrap results & figures
│   ├── figures/                    # Checkpoint 1–6 publication figures
│   ├── logs/                       # Training logs & experiment configurations
│   ├── metrics/                    # Checkpoint 5 CPCB & Checkpoint 6 Spark/storage metrics
│   ├── models/                     # Frozen checkpoints (Joint TCN v2, Checkpoint 4 Conformal, Calibration baselines)
│   ├── predictions/                # Historical model test prediction tables
│   ├── reports/                    # Final manuscript (v3), phase audits, and methodology reports
│   └── spark-warehouse/            # Hive warehouse tables (pm25_research.db) queried via DuckDB
├── src/
│   ├── aqi/                        # CPCB PM2.5 breakpoints, 24h completeness windowing & category evaluator
│   ├── bigdata/                    # PySpark ingestion, preprocessing, aggregation, Hive & benchmarking
│   ├── data/                       # Ingestion, audit validation, preprocessing & cyclical feature engineering
│   ├── evaluation/                 # Phase 3C benchmark runner, block bootstrap, propagation & conformal evaluators
│   ├── missingness/                # MCAR, contiguous block, MAR-RH, MAR-Lag & MNAR-PM mask generators
│   ├── models/                     # Calibration regressors, imputers, sequential Pipelines A/B/D & Causal Joint TCN v2
│   ├── training/                   # Joint TCN v1 & causal v2 training pipelines
│   ├── uncertainty/                # Chronological 4-way split manager, split/regime & temporal block conformal
│   └── visualization/              # Matplotlib publication plotting modules
├── tests/                          # 111 unit, causality, input-isolation, SQL security, SHA-256 & UI tests
└── run_checkpoint[1-6].py          # End-to-end checkpoint orchestration scripts
```

---

## Quickstart

### 1. Environment & Dependencies

Requires Python 3.10+ (developed and verified on Python 3.13) with core scientific ML dependencies (`torch`, `scikit-learn`, `xgboost`, `pandas`, `numpy`, `pyarrow`, `openpyxl`, `joblib`, `matplotlib`, `seaborn`, `pyyaml`, `pyspark`) and dashboard dependencies:

```bash
pip install -r dashboard/requirements-dashboard.txt
```

### 2. Launch the Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard runs locally on `http://localhost:8501` and exposes eight integrated views:
- **Home** & **01 Research Overview**: Four-way dataset partition, four-pipeline architecture, and reconciled findings.
- **02 Dataset Explorer**: Independent PM-channel availability, partial/complete/station outages, heatmaps, and exports.
- **03 Calibration & Reconstruction Lab**: All 4 pipelines across all 9 missingness scenarios with full/missing/common-support modes and paired moving-block bootstrap sensitivity (`12h–72h`).
- **04 Joint TCN Explorer**: `ChannelLayerNorm1d` architecture, 16-epoch losses, `144` causality checks, and BAM-isolated synthetic outage demonstration.
- **05 Uncertainty Analysis**: 90%/95% intervals, pointwise vs. simultaneous `12h` (`63.83%`) & `24h` (`74.58%`) block coverage, and `>250 µg/m³` saturation diagnostics.
- **06 CPCB Reliability**: 24-hour (`>= 18/24h`) regulatory PM2.5 confusion matrices, `0/4` Severe recall diagnostics, and boundary stability analysis.
- **07 Big Data Analytics**: PySpark throughput/storage comparisons and guarded read-only DuckDB SQL console over `outputs/spark-warehouse/pm25_research.db`.

### 3. Run the Test Suite

```bash
python3 -m unittest discover tests -v
```

Verifies `111` tests covering data leakage isolation, future-perturbation causality gates, moving-block bootstrap sensitivity, CPCB window rules, DuckDB SQL injection guardrails, simulator BAM isolation, SHA-256 integrity of frozen artifacts, and all 8 Streamlit pages via `AppTest`. *(Note: inside restricted environments without loopback sockets, PySpark's `test_spark_and_pandas_numerical_consistency` skips cleanly.)*

---

## Frozen Artifact Provenance & Excluded Files

### Preserved SHA-256 Checksums

| Artifact | Role | SHA-256 |
| :--- | :--- | :--- |
| `outputs/models/joint_tcn/joint_tcn_v2_causal_checkpoint.pt` | Phase 3C Causal Joint TCN v2 | `4a0c5704257fd556e53cfc23b1c26439881009f8aabb423b3f6f2752e7c5217c` |
| `outputs/models/conformal/frozen_joint_tcn_checkpoint.pt` | Checkpoint 4/5 Historical TCN | `3eae6ea2b0f6c7abdbca72df672c61f068d78b4e49c9006e0eb54ac8a2fa7350` |
| `outputs/experiments/phase3C/benchmark_predictions.parquet` | Phase 3C Benchmark Predictions | `87e5263742403644d96605f9bc742de6e5009a29d8eb5509c149b8b154a4f662` |

### Regenerating Excluded Benchmark Scales

Synthetic multi-scale benchmark datasets `5x`, `10x`, `25x`, and `50x` under `data/benchmark_scales/` (`scale_50x.csv` alone is `122 MB`, exceeding GitHub's `100 MB` file limit) are excluded via `.gitignore` along with Spark Derby runtime lock files (`metastore_db/`, `derby.log`), temporary test files (`data/benchmark_scales_test/`, `scratch/`), and the duplicate raw download folder `doi_10_6078_D1RQ70__v20230814/`. `scale_1x.parquet` and `scale_1x.csv` are included, and larger scale files can be deterministically regenerated locally by running:

```bash
python3 run_checkpoint6.py
```

---

## Citation & Data Attribution

- Underlying collocated field measurement dataset: Campmier, M. et al. (2023), *Dryad Digital Repository* (`DOI: 10.6078/D1RQ70`; *Atmospheric Measurement Techniques*, `10.5194/amt-2023-35`).
- Full technical manuscript and empirical tables: [`outputs/reports/airwise_phase3C_final_manuscript_v3.md`](outputs/reports/airwise_phase3C_final_manuscript_v3.md).
