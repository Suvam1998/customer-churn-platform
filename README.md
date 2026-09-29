# Real-Time Customer Churn Prediction & Intelligent Retention Platform

> M.Tech AI & Data Science capstone — an end-to-end platform that predicts
> customer churn, explains predictions, estimates revenue at risk, and
> generates intelligent retention recommendations, with a simulated
> real-time event layer.

**Data provenance (important):** Customer records and historical churn labels
come from the **real IBM Telco Customer Churn dataset**. The **real-time events**
in this platform are **SIMULATED**, because the source dataset is historical
rather than a live event stream. Simulated components are always labelled as
such.

---

## Build status (by phase)

The project is built incrementally in 27 phases (see the master plan).

| Phase | Description | Status |
|------:|-------------|--------|
| 1 | Project foundation (structure, config, health API, dashboard shell, tests) | ✅ **Complete** |
| 2 | Real dataset ingestion & profiling | ✅ **Complete** |
| 3 | Data validation & preprocessing | ✅ **Complete** |
| 4 | EDA | ✅ **Complete** |
| 5 | Feature engineering | ✅ **Complete** |
| 6 | Baseline model | ✅ **Complete** |
| 7 | Advanced models | ⏳ NOT YET EXECUTED |
| 8 | Hyperparameter tuning | ⏳ NOT YET EXECUTED |
| 9 | Calibration | ⏳ NOT YET EXECUTED |
| 10 | Explainable AI (SHAP) | ⏳ NOT YET EXECUTED |
| 11 | Customer segmentation | ⏳ NOT YET EXECUTED |
| 12 | CLV & revenue-at-risk | ⏳ NOT YET EXECUTED |
| 13 | Retention engine | ⏳ NOT YET EXECUTED |
| 14 | PostgreSQL | ⏳ NOT YET EXECUTED |
| 15 | FastAPI (full) | ⏳ NOT YET EXECUTED |
| 16 | Real-time event simulator | ⏳ NOT YET EXECUTED |
| 17 | Kafka integration | ⏳ NOT YET EXECUTED |
| 18 | Streamlit dashboard (full) | ⏳ NOT YET EXECUTED |
| 19 | MLflow | ⏳ NOT YET EXECUTED |
| 20 | Monitoring & drift | ⏳ NOT YET EXECUTED |
| 21 | Databricks | ⏳ NOT YET EXECUTED |
| 22 | Snowflake | ⏳ NOT YET EXECUTED |
| 23 | Docker | ⏳ NOT YET EXECUTED |
| 24 | Testing | ⏳ NOT YET EXECUTED |
| 25–27 | Integration, documentation, demo | ⏳ NOT YET EXECUTED |

No model metrics, SHAP values, or financial figures are reported yet because
those phases have not been executed. This project follows a strict **honesty
rule**: every reported number originates from actual execution.

### Phase 2 — dataset profile (executed)

Downloaded from the public IBM mirror into `data/raw/telco_customer_churn.csv`.
Reproduce with `python scripts/download_data.py`. Artifacts:
`results/data_profile.json`, `docs/data_dictionary.md`.

| Fact | Value |
|------|-------|
| Rows | 7,043 |
| Columns | 21 |
| Duplicate rows / IDs | 0 / 0 |
| Missing (NaN) | none |
| `TotalCharges` blanks | 11 (all `tenure == 0`; stored as text, handled in preprocessing — not dropped) |
| Target `Churn` = No | 5,174 (73.46%) |
| Target `Churn` = Yes | 1,869 (26.54%) |

Class imbalance (~26.5% positive) is confirmed and drives the Phase 16 imbalance
handling and the choice of PR-AUC / recall over accuracy.

### Phase 3 — validation & preprocessing (executed)

Reproduce with `python scripts/validate_data.py`. Artifact:
`results/validation_report.json`.

- **Validation:** all 24 schema/quality checks PASS (required columns, unique
  IDs, valid categorical/target values, non-negative charges, documented
  `TotalCharges` blanks).
- **Cleaning (documented, non-leaky):** `TotalCharges` text→numeric with the 11
  `tenure == 0` blanks set to `0.0`; `Churn` encoded Yes=1/No=0; **no rows
  dropped**.
- **Split (stratified 70/15/15, seed 42):** train 4,929 / val 1,057 / test 1,057,
  churn rate 0.2654 / 0.2658 / 0.2649.
- **Preprocessing:** `ColumnTransformer` (median-impute+scale numerics;
  most-frequent-impute+one-hot categoricals) → **45 features**, fit on **train
  only** (leakage-safe), `handle_unknown="ignore"` for unseen categories.

### Phase 4 — EDA (executed)

Reproduce with `python scripts/run_eda.py`. Artifacts: 12 figures in
`docs/figures/` + `docs/eda_summary.md`.

Key findings (real numbers):

| Signal | Finding |
|--------|---------|
| Contract | Month-to-month **42.7%** churn vs one-year **11.3%** vs two-year **2.8%** |
| Tenure ↔ churn | Pearson **−0.35** (longer tenure → lower churn) |
| MonthlyCharges ↔ churn | **+0.19** (higher monthly charges → more churn) |
| TotalCharges ↔ churn | **−0.20**; note tenure↔TotalCharges **0.83** (collinearity) |

These motivate the Phase 5 engineered features (tenure bands, contract/payment
risk indicators, per-month charge ratios).

### Phase 5 — feature engineering (executed)

Reproduce with `python scripts/build_features.py`. Artifacts:
`data/features/features.parquet`, `docs/feature_dictionary.md`.

- **13 engineered features** (row-wise, leakage-safe): `average_charge_per_month`,
  `total_services`, `service_adoption_score`, `is_month_to_month`,
  `is_long_term_contract`, `has_tech_support/online_security/online_backup/`
  `device_protection`, `has_streaming`, and `payment/contract/tenure_risk_indicator`.
- Base 45 → **58 preprocessed features** with the engineered set, still fit on
  **train only**.
- **Leakage guarantee:** every feature is a pure function of the customer's own
  row (no target, no cross-row/global statistics); a test asserts features are
  identical with the target column removed. Risk indicators encode telecom
  domain knowledge, not label-fitted thresholds.
- The base vs. engineered sets are both available via
  `prepare_data(include_engineered=...)` for the Phase 52 A/B experiments.

### Phase 6 — baseline Logistic Regression (executed)

Reproduce with `python scripts/train_baseline.py`. Metrics reported on the
**validation** set (test untouched). Artifacts: `results/baseline_metrics.json`,
`docs/figures/baseline/` (confusion, ROC, PR, calibration).

| Model (validation) | ROC-AUC | PR-AUC | Precision | Recall | F1 | Log Loss | Brier |
|--------------------|--------:|-------:|----------:|-------:|---:|---------:|------:|
| LogReg (base features) | 0.8297 | 0.6282 | 0.6388 | 0.5160 | 0.5709 | 0.4358 | 0.1408 |
| LogReg (engineered) — **baseline** | **0.8338** | **0.6448** | 0.6651 | 0.5160 | 0.5812 | 0.4308 | 0.1388 |

- **RQ1 (Experiment A vs B):** engineered features improve ROC-AUC **+0.0041**
  and PR-AUC **+0.0166** over base features — a modest but consistent gain.
- Accuracy is deliberately not the headline (imbalanced data); ROC-AUC / PR-AUC /
  recall / Brier lead. LR is already fairly well-calibrated (Brier 0.139),
  giving Phase 9 a reference point.
- Confusion @0.5 (engineered): TN=703, FP=73, FN=136, TP=145 — recall 0.52 at
  the default threshold motivates the Phase 54 threshold analysis.

---

## Research questions

- **RQ1** — How much do engineered behavioral features improve churn prediction?
- **RQ2** — Which ML model provides the best predictive and calibrated performance?
- **RQ3** — How does customer value change retention prioritization vs. churn probability alone?
- **RQ4** — Can simulated real-time behavioral events update customer risk scores?
- **RQ5** — How can Explainable AI improve the interpretability of churn predictions?
- **RQ6** — How does model drift affect a deployed churn prediction system?

---

## Technology stack

Python · Pandas · NumPy · PyArrow · PySpark/Delta · scikit-learn · XGBoost ·
LightGBM · CatBoost · (TF/PyTorch) · SHAP · FastAPI · Pydantic · Uvicorn ·
PostgreSQL/SQLAlchemy · Kafka (with local fallback) · Streamlit · Plotly ·
MLflow · Evidently · Docker · pytest · Snowflake-compatible SQL.

---

## Repository layout

```
customer-churn-platform/
├── api/            FastAPI app (main, routes, schemas, services)
├── dashboard/      Streamlit app (pages, components)
├── src/            Library code (ingestion, validation, preprocessing,
│                   features, models, explainability, segmentation,
│                   retention, streaming, monitoring, config, risk)
├── configs/        config.yaml + model_config.yaml (all thresholds live here)
├── database/       PostgreSQL schema + seed
├── mlops/          train / evaluate / register / retrain pipelines
├── databricks/     PySpark Bronze→Silver→Gold pipelines
├── snowflake/      Snowflake-compatible SQL (optional)
├── notebooks/      EDA & modelling notebooks
├── tests/          pytest suite
├── scripts/        setup / download / train / run entrypoints
├── docs/           architecture, data dictionary, methodology, research
└── data/           raw / processed / features / samples
```

---

## Python version note

The primary objective is reproducibility. The development host runs **Python
3.14**, but several ML libraries (CatBoost, TensorFlow, numba→SHAP) may not yet
publish wheels for 3.14. To keep the platform reproducible:

- **Recommended:** use **Python 3.11 or 3.12** for the local virtualenv, or run
  via Docker (the image pins Python 3.12).
- Phase 1 (foundation) installs and runs cleanly on Python 3.10–3.14 using only
  the CORE dependency group.
- Heavier dependencies are exercised in their own phases and version-checked
  against the interpreter at that time.

---

## Quickstart (Windows PowerShell)

```powershell
# from the customer-churn-platform/ directory

# 1. Create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (Phase 1 needs only the CORE group; see note above)
pip install -r requirements.txt

# 3. Configure environment
Copy-Item .env.example .env

# 4. Run the API
uvicorn api.main:app --reload
#    -> http://localhost:8000/health   and   http://localhost:8000/docs

# 5. Run the dashboard (new terminal, venv activated)
streamlit run dashboard/app.py
#    -> http://localhost:8501

# 6. Run tests
pytest
```

### Docker

```powershell
docker compose up --build           # api + dashboard + postgres + mlflow
docker compose --profile streaming up --build   # + kafka + zookeeper
```

---

## License

MIT (see `LICENSE`). The IBM Telco Customer Churn dataset is used under IBM's
educational/sample terms; refer to the dataset source for details.
