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
| 7 | Advanced models | ✅ **Complete** |
| 8 | Hyperparameter tuning | ✅ **Complete** |
| 9 | Calibration | ✅ **Complete** |
| 10 | Explainable AI (SHAP) | ✅ **Complete** |
| 11 | Customer segmentation | ✅ **Complete** |
| 12 | CLV & revenue-at-risk | ✅ **Complete** |
| 13 | Retention engine | ✅ **Complete** |
| 14 | PostgreSQL | ✅ **Complete** |
| 15 | FastAPI (full) | ✅ **Complete** |
| 16 | Real-time event simulator | ✅ **Complete** |
| 17 | Kafka integration | ✅ **Complete** |
| 18 | Streamlit dashboard (full) | ✅ **Complete** |
| 19 | MLflow | ✅ **Complete** |
| 20 | Monitoring & drift | ✅ **Complete** |
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

### Phase 7 — advanced models (executed)

Reproduce with `python scripts/train_models.py`. All 7 models trained on the
**same engineered split**, evaluated on **validation** (test untouched).
Artifacts: `results/model_comparison.{csv,json}`,
`docs/figures/models/model_comparison.png`. Values are **untuned defaults**
(tuning is Phase 8).

| Model (validation) | ROC-AUC | PR-AUC | Precision | Recall | F1 | Log Loss | Brier | Train s |
|--------------------|--------:|-------:|----------:|-------:|---:|---------:|------:|--------:|
| random_forest | **0.8346** | 0.6424 | 0.6881 | 0.4947 | 0.5756 | 0.4294 | 0.1380 | 0.56 |
| logistic_regression | 0.8338 | **0.6448** | 0.6651 | 0.5160 | 0.5812 | 0.4308 | 0.1388 | 0.14 |
| catboost | 0.8327 | 0.6379 | 0.6481 | 0.4982 | 0.5634 | 0.4350 | 0.1390 | 0.90 |
| neural_network (MLP) | 0.8305 | 0.6389 | 0.6931 | 0.4662 | 0.5574 | 0.4342 | 0.1398 | 0.38 |
| xgboost | 0.8265 | 0.6321 | 0.6806 | 0.5231 | **0.5915** | 0.4499 | 0.1426 | 0.51 |
| decision_tree | 0.8210 | 0.5856 | 0.6284 | 0.4093 | 0.4957 | 0.4468 | 0.1452 | 0.07 |
| lightgbm | 0.8201 | 0.6294 | 0.6256 | 0.5053 | 0.5591 | 0.4680 | 0.1473 | 0.38 |

- **RQ2 (provisional):** scores cluster tightly (ROC-AUC 0.820–0.835) — realistic
  for Telco churn, where LR is competitive with boosting. RF leads ROC-AUC, LR
  leads PR-AUC, XGBoost leads F1. Final selection follows Phase 8 (tuning) + 9
  (calibration), on validation.
- **Neural network:** scikit-learn `MLPClassifier` is used (not TensorFlow/PyTorch)
  to keep the platform reproducible on Python 3.14 — a documented substitution,
  not a fabrication.
- **Python 3.14 note:** XGBoost, LightGBM, and CatBoost all installed and ran on
  3.14 (nothing skipped); the registry still skips-with-note if a wheel is ever
  unavailable.

### Phase 8 — hyperparameter tuning (executed)

Reproduce with `python scripts/tune_models.py`. `RandomizedSearchCV` with
**3-fold CV on the training split**; models compared on **validation**; test
untouched. Artifacts: `results/tuning_results.json`,
`models/tuned_best_model.joblib` (+ meta).

| Model | CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Val F1 | Val Brier | Tune s |
|-------|-----------:|------------:|-----------:|-------:|----------:|-------:|
| **catboost** (best) | 0.8469 | **0.8371** | 0.6442 | 0.5762 | **0.1372** | 8.5 |
| xgboost | 0.8461 | 0.8369 | 0.6405 | 0.5878 | 0.1377 | 2.7 |
| logistic_regression | 0.8442 | 0.8338 | 0.6437 | 0.5737 | 0.1389 | 10.5 |
| random_forest | 0.8464 | 0.8332 | 0.6381 | **0.6104** | 0.1657 | 7.2 |
| lightgbm | 0.8349 | 0.8310 | 0.6333 | 0.5592 | 0.1410 | 2.9 |

- **RQ2:** tuning lifted XGBoost most (0.8265 → 0.8369). **CatBoost** leads
  validation ROC-AUC (0.8371) with the best Brier (0.1372) — best-calibrated of
  the strong models. Best CatBoost params: `lr=0.03, iterations=200, depth=4`.
- **Honest caveat:** RandomForest has the best F1 (0.6104) but the **worst Brier
  (0.1657)** — poorly calibrated. Since this platform needs trustworthy churn
  *probabilities*, calibration quality (Phase 9) matters as much as ranking.
- The best tuned pipeline is persisted as the model artifact for downstream
  phases (SHAP, retention, API). Test set still reserved for the final
  single evaluation.

### Phase 9 — probability calibration (executed)

Reproduce with `python scripts/calibrate_model.py`. Compares uncalibrated /
Platt (sigmoid) / isotonic on **validation** (Brier-selected; test untouched).
Artifacts: `results/calibration_results.json`,
`docs/figures/calibration/reliability_comparison.png`,
`models/production_model.joblib` (+ meta).

| Variant (validation) | ROC-AUC | PR-AUC | F1 | Log Loss | Brier |
|----------------------|--------:|-------:|---:|---------:|------:|
| **uncalibrated** (selected) | 0.8371 | 0.6442 | 0.5762 | 0.4266 | **0.1372** |
| platt_sigmoid | 0.8368 | 0.6445 | 0.5750 | 0.4308 | 0.1379 |
| isotonic | 0.8367 | 0.6423 | 0.5672 | 0.4614 | 0.1381 |

- **RQ (calibration):** CatBoost's **native probabilities are already
  well-calibrated** (Brier 0.1372). Neither Platt nor isotonic improves Brier or
  log loss on validation, so the data-driven choice is to keep the uncalibrated
  probabilities — an honest negative result, not a failure. The reliability
  overlay shows all three variants hugging the diagonal.
- **Production model:** `models/production_model.joblib`, version
  `catboost-uncalibrated-2026-09-29`, selected by validation Brier. This is the
  artifact SHAP, the retention engine, and the API load.

### Phase 10 — Explainable AI / SHAP (executed)

Reproduce with `python scripts/run_explainability.py`. `TreeExplainer` on the
tuned CatBoost; explanations are 100% SHAP-generated (nothing hardcoded).
Artifacts: `results/shap_global_importance.csv`,
`results/explanations_sample.json`, `docs/figures/shap/shap_summary.png`.

- **Global drivers (mean |SHAP|):** `tenure` (0.347) ≫ `InternetService_Fiber
  optic` (0.261), `TechSupport_No` (0.179), `OnlineSecurity_No` (0.172),
  `contract_risk_indicator` (0.160), `MonthlyCharges` (0.151),
  `is_month_to_month` (0.145). Engineered risk features rank among the top
  drivers, corroborating RQ1 and the EDA.
- **Local example — customer `7590-VHVEG`** (tenure 1, churn_prob **0.647**):
  top churn-increasing factors are low `tenure` (+0.79), `TotalCharges`,
  `TechSupport_No`, `OnlineSecurity_No`, `Electronic check`; protective factors
  include not having fiber optic and lower `MonthlyCharges`.
- **Correctness (RQ5):** a test verifies **SHAP additivity** —
  `sigmoid(base + Σ SHAP)` reproduces the model's predicted probability
  (atol 1e-2). Directionality on the beeswarm is correct (low tenure → higher
  churn SHAP).

### Phase 11 — customer segmentation (executed)

Reproduce with `python scripts/run_segmentation.py`. K-Means on
[tenure, MonthlyCharges, TotalCharges, service_adoption_score,
churn_probability]; **k=3 chosen by silhouette** (0.4261). Artifacts:
`results/segmentation_profile.json`, `data/features/segments.parquet`,
`docs/figures/segmentation/{k_selection,pca_clusters}.png`.

| Cluster | Size | Avg tenure | Avg monthly | Avg churn prob | Actual churn | Label |
|--------:|-----:|-----------:|------------:|---------------:|-------------:|-------|
| 1 | 2,392 (34%) | 11.1 | 75.08 | 0.5336 | 0.5464 | **High-value/high-risk** |
| 2 | 2,341 (33%) | 56.7 | 88.92 | 0.1490 | 0.1427 | High-value/low-risk |
| 0 | 2,310 (33%) | 29.8 | 29.60 | 0.1028 | 0.0987 | Low-value/low-risk |

- **k is data-driven** (Elbow + Silhouette), not assumed. Silhouette peaks at
  k=3; a distinct "Low-value/high-risk" cluster does not emerge in this data.
- Labels are assigned **after** profiling, relative to overall medians.
- **Cross-check:** each cluster's mean predicted churn probability closely
  matches its **actual** churn rate (0.53↔0.55, 0.15↔0.14, 0.10↔0.10),
  independently validating both the model and the segmentation. Cluster 1
  (new, high-spend, high-risk) is the priority retention target — feeds
  Phase 12/13.

### Phase 12 — CLV & revenue-at-risk (executed)

Reproduce with `python scripts/run_value.py`. Artifacts:
`results/value_summary.json`, `data/features/customer_value.parquet`.
**All monetary values are ESTIMATES** from configurable assumptions
(`expected_lifetime_months=24`, `gross_margin=0.30`) — not observed financials.

- **Formulas:** `CLV = MonthlyCharges × 24 × 0.30`;
  `revenue_at_risk = churn_probability × CLV`; `priority_score = revenue_at_risk`.
- **Executed estimates:** total CLV ≈ **3,284,040**; total revenue-at-risk ≈
  **1,000,723** (currency units). Risk levels: LOW 5,621 / MEDIUM 926 / HIGH 415
  / CRITICAL 81.
- **By segment:** the High-value/high-risk cluster concentrates **~70%** of the
  estimated revenue-at-risk (705k of 1.00M) — the clear priority target.
- **RQ3 (value vs. probability):** ranking the top 10% by *revenue-at-risk*
  vs. *churn probability alone* overlaps only **77.3%** — **22.7% (160
  customers)** are newly prioritised once value is considered. Customer value
  materially changes retention prioritisation.

### Phase 13 — retention decision engine (executed)

Reproduce with `python scripts/run_retention.py`. Ordered, documented,
config-driven rules (**not** `if prob>0.5`) emit priority, urgency, action, and
reason from churn/value/contract/payment/support/segment signals. Business-sim
values are **estimates**. Artifacts: `results/retention_summary.json`,
`data/features/retention_recommendations.parquet`.

- **Actionable customers:** 1,422 of 7,043 (the other 5,621 are LOW risk →
  `no_intervention`).
- **Action mix:** payment_assistance 984, onboarding_assistance 302,
  account_manager_contact 81, plan_optimization 55.
- **Urgency:** LOW 5,621 / MEDIUM 926 / HIGH 415 / IMMEDIATE 81.
- **Business simulation (EST):** total intervention cost ≈ 48,990; expected
  retained value ≈ 188,608; net ≈ 139,618 — assuming a configurable 35%
  intervention effectiveness. Clearly labelled as an estimate, not a realised
  outcome.
- **Honest note:** the electronic-check payment driver is common among at-risk
  customers, so `payment_assistance` dominates the ordered ruleset on this
  dataset; the engine still supports the full action catalogue for other
  signal patterns.

### Phase 14 — PostgreSQL persistence (executed)

Reproduce with `python scripts/init_db.py`. Full schema
(`database/schema.sql`) + SQLAlchemy ORM (`src/db/models.py`) for 10 tables:
customers, subscriptions, customer_events, customer_features, predictions,
risk_scores, recommendations, model_versions, experiments, monitoring_metrics.

- **PostgreSQL-compatible, SQLite-fallback:** the session layer uses
  `DATABASE_URL` / `POSTGRES_*` if set, else a local SQLite file — so the
  platform runs with **no PostgreSQL installed**. Generic `JSON` columns work on
  both.
- **Seeded from real data + artifacts:** 7,043 customers & subscriptions from
  the IBM Telco dataset; 7,043 predictions/risk_scores/recommendations from the
  Phase 13 artifacts; 1 production model_version from the Phase 9 metadata.
- Events carry an `is_simulated` flag (real-time events are simulated —
  Phase 16+).

### Phase 15 — FastAPI (full) (executed)

Run with `uvicorn api.main:app --reload` → `http://localhost:8000/docs`.
Endpoints (all Pydantic-typed, proper status codes): `POST /predict`
(existing id or ad-hoc raw features), `POST /event`, `GET /customer/{id}`,
`/customer/{id}/explanation`, `/customer/{id}/recommendation`,
`GET /customers/high-risk`, `GET /dashboard/metrics`, `GET /model/metrics`,
`GET /monitoring/drift` (stub → Phase 20), `POST /retrain` (202, offline
pipeline → Phase 43).

- **Model service layer** (`api/services/model_service.py`) loads the production
  model once and precomputes a scored per-customer table (fast reads, DB-optional);
  SHAP explainer is lazy-loaded.
- **Live-verified** (real HTTP): `/dashboard/metrics` → 7,043 customers,
  churn 0.2654, 496 high-risk, revenue-at-risk ≈ 1,000,723, version
  `catboost-uncalibrated-2026-09-29`; `/predict` for `7590-VHVEG` → prob 0.6468,
  MEDIUM, `payment_assistance`, factors [tenure, TotalCharges, TechSupport No].
- `/predict` responses are fully dynamic; explanations come from SHAP (not
  hardcoded). `/event` returns a SIMULATED-labelled risk update (documented
  heuristic delta; full streaming re-inference is Phase 16/17).

### Phase 16 — real-time event simulator (executed)

Reproduce with `python scripts/run_event_simulation.py --n 25 --seed 7`.
Generates **SIMULATED** events over **real** customer IDs and updates risk.

- **Event generator** (`src/streaming/event_generator.py`): 9 event types
  (login, purchase, payment_failed, support_ticket, complaint, plan_upgrade,
  plan_downgrade, cancellation_attempt, inactivity) with weighted sampling and
  ISO timestamps; IDs are drawn only from the real dataset (never invented).
- **Risk-update flow** (`src/streaming/risk_update.py`): shared by the API and
  the simulator. Bounded, documented per-event deltas; `RiskState` accumulates
  successive events; each update yields previous/new probability, risk change,
  new risk level, revenue-at-risk, and a fresh recommendation.
- **Executed:** 25 events applied and persisted to `customer_events` +
  `risk_scores`; e.g. `7764-BDPEE` 0.151→0.301 (+0.15, payment_failed);
  `3027-ZTDHO` support_ticket kept CRITICAL → `account_manager_contact`.
- The API `/event` route was refactored onto this shared logic. Every record is
  flagged `is_simulated=True`.

### Phase 17 — Kafka integration (executed)

Reproduce with `python scripts/run_streaming.py --n 30`. Pipeline:
Event Generator → Producer → `customer-events` → Consumer → risk update →
`customer-risk-updates` → DB.

- **Broker abstraction** (`src/streaming/broker.py`): `KafkaBroker`
  (lazy `kafka-python`) and a **file-backed `LocalBroker`** default so it runs
  with **no Kafka installed**. `get_broker()` uses Kafka when
  `streaming.mode=kafka` + broker available, else falls back with a warning.
- **Producer/Consumer** (`producer.py`, `consumer.py`) are backend-agnostic;
  the consumer applies the shared `RiskState`, publishes to
  `customer-risk-updates`, and persists events/risk scores.
- **Executed (local backend):** produced 30 → `customer-events`, consumed →
  30 risk updates → `customer-risk-updates`; e.g. `3068-OMWZA` payment_failed
  0.922→1.0 CRITICAL → `account_manager_contact`.
- **Kafka mode:** `pip install kafka-python` + `docker compose --profile
  streaming up`, then set `STREAMING_MODE=kafka`.

### Phase 18 — Streamlit dashboard (executed)

Run with `streamlit run dashboard/app.py` (start the API first). Native
multipage layout with **7 pages**:

1. **Overview** — KPI tiles (customers, churn rate, high-risk, revenue-at-risk,
   avg prob, model version/metrics) + risk/churn charts + segment table.
2. **Customer Risk** — filterable table (risk, segment, tenure, min prob) with
   progress-bar churn column and recommendations.
3. **Customer 360** — profile, risk/value, SHAP explanation chart,
   recommendation, and a **simulate-event** control.
4. **Real-Time Monitor** — inject SIMULATED events, session event log, and
   recent `customer-risk-updates` from the streaming topic.
5. **Segmentation** — cluster profiles + PCA / k-selection figures.
6. **Model Performance** — production metrics, comparison table, and diagnostic
   figures (comparison, SHAP, ROC/PR/confusion, calibration).
7. **Monitoring** — drift status, data-quality report, prediction distribution.

- **Shared client** (`dashboard/lib/api_client.py`) calls the API and degrades
  gracefully (banner) when it's down; pages also read artifacts directly.
- **Validated with Streamlit `AppTest`** — every page renders headlessly with
  no uncaught exception (this caught and fixed a real broker-payload bug). The
  full stack was smoke-tested live (Streamlit + API both healthy).

### Phase 19 — MLflow tracking + registry (executed)

Reproduce with `python scripts/run_mlflow.py`; view with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

- **Experiment tracking:** logs **15 runs** into the `churn-retention`
  experiment — 7 comparison (Phase 7), 5 tuning (Phase 8), 3 calibration
  (Phase 9) — each with params, metrics, training duration, dataset version, and
  preprocessing version tags. Every metric is a real executed result.
- **Model registry:** the production model is logged (cloudpickle) and
  registered as **`churn-retention-model` v1**.
- **Stages via aliases:** MLflow 3.x removed model *stages*, so Development /
  Staging / Production are expressed as registry **aliases** + a `stage` tag.
- **Validation-gated promotion (no auto-promote):** promotion to `production`
  requires `roc_auc ≥ 0.80` **and** `brier ≤ 0.20`. Executed: **PASS**
  (roc_auc 0.8371, brier 0.1372) → promoted to production; otherwise it stays at
  `staging`.
- Backend upgraded to SQLite automatically (the registry needs a DB store, not a
  file store).

### Phase 20 — monitoring & drift (executed)

Reproduce with `python scripts/run_monitoring.py`. Custom **PSI** drift metrics
(Evidently optional), configurable thresholds, wired into
`GET /monitoring/drift` and the Monitoring page. Artifacts:
`results/monitoring_report.json`, `docs/figures/monitoring/drift_{real,simulated}.png`.

- **REAL (train vs test):** `STABLE` — 0/14 features drifted, prediction-drift
  PSI 0.007. Correct: a static dataset shows no real drift (reported honestly).
- **SIMULATED injected shift:** `RETRAINING_REQUIRED` — 4/14 features drifted
  (tenure, MonthlyCharges, PaymentMethod, Contract — exactly the injected ones),
  prediction-drift PSI 0.164 crosses the threshold. Clearly labelled synthetic,
  purely to demonstrate the detector firing.
- **Thresholds** are configurable (`monitoring.drift_threshold=0.15`,
  `max_drifted_share=0.30`). PSI bands: <0.1 stable, 0.1–0.25 moderate, >0.25
  significant.
- **`RETRAINING_REQUIRED` is a signal only — no model is auto-deployed**
  (promotion requires validation, Phase 42/43).

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
