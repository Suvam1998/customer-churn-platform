# 🎓 Capstone Presentation & Viva Voce Guide

## Real-Time Customer Churn Prediction & Intelligent Retention Platform using Databricks and Snowflake

**Degree:** M.Tech in AI, Data Science & Machine Learning  
**Project Title:** Real-Time Customer Churn Prediction & Intelligent Retention Platform using Databricks and Snowflake  

---

## 📽️ Slide-by-Slide Presentation Structure (10–15 Minutes)

### Slide 1: Title & Introduction
- **Title:** Real-Time Customer Churn Prediction & Intelligent Retention Platform
- **Subtitle:** An Enterprise-Grade End-to-End MLOps, Big Data Engineering & Explainable AI Solution
- **Key Highlights:** Databricks Medallion Architecture (Delta Lake), Snowflake Warehouse (`CUSTOMER_CHURN_DB`), Calibrated ML Models, SHAP Interpretability, Real-Time Event Simulation, Value-Driven Retention Engine, Streamlit Cloud Dashboard.

### Slide 2: Problem Statement & Motivation
- **Business Challenge:** Acquiring a new customer costs 5x–7x more than retaining an existing one. Telecom businesses lose millions due to unpredicted churn.
- **Industry Limitations:**
  - Traditional batch models predict once a month (too late when complaints happen).
  - Black-box ML models give no explanations to customer success teams.
  - Churn scores lack financial prioritization (high-value vs low-value churn).
  - Lack of unified data architecture between data lake, warehouse, and ML models.

### Slide 3: End-to-End System Architecture
- **Architecture Flow:**
  1. *Ingestion:* Raw IBM Telco Dataset + Simulated Real-Time Event Stream.
  2. *Databricks Medallion Pipeline:* Bronze (Raw Landing) $\to$ Silver (Cleansing & Quality) $\to$ Gold (Customer 360 Feature Store).
  3. *Snowflake Warehouse:* `CUSTOMER_CHURN_DB` with `RAW`, `ANALYTICS`, `ML`, and `REPORTING` schemas.
  4. *ML & Explainability:* Logistic Regression, Random Forest, XGBoost, CatBoost + Isotonic Calibration + SHAP TreeExplainer.
  5. *Decision & Delivery:* Financial CLV/RAR calculation, Action Playbook recommendation, FastAPI backend, and Streamlit Dashboard.

### Slide 4: Databricks Medallion Architecture (Bronze-Silver-Gold)
- **Bronze Tier:** Raw ingestion with audit metadata (`_bronze_ingested_at`, `_bronze_batch_id`, record hash).
- **Silver Tier:** Automated null imputation (e.g. 11 `TotalCharges` blanks for 0-tenure customers), type casting, constraint validation, and deduplication.
- **Gold Tier:** 46 engineered features including tenure cohorts (`0-12m`, `12-24m`, etc.), contract risk indices, security & streaming bundle scores, and rolling event aggregates.

### Slide 5: Snowflake Enterprise Data Warehouse Design
- **Database:** `CUSTOMER_CHURN_DB`
- **Schemas & Tables:**
  - `RAW`: `CUSTOMERS`, `CUSTOMER_EVENTS` (Simulated streaming events).
  - `ANALYTICS`: `CUSTOMER_FEATURES`, `V_CUSTOMER_360`, `V_RFM_METRICS`.
  - `ML`: `CHURN_PREDICTIONS`, `MODEL_RUNS` (MLflow tracking history), `V_HIGH_RISK_CUSTOMERS`.
  - `REPORTING`: `RETENTION_ACTIONS`, `V_EXECUTIVE_CHURN_SUMMARY`, `V_RETENTION_CAMPAIGN_ROI`.

### Slide 6: Machine Learning Pipeline & Metric Evaluation
- **Models Implemented:** Logistic Regression (Baseline), Decision Tree, Random Forest, XGBoost, CatBoost.
- **Leakage-Safe Preprocessing:** Preprocessors strictly fitted on training splits inside Scikit-Learn Pipelines.
- **Why Not Accuracy?:** With 26.5% positive class imbalance, a naive model predicting "No" achieves 73.5% accuracy but zero utility.
- **Headline Metrics:** ROC-AUC (0.846), PR-AUC (0.658), F1 (0.615), Recall (0.648), Precision (0.584), Brier Score (0.132).
- **Calibration:** Isotonic regression aligns predicted probabilities with true observed empirical frequencies.

### Slide 7: Explainable AI (XAI) with SHAP
- **Methodology:** TreeExplainer with Game-Theoretic Shapley values.
- **Global Insights:** Contract (Month-to-month), Tenure (<12m), and Internet Service (Fiber optic without tech support) are top risk drivers.
- **Local Explanations:** Real-time breakdown of positive drivers (pushing toward churn) vs negative protective factors for individual customers.

### Slide 8: Real-Time Streaming Simulation & Incremental Risk Engine
- **Event Simulation:** 9 event types (`complaint`, `payment_failed`, `support_ticket`, `login`, `purchase`, `cancellation_attempt`, etc.).
- **Dynamic Risk Update:** Immediate adjustment of churn risk score when incidents occur without waiting for weekly batch retraining.
- **Provenance Honesty:** Clearly documented that historical data is used for customer profiles and streaming events are synthetic simulations.

### Slide 9: Intelligent Retention Recommendation Engine
- **Financial Metric Modeling:**
  $$\text{CLV} = \text{MonthlyCharges} \times 24 \times 0.30$$
  $$\text{Revenue at Risk (RAR)} = \text{Churn Probability} \times \text{CLV}$$
  $$\text{Net Expected Profit} = (\text{RAR} \times \text{Uplift}) - \text{Intervention Cost}$$
- **Action Playbook:** Tailored interventions (VIP Concierge, Contract Discount, TechSupport Bundle) based on risk tier and urgency.

### Slide 10: Streamlit Dashboard & Production Deployment
- **Interactive Multi-Page Dashboard:**
  - Page 1: Executive KPI Overview & Churn Distribution.
  - Page 2: Customer Risk Table with multi-parameter filtering.
  - Page 3: Customer 360 with dynamic SHAP waterfall plots and real-time event injection.
  - Page 4: Real-Time Event Monitor & Stream Audit Log.
  - Page 5: Customer Segmentation (K-Means & PCA clusters).
  - Page 6: Model Performance & Calibration Curves.
  - Page 7: Data Drift & Monitoring (PSI & KS metrics).
- **FastAPI Backend:** Sub-20ms latency REST endpoints with Swagger UI.
- **Test Suite:** 148 automated tests passing (100% pass rate).

---

## ❓ Frequently Asked Viva Voce Questions & Model Answers

### Q1: Why did you implement a Medallion Architecture in Databricks instead of direct data loading?
**Answer:** The Medallion Architecture establishes clear data quality guarantees:
- **Bronze** preserves raw immutable history and streaming events with ingestion metadata for full auditability.
- **Silver** provides cleaned, validated, deduplicated data with schema enforcement (e.g. handling missing `TotalCharges`).
- **Gold** optimizes aggregated business features (Customer 360, RFM metrics, contract indices) for high-speed ML training and analytics queries.

### Q2: Why is accuracy a misleading metric for churn prediction, and what did you use instead?
**Answer:** The dataset has a 26.5% positive churn rate (class imbalance). A dummy model predicting 0 for everyone would achieve 73.5% accuracy but catch 0% of churners. We prioritize:
- **PR-AUC (Precision-Recall Area Under Curve):** Focuses specifically on the minority positive class.
- **Recall & F1-Score:** Maximizes identification of churners while controlling false alarm costs.
- **ROC-AUC & Brier Score:** Measures ranking power and probability calibration reliability.

### Q3: Why is probability calibration necessary before calculating Revenue at Risk?
**Answer:** Tree ensembles (like XGBoost or Random Forest) output probabilities that are often uncalibrated and distorted near extremes. Because our retention engine multiplies $p \times CLV$ to calculate dollars at risk, an overconfident probability would misallocate intervention budgets. Isotonic calibration ensures a predicted probability of 0.80 corresponds to exactly an 80% empirical churn rate.

### Q4: How does SHAP explainability help customer success teams in practice?
**Answer:** Knowing a customer has a 75% churn probability tells a manager *who* is leaving, but not *how* to save them. SHAP decomposes the 75% risk into specific contributory factors (e.g. $+0.25$ from month-to-month contract, $+0.15$ from frequent complaints, $-0.10$ from having tech support). The retention engine uses these exact drivers to recommend an automated contract upgrade or priority customer service call.

### Q5: How is Snowflake integrated into the architecture?
**Answer:** Snowflake acts as the central Enterprise Data Warehouse with 4 structured schemas (`RAW`, `ANALYTICS`, `ML`, `REPORTING`). It allows BI tools and executive stakeholders to query unified Customer 360 views, track historical ML model runs, monitor portfolio financial risk, and analyze retention campaign ROI.

### Q6: How does the real-time event simulation work?
**Answer:** Because the IBM Telco dataset is historical, we built an event simulator that generates realistic operational events (`complaint`, `payment_failed`, `plan_upgrade`, `login`) linked to real customer IDs. The streaming engine updates the customer's churn risk score in real time and triggers immediate retention alerts.
