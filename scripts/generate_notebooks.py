"""Notebook generator script.

Generates 5 fully documented, executable Jupyter notebooks:
1. notebooks/01_databricks_medallion_pipeline.ipynb
2. notebooks/02_exploratory_data_analysis.ipynb
3. notebooks/03_model_training_and_mlflow.ipynb
4. notebooks/04_shap_explainability.ipynb
5. notebooks/05_snowflake_analytics_and_retention.ipynb
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def generate_all_notebooks():
    # -------------------------------------------------------------
    # Notebook 1: Databricks Medallion Pipeline
    # -------------------------------------------------------------
    nb1_cells = [
        md_cell("""# 🏗️ Databricks Medallion Architecture (Bronze -> Silver -> Gold)
## Real-Time Customer Churn Prediction & Intelligent Retention Platform

This notebook demonstrates the end-to-end **Databricks Delta Lake Medallion Pipeline**:
1. **Bronze Tier**: Raw ingestion of IBM Telco customer data and simulated streaming events with metadata lineage (`_ingested_at`, `_batch_id`).
2. **Silver Tier**: Data cleaning, constraint validation, type casting, missing value handling, and deduplication.
3. **Gold Tier**: High-value aggregated customer feature store (RFM metrics, service bundle scores, contract risk factors, Customer 360).
"""),
        code_cell("""import sys
from pathlib import Path

# Setup project root
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from databricks.bronze_ingestion import BronzeIngestionPipeline
from databricks.silver_cleaning import SilverCleaningPipeline
from databricks.gold_feature_store import GoldFeatureStorePipeline
from databricks.pipeline_orchestrator import MedallionPipelineOrchestrator

print("Databricks Medallion Architecture modules loaded successfully!")
"""),
        md_cell("### 1. Execute Bronze Ingestion Tier"),
        code_cell("""bronze = BronzeIngestionPipeline()
bronze_df = bronze.ingest_customers_raw()
print(f"Bronze Delta Table Record Count: {len(bronze_df)}")
bronze_df[["customerID", "gender", "tenure", "MonthlyCharges", "_bronze_ingested_at", "_bronze_batch_id"]].head()
"""),
        md_cell("### 2. Execute Silver Cleansing & Validation Tier"),
        code_cell("""silver = SilverCleaningPipeline()
silver_df, quality_report = silver.clean_customers(bronze_df)
print("Silver Quality Report:", quality_report)
silver_df[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn_Numeric", "_silver_processed_at"]].head()
"""),
        md_cell("### 3. Execute Gold Feature Store & Aggregation Tier"),
        code_cell("""gold = GoldFeatureStorePipeline()
gold_df = gold.build_gold_features(silver_df)
print(f"Gold Features Shape: {gold_df.shape}")
gold_df[["customerID", "tenure_cohort", "contract_risk_index", "security_bundle_score", "total_active_services", "estimated_clv"]].head()
"""),
        md_cell("### 4. End-to-End Orchestration Summary"),
        code_cell("""orchestrator = MedallionPipelineOrchestrator()
summary = orchestrator.run_pipeline()
print("Pipeline Status:", summary["status"])
print(f"Total Runtime: {summary['total_runtime_seconds']}s")
""")
    ]
    (NOTEBOOKS_DIR / "01_databricks_medallion_pipeline.ipynb").write_text(
        json.dumps(make_notebook(nb1_cells), indent=2), encoding="utf-8"
    )

    # -------------------------------------------------------------
    # Notebook 2: Exploratory Data Analysis
    # -------------------------------------------------------------
    nb2_cells = [
        md_cell("""# 📊 Exploratory Data Analysis (EDA)
## IBM Telco Customer Churn Dataset

This notebook explores customer behavior, contract patterns, service subscriptions, and their relationship with customer churn.
"""),
        code_cell("""import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.ingestion.load_data import load_raw
from src.preprocessing.preprocess import clean_raw

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
df = clean_raw(load_raw())
print(f"Loaded Cleaned Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()
"""),
        md_cell("### 1. Target Distribution & Overall Churn Rate"),
        code_cell("""churn_counts = df["Churn"].value_counts()
churn_rate = df["Churn_Numeric"].mean()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].pie(churn_counts, labels=churn_counts.index, autopct="%.1f%%", colors=["#4C72B0", "#DD8452"], explode=(0, 0.08))
ax[0].set_title("Churn Class Distribution")

sns.countplot(x="Churn", data=df, ax=ax[1], palette=["#4C72B0", "#DD8452"])
ax[1].set_title(f"Churn Count (Overall Rate: {churn_rate:.1%})")
plt.tight_layout()
plt.show()
"""),
        md_cell("### 2. Churn by Contract Type & Payment Method"),
        code_cell("""fig, ax = plt.subplots(1, 2, figsize=(14, 5))
contract_churn = df.groupby("Contract")["Churn_Numeric"].mean().reset_index()
sns.barplot(x="Contract", y="Churn_Numeric", data=contract_churn, ax=ax[0], palette="Blues_r")
ax[0].set_title("Churn Rate by Contract Type")
ax[0].set_ylabel("Churn Rate")

payment_churn = df.groupby("PaymentMethod")["Churn_Numeric"].mean().reset_index()
sns.barplot(x="PaymentMethod", y="Churn_Numeric", data=payment_churn, ax=ax[1], palette="Reds_r")
ax[1].set_title("Churn Rate by Payment Method")
ax[1].set_ylabel("Churn Rate")
ax[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()
"""),
        md_cell("### 3. Tenure vs Monthly Charges Distribution"),
        code_cell("""plt.figure(figsize=(10, 6))
sns.scatterplot(x="tenure", y="MonthlyCharges", hue="Churn", data=df, alpha=0.6, palette=["#4C72B0", "#DD8452"])
plt.title("Tenure vs Monthly Charges (Colored by Churn)")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges ($)")
plt.show()
""")
    ]
    (NOTEBOOKS_DIR / "02_exploratory_data_analysis.ipynb").write_text(
        json.dumps(make_notebook(nb2_cells), indent=2), encoding="utf-8"
    )

    # -------------------------------------------------------------
    # Notebook 3: Model Training & MLflow Tracking
    # -------------------------------------------------------------
    nb3_cells = [
        md_cell("""# 🤖 Machine Learning Pipeline & MLflow Tracking
## Logistic Regression, Random Forest, and XGBoost with Calibration

In this notebook, we train, evaluate, tune, and calibrate multiple churn classification models:
- **Baseline Models**: Logistic Regression, Random Forest, XGBoost
- **Leakage Prevention**: Stratified splitting before preprocessor fitting
- **Comprehensive Metrics**: ROC-AUC, PR-AUC, F1-Score, Recall, Precision, Log Loss, Brier Score
- **Calibration**: Isotonic & Sigmoid Platt Scaling
- **MLflow Tracking**: Experiment tracking and artifact registry
"""),
        code_cell("""import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.models.train import train_and_evaluate_all, load_splits
from src.models.metrics import metrics_row
from src.models.calibration import evaluate_calibrators, build_tuned_estimator

splits, results, pipelines, skipped = train_and_evaluate_all(include_engineered=True)
print(f"Trained {len(results)} models successfully!")
"""),
        md_cell("### 1. Model Comparison Table (Validation Set)"),
        code_cell("""comparison_rows = [metrics_row(name, m) for name, m in results.items()]
comp_df = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False)
comp_df
"""),
        md_cell("### 2. Probability Calibration Analysis"),
        code_cell("""cal_results, best_cal_name = evaluate_calibrators()
print(f"Best Calibration Method: {best_cal_name}")
pd.DataFrame([
    {"Method": name, "Brier Score": res["brier"], "ROC-AUC": res["roc_auc"]}
    for name, res in cal_results.items()
])
""")
    ]
    (NOTEBOOKS_DIR / "03_model_training_and_mlflow.ipynb").write_text(
        json.dumps(make_notebook(nb3_cells), indent=2), encoding="utf-8"
    )

    # -------------------------------------------------------------
    # Notebook 4: SHAP Explainability
    # -------------------------------------------------------------
    nb4_cells = [
        md_cell("""# 🔍 Explainable AI (XAI) with SHAP
## Global Feature Importance & Customer-Level Local Explanations

Using TreeExplainer with SHAP:
- **Global Explanations**: Summary bar plots, feature impact rankings
- **Local Explanations**: Why a specific customer is predicted at risk of churn
- **Actionable Insights**: Identifying top drivers (contract type, tenure, fiber optic, tech support)
"""),
        code_cell("""import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.explainability.shap_explainer import ChurnExplainer, engineered_dataset, get_customer_features
from src.features.feature_engineering import FEATURED_FEATURE_COLUMNS

explainer = ChurnExplainer.from_tuned()
df_eng = engineered_dataset()
print("SHAP Explainer initialized successfully!")
"""),
        md_cell("### 1. Global Feature Importance (Top 15 Drivers)"),
        code_cell("""sample_df = df_eng[FEATURED_FEATURE_COLUMNS].sample(300, random_state=42)
global_imp = explainer.global_importance(sample_df, top_n=15)
global_imp
"""),
        md_cell("### 2. Individual Customer Local Explanation"),
        code_cell("""sample_cid = "7590-VHVEG"
feat_row, actual_label = get_customer_features(sample_cid, df_eng)
explanation = explainer.explain_customer(feat_row, sample_cid)

print(f"Customer ID: {explanation['customer_id']}")
print(f"Predicted Churn Probability: {explanation['churn_probability']:.3f}")
print("\\nTop Positive Factors (Increasing Churn Risk):")
for f in explanation['top_positive_factors']:
    print(f"  - {f['label']}: +{f['shap']:.4f}")

print("\\nTop Protective Factors (Reducing Churn Risk):")
for f in explanation['top_negative_factors']:
    print(f"  - {f['label']}: {f['shap']:.4f}")
""")
    ]
    (NOTEBOOKS_DIR / "04_shap_explainability.ipynb").write_text(
        json.dumps(make_notebook(nb4_cells), indent=2), encoding="utf-8"
    )

    # -------------------------------------------------------------
    # Notebook 5: Snowflake Analytics & Retention
    # -------------------------------------------------------------
    nb5_cells = [
        md_cell("""# ❄️ Snowflake Data Warehouse Analytics & Retention Engine
## CUSTOMER_CHURN_DB Queries & Value-Based Decisioning

This notebook demonstrates:
1. Connecting to Snowflake `CUSTOMER_CHURN_DB` (with automatic local mock fallback)
2. Querying `RAW`, `ANALYTICS`, `ML`, and `REPORTING` schemas
3. Retention Decision Engine: Expected Retained Value, Intervention Costs, and Net Preserved ROI
"""),
        code_cell("""import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from snowflake.snowflake_client import SnowflakeClient
from snowflake.load_data import SnowflakeDataLoader

client = SnowflakeClient()
loader = SnowflakeDataLoader()
summary = loader.load_all()
print(f"Snowflake Mode: {summary['mode']}, Database: {summary['database']}")
"""),
        md_cell("### 1. Query Executive Churn Summary"),
        code_cell("""exec_summary = client.get_executive_summary()
pd.DataFrame([exec_summary])
"""),
        md_cell("### 2. Query High-Risk Customer Priorities"),
        code_cell("""q_high_risk = \"\"\"
SELECT
    p.CUSTOMER_ID,
    p.CHURN_PROBABILITY,
    p.RISK_LEVEL,
    p.REVENUE_AT_RISK,
    p.ESTIMATED_CLV,
    r.RECOMMENDED_ACTION,
    r.URGENCY,
    r.ESTIMATED_NET_VALUE
FROM ML_CHURN_PREDICTIONS p
JOIN REPORTING_RETENTION_ACTIONS r ON p.CUSTOMER_ID = r.CUSTOMER_ID
WHERE p.RISK_LEVEL IN ('HIGH', 'CRITICAL')
ORDER BY p.REVENUE_AT_RISK DESC
LIMIT 10;
\"\"\"
high_risk_df = client.query_df(q_high_risk)
high_risk_df
"""),
        md_cell("### 3. Retention Campaign Portfolio ROI"),
        code_cell("""q_roi = \"\"\"
SELECT
    RECOMMENDED_ACTION,
    URGENCY,
    COUNT(*) AS TOTAL_ACTIONS,
    ROUND(SUM(REVENUE_AT_RISK), 2) AS REVENUE_AT_RISK_COVERED,
    ROUND(SUM(ESTIMATED_INTERVENTION_COST), 2) AS TOTAL_BUDGET,
    ROUND(SUM(ESTIMATED_RETAINED_VALUE), 2) AS EXPECTED_RETURN,
    ROUND(SUM(ESTIMATED_NET_VALUE), 2) AS NET_PRESERVED_PROFIT
FROM REPORTING_RETENTION_ACTIONS
GROUP BY RECOMMENDED_ACTION, URGENCY
ORDER BY NET_PRESERVED_PROFIT DESC;
\"\"\"
roi_df = client.query_df(q_roi)
roi_df
""")
    ]
    (NOTEBOOKS_DIR / "05_snowflake_analytics_and_retention.ipynb").write_text(
        json.dumps(make_notebook(nb5_cells), indent=2), encoding="utf-8"
    )
    print("Generated 5 Jupyter Notebooks successfully in notebooks/ directory!")


if __name__ == "__main__":
    generate_all_notebooks()
