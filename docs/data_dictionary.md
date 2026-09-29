# Data Dictionary — IBM Telco Customer Churn

> Generated from the real downloaded dataset. Descriptions are the standard IBM Telco churn definitions; dtype and example come from the actual file.

- **Rows:** 7,043
- **Columns:** 21
- **Target column:** `Churn` (Yes = churned, No = retained)

| # | Column | Dtype | Non-null | Example | Description |
|--:|--------|-------|---------:|---------|-------------|
| 1 | `customerID` | str | 7,043 | 7590-VHVEG | Unique customer identifier. |
| 2 | `gender` | str | 7,043 | Female | Customer gender (Male/Female). |
| 3 | `SeniorCitizen` | int64 | 7,043 | 0 | Whether the customer is a senior citizen (0/1). |
| 4 | `Partner` | str | 7,043 | Yes | Whether the customer has a partner (Yes/No). |
| 5 | `Dependents` | str | 7,043 | No | Whether the customer has dependents (Yes/No). |
| 6 | `tenure` | int64 | 7,043 | 1 | Number of months the customer has stayed with the company. |
| 7 | `PhoneService` | str | 7,043 | No | Whether the customer has phone service (Yes/No). |
| 8 | `MultipleLines` | str | 7,043 | No phone service | Whether the customer has multiple lines (Yes/No/No phone service). |
| 9 | `InternetService` | str | 7,043 | DSL | Customer's internet service provider (DSL/Fiber optic/No). |
| 10 | `OnlineSecurity` | str | 7,043 | No | Whether the customer has online security (Yes/No/No internet service). |
| 11 | `OnlineBackup` | str | 7,043 | Yes | Whether the customer has online backup (Yes/No/No internet service). |
| 12 | `DeviceProtection` | str | 7,043 | No | Whether the customer has device protection (Yes/No/No internet service). |
| 13 | `TechSupport` | str | 7,043 | No | Whether the customer has tech support (Yes/No/No internet service). |
| 14 | `StreamingTV` | str | 7,043 | No | Whether the customer has streaming TV (Yes/No/No internet service). |
| 15 | `StreamingMovies` | str | 7,043 | No | Whether the customer has streaming movies (Yes/No/No internet service). |
| 16 | `Contract` | str | 7,043 | Month-to-month | Contract term (Month-to-month/One year/Two year). |
| 17 | `PaperlessBilling` | str | 7,043 | Yes | Whether the customer has paperless billing (Yes/No). |
| 18 | `PaymentMethod` | str | 7,043 | Electronic check | Payment method (Electronic check/Mailed check/Bank transfer/Credit card). |
| 19 | `MonthlyCharges` | float64 | 7,043 | 29.85 | The amount charged to the customer monthly. |
| 20 | `TotalCharges` | str | 7,043 | 29.85 | The total amount charged to the customer (text in raw file; blanks when tenure==0). |
| 21 | `Churn` | str | 7,043 | No | TARGET — whether the customer churned (Yes/No). |
