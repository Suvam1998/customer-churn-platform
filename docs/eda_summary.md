# EDA Summary — IBM Telco Customer Churn

> Generated from the real cleaned dataset (`python scripts/run_eda.py`). Figures live in `docs/figures/`.

## Overall churn

- Customers: **7,043**
- Churned: **1,869** (26.54%)
- Retained: **5,174**

## Churn rate by contract

| Contract       |   n_customers |   n_churned |   churn_rate |
|:---------------|--------------:|------------:|-------------:|
| Month-to-month |          3875 |        1655 |       0.4271 |
| One year       |          1473 |         166 |       0.1127 |
| Two year       |          1695 |          48 |       0.0283 |

## Churn rate by payment method

| PaymentMethod             |   n_customers |   n_churned |   churn_rate |
|:--------------------------|--------------:|------------:|-------------:|
| Electronic check          |          2365 |        1071 |       0.4529 |
| Mailed check              |          1612 |         308 |       0.1911 |
| Bank transfer (automatic) |          1544 |         258 |       0.1671 |
| Credit card (automatic)   |          1522 |         232 |       0.1524 |

## Churn rate by internet service

| InternetService   |   n_customers |   n_churned |   churn_rate |
|:------------------|--------------:|------------:|-------------:|
| Fiber optic       |          3096 |        1297 |       0.4189 |
| DSL               |          2421 |         459 |       0.1896 |
| No                |          1526 |         113 |       0.074  |

## Churn rate by tenure band

| tenure_band   |   n_customers |   n_churned |   churn_rate |
|:--------------|--------------:|------------:|-------------:|
| 0-12          |          2186 |        1037 |       0.4744 |
| 13-24         |          1024 |         294 |       0.2871 |
| 25-36         |           832 |         180 |       0.2163 |
| 37-48         |           762 |         145 |       0.1903 |
| 49-60         |           832 |         120 |       0.1442 |
| 61-72         |          1407 |          93 |       0.0661 |

## Numeric summary by churn outcome

```
      tenure        MonthlyCharges        TotalCharges         
        mean median           mean median         mean   median
Churn                                                          
0      37.57   38.0          61.27  64.43      2549.91  1679.52
1      17.98   10.0          74.44  79.65      1531.80   703.55
```

## Correlation with churn (numeric)

```
Churn             1.000
MonthlyCharges    0.193
SeniorCitizen     0.151
TotalCharges     -0.198
tenure           -0.352
```
