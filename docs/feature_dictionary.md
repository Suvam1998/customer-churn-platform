# Feature Dictionary — Engineered Features

> Leakage-safe, row-wise features (no target, no cross-row statistics). See `src/features/feature_engineering.py`.

| Feature | Group | Rationale |
|---------|-------|-----------|
| `average_charge_per_month` | value | TotalCharges / tenure (falls back to MonthlyCharges when tenure==0). Captures effective spend rate vs. headline monthly price. |
| `total_services` | engagement | Count of subscribed services among the 9 core/add-on services (0-9). Engagement/stickiness proxy. |
| `service_adoption_score` | engagement | total_services / 9 in [0,1]. Normalised engagement; higher adoption tends to reduce churn. |
| `is_month_to_month` | contract | 1 if Contract == 'Month-to-month'. The highest-churn contract type (EDA: 42.7%). |
| `is_long_term_contract` | contract | 1 if Contract in {'One year','Two year'}. Long-term commitment lowers churn. |
| `has_tech_support` | service | 1 if TechSupport == 'Yes'. Support access is protective against churn. |
| `has_online_security` | service | 1 if OnlineSecurity == 'Yes'. Security add-on correlates with retention. |
| `has_online_backup` | service | 1 if OnlineBackup == 'Yes'. |
| `has_device_protection` | service | 1 if DeviceProtection == 'Yes'. |
| `has_streaming` | engagement | 1 if the customer streams TV or movies. Higher engagement signal. |
| `payment_risk_indicator` | risk | 1 if PaymentMethod == 'Electronic check' (domain: highest-churn payment method). |
| `contract_risk_indicator` | risk | 1 if month-to-month contract (domain risk flag). |
| `tenure_risk_indicator` | risk | 1 if tenure < 12 months (domain: new customers churn most). |
