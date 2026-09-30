-- ==============================================================================
-- 06_queries_and_views.sql
-- Production Analytical Queries & Business Reporting Script
-- Database: CUSTOMER_CHURN_DB
-- ==============================================================================

USE DATABASE CUSTOMER_CHURN_DB;

-- ==============================================================================
-- 1. Executive Churn Risk & Financial Exposure Analysis
-- ==============================================================================
SELECT
    p.RISK_LEVEL,
    COUNT(p.CUSTOMER_ID) AS CUSTOMER_COUNT,
    ROUND(COUNT(p.CUSTOMER_ID) * 100.0 / SUM(COUNT(p.CUSTOMER_ID)) OVER(), 2) AS SHARE_OF_BASE_PCT,
    ROUND(AVG(p.CHURN_PROBABILITY), 4) AS AVG_CHURN_PROBABILITY,
    ROUND(SUM(p.REVENUE_AT_RISK), 2) AS TOTAL_REVENUE_AT_RISK,
    ROUND(AVG(p.REVENUE_AT_RISK), 2) AS AVG_REVENUE_AT_RISK_PER_CUST,
    ROUND(SUM(p.ESTIMATED_CLV), 2) AS TOTAL_PORTFOLIO_CLV
FROM CUSTOMER_CHURN_DB.ML.CHURN_PREDICTIONS p
GROUP BY p.RISK_LEVEL
ORDER BY CASE p.RISK_LEVEL
    WHEN 'CRITICAL' THEN 1
    WHEN 'HIGH' THEN 2
    WHEN 'MEDIUM' THEN 3
    WHEN 'LOW' THEN 4
END;

-- ==============================================================================
-- 2. Cohort Retention & Contract Vulnerability Matrix
-- ==============================================================================
SELECT
    f.TENURE_COHORT,
    c.CONTRACT,
    c.PAYMENT_METHOD,
    COUNT(*) AS TOTAL_CUSTOMERS,
    SUM(CASE WHEN p.RISK_LEVEL IN ('HIGH', 'CRITICAL') THEN 1 ELSE 0 END) AS HIGH_RISK_CUSTOMERS,
    ROUND(SUM(CASE WHEN p.RISK_LEVEL IN ('HIGH', 'CRITICAL') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS HIGH_RISK_RATE_PCT,
    ROUND(AVG(c.MONTHLY_CHARGES), 2) AS AVG_MONTHLY_CHARGES,
    ROUND(SUM(p.REVENUE_AT_RISK), 2) AS TOTAL_EXPOSURE
FROM CUSTOMER_CHURN_DB.RAW.CUSTOMERS c
JOIN CUSTOMER_CHURN_DB.ANALYTICS.CUSTOMER_FEATURES f ON c.CUSTOMER_ID = f.CUSTOMER_ID
JOIN CUSTOMER_CHURN_DB.ML.CHURN_PREDICTIONS p ON c.CUSTOMER_ID = p.CUSTOMER_ID
GROUP BY f.TENURE_COHORT, c.CONTRACT, c.PAYMENT_METHOD
HAVING COUNT(*) >= 20
ORDER BY TOTAL_EXPOSURE DESC
LIMIT 20;

-- ==============================================================================
-- 3. Top 10 High-Value Urgent Customers for Immediate Retention Intervention
-- ==============================================================================
SELECT
    p.CUSTOMER_ID,
    c.TENURE AS TENURE_MONTHS,
    c.CONTRACT,
    c.PAYMENT_METHOD,
    c.INTERNET_SERVICE,
    c.MONTHLY_CHARGES,
    p.CHURN_PROBABILITY,
    p.RISK_LEVEL,
    p.REVENUE_AT_RISK,
    r.RECOMMENDED_ACTION,
    r.URGENCY,
    r.REASON,
    r.ESTIMATED_NET_VALUE
FROM CUSTOMER_CHURN_DB.ML.CHURN_PREDICTIONS p
JOIN CUSTOMER_CHURN_DB.RAW.CUSTOMERS c ON p.CUSTOMER_ID = c.CUSTOMER_ID
JOIN CUSTOMER_CHURN_DB.REPORTING.RETENTION_ACTIONS r ON p.CUSTOMER_ID = r.CUSTOMER_ID
WHERE p.RISK_LEVEL IN ('HIGH', 'CRITICAL')
  AND r.URGENCY IN ('IMMEDIATE', 'HIGH')
ORDER BY p.REVENUE_AT_RISK DESC
LIMIT 10;

-- ==============================================================================
-- 4. Simulated Real-Time Streaming Incident Impact Analysis
-- ==============================================================================
SELECT
    e.EVENT_TYPE,
    COUNT(e.EVENT_ID) AS EVENT_COUNT,
    COUNT(DISTINCT e.CUSTOMER_ID) AS AFFECTED_CUSTOMERS,
    ROUND(AVG(p.CHURN_PROBABILITY), 4) AS AVG_CHURN_PROB_AFTER_EVENT,
    SUM(CASE WHEN p.RISK_LEVEL = 'CRITICAL' THEN 1 ELSE 0 END) AS CRITICAL_ESCALATIONS,
    ROUND(SUM(p.REVENUE_AT_RISK), 2) AS TOTAL_ASSOCIATED_RISK
FROM CUSTOMER_CHURN_DB.RAW.CUSTOMER_EVENTS e
JOIN CUSTOMER_CHURN_DB.ML.CHURN_PREDICTIONS p ON e.CUSTOMER_ID = p.CUSTOMER_ID
GROUP BY e.EVENT_TYPE
ORDER BY TOTAL_ASSOCIATED_RISK DESC;

-- ==============================================================================
-- 5. Retention Strategy ROI & Action Breakdown
-- ==============================================================================
SELECT
    RECOMMENDED_ACTION,
    URGENCY,
    COUNT(*) AS TOTAL_ACTIONS,
    ROUND(SUM(REVENUE_AT_RISK), 2) AS REVENUE_AT_RISK_COVERED,
    ROUND(SUM(ESTIMATED_INTERVENTION_COST), 2) AS TOTAL_BUDGET_REQUIRED,
    ROUND(SUM(ESTIMATED_RETAINED_VALUE), 2) AS TOTAL_EXPECTED_RETURN,
    ROUND(SUM(ESTIMATED_NET_VALUE), 2) AS NET_PRESERVED_PROFIT
FROM CUSTOMER_CHURN_DB.REPORTING.RETENTION_ACTIONS
GROUP BY RECOMMENDED_ACTION, URGENCY
ORDER BY NET_PRESERVED_PROFIT DESC;
