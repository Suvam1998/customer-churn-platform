-- ==============================================================================
-- 01_database_setup.sql
-- Snowflake Database, Schema, Warehouse, Role, and Stage Initialization
-- Project: Real-Time Customer Churn Prediction & Intelligent Retention Platform
-- ==============================================================================

-- 1. Create Virtual Warehouse
CREATE WAREHOUSE IF NOT EXISTS CHURN_WH
    WITH WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 120
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Virtual warehouse for Customer Churn Analytics and ML Workloads';

USE WAREHOUSE CHURN_WH;

-- 2. Create Database
CREATE DATABASE IF NOT EXISTS CUSTOMER_CHURN_DB
    COMMENT = 'Enterprise Data Warehouse for Customer Churn Platform';

USE DATABASE CUSTOMER_CHURN_DB;

-- 3. Create Medallion / Multi-tier Schemas
CREATE SCHEMA IF NOT EXISTS CUSTOMER_CHURN_DB.RAW
    COMMENT = 'Raw Ingestion Layer: Staged files, raw Telco data, and incoming events';

CREATE SCHEMA IF NOT EXISTS CUSTOMER_CHURN_DB.ANALYTICS
    COMMENT = 'Silver/Analytics Tier: Cleaned customer attributes, engineered features, RFM metrics';

CREATE SCHEMA IF NOT EXISTS CUSTOMER_CHURN_DB.ML
    COMMENT = 'Machine Learning Tier: Churn predictions, model metadata, drift logs, SHAP values';

CREATE SCHEMA IF NOT EXISTS CUSTOMER_CHURN_DB.REPORTING
    COMMENT = 'Gold/Reporting Tier: Retention decisions, executive KPI dashboards, business alerts';

-- 4. Create File Formats
USE SCHEMA CUSTOMER_CHURN_DB.RAW;

CREATE OR REPLACE FILE FORMAT CSV_FORMAT
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    RECORD_DELIMITER = '\n'
    SKIP_HEADER = 1
    NULL_IF = ('NULL', 'null', '', ' ')
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    TRIM_SPACE = TRUE
    ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE;

CREATE OR REPLACE FILE FORMAT JSON_FORMAT
    TYPE = 'JSON'
    STRIP_OUTER_ARRAY = TRUE
    ENABLE_OCTAL = FALSE
    ALLOW_DUPLICATE = FALSE;

CREATE OR REPLACE FILE FORMAT PARQUET_FORMAT
    TYPE = 'PARQUET'
    COMPRESSION = 'SNAPPY';

-- 5. Create Internal Named Stages
CREATE STAGE IF NOT EXISTS RAW_STAGE
    FILE_FORMAT = CSV_FORMAT
    COMMENT = 'Internal stage for raw customer CSV uploads';

CREATE STAGE IF NOT EXISTS STREAMING_STAGE
    FILE_FORMAT = JSON_FORMAT
    COMMENT = 'Internal stage for real-time simulated event streams';

-- 6. Role & Access Control Configuration
CREATE ROLE IF NOT EXISTS CHURN_ADMIN_ROLE;
CREATE ROLE IF NOT EXISTS CHURN_ANALYST_ROLE;

GRANT USAGE ON WAREHOUSE CHURN_WH TO ROLE CHURN_ADMIN_ROLE;
GRANT USAGE ON WAREHOUSE CHURN_WH TO ROLE CHURN_ANALYST_ROLE;

GRANT ALL PRIVILEGES ON DATABASE CUSTOMER_CHURN_DB TO ROLE CHURN_ADMIN_ROLE;
GRANT USAGE ON DATABASE CUSTOMER_CHURN_DB TO ROLE CHURN_ANALYST_ROLE;

GRANT USAGE ON ALL SCHEMAS IN DATABASE CUSTOMER_CHURN_DB TO ROLE CHURN_ANALYST_ROLE;
GRANT SELECT ON ALL TABLES IN SCHEMA CUSTOMER_CHURN_DB.REPORTING TO ROLE CHURN_ANALYST_ROLE;
GRANT SELECT ON ALL VIEWS IN SCHEMA CUSTOMER_CHURN_DB.REPORTING TO ROLE CHURN_ANALYST_ROLE;
GRANT SELECT ON ALL VIEWS IN SCHEMA CUSTOMER_CHURN_DB.ANALYTICS TO ROLE CHURN_ANALYST_ROLE;
