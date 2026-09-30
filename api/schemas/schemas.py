"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---- prediction ----------------------------------------------------------

class RawCustomer(BaseModel):
    """Raw Telco-style customer features for ad-hoc scoring."""
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = Field(0, ge=0)
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = Field(70.0, ge=0)
    TotalCharges: float = Field(70.0, ge=0)


class PredictRequest(BaseModel):
    customer_id: str | None = Field(None, description="Existing customer id to score")
    customer: RawCustomer | None = Field(None, description="Ad-hoc customer features")


class RiskFactor(BaseModel):
    factor: str
    contribution: float


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    estimated_clv: float
    revenue_at_risk: float
    top_risk_factors: list[str]
    recommended_action: str
    model_version: str
    prediction_timestamp: str
    prediction_latency_ms: float


# ---- events --------------------------------------------------------------

EventType = Literal[
    "login", "purchase", "payment_failed", "support_ticket", "complaint",
    "plan_upgrade", "plan_downgrade", "cancellation_attempt", "inactivity",
]


class EventRequest(BaseModel):
    customer_id: str
    event_type: EventType
    value: float = 1.0
    timestamp: str | None = None


class EventResponse(BaseModel):
    accepted: bool
    is_simulated: bool
    customer_id: str
    event_type: str
    previous_probability: float | None
    new_probability: float
    risk_change: float | None
    risk_level: str
    recommended_action: str
    note: str


# ---- customer views ------------------------------------------------------

class CustomerResponse(BaseModel):
    customer_id: str
    tenure: int
    contract: str
    payment_method: str
    internet_service: str
    monthly_charges: float
    total_charges: float
    churn_probability: float
    risk_level: str
    estimated_clv: float
    revenue_at_risk: float
    segment: str | None
    actual_churn: int


class ExplanationResponse(BaseModel):
    customer_id: str
    churn_probability: float
    top_positive_factors: list[RiskFactor]
    top_negative_factors: list[RiskFactor]


class RecommendationResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    urgency: str
    recommended_action: str
    reason: str
    revenue_at_risk: float
    estimated_intervention_cost: float
    estimated_expected_retained_value: float
    estimated_net_value: float


class HighRiskItem(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    revenue_at_risk: float
    segment: str | None
    recommended_action: str


# ---- metrics / monitoring / retrain -------------------------------------

class DashboardMetrics(BaseModel):
    total_customers: int
    churned_customers: int
    churn_rate: float
    high_risk_customers: int
    total_revenue_at_risk: float
    average_churn_probability: float
    model_version: str
    risk_level_counts: dict


class ModelMetrics(BaseModel):
    model_version: str
    base_model: str | None
    calibration: str | None
    metrics: dict


class DriftResponse(BaseModel):
    status: str
    scenario: str
    note: str
    n_features: int
    n_drifted: int
    share_drifted: float
    threshold: float
    drifted_features: list[str]
    prediction_drift_psi: float | None


class RetrainResponse(BaseModel):
    status: str
    note: str
