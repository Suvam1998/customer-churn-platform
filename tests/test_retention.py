"""Phase 13 tests: retention decision engine rules + business simulation."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import get_config
from src.retention.recommendations import (
    ACCOUNT_MANAGER_CONTACT,
    NO_INTERVENTION,
    ONBOARDING_ASSISTANCE,
    PAYMENT_ASSISTANCE,
    PLAN_OPTIMIZATION,
    TECHNICAL_SUPPORT_INTERVENTION,
    decide_for_dataframe,
    make_decision,
    recommend_action,
)


def _f(**kwargs) -> dict:
    base = dict(
        customerID="X", churn_probability=0.6, risk_level="HIGH",
        estimated_clv=300.0, revenue_at_risk=180.0, priority_score=180.0,
        payment_risk_indicator=0, is_month_to_month=0, tenure=40,
        has_internet=1, has_tech_support=1, has_online_security=1,
        service_adoption_score=0.5, segment_label="Low-value/high-risk",
    )
    base.update(kwargs)
    return base


def test_low_risk_no_intervention():
    action, _ = recommend_action(_f(risk_level="LOW"))
    assert action == NO_INTERVENTION


def test_critical_high_value_escalates():
    action, _ = recommend_action(_f(risk_level="CRITICAL", estimated_clv=700,
                                    payment_risk_indicator=1))
    assert action == ACCOUNT_MANAGER_CONTACT


def test_payment_risk_gets_payment_assistance():
    action, _ = recommend_action(_f(payment_risk_indicator=1))
    assert action == PAYMENT_ASSISTANCE


def test_new_month_to_month_onboarding():
    action, _ = recommend_action(_f(is_month_to_month=1, tenure=3))
    assert action == ONBOARDING_ASSISTANCE


def test_month_to_month_plan_optimization():
    action, _ = recommend_action(_f(is_month_to_month=1, tenure=30))
    assert action == PLAN_OPTIMIZATION


def test_no_tech_support_intervention():
    action, _ = recommend_action(_f(has_tech_support=0))
    assert action == TECHNICAL_SUPPORT_INTERVENTION


def test_urgency_mapping():
    assert make_decision(_f(risk_level="CRITICAL", estimated_clv=100)).urgency == "IMMEDIATE"
    assert make_decision(_f(risk_level="HIGH")).urgency == "HIGH"
    assert make_decision(_f(risk_level="MEDIUM")).urgency == "MEDIUM"


def test_business_simulation_math():
    cfg = get_config()
    eff = cfg.get("retention.intervention_effectiveness", 0.35)
    d = make_decision(_f(risk_level="HIGH", revenue_at_risk=200.0,
                         payment_risk_indicator=1))
    # payment_assistance cost from config action_costs.
    cost = cfg.get("retention.action_costs", {})["payment_assistance"]
    assert d.estimated_expected_retained_value == pytest.approx(200.0 * eff, rel=1e-3)
    assert d.estimated_intervention_cost == pytest.approx(cost)
    assert d.estimated_net_value == pytest.approx(200.0 * eff - cost, rel=1e-3)


def test_no_intervention_zero_cost_and_retained():
    d = make_decision(_f(risk_level="LOW"))
    assert d.recommended_action == NO_INTERVENTION
    assert d.estimated_intervention_cost == 0.0
    assert d.estimated_expected_retained_value == 0.0


def test_decide_for_dataframe_batch():
    df = pd.DataFrame([_f(customerID="a", risk_level="LOW"),
                       _f(customerID="b", risk_level="HIGH", payment_risk_indicator=1)])
    out = decide_for_dataframe(df)
    assert len(out) == 2
    assert set(["recommended_action", "urgency", "reason",
                "estimated_net_value"]).issubset(out.columns)
