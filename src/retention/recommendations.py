"""Retention decision engine.

Turns a customer's churn probability, value, and behavioural signals into a
structured decision: priority, urgency, recommended action, and a human reason.
The rules are ordered, documented, and driven by configurable thresholds in
``configs/config.yaml`` — this is deliberately NOT ``if prob > 0.5: discount``.

A clearly-labelled business SIMULATION accompanies each decision (intervention
cost, expected retained value). These are ESTIMATES from assumptions, not
observed outcomes — the dataset has no realised retention results.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from src.config import Config, get_config

# Action catalogue.
NO_INTERVENTION = "no_intervention"
PAYMENT_ASSISTANCE = "payment_assistance"
PLAN_OPTIMIZATION = "plan_optimization"
PERSONALIZED_DISCOUNT = "personalized_discount"
TECHNICAL_SUPPORT_INTERVENTION = "technical_support_intervention"
ACCOUNT_MANAGER_CONTACT = "account_manager_contact"
LOYALTY_REWARD = "loyalty_reward"
FEATURE_EDUCATION = "feature_education"
ONBOARDING_ASSISTANCE = "onboarding_assistance"

_URGENCY_BY_RISK = {
    "LOW": "LOW",
    "MEDIUM": "MEDIUM",
    "HIGH": "HIGH",
    "CRITICAL": "IMMEDIATE",
}


@dataclass
class RetentionDecision:
    customer_id: str
    churn_probability: float
    risk_level: str
    estimated_clv: float
    revenue_at_risk: float
    priority_score: float
    urgency: str
    recommended_action: str
    reason: str
    # Business SIMULATION (estimates).
    estimated_intervention_cost: float
    estimated_expected_retained_value: float
    estimated_net_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def _is_high_value(f: dict, cfg: Config) -> bool:
    seg = str(f.get("segment_label", "") or "")
    threshold = cfg.get("retention.high_value_clv_threshold", 500.0)
    return seg.startswith("High-value") or float(f.get("estimated_clv", 0)) >= threshold


def recommend_action(f: dict, cfg: Config | None = None) -> tuple[str, str]:
    """Return (action, reason) from ordered, documented rules.

    ``f`` keys used: risk_level, payment_risk_indicator, is_month_to_month,
    tenure, has_internet, has_tech_support, has_online_security,
    service_adoption_score, segment_label, estimated_clv.
    """
    cfg = cfg or get_config()

    # 1. Low risk -> monitor only.
    if f.get("risk_level") == "LOW":
        return NO_INTERVENTION, "Low churn risk; monitor only, no intervention needed."

    high_value = _is_high_value(f, cfg)

    # 2. Critical risk on a high-value customer -> human escalation.
    if f.get("risk_level") == "CRITICAL" and high_value:
        return (ACCOUNT_MANAGER_CONTACT,
                "Critical churn risk on a high-value customer; escalate to an "
                "account manager for immediate personal outreach.")

    # 3. Payment friction (electronic-check payers churn most).
    if int(f.get("payment_risk_indicator", 0)) == 1:
        return (PAYMENT_ASSISTANCE,
                "Electronic-check payer with elevated churn risk; offer payment "
                "assistance / auto-pay migration to reduce billing friction.")

    # 4. New month-to-month customer in the early-tenure danger zone.
    if int(f.get("is_month_to_month", 0)) == 1 and float(f.get("tenure", 99)) < 12:
        return (ONBOARDING_ASSISTANCE,
                "New month-to-month customer within the high-churn first year; "
                "provide onboarding assistance and early engagement.")

    # 5. Month-to-month contract -> encourage a longer commitment.
    if int(f.get("is_month_to_month", 0)) == 1:
        return (PLAN_OPTIMIZATION,
                "Month-to-month contract; propose an annual plan with incentives "
                "to lengthen commitment.")

    # 6. Internet customer without tech support.
    if int(f.get("has_internet", 1)) == 1 and int(f.get("has_tech_support", 0)) == 0:
        return (TECHNICAL_SUPPORT_INTERVENTION,
                "Internet plan without tech support; proactive technical support "
                "can address unresolved issues driving churn.")

    # 7. Internet customer without online security.
    if int(f.get("has_internet", 1)) == 1 and int(f.get("has_online_security", 0)) == 0:
        return (FEATURE_EDUCATION,
                "No online-security add-on; educate on protective features to "
                "increase engagement and stickiness.")

    # 8. High-value at-risk without a single dominant driver -> loyalty reward.
    if high_value:
        return (LOYALTY_REWARD,
                "High-value at-risk customer; offer a loyalty reward to reinforce "
                "the relationship.")

    # 9. Fallback for remaining at-risk customers.
    return (PERSONALIZED_DISCOUNT,
            "Elevated churn risk without a dominant driver; a targeted "
            "personalised discount is the most cost-effective lever.")


def make_decision(f: dict, cfg: Config | None = None) -> RetentionDecision:
    """Build a full retention decision (rules + business simulation)."""
    cfg = cfg or get_config()
    action, reason = recommend_action(f, cfg)

    risk_level = f.get("risk_level", "LOW")
    urgency = _URGENCY_BY_RISK.get(risk_level, "LOW")

    revenue_at_risk = float(f.get("revenue_at_risk", 0.0))
    effectiveness = cfg.get("retention.intervention_effectiveness", 0.35)
    action_costs = cfg.get("retention.action_costs", {}) or {}
    cost = float(action_costs.get(action, cfg.get("retention.default_intervention_cost", 50.0)))

    # Business SIMULATION (estimates).
    if action == NO_INTERVENTION:
        expected_retained = 0.0
        cost = 0.0
    else:
        expected_retained = round(revenue_at_risk * effectiveness, 2)
    net_value = round(expected_retained - cost, 2)

    return RetentionDecision(
        customer_id=str(f.get("customerID", f.get("customer_id", ""))),
        churn_probability=round(float(f.get("churn_probability", 0.0)), 4),
        risk_level=risk_level,
        estimated_clv=round(float(f.get("estimated_clv", 0.0)), 2),
        revenue_at_risk=round(revenue_at_risk, 2),
        priority_score=round(float(f.get("priority_score", revenue_at_risk)), 2),
        urgency=urgency,
        recommended_action=action,
        reason=reason,
        estimated_intervention_cost=round(cost, 2),
        estimated_expected_retained_value=expected_retained,
        estimated_net_value=net_value,
    )


def decide_for_dataframe(value_table, cfg: Config | None = None):
    """Vectorised-ish batch: return a DataFrame of decisions for every row.

    ``value_table`` must contain the value columns (Phase 12) and the
    engineered behavioural flags.
    """
    import pandas as pd

    cfg = cfg or get_config()
    records = [make_decision(row, cfg).to_dict()
               for row in value_table.to_dict(orient="records")]
    return pd.DataFrame(records)
