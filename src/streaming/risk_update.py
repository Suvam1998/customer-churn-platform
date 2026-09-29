"""Risk-update logic for simulated real-time events.

A single source of truth for how a SIMULATED event nudges a customer's churn
probability, used by both the API (`POST /event`) and the streaming
consumer/simulator. The model is not retrained per event; instead a small,
bounded, documented heuristic delta adjusts the probability, then risk level,
revenue-at-risk, and the recommended action are recomputed.

``RiskState`` keeps per-customer current probability in memory so repeated
events accumulate (e.g. successive payment failures keep raising risk).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.config import Config, get_config
from src.retention.recommendations import make_decision
from src.risk import classify_risk

# Bounded heuristic deltas on churn probability per SIMULATED event type.
EVENT_DELTAS: dict[str, float] = {
    "payment_failed": +0.15,
    "complaint": +0.12,
    "support_ticket": +0.05,
    "cancellation_attempt": +0.30,
    "inactivity": +0.08,
    "plan_downgrade": +0.06,
    "login": -0.03,
    "purchase": -0.05,
    "plan_upgrade": -0.10,
}


def adjusted_probability(previous: float, event_type: str) -> float:
    """Return the clipped probability after applying the event's heuristic delta."""
    return min(1.0, max(0.0, float(previous) + EVENT_DELTAS.get(event_type, 0.0)))


@dataclass
class RiskUpdate:
    customer_id: str
    event_type: str
    timestamp: str
    previous_probability: float
    new_probability: float
    risk_change: float
    previous_risk_level: str
    new_risk_level: str
    revenue_at_risk: float
    recommended_action: str
    model_version: str
    is_simulated: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class RiskState:
    """In-memory current churn probability per customer, seeded from the service."""

    def __init__(self, service, cfg: Config | None = None) -> None:
        self.service = service
        self.cfg = cfg or get_config()
        self.prob: dict[str, float] = service.table["churn_probability"].to_dict()

    def current(self, customer_id: str) -> float:
        if customer_id not in self.prob:
            raise KeyError(customer_id)
        return self.prob[customer_id]

    def apply(self, event: dict) -> RiskUpdate:
        cid = event["customer_id"]
        etype = event["event_type"]
        prev = self.current(cid)
        new = adjusted_probability(prev, etype)
        self.prob[cid] = new

        row = self.service.table.loc[cid]
        clv = float(row["estimated_clv"])
        rar = round(new * clv, 2)
        new_level = classify_risk(new)

        feat = row.to_dict()
        feat.update({
            "churn_probability": new, "risk_level": new_level,
            "estimated_clv": clv, "revenue_at_risk": rar, "priority_score": rar,
        })
        action = make_decision(feat, cfg=self.cfg).recommended_action

        ts = event.get("timestamp") or datetime.now(timezone.utc).isoformat()
        return RiskUpdate(
            customer_id=cid,
            event_type=etype,
            timestamp=ts,
            previous_probability=round(prev, 4),
            new_probability=round(new, 4),
            risk_change=round(new - prev, 4),
            previous_risk_level=classify_risk(prev),
            new_risk_level=new_level,
            revenue_at_risk=rar,
            recommended_action=action,
            model_version=getattr(self.service, "model_version", "unknown"),
        )
