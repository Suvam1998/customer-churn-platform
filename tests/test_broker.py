"""Phase 17 tests: broker abstraction + producer/consumer pipeline."""
from __future__ import annotations

import pytest

from src.config import get_config
from src.streaming.broker import LocalBroker, get_broker


def test_local_broker_roundtrip(tmp_path):
    broker = LocalBroker(tmp_path)
    broker.produce("t", {"a": 1}, key="k1")
    broker.produce("t", {"a": 2}, key="k2")

    msgs = list(broker.consume("t"))
    assert [m.value["a"] for m in msgs] == [1, 2]
    assert [m.offset for m in msgs] == [0, 1]
    assert msgs[0].key == "k1"
    assert broker.count("t") == 2


def test_local_broker_from_offset(tmp_path):
    broker = LocalBroker(tmp_path)
    for i in range(5):
        broker.produce("t", {"i": i})
    tail = [m.value["i"] for m in broker.consume("t", from_offset=3)]
    assert tail == [3, 4]


def test_local_broker_purge(tmp_path):
    broker = LocalBroker(tmp_path)
    broker.produce("t", {"x": 1})
    broker.purge("t")
    assert broker.count("t") == 0


def test_get_broker_local_mode(monkeypatch):
    monkeypatch.delenv("STREAMING_MODE", raising=False)
    broker = get_broker(get_config())
    assert broker.backend == "local"


def test_producer_writes_events(tmp_path):
    from src.streaming.producer import EventProducer

    broker = LocalBroker(tmp_path)
    producer = EventProducer(broker=broker)
    events = producer.generate_and_send(10, seed=3)
    assert len(events) == 10
    assert broker.count(producer.topic) == 10


def _model_ready() -> bool:
    return (get_config().resolve_path("paths.models") / "production_model.joblib").exists()


@pytest.mark.skipif(not _model_ready(), reason="run Phase 9 first")
def test_end_to_end_pipeline(tmp_path):
    from api.services.model_service import ChurnService
    from src.streaming.consumer import EventConsumer
    from src.streaming.producer import EventProducer

    broker = LocalBroker(tmp_path)
    svc = ChurnService()
    producer = EventProducer(broker=broker)
    consumer = EventConsumer(svc, broker=broker)

    producer.generate_and_send(15, seed=5)
    updates = consumer.process_available(persist=False)

    assert len(updates) == 15
    # Risk updates were published to the risk topic.
    assert broker.count(consumer.risk_topic) == 15
    for u in updates:
        assert 0.0 <= u.new_probability <= 1.0
        assert u.is_simulated is True
        assert u.recommended_action
