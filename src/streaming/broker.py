"""Message-broker abstraction with a Kafka backend and a local fallback.

The platform targets Apache Kafka but must run with NO Kafka installed. Both
backends share the same interface (``produce`` / ``consume``), so the producer,
consumer, and demo code are backend-agnostic.

  * ``LocalBroker`` (default) — file-backed JSONL "topics" under ``data/stream/``.
    Durable and cross-process, requires no services.
  * ``KafkaBroker`` — real Kafka via ``kafka-python`` (lazy import). Selected
    when ``streaming.mode == "kafka"`` and the client/broker are available;
    otherwise the factory falls back to ``LocalBroker`` with a warning.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.config import Config, get_config

logger = logging.getLogger("churn.broker")


@dataclass
class Message:
    topic: str
    key: str | None
    value: dict
    offset: int


class LocalBroker:
    """File-backed JSONL broker (append-only log per topic)."""

    backend = "local"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, topic: str) -> Path:
        return self.root / f"{topic}.jsonl"

    def produce(self, topic: str, value: dict, key: str | None = None) -> None:
        rec = {"key": key, "value": value}
        with self._path(topic).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def consume(self, topic: str, from_offset: int = 0) -> Iterator[Message]:
        path = self._path(topic)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for offset, line in enumerate(fh):
                if offset < from_offset or not line.strip():
                    continue
                rec = json.loads(line)
                yield Message(topic=topic, key=rec.get("key"),
                              value=rec["value"], offset=offset)

    def count(self, topic: str) -> int:
        path = self._path(topic)
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def purge(self, topic: str) -> None:
        self._path(topic).unlink(missing_ok=True)


class KafkaBroker:
    """Real Kafka backend (lazy kafka-python). Requires a running broker."""

    backend = "kafka"

    def __init__(self, bootstrap_servers: str) -> None:
        from kafka import KafkaProducer  # lazy: only needed in kafka mode

        self.bootstrap_servers = bootstrap_servers
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: (k or "").encode("utf-8"),
        )

    def produce(self, topic: str, value: dict, key: str | None = None) -> None:
        self._producer.send(topic, value=value, key=key)
        self._producer.flush()

    def consume(self, topic: str, from_offset: int = 0) -> Iterator[Message]:
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            consumer_timeout_ms=2000,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        for msg in consumer:
            yield Message(topic=topic,
                          key=msg.key.decode("utf-8") if msg.key else None,
                          value=msg.value, offset=msg.offset)
        consumer.close()


def get_broker(cfg: Config | None = None):
    """Return the configured broker, falling back to LocalBroker."""
    cfg = cfg or get_config()
    mode = cfg.get("streaming.mode", "local")
    if mode == "kafka":
        try:
            broker = KafkaBroker(cfg.get("streaming.bootstrap_servers", "localhost:9092"))
            logger.info("using KafkaBroker (%s)", broker.bootstrap_servers)
            return broker
        except Exception as exc:  # pragma: no cover - needs kafka
            logger.warning("Kafka unavailable (%s); falling back to LocalBroker", exc)
    return LocalBroker(cfg.root / "data" / "stream")
