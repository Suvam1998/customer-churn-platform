"""Engine / session management.

Resolves the database URL in this order:
  1. ``DATABASE_URL`` env var (e.g. a PostgreSQL DSN in production/Docker), else
  2. build a PostgreSQL DSN from POSTGRES_* env vars if POSTGRES_HOST is set, else
  3. a local SQLite file under ``data/`` — so the platform runs with NO Postgres.

This keeps local development and CI dependency-free while remaining fully
PostgreSQL-compatible.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import Config, get_config
from src.db.models import Base

logger = logging.getLogger("churn.db")


def get_database_url(cfg: Config | None = None) -> str:
    cfg = cfg or get_config()

    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]

    if os.getenv("POSTGRES_HOST"):
        user = os.getenv("POSTGRES_USER", "churn_user")
        pwd = os.getenv("POSTGRES_PASSWORD", "")
        host = os.environ["POSTGRES_HOST"]
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "churn")
        return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

    # SQLite fallback.
    data_dir = cfg.root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(data_dir / 'churn.db').as_posix()}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, future=True, connect_args=connect_args)
    logger.info("database engine: %s", engine.url.render_as_string(hide_password=True))
    return engine


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, class_=Session)


def get_session() -> Session:
    """Return a new Session (caller manages the lifecycle)."""
    return _session_factory()()


def init_db(engine: Engine | None = None, drop: bool = False) -> None:
    """Create all tables (optionally dropping first)."""
    engine = engine or get_engine()
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
