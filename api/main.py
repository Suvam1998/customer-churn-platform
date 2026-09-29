"""FastAPI application entrypoint.

Phase 1 wires up the app factory, structured request logging, a global error
handler, and the health router. Prediction / customer / event / metric routers
are added in later phases.
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src import __version__
from src.config import get_config
from api.routes import customers, events, health, metrics, predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("churn.api")


def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(
        title=cfg.get("api.title", "Churn & Retention Platform API"),
        version=__version__,
        description=(
            "Real-Time Customer Churn Prediction & Intelligent Retention "
            "Platform. NOTE: real-time events are SIMULATED (see README)."
        ),
    )

    @app.middleware("http")
    async def add_latency_header(request: Request, call_next):
        """Structured request log + prediction/API latency header."""
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Process-Time-ms"] = str(latency_ms)
        logger.info(
            "request path=%s method=%s status=%s latency_ms=%s",
            request.url.path, request.method, response.status_code, latency_ms,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled error path=%s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "path": request.url.path},
        )

    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(customers.router)
    app.include_router(events.router)
    app.include_router(metrics.router)

    @app.get("/", tags=["root"], summary="API root")
    def root() -> dict:
        return {
            "name": app.title,
            "version": app.version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
