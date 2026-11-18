"""FastAPI service exposing the feed, scoring and evaluation metrics, plus the web UI."""

from __future__ import annotations

import logging
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .service import FeedService

log = logging.getLogger("start_invest.api")
DEFAULT_WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class ScoreRequest(BaseModel):
    startup_id: int | None = Field(None, description="Score a startup already in the pool")
    features: dict | None = Field(None, description="Or score an ad-hoc feature dict; unknown fields are ignored")


class FeedRequest(BaseModel):
    thesis: str | None = Field(None, max_length=4000, description="Optional thesis text to re-personalise the feed")
    k: int = Field(10, ge=1, le=50)


def create_app(bundle_path: str | None = None, web_dir: str | Path | None = None) -> FastAPI:
    path = bundle_path or os.environ.get("START_INVEST_BUNDLE", "artifacts/bundle.joblib")
    web = Path(web_dir or os.environ.get("START_INVEST_WEB", DEFAULT_WEB_DIR))
    app = FastAPI(
        title="Start-Invest",
        version=__version__,
        description="Personalised startup-investment feed with explanations and confidence estimates. All data is synthetic.",
    )

    origins = [o for o in os.environ.get("START_INVEST_CORS_ORIGINS", "").split(",") if o]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = rid
        response.headers["server-timing"] = f"app;dur={ms:.1f}"
        log.info("%s %s %s %.1fms rid=%s", request.method, request.url.path, response.status_code, ms, rid)
        return response

    @lru_cache(maxsize=1)
    def svc() -> FeedService:
        return FeedService(path)

    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok", "version": __version__, "embedding_backend": svc().metrics["embedding_backend"]}

    @app.get("/metrics", tags=["model"])
    def metrics():
        return svc().metrics

    @app.get("/investors", tags=["investors"])
    def investors(limit: int = Query(50, ge=1, le=500)):
        return svc().list_investors(limit)

    @app.get("/investors/{investor_id}", tags=["investors"])
    def investor(investor_id: int):
        try:
            return svc().investor(investor_id)
        except KeyError:
            raise HTTPException(404, f"No investor with id {investor_id}")

    @app.get("/investors/{investor_id}/feed", tags=["feed"])
    def feed(investor_id: int, k: int = Query(10, ge=1, le=50)):
        try:
            return svc().feed(investor_id, k)
        except KeyError:
            raise HTTPException(404, f"No investor with id {investor_id}")

    @app.post("/investors/{investor_id}/feed", tags=["feed"])
    def feed_with_thesis(investor_id: int, req: FeedRequest):
        try:
            return svc().feed(investor_id, req.k, thesis_override=req.thesis)
        except KeyError:
            raise HTTPException(404, f"No investor with id {investor_id}")

    @app.post("/score", tags=["model"])
    def score(req: ScoreRequest):
        if req.startup_id is None and not req.features:
            raise HTTPException(422, "Send either startup_id or features")
        try:
            return svc().score_startup(req.startup_id, req.features)
        except KeyError:
            raise HTTPException(404, f"No startup with id {req.startup_id}")

    if web.is_dir():

        @app.get("/", include_in_schema=False)
        def ui():
            return FileResponse(web / "index.html")

        app.mount("/web", StaticFiles(directory=web), name="web")

    return app


app = create_app()
