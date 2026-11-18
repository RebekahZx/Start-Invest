"""FastAPI service exposing the feed, scoring and evaluation metrics, plus the demo UI."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .service import FeedService

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class ScoreRequest(BaseModel):
    startup_id: int | None = Field(None, description="Score a startup already in the pool")
    features: dict | None = Field(None, description="Or score an ad-hoc feature dict")


class FeedRequest(BaseModel):
    thesis: str | None = Field(None, description="Optional thesis text to re-personalise the feed")
    k: int = Field(10, ge=1, le=50)


def create_app(bundle_path: str | None = None) -> FastAPI:
    path = bundle_path or os.environ.get("STARTUP_MATCH_BUNDLE", "artifacts/bundle.joblib")
    app = FastAPI(title="Startup Match", version="0.2.0",
                  description="Personalised startup-investment feed with explanations and confidence estimates.")

    @lru_cache(maxsize=1)
    def svc() -> FeedService:
        return FeedService(path)

    @app.get("/health")
    def health():
        return {"status": "ok", "embedding_backend": svc().metrics["embedding_backend"]}

    @app.get("/metrics")
    def metrics():
        return svc().metrics

    @app.get("/investors")
    def investors(limit: int = Query(50, ge=1, le=500)):
        return svc().list_investors(limit)

    @app.get("/investors/{investor_id}")
    def investor(investor_id: int):
        try:
            return svc().investor(investor_id)
        except KeyError:
            raise HTTPException(404, f"Unknown investor {investor_id}")

    @app.get("/investors/{investor_id}/feed")
    def feed(investor_id: int, k: int = Query(10, ge=1, le=50)):
        try:
            return svc().feed(investor_id, k)
        except KeyError:
            raise HTTPException(404, f"Unknown investor {investor_id}")

    @app.post("/investors/{investor_id}/feed")
    def feed_with_thesis(investor_id: int, req: FeedRequest):
        try:
            return svc().feed(investor_id, req.k, thesis_override=req.thesis)
        except KeyError:
            raise HTTPException(404, f"Unknown investor {investor_id}")

    @app.post("/score")
    def score(req: ScoreRequest):
        if req.startup_id is None and not req.features:
            raise HTTPException(422, "Provide startup_id or features")
        try:
            return svc().score_startup(req.startup_id, req.features)
        except KeyError:
            raise HTTPException(404, f"Unknown startup {req.startup_id}")

    @app.get("/")
    def ui():
        return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
