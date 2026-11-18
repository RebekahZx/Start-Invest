"""Export a static snapshot of the API so the web UI can run without a server (e.g. GitHub Pages)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import FeedService


def export(bundle: str, out: str, n_investors: int = 30, k: int = 25) -> Path:
    svc = FeedService(bundle)
    investors = svc.list_investors(n_investors)
    feeds = {str(i["investor_id"]): svc.feed(i["investor_id"], k) for i in investors}
    startup_ids = sorted({x["startup_id"] for f in feeds.values() for x in f})
    scores = {str(s): svc.score_startup(startup_id=s) for s in startup_ids}
    payload = {"mode": "static", "investors": investors, "feeds": feeds, "scores": scores, "metrics": svc.metrics}
    path = Path(out)
    path.write_text(json.dumps(payload, separators=(",", ":"), default=str))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="artifacts/bundle.joblib")
    ap.add_argument("--out", default="web/demo-data.json")
    ap.add_argument("--investors", type=int, default=30)
    ap.add_argument("--k", type=int, default=25)
    a = ap.parse_args()
    p = export(a.bundle, a.out, a.investors, a.k)
    print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
