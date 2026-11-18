"""Inference service: personalised feeds and single-startup scoring from a trained bundle."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .explain import EVALUATOR_PHRASES, RANKER_PHRASES, reasons
from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from .ranker import pair_features


class FeedService:
    def __init__(self, bundle_path: str | Path = "artifacts/bundle.joblib"):
        b = joblib.load(bundle_path)
        self.evaluator, self.ranker, self.encoder = b["evaluator"], b["ranker"], b["encoder"]
        self.investors: pd.DataFrame = b["investors"].set_index("investor_id", drop=False)
        self.pool: pd.DataFrame = b["pool"]
        self.pool_features: pd.DataFrame = b["pool_features"].set_index("startup_id", drop=False)
        self.startup_emb, self.investor_emb = b["startup_emb"], b["investor_emb"]
        self.metrics = b["metrics"]

    # ---------------------------------------------------------------- feeds
    def investor(self, investor_id: int) -> dict:
        inv = self.investors.loc[investor_id]
        return {"investor_id": int(inv.investor_id), "name": inv["name"], "thesis": inv.thesis,
                "preferred_sectors": list(inv.preferred_sectors), "stages": list(inv.stages),
                "ticket_min": float(inv.ticket_min), "ticket_max": float(inv.ticket_max),
                "risk_appetite": round(float(inv.risk_appetite), 3)}

    def list_investors(self, limit: int = 50) -> list[dict]:
        return [self.investor(int(i)) for i in self.investors.index[:limit]]

    def feed(self, investor_id: int, k: int = 10, thesis_override: str | None = None) -> list[dict]:
        if investor_id not in self.investors.index:
            raise KeyError(investor_id)
        pairs = pd.DataFrame({"investor_id": investor_id, "startup_id": self.pool["startup_id"]})
        inv_emb = dict(self.investor_emb)
        if thesis_override:
            inv_emb[investor_id] = self.encoder.encode([thesis_override])[0]
        cols = [c for c in self.pool.columns if c not in ("name", "sector", "country", "description")]
        df = pair_features(pairs, self.investors.reset_index(drop=True), self.pool[cols], inv_emb, self.startup_emb)
        df["score"] = self.ranker.score(df)
        top = df.nlargest(k, "score")
        contrib = self.ranker.shap_values(top)
        meta = self.pool.set_index("startup_id")
        out = []
        for idx, row in top.iterrows():
            m = meta.loc[row.startup_id]
            out.append({
                "startup_id": int(row.startup_id), "name": m["name"], "sector": m["sector"], "country": m["country"],
                "description": m["description"], "next_stage": row.next_stage,
                "score": round(float(row.score), 4),
                "p_success": round(float(row.p_success), 3),
                "uncertainty": round(float(row.p_uncertainty), 3),
                "confidence": _confidence_label(float(row.p_uncertainty)),
                "text_sim": round(float(row.text_sim), 3),
                "reasons": reasons(contrib.loc[idx], RANKER_PHRASES, top_k=4),
            })
        return out

    # ------------------------------------------------------------- scoring
    def score_startup(self, startup_id: int | None = None, features: dict | None = None) -> dict:
        """Score a known startup by id, or an ad-hoc feature dict (missing numerics allowed)."""
        if startup_id is not None:
            row = self.pool_features.loc[[startup_id], FEATURES]
        else:
            row = pd.DataFrame([{f: (features or {}).get(f, np.nan) for f in FEATURES}])
            for c in NUMERIC_FEATURES:
                row[c] = pd.to_numeric(row[c], errors="coerce")
            for c in CATEGORICAL_FEATURES:
                row[c] = row[c].astype(str)
        p, std = self.evaluator.predict(row)
        sv = self.evaluator.shap_values(row).iloc[0]
        values = row.iloc[0].replace("nan", np.nan)
        missing = [f for f in FEATURES if pd.isna(values[f])]
        return {"p_success": round(float(p[0]), 3), "uncertainty": round(float(std[0]), 3),
                "confidence": _confidence_label(float(std[0])),
                "missing_features": missing,
                "reasons": reasons(sv, EVALUATOR_PHRASES, top_k=5, values=values)}


def _confidence_label(std: float) -> str:
    return "high" if std < 0.03 else ("medium" if std < 0.07 else "low")
