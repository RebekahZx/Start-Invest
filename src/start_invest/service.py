"""Inference service: personalised feeds and single-startup scoring from a trained bundle."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .explain import EVALUATOR_LABELS, EVALUATOR_PHRASES, RANKER_LABELS, RANKER_PHRASES, contributions, reasons
from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from .ranker import pair_features

META_COLS = ("name", "sector", "country", "description")


class FeedService:
    def __init__(self, bundle_path: str | Path = "artifacts/bundle.joblib"):
        b = joblib.load(bundle_path)
        self.evaluator, self.ranker, self.encoder = b["evaluator"], b["ranker"], b["encoder"]
        self.investors: pd.DataFrame = b["investors"].set_index("investor_id", drop=False)
        self.pool: pd.DataFrame = b["pool"]
        self.pool_features: pd.DataFrame = b["pool_features"].set_index("startup_id", drop=False)
        self.startup_emb, self.investor_emb = b["startup_emb"], b["investor_emb"]
        self.metrics = b["metrics"]
        self.featured_investor_ids: list[int] = b.get("test_investor_ids", [])
        # Confidence is relative: terciles of ensemble disagreement across the pool.
        self._conf_cuts = np.quantile(self.pool["p_uncertainty"], [1 / 3, 2 / 3])

    # -------------------------------------------------------------- investors
    def investor(self, investor_id: int) -> dict:
        inv = self.investors.loc[investor_id]
        return {
            "investor_id": int(inv.investor_id),
            "name": inv["name"],
            "thesis": inv.thesis,
            "preferred_sectors": list(inv.preferred_sectors),
            "stages": list(inv.stages),
            "ticket_min": float(inv.ticket_min),
            "ticket_max": float(inv.ticket_max),
            "risk_appetite": round(float(inv.risk_appetite), 3),
            "held_out": int(inv.investor_id) in self.featured_investor_ids,
        }

    def list_investors(self, limit: int = 50) -> list[dict]:
        """Held-out investors first: the ranker never trained on their relevance labels."""
        ids = self.featured_investor_ids + [int(i) for i in self.investors.index if int(i) not in set(self.featured_investor_ids)]
        return [self.investor(i) for i in ids[:limit]]

    # ------------------------------------------------------------------- feed
    def feed(self, investor_id: int, k: int = 10, thesis_override: str | None = None) -> list[dict]:
        if investor_id not in self.investors.index:
            raise KeyError(investor_id)
        pairs = pd.DataFrame({"investor_id": investor_id, "startup_id": self.pool["startup_id"]})
        inv_emb = dict(self.investor_emb)
        if thesis_override:
            inv_emb[investor_id] = self.encoder.encode([thesis_override])[0]
        cols = [c for c in self.pool.columns if c not in META_COLS]
        df = pair_features(pairs, self.investors.reset_index(drop=True), self.pool[cols], inv_emb, self.startup_emb)
        df["score"] = self.ranker.score(df)
        top = df.nlargest(k, "score")
        contrib = self.ranker.shap_values(top)
        meta = self.pool.set_index("startup_id")
        out = []
        for rank, (idx, row) in enumerate(top.iterrows(), start=1):
            m = meta.loc[row.startup_id]
            items = contributions(contrib.loc[idx], RANKER_LABELS, RANKER_PHRASES)
            top_reasons = reasons(contrib.loc[idx], RANKER_PHRASES, top_k=4)
            # SHAP is relative to an average candidate: a ticket that is outside the range but
            # closer than most can still contribute positively. Never claim it "matches".
            if row.ticket_gap > 0:
                for c in items + top_reasons:
                    if c["feature"] == "ticket_gap" and c["impact"] > 0:
                        c["text"] = "Ticket is outside your range, but closer than most"
            out.append(
                {
                    "rank": rank,
                    "startup_id": int(row.startup_id),
                    "name": m["name"],
                    "sector": m["sector"],
                    "country": m["country"],
                    "description": m["description"],
                    "next_stage": row.next_stage,
                    "score": round(float(row.score), 4),
                    "p_success": round(float(row.p_success), 3),
                    "uncertainty": round(float(row.p_uncertainty), 4),
                    "confidence": self._confidence_label(float(row.p_uncertainty)),
                    "text_sim": round(float(row.text_sim), 3),
                    "facts": {
                        "total_raised_usd": round(float(np.expm1(row.log_total_raised)), 0),
                        "last_round_usd": round(float(np.expm1(row.log_last_round)), 0),
                        "months_since_last_round": round(float(row.months_since_last_round), 1),
                        "revenue_growth_yoy": round(float(row.revenue_growth_yoy), 3),
                        "stage_fit": bool(row.stage_fit),
                        "expected_ticket_usd": round(float(row.expected_ticket), 0),
                        "ticket_in_range": bool(row.ticket_gap == 0),
                    },
                    "reasons": top_reasons,
                    "contributions": items,
                }
            )
        return out

    # ---------------------------------------------------------------- scoring
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
        return {
            "p_success": round(float(p[0]), 3),
            "uncertainty": round(float(std[0]), 4),
            "confidence": self._confidence_label(float(std[0])),
            "missing_features": missing,
            "reasons": reasons(sv, EVALUATOR_PHRASES, top_k=5, values=values),
            "contributions": contributions(sv, EVALUATOR_LABELS, EVALUATOR_PHRASES, values=values),
        }

    def _confidence_label(self, std: float) -> str:
        lo, hi = self._conf_cuts
        return "high" if std <= lo else ("medium" if std <= hi else "low")
