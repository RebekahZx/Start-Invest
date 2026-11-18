"""Turn SHAP contributions into short, human-readable reasons."""

from __future__ import annotations

import pandas as pd

# feature -> (phrase when it pushes the score up, phrase when it pushes it down)
EVALUATOR_PHRASES = {
    "total_prior_exits": ("Founders have prior exits", "No prior founder exits"),
    "max_founder_years": ("Experienced founding team", "Relatively inexperienced team"),
    "has_technical_founder": ("Has a technical founder", "No technical founder"),
    "revenue_growth_yoy": ("Strong revenue growth", "Weak revenue-growth evidence"),
    "log_customers": ("Large customer base", "Small customer base"),
    "hiring_growth_6m": ("Hiring momentum", "Flat or shrinking hiring"),
    "web_traffic_growth_6m": ("Growing product usage", "Declining product usage"),
    "months_since_last_round": ("Recently raised", "Long time since last round"),
    "log_total_raised": ("Well-funded so far", "Little capital raised so far"),
    "log_last_round": ("Large last round", "Small last round"),
    "n_rounds": ("Track record of raising", "Few rounds raised"),
    "last_stage_idx": ("Later-stage company", "Very early stage"),
    "avg_months_between_rounds": ("Fast fundraising cadence", "Slow fundraising cadence"),
    "months_since_founding": ("Established operating history", "Young company"),
    "log_headcount": ("Sizeable team", "Small team"),
    "n_founders": ("Multi-founder team", "Solo or small founding team"),
    "sector": ("Sector with strong funding activity", "Sector with weaker funding activity"),
    "country": ("Strong regional funding market", "Weaker regional funding market"),
}

RANKER_PHRASES = {
    "text_sim": ("Strong fit with your investment thesis", "Weak fit with your investment thesis"),
    "stage_fit": ("Raising at a stage you invest in", "Outside your stage focus"),
    "ticket_gap": ("Round size matches your ticket range", "Round size outside your ticket range"),
    "p_success": ("High estimated follow-on probability", "Low estimated follow-on probability"),
    "p_uncertainty": ("Estimate confidence suits your profile", "Estimate is uncertain"),
    "risk_appetite": ("Suits your risk appetite", "Less suited to your risk appetite"),
    "p_x_risk": ("Risk/return profile matches your style", "Risk/return profile mismatches your style"),
    "log_total_raised": ("Well-funded so far", "Little capital raised so far"),
    "revenue_growth_yoy": ("Strong revenue growth", "Weak revenue-growth evidence"),
    "months_since_last_round": ("Likely raising soon", "Not obviously raising soon"),
}


def reasons(shap_row: pd.Series, phrases: dict, top_k: int = 3, min_abs: float = 0.02, values: pd.Series | None = None) -> list[dict]:
    """Top contributions by magnitude, each as {'text', 'direction', 'impact'}.

    If `values` (the model inputs) is given, features that were missing are
    skipped: a contribution from an unknown value is not a reason to show a user.
    """
    out = []
    for feat, _ in shap_row.abs().sort_values(ascending=False).items():
        if len(out) >= top_k or abs(shap_row[feat]) < min_abs:
            break
        if values is not None and feat in values and pd.isna(values[feat]):
            continue
        up, down = phrases.get(feat, (f"{feat} raises the score", f"{feat} lowers the score"))
        positive = shap_row[feat] > 0
        text = up if positive else down
        out.append({"feature": feat, "text": text, "direction": "+" if positive else "-", "impact": round(float(shap_row[feat]), 4)})
    return out


# Short names used where every contribution is shown (tables, waterfalls).
RANKER_LABELS = {
    "text_sim": "Thesis fit",
    "stage_fit": "Stage",
    "ticket_gap": "Ticket size",
    "p_success": "Follow-on odds",
    "p_uncertainty": "Estimate certainty",
    "risk_appetite": "Your risk appetite",
    "p_x_risk": "Risk/return fit",
    "log_total_raised": "Capital raised",
    "revenue_growth_yoy": "Revenue growth",
    "months_since_last_round": "Raise timing",
}
EVALUATOR_LABELS = {
    "n_rounds": "Rounds raised",
    "log_total_raised": "Capital raised",
    "log_last_round": "Last round size",
    "last_stage_idx": "Stage",
    "months_since_last_round": "Time since last round",
    "months_since_founding": "Company age",
    "avg_months_between_rounds": "Fundraising cadence",
    "n_founders": "Founders",
    "max_founder_years": "Founder experience",
    "total_prior_exits": "Prior exits",
    "has_technical_founder": "Technical founder",
    "revenue_growth_yoy": "Revenue growth",
    "log_customers": "Customers",
    "log_headcount": "Team size",
    "hiring_growth_6m": "Hiring growth",
    "web_traffic_growth_6m": "Usage growth",
    "sector": "Sector",
    "country": "Country",
}


def contributions(shap_row: pd.Series, labels: dict, phrases: dict, values: pd.Series | None = None) -> list[dict]:
    """Every contribution, largest first, with a short label and a sentence. Missing inputs are dropped."""
    out = []
    for feat in shap_row.abs().sort_values(ascending=False).index:
        if values is not None and feat in values and pd.isna(values[feat]):
            continue
        v = float(shap_row[feat])
        up, down = phrases.get(feat, (feat, feat))
        out.append({"feature": feat, "label": labels.get(feat, feat), "impact": round(v, 4), "text": up if v > 0 else down})
    return out
