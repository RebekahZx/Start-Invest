"""Point-in-time feature construction.

Every feature is computed only from events dated on or before the snapshot,
so the evaluator can never see the future it is asked to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STAGES, TRAIN_END, VAL_END

NUMERIC_FEATURES = [
    "n_rounds",
    "log_total_raised",
    "log_last_round",
    "last_stage_idx",
    "months_since_last_round",
    "months_since_founding",
    "avg_months_between_rounds",
    "n_founders",
    "max_founder_years",
    "total_prior_exits",
    "has_technical_founder",
    "revenue_growth_yoy",
    "log_customers",
    "log_headcount",
    "hiring_growth_6m",
    "web_traffic_growth_6m",
]
CATEGORICAL_FEATURES = ["sector", "country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(snapshots: pd.DataFrame, startups: pd.DataFrame, founders: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """Return one row per snapshot with FEATURES plus ids, dates and label (if present)."""
    snap = snapshots.merge(startups[["startup_id", "sector", "country", "founded"]], on="startup_id", how="left")

    # rounds visible at the snapshot: date <= snapshot_date (point-in-time join)
    r = rounds.merge(snap[["startup_id", "snapshot_date"]], on="startup_id", how="inner")
    r = r[r["date"] <= r["snapshot_date"]].sort_values(["startup_id", "date"])
    agg = r.groupby("startup_id").agg(
        n_rounds=("stage_idx", "size"),
        total_raised=("amount_usd", "sum"),
        last_round=("amount_usd", "last"),
        last_stage_idx=("stage_idx", "max"),
        last_date=("date", "max"),
        first_date=("date", "min"),
    )
    snap = snap.merge(agg, on="startup_id", how="left")
    snap["n_rounds"] = snap["n_rounds"].fillna(0).astype(int)
    snap["log_total_raised"] = np.log1p(snap["total_raised"].fillna(0.0))
    snap["log_last_round"] = np.log1p(snap["last_round"].fillna(0.0))
    snap["last_stage_idx"] = snap["last_stage_idx"].fillna(-1)
    days = lambda a, b: (a - b).dt.days / 30.44
    snap["months_since_last_round"] = days(snap["snapshot_date"], snap["last_date"])
    snap["months_since_founding"] = days(snap["snapshot_date"], snap["founded"])
    span = days(snap["last_date"], snap["first_date"])
    snap["avg_months_between_rounds"] = np.where(snap["n_rounds"] > 1, span / (snap["n_rounds"] - 1).clip(lower=1), np.nan)

    f = founders.groupby("startup_id").agg(
        n_founders=("years_experience", "size"),
        max_founder_years=("years_experience", "max"),
        total_prior_exits=("prior_exits", "sum"),
        has_technical_founder=("technical", "max"),
    )
    snap = snap.merge(f, on="startup_id", how="left")
    snap["has_technical_founder"] = snap["has_technical_founder"].astype(float)
    snap["log_customers"] = np.log1p(snap["customers"])
    snap["log_headcount"] = np.log1p(snap["headcount"])

    for c in CATEGORICAL_FEATURES:
        snap[c] = pd.Categorical(snap[c].astype(str))

    keep = ["startup_id", "snapshot_date"] + FEATURES + (["label"] if "label" in snap else [])
    return snap[keep].reset_index(drop=True)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train < 2022, validation = 2022, test = 2023 to mid-2024 (by snapshot date)."""
    tr = df[df["snapshot_date"] < TRAIN_END]
    va = df[(df["snapshot_date"] >= TRAIN_END) & (df["snapshot_date"] < VAL_END)]
    te = df[df["snapshot_date"] >= VAL_END]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def next_stage(last_stage_idx: float) -> str:
    i = int(min(max(last_stage_idx + 1, 0), len(STAGES) - 1))
    return STAGES[i]
