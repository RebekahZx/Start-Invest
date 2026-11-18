"""Central configuration: dates, horizons and splits."""

from dataclasses import dataclass

import pandas as pd

DATA_END = pd.Timestamp("2026-06-30")
HORIZON_DAYS = 730  # label: raises another round within 24 months of the snapshot

# Time-based split for the startup evaluator (by snapshot date).
TRAIN_END = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2023-01-01")
TEST_END = DATA_END - pd.Timedelta(days=HORIZON_DAYS)  # label must be fully observable

STAGES = ["pre-seed", "seed", "series-a", "series-b", "series-c", "series-d"]


@dataclass
class PipelineConfig:
    n_startups: int = 6000
    n_investors: int = 400
    candidates_per_investor: int = 60
    n_bagged_models: int = 5
    embedding_backend: str = "auto"  # "auto" | "sentence-transformers" | "lsa"
    seed: int = 7
