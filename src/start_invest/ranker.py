"""Investor-startup matching and learning-to-rank.

Investors are described by a free-text thesis, a stage range, a ticket range
and a risk appetite. Each (investor, startup) pair gets features from three
sources: thesis-to-startup text similarity, mandate fit (stage, ticket), and
the evaluator's out-of-sample success estimate. A LambdaMART ranker learns how
to trade these off per investor profile, instead of multiplying two scores.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from .data import STAGE_AMOUNT_USD
from .features import next_stage

PAIR_FEATURES = [
    "text_sim",
    "stage_fit",
    "ticket_gap",
    "p_success",
    "p_uncertainty",
    "risk_appetite",
    "p_x_risk",
    "log_total_raised",
    "revenue_growth_yoy",
    "months_since_last_round",
]


def startup_text(startups: pd.DataFrame, founders: pd.DataFrame) -> pd.Series:
    bios = founders.groupby("startup_id")["bio"].apply(" ".join)
    s = startups.set_index("startup_id")
    return (s["description"] + " Founders: " + bios.reindex(s.index).fillna("")).rename("text")


def _ticket_gap(expected_ticket: np.ndarray, tmin: np.ndarray, tmax: np.ndarray) -> np.ndarray:
    """0 when the ticket falls inside the investor's range, else |log distance| to it."""
    below = np.log(tmin / expected_ticket).clip(min=0)
    above = np.log(expected_ticket / tmax).clip(min=0)
    return below + above


def build_candidates(investors: pd.DataFrame, pool: pd.DataFrame, per_investor: int, rng: np.random.Generator, sector_of: pd.Series) -> pd.DataFrame:
    """Sample a candidate list per investor: ~40% from preferred sectors, rest random."""
    rows = []
    ids = pool["startup_id"].to_numpy()
    sectors = sector_of.reindex(ids).to_numpy()
    for inv in investors.itertuples():
        in_pref = ids[np.isin(sectors, inv.preferred_sectors)]
        k_pref = min(len(in_pref), int(per_investor * 0.4))
        chosen = set(rng.choice(in_pref, size=k_pref, replace=False)) if k_pref else set()
        rest = ids[~np.isin(ids, list(chosen))]
        chosen |= set(rng.choice(rest, size=per_investor - len(chosen), replace=False))
        rows += [(inv.investor_id, sid) for sid in chosen]
    return pd.DataFrame(rows, columns=["investor_id", "startup_id"])


def pair_features(
    pairs: pd.DataFrame, investors: pd.DataFrame, pool: pd.DataFrame, inv_emb: dict[int, np.ndarray], st_emb: dict[int, np.ndarray]
) -> pd.DataFrame:
    """Observable features for each (investor, startup) pair. `pool` holds evaluator outputs."""
    df = pairs.merge(investors[["investor_id", "stages", "ticket_min", "ticket_max", "risk_appetite"]], on="investor_id").merge(pool, on="startup_id")
    A = np.vstack([inv_emb[i] for i in df["investor_id"]])
    B = np.vstack([st_emb[s] for s in df["startup_id"]])
    df["text_sim"] = (A * B).sum(axis=1)
    nxt = df["last_stage_idx"].map(next_stage)
    df["next_stage"] = nxt
    df["stage_fit"] = [float(s in st) for s, st in zip(nxt, df["stages"], strict=True)]
    # observable proxy for the next round's size: last round x step-up, investor takes ~15%
    expected_round = np.expm1(df["log_last_round"]).clip(lower=1e5) * 2.5
    df["expected_ticket"] = 0.15 * expected_round
    df["ticket_gap"] = _ticket_gap(df["expected_ticket"].to_numpy(), df["ticket_min"].to_numpy(), df["ticket_max"].to_numpy())
    df["p_x_risk"] = df["p_success"] * (df["risk_appetite"] - 0.5)
    return df


def true_relevance(df: pd.DataFrame, investors: pd.DataFrame, startups: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Hidden ground truth: graded 0-3 interest, from true sector, mandate fit and quality.

    Uses latent variables the model never sees directly (true sector preference,
    true quality, true next-round size), so the ranker has to infer them.
    """
    inv = investors.set_index("investor_id")
    st = startups.set_index("startup_id")
    sector_match = np.array([st.at[s, "sector"] in inv.at[i, "preferred_sectors"] for i, s in zip(df["investor_id"], df["startup_id"], strict=True)], float)
    q = st.loc[df["startup_id"], "q_true"].to_numpy()
    risk = df["risk_appetite"].to_numpy()
    true_ticket = np.array([0.15 * STAGE_AMOUNT_USD[n] for n in df["next_stage"]]) * np.exp(0.35 * q)
    ticket_ok = (_ticket_gap(true_ticket, df["ticket_min"].to_numpy(), df["ticket_max"].to_numpy()) < 0.5).astype(float)
    traction = df["revenue_growth_yoy"].clip(-1, 3).to_numpy()
    score = (
        2.4 * sector_match
        + 1.6 * df["stage_fit"].to_numpy()
        + 0.9 * ticket_ok
        + (0.3 + 1.0 * risk) * q
        + (1.0 - risk) * 0.5 * traction
        + rng.normal(0, 0.6, len(df))
    )
    # grade within each investor: top 10% -> 3, next 15% -> 2, next 25% -> 1
    pct = pd.Series(score).groupby(df["investor_id"].to_numpy()).rank(pct=True, ascending=False).to_numpy()
    return np.select([pct <= 0.10, pct <= 0.25, pct <= 0.50], [3, 2, 1], 0)


class FeedRanker:
    def __init__(self, seed: int = 7):
        self.seed = seed
        self.model: lgb.Booster | None = None
        self._explainer: shap.TreeExplainer | None = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> FeedRanker:
        train = train.sort_values("investor_id")
        val = val.sort_values("investor_id")
        params = dict(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[10],
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            feature_fraction=0.9,
            seed=self.seed,
            verbose=-1,
        )
        dtr = lgb.Dataset(train[PAIR_FEATURES], train["relevance"], group=train.groupby("investor_id").size().to_numpy())
        dva = lgb.Dataset(val[PAIR_FEATURES], val["relevance"], group=val.groupby("investor_id").size().to_numpy(), reference=dtr)
        self.model = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        self._explainer = None
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[PAIR_FEATURES], num_iteration=self.model.best_iteration)

    def shap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return pd.DataFrame(np.asarray(self._explainer.shap_values(df[PAIR_FEATURES])), columns=PAIR_FEATURES, index=df.index)

    def __getstate__(self):
        return {"seed": self.seed, "model": self.model.model_to_string(), "_explainer": None, "best_iteration": self.model.best_iteration}

    def __setstate__(self, state):
        self.seed = state["seed"]
        self.model = lgb.Booster(model_str=state["model"])
        self.model.best_iteration = state["best_iteration"]
        self._explainer = None
