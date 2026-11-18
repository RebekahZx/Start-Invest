"""End-to-end training and evaluation; produces a single artifact bundle."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import PipelineConfig
from .data import Dataset, generate
from .embeddings import TextEncoder
from .evaluator import StartupEvaluator
from .features import build_features, time_split
from .metrics import bootstrap_ci, classification_report, paired_bootstrap, per_group_ndcg, reliability_bins
from .ranker import FeedRanker, build_candidates, pair_features, startup_text, true_relevance

log = logging.getLogger(__name__)


def run(cfg: PipelineConfig, out_dir: str | Path = "artifacts") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    # 1. data -------------------------------------------------------------
    ds: Dataset = generate(cfg.n_startups, cfg.n_investors, cfg.seed)
    X = build_features(ds.snapshots, ds.startups, ds.founders, ds.rounds)
    train, val, test = time_split(X)
    log.info("snapshots train/val/test = %d/%d/%d", len(train), len(val), len(test))

    # 2. startup evaluator --------------------------------------------------
    evaluator = StartupEvaluator(cfg.n_bagged_models, cfg.seed).fit(train, val)
    p_raw, _ = evaluator.predict(test, calibrated=False)
    p_cal, p_std = evaluator.predict(test)
    y = test["label"].to_numpy()
    eval_metrics = {
        "uncalibrated": classification_report(y, p_raw),
        "calibrated": classification_report(y, p_cal),
        "baseline_last_round_recency_auc": float(roc_auc_score(y, -test["months_since_last_round"].to_numpy())),
        "random_split_auc_for_comparison": _random_split_auc(X, cfg),
        "top_features": evaluator.feature_importance().head(8).round(1).to_dict(),
        "reliability": {"uncalibrated": reliability_bins(y, p_raw), "calibrated": reliability_bins(y, p_cal)},
    }

    # 3. text embeddings ----------------------------------------------------
    texts = startup_text(ds.startups, ds.founders)
    encoder = TextEncoder(cfg.embedding_backend, seed=cfg.seed).fit(list(texts) + list(ds.investors["thesis"]))
    st_emb = dict(zip(texts.index, encoder.encode(list(texts)), strict=True))
    inv_emb = dict(zip(ds.investors["investor_id"], encoder.encode(list(ds.investors["thesis"])), strict=True))

    # 4. ranking data: candidates drawn from test-period startups (out-of-sample p)
    pool = test.copy()
    pool["p_success"], pool["p_uncertainty"] = p_cal, p_std
    pool = pool[
        ["startup_id", "p_success", "p_uncertainty", "log_total_raised", "log_last_round", "last_stage_idx", "revenue_growth_yoy", "months_since_last_round"]
    ]
    sector_of = ds.startups.set_index("startup_id")["sector"]
    pairs = build_candidates(ds.investors, pool, min(cfg.candidates_per_investor, len(pool)), rng, sector_of)
    P = pair_features(pairs, ds.investors, pool, inv_emb, st_emb)
    P["relevance"] = true_relevance(P, ds.investors, ds.startups, rng)
    sector_hit = [
        sector_of[s] in prefs
        for s, prefs in zip(P["startup_id"], P["investor_id"].map(ds.investors.set_index("investor_id")["preferred_sectors"]), strict=True)
    ]
    embedding_sector_auc = float(roc_auc_score(sector_hit, P["text_sim"]))

    inv_ids = rng.permutation(ds.investors["investor_id"].to_numpy())
    n_tr, n_va = int(0.6 * len(inv_ids)), int(0.2 * len(inv_ids))
    split = {"train": inv_ids[:n_tr], "val": inv_ids[n_tr : n_tr + n_va], "test": inv_ids[n_tr + n_va :]}
    part = {k: P[P["investor_id"].isin(v)].copy() for k, v in split.items()}

    ranker = FeedRanker(cfg.seed).fit(part["train"], part["val"])
    T = part["test"]
    T["score_ranker"] = ranker.score(T)
    T["score_random"] = rng.random(len(T))
    T["score_popularity"] = T["log_total_raised"]
    T["score_similarity"] = T["text_sim"]
    T["score_success"] = T["p_success"]
    T["score_naive_product"] = T["text_sim"].clip(lower=0) * T["p_success"]
    # strongest hand-built baseline: equal-weight z-scores of every mandate signal, per investor
    z = lambda c: T.groupby("investor_id")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    T["score_handtuned"] = z("text_sim") + z("p_success") + T["stage_fit"] - z("ticket_gap")
    names = {
        "ranker": "LambdaMART ranker",
        "random": "Random",
        "popularity": "Popularity (total raised)",
        "similarity": "Text similarity only",
        "success": "Success probability only",
        "naive_product": "Similarity x success (naive product)",
        "handtuned": "Hand-tuned mandate score (sim + success + stage + ticket)",
    }
    rank_metrics, per_inv = {}, {}
    for key, label in names.items():
        for k in (5, 10):
            per_inv[(key, k)] = per_group_ndcg(T, f"score_{key}", k)
        m10, lo, hi = bootstrap_ci(per_inv[(key, 10)])
        rank_metrics[label] = {"ndcg@5": round(float(per_inv[(key, 5)].mean()), 4), "ndcg@10": round(m10, 4), "ndcg@10_ci95": [round(lo, 4), round(hi, 4)]}
    best_base = max((k for k in names if k != "ranker"), key=lambda k: per_inv[(k, 10)].mean())
    comparison = paired_bootstrap(per_inv[("ranker", 10)], per_inv[(best_base, 10)])

    metrics = {
        "config": vars(cfg),
        "embedding_backend": encoder.backend,
        "embedding_sector_match_auc": round(embedding_sector_auc, 4),
        "data": {
            "startups": len(ds.startups),
            "rounds": len(ds.rounds),
            "investors": len(ds.investors),
            "snapshots": {"train": len(train), "val": len(val), "test": len(test)},
            "ranking_pairs": len(P),
            "test_investors": int(T["investor_id"].nunique()),
        },
        "evaluator": eval_metrics,
        "ranking": rank_metrics,
        "ranker_vs_best_baseline": {"baseline": names[best_base], **{k: round(v, 4) for k, v in comparison.items()}},
    }

    # 5. artifact bundle for the API ------------------------------------------
    feed_pool = pool.merge(ds.startups[["startup_id", "name", "sector", "country", "description"]], on="startup_id")
    bundle = {
        "evaluator": evaluator,
        "ranker": ranker,
        "encoder": encoder,
        "investors": ds.investors,
        "pool": feed_pool,
        "pool_features": test,
        "startup_emb": st_emb,
        "investor_emb": inv_emb,
        "metrics": metrics,
        "test_investor_ids": [int(i) for i in split["test"]],
    }
    joblib.dump(bundle, out / "bundle.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def _random_split_auc(X: pd.DataFrame, cfg: PipelineConfig) -> float:
    """The optimistic number you get if you ignore time: shown only to quantify the gap."""
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(X, test_size=0.3, random_state=cfg.seed, stratify=X["label"])
    tr, va = train_test_split(tr, test_size=0.2, random_state=cfg.seed, stratify=tr["label"])
    ev = StartupEvaluator(1, cfg.seed).fit(tr, va)
    p, _ = ev.predict(te, calibrated=False)
    return classification_report(te["label"].to_numpy(), p)["auc"]
