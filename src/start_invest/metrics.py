"""Evaluation metrics with uncertainty, so improvements are tested rather than assumed."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def dcg(rel: np.ndarray, k: int) -> float:
    rel = np.asarray(rel, float)[:k]
    return float(((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2))).sum())


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-np.asarray(scores), kind="stable")
    ideal = dcg(np.sort(relevance)[::-1], k)
    return dcg(np.asarray(relevance)[order], k) / ideal if ideal > 0 else 0.0


def per_group_ndcg(df: pd.DataFrame, score_col: str, k: int = 10, group: str = "investor_id") -> np.ndarray:
    return np.array([ndcg_at_k(g["relevance"].to_numpy(), g[score_col].to_numpy(), k) for _, g in df.groupby(group)])


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> dict:
    """Mean difference a-b with 95% CI and one-sided p-value for a > b, resampling groups."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_value": float((means <= 0).mean()),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_report(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "base_rate": float(y.mean()),
        "n": len(y),
    }


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Mean predicted vs observed rate per probability bin (for reliability plots)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            out.append(
                {
                    "bin_low": float(edges[b]),
                    "bin_high": float(edges[b + 1]),
                    "n": int(m.sum()),
                    "mean_predicted": float(p[m].mean()),
                    "observed_rate": float(y[m].mean()),
                }
            )
    return out
