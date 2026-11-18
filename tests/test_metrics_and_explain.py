import numpy as np
import pandas as pd
import pytest

from startup_match.explain import reasons
from startup_match.metrics import expected_calibration_error, ndcg_at_k, paired_bootstrap


def test_ndcg_perfect_and_reversed():
    rel = np.array([3, 2, 1, 0])
    assert ndcg_at_k(rel, np.array([4, 3, 2, 1]), k=4) == pytest.approx(1.0)
    assert ndcg_at_k(rel, np.array([1, 2, 3, 4]), k=4) < 0.6


def test_ndcg_known_value():
    # one relevant item (gain 1) at rank 2: DCG = 1/log2(3), IDCG = 1
    assert ndcg_at_k(np.array([0, 1, 0]), np.array([3, 2, 1]), k=3) == pytest.approx(1 / np.log2(3))


def test_ndcg_all_zero_relevance():
    assert ndcg_at_k(np.zeros(5), np.arange(5), k=3) == 0.0


def test_ece_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(1)
    b = rng.uniform(size=200)
    r = paired_bootstrap(b + 0.05, b)
    assert r["ci_low"] > 0 and r["p_value"] < 0.01


def test_reasons_order_sign_and_missing():
    sv = pd.Series({"revenue_growth_yoy": 0.5, "months_since_last_round": -0.8, "log_customers": 0.01})
    phrases = {"revenue_growth_yoy": ("Strong revenue growth", "Weak revenue"),
               "months_since_last_round": ("Recently raised", "Long time since last round")}
    r = reasons(sv, phrases, top_k=3)
    assert [x["feature"] for x in r] == ["months_since_last_round", "revenue_growth_yoy"]  # tiny one dropped
    assert r[0]["direction"] == "-" and r[0]["text"] == "Long time since last round"
    r2 = reasons(sv, phrases, values=pd.Series({"months_since_last_round": np.nan, "revenue_growth_yoy": 1.0}))
    assert [x["feature"] for x in r2] == ["revenue_growth_yoy"]
