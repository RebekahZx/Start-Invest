import joblib
import pytest
from fastapi.testclient import TestClient

from start_invest.api import create_app


def test_evaluator_beats_chance_and_is_calibrated(trained):
    _, m = trained
    cal = m["evaluator"]["calibrated"]
    assert cal["auc"] > 0.65
    assert cal["ece"] < 0.08


def test_ranker_beats_every_baseline(trained):
    _, m = trained
    ranker = m["ranking"]["LambdaMART ranker"]["ndcg@10"]
    for name, r in m["ranking"].items():
        if name != "LambdaMART ranker":
            assert ranker > r["ndcg@10"], name
    assert m["ranker_vs_best_baseline"]["ci_low"] > 0


def test_embeddings_capture_sector_fit(trained):
    _, m = trained
    assert m["embedding_sector_match_auc"] > 0.75


def test_bundle_round_trips(trained):
    out, _ = trained
    b = joblib.load(out / "bundle.joblib")
    p, sd = b["evaluator"].predict(b["pool_features"].head(5))
    assert ((p >= 0) & (p <= 1)).all() and (sd >= 0).all()


@pytest.fixture(scope="module")
def client(trained):
    out, _ = trained
    return TestClient(create_app(str(out / "bundle.joblib")))


def test_health_and_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "ranking" in client.get("/metrics").json()


def test_feed_contract(client):
    inv = client.get("/investors?limit=1").json()[0]
    feed = client.get(f"/investors/{inv['investor_id']}/feed?k=5").json()
    assert len(feed) == 5
    assert [x["score"] for x in feed] == sorted((x["score"] for x in feed), reverse=True)
    for x in feed:
        assert 0 <= x["p_success"] <= 1
        assert x["confidence"] in {"high", "medium", "low"}
        assert 1 <= len(x["reasons"]) <= 4


def test_thesis_override_changes_feed(client):
    inv = client.get("/investors?limit=1").json()[0]
    iid = inv["investor_id"]
    health = client.post(f"/investors/{iid}/feed", json={"k": 10, "thesis": "clinical automation, medical imaging and patient care for hospitals"}).json()
    farms = client.post(f"/investors/{iid}/feed", json={"k": 10, "thesis": "crop yield prediction, soil sensing and farm marketplaces for farmers"}).json()
    share = lambda feed, s: sum(x["sector"] == s for x in feed) / len(feed)
    assert share(health, "healthtech") > share(farms, "healthtech")
    assert share(farms, "agritech") > share(health, "agritech")


def test_score_endpoints(client):
    inv = client.get("/investors?limit=1").json()[0]
    sid = client.get(f"/investors/{inv['investor_id']}/feed?k=1").json()[0]["startup_id"]
    r = client.post("/score", json={"startup_id": sid}).json()
    assert 0 <= r["p_success"] <= 1 and r["reasons"]
    adhoc = client.post("/score", json={"features": {"sector": "fintech", "country": "IN", "revenue_growth_yoy": 1.5}}).json()
    assert "months_since_last_round" in adhoc["missing_features"]
    assert all(x["feature"] not in adhoc["missing_features"] for x in adhoc["reasons"])


def test_errors(client):
    assert client.get("/investors/999999/feed").status_code == 404
    assert client.post("/score", json={}).status_code == 422
