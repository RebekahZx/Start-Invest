import pytest

from startup_match.config import PipelineConfig
from startup_match.data import generate
from startup_match.pipeline import run


@pytest.fixture(scope="session")
def small_data():
    return generate(n_startups=1500, n_investors=60, seed=3)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    cfg = PipelineConfig(n_startups=2500, n_investors=120, candidates_per_investor=40,
                         n_bagged_models=2, embedding_backend="lsa", seed=11)
    metrics = run(cfg, out)
    return out, metrics
