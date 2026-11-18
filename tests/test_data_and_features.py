import numpy as np
import pandas as pd

from startup_match.config import HORIZON_DAYS, TEST_END, TRAIN_END, VAL_END
from startup_match.features import FEATURES, build_features, time_split


def test_shapes_and_label_rate(small_data):
    d = small_data
    assert len(d.startups) == 1500 and len(d.investors) == 60
    assert d.snapshots["snapshot_date"].max() < TEST_END
    rate = d.snapshots["label"].mean()
    assert 0.1 < rate < 0.6, rate


def test_label_matches_definition(small_data):
    d = small_data
    s = d.snapshots.iloc[:200]
    for row in s.itertuples():
        dates = d.rounds.loc[d.rounds.startup_id == row.startup_id, "date"]
        expect = ((dates > row.snapshot_date) & (dates <= row.snapshot_date + pd.Timedelta(days=HORIZON_DAYS))).any()
        assert row.label == int(expect)


def test_features_ignore_future_rounds(small_data):
    """Point-in-time correctness: adding or removing rounds after the snapshot must not change features."""
    d = small_data
    base = build_features(d.snapshots, d.startups, d.founders, d.rounds)
    snap = d.snapshots.set_index("startup_id")["snapshot_date"]
    future = d.rounds["date"] > d.rounds["startup_id"].map(snap)
    assert future.any()
    without_future = build_features(d.snapshots, d.startups, d.founders, d.rounds[~future])
    num = [f for f in FEATURES if f not in ("sector", "country")]
    pd.testing.assert_frame_equal(base[num], without_future[num])

    fake = d.rounds[future].copy()
    fake["amount_usd"] *= 100
    with_fake = build_features(d.snapshots, d.startups, d.founders, pd.concat([d.rounds[~future], fake]))
    pd.testing.assert_frame_equal(base[num], with_fake[num])


def test_time_split_is_ordered_and_disjoint(small_data):
    d = small_data
    tr, va, te = time_split(build_features(d.snapshots, d.startups, d.founders, d.rounds))
    assert tr["snapshot_date"].max() < TRAIN_END <= va["snapshot_date"].min()
    assert va["snapshot_date"].max() < VAL_END <= te["snapshot_date"].min()
    assert not set(tr.startup_id) & set(te.startup_id)


def test_generation_is_deterministic():
    from startup_match.data import generate
    a, b = generate(300, 10, seed=5), generate(300, 10, seed=5)
    pd.testing.assert_frame_equal(a.rounds, b.rounds)
    assert np.array_equal(a.snapshots["label"], b.snapshots["label"])
