"""Startup evaluator: P(raises another round within 24 months).

A bagged ensemble of LightGBM classifiers, early-stopped on a later time
window, then Platt-calibrated (logistic regression on the log-odds) on that
same window. Platt scaling is smooth and monotone, so it never collapses many
startups onto the same probability the way isotonic steps can. The spread
across the ensemble is reported as a confidence estimate.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from .features import CATEGORICAL_FEATURES, FEATURES

PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
)


class StartupEvaluator:
    def __init__(self, n_models: int = 5, seed: int = 7):
        self.n_models = n_models
        self.seed = seed
        self.models: list[lgb.Booster] = []
        self.calibrator: LogisticRegression | None = None
        self.categories: dict[str, list[str]] = {}
        self._explainers: list[shap.TreeExplainer] | None = None

    # ------------------------------------------------------------------ data
    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].copy()
        for c in CATEGORICAL_FEATURES:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.categories[c])
        return X

    # ----------------------------------------------------------------- train
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> StartupEvaluator:
        self.categories = {c: sorted(pd.concat([train[c], val[c]]).astype(str).unique()) for c in CATEGORICAL_FEATURES}
        Xtr, Xva = self._prep(train), self._prep(val)
        self.models = []
        for i in range(self.n_models):
            p = dict(PARAMS, seed=self.seed + i)
            dtr = lgb.Dataset(Xtr, train["label"], categorical_feature=CATEGORICAL_FEATURES)
            dva = lgb.Dataset(Xva, val["label"], reference=dtr)
            m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            self.models.append(m)
        raw = self._raw(Xva).mean(axis=0)
        self.calibrator = LogisticRegression(C=1e4).fit(_logit(raw)[:, None], val["label"])
        self._explainers = None
        return self

    # --------------------------------------------------------------- predict
    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        return np.vstack([m.predict(X, num_iteration=m.best_iteration) for m in self.models])

    def predict(self, df: pd.DataFrame, calibrated: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return (probability, uncertainty = std across the ensemble)."""
        raw = self._raw(self._prep(df))
        mean, std = raw.mean(axis=0), raw.std(axis=0)
        if calibrated and self.calibrator is not None:
            mean = self.calibrator.predict_proba(_logit(mean)[:, None])[:, 1]
        return mean, std

    # --------------------------------------------------------------- explain
    def shap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contributions (log-odds), averaged across the ensemble."""
        if self._explainers is None:
            self._explainers = [shap.TreeExplainer(m) for m in self.models]
        X = self._prep(df)
        vals = np.mean([_positive_class(e, X) for e in self._explainers], axis=0)
        return pd.DataFrame(vals, columns=FEATURES, index=df.index)

    def feature_importance(self) -> pd.Series:
        imp = np.mean([m.feature_importance("gain") for m in self.models], axis=0)
        return pd.Series(imp, index=FEATURES).sort_values(ascending=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_explainers"] = None
        state["models"] = [m.model_to_string() for m in self.models]
        return state

    def __setstate__(self, state):
        state["models"] = [lgb.Booster(model_str=s) for s in state["models"]]
        self.__dict__.update(state)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _positive_class(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for the positive class, across SHAP versions' output formats."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        v = explainer.shap_values(X)
    if isinstance(v, list):
        v = v[-1]
    v = np.asarray(v)
    return v[..., -1] if v.ndim == 3 else v
