"""`lightgbm`: two gradient-boosted tree boosters (L2 regression for the return, binary
log-loss for the direction) with the small-N settings of docs/design.md §7.

* NaN features are handled natively (missing-value splits); constant columns are never split.
* Small-N caps are enforced regardless of the parameters passed: num_leaves <= 7 and
  min_data_in_leaf >= 20; feature and bagging fractions default to 0.8 with L2 on leaf values.
* The number of rounds is chosen by early stopping on a seeded inner split (`valid_fraction`
  of the rows held out), then the booster is refitted on all rows with that many rounds.
* Determinism: `seed`, `deterministic=True`, `force_row_wise=True`, `num_threads=1`.
* A head with fewer than MIN_TRAIN_ROWS usable rows, or a direction head with one class,
  falls back to the training base rate for that head (SmallSampleWarning).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import MIN_TRAIN_ROWS, BaseModel, clip_proba, register

MAX_NUM_LEAVES = 7
MIN_DATA_IN_LEAF = 20
DEFAULT_PARAMS: dict[str, Any] = {
    "num_leaves": MAX_NUM_LEAVES,
    "max_depth": 3,
    "min_data_in_leaf": MIN_DATA_IN_LEAF,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "min_sum_hessian_in_leaf": 1e-3,
    "max_bin": 63,
}


@register("lightgbm")
class LightGBMModel(BaseModel):
    def __init__(self, *, seed: int = 7, num_boost_round: int = 500, early_stopping_rounds: int = 30,
                 valid_fraction: float = 0.2, **lgb_params):
        super().__init__(seed=seed, num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping_rounds, valid_fraction=valid_fraction,
                         **lgb_params)
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.valid_fraction = float(valid_fraction)
        merged = {**DEFAULT_PARAMS, **lgb_params}
        merged["num_leaves"] = int(min(int(merged["num_leaves"]), MAX_NUM_LEAVES))
        merged["min_data_in_leaf"] = int(max(int(merged["min_data_in_leaf"]), MIN_DATA_IN_LEAF))
        self.lgb_params: dict[str, Any] = merged
        self.booster_return_: Any = None
        self.booster_direction_: Any = None
        self.best_iterations_: dict[str, int] = {}

    def _params(self, objective: str, metric: str) -> dict[str, Any]:
        return {**self.lgb_params, "objective": objective, "metric": metric, "seed": self.seed,
                "deterministic": True, "force_row_wise": True, "num_threads": 1, "verbose": -1}

    def _train(self, M: np.ndarray, y: np.ndarray, *, objective: str, metric: str) -> Any:
        import lightgbm as lgb

        params = self._params(objective, metric)
        n = len(y)
        perm = np.random.default_rng(self.seed).permutation(n)
        n_valid = int(min(max(round(self.valid_fraction * n), 5), n // 2))
        valid_idx, train_idx = perm[:n_valid], perm[n_valid:]
        dtrain = lgb.Dataset(M[train_idx], label=y[train_idx], params=params)
        dvalid = lgb.Dataset(M[valid_idx], label=y[valid_idx], reference=dtrain, params=params)
        probe = lgb.train(params, dtrain, num_boost_round=self.num_boost_round, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
        best = max(int(probe.best_iteration or 0), 1)
        self.best_iterations_[objective] = best
        return lgb.train(params, lgb.Dataset(M, label=y, params=params), num_boost_round=best)

    def fit(self, X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series) -> BaseModel:
        M, r, d = self._fit_base_rates(X, y_return, y_direction)
        self.booster_return_ = self.booster_direction_ = None
        self.best_iterations_ = {}
        if M.shape[1] == 0:
            self._small_sample("return", 0)
            self._small_sample("direction", 0)
            return self
        ok = np.isfinite(r)
        if ok.sum() >= MIN_TRAIN_ROWS:
            self.booster_return_ = self._train(M[ok], r[ok], objective="regression", metric="l2")
        else:
            self._small_sample("return", int(ok.sum()))
        lab = np.isfinite(d) & (d != 0)
        y = (d[lab] > 0).astype(float)
        if lab.sum() >= MIN_TRAIN_ROWS and 0 < y.sum() < len(y):
            self.booster_direction_ = self._train(M[lab], y, objective="binary", metric="binary_logloss")
        else:
            self._small_sample("direction", int(lab.sum()))
        return self

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self.booster_return_ is None:
            return self._base_return(len(X))
        return np.asarray(self.booster_return_.predict(self._matrix(X), num_threads=1), dtype=float)

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self.booster_direction_ is None:
            return self._base_proba(len(X))
        return clip_proba(self.booster_direction_.predict(self._matrix(X), num_threads=1))

    # ---- introspection ------------------------------------------------------------------------
    def feature_importance(self) -> pd.Series | None:
        """Mean of the normalised gain importances over the fitted boosters."""
        parts = []
        for booster in (self.booster_return_, self.booster_direction_):
            if booster is None:
                continue
            g = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            parts.append(g / g.sum() if g.sum() > 0 else g)
        if not parts:
            return None
        return pd.Series(np.mean(parts, axis=0), index=self.feature_names_, name="importance")

    def contributions(self, X: pd.DataFrame, head: str = "return") -> pd.DataFrame:
        """Per-row SHAP-style contributions (plus a `bias` column) of the return head (sums to
        r_hat) or the direction head (sums to the logit of p_up)."""
        self._check_fitted()
        booster = self.booster_return_ if head == "return" else self.booster_direction_
        if booster is None:
            raise RuntimeError(f"{head} head fell back to the base rate; no contributions")
        contrib = booster.predict(self._matrix(X), pred_contrib=True, num_threads=1)
        return pd.DataFrame(contrib, columns=[*self.feature_names_, "bias"], index=X.index)
