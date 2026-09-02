"""`ensemble`: (weighted) mean of the members' p_up, r_hat and magnitude.

Members are given by registry name (default `linear` + `lightgbm`, each built with the
ensemble's seed and any `member_params[name]`) or as already-constructed BaseModel instances.
Every member is fitted on the same (X, y_return, y_direction). The magnitude is the mean of the
members' `predict_magnitude`, not |mean r_hat|, so a magnitude-only member (e.g. hist_abs_mean)
contributes its magnitude even though its r_hat is 0.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import BaseModel, clip_proba, make_model, register

DEFAULT_MEMBERS = ("linear", "lightgbm")


@register("ensemble")
class Ensemble(BaseModel):
    def __init__(self, *, seed: int = 7, members: Sequence[str | BaseModel] = DEFAULT_MEMBERS,
                 member_params: dict[str, dict] | None = None, weights: Sequence[float] | None = None,
                 **params):
        super().__init__(seed=seed, members=tuple(members), member_params=member_params,
                         weights=None if weights is None else tuple(weights), **params)
        if len(members) == 0:
            raise ValueError("ensemble needs at least one member")
        member_params = member_params or {}
        self.members_: list[BaseModel] = [
            m if isinstance(m, BaseModel) else make_model(m, seed=seed, **member_params.get(m, {}))
            for m in members
        ]
        w = np.ones(len(self.members_)) if weights is None else np.asarray(weights, dtype=float)
        if len(w) != len(self.members_) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per member, not all zero")
        self.weights_: np.ndarray = w / w.sum()

    @property
    def member_names(self) -> list[str]:
        return [m.name for m in self.members_]

    def fit(self, X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series) -> BaseModel:
        self._fit_base_rates(X, y_return, y_direction)
        for m in self.members_:
            m.fit(X, y_return, y_direction)
        return self

    def _mean(self, preds: list[np.ndarray]) -> np.ndarray:
        stack = np.vstack([np.asarray(p, dtype=float) for p in preds])
        return np.average(stack, axis=0, weights=self.weights_)

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return clip_proba(self._mean([m.predict_proba_up(X) for m in self.members_]))

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._mean([m.predict_return(X) for m in self.members_])

    def predict_magnitude(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._mean([m.predict_magnitude(X) for m in self.members_])

    def feature_importance(self) -> pd.Series | None:
        parts = [(m.feature_importance(), w) for m, w in zip(self.members_, self.weights_, strict=True)]
        parts = [(s, w) for s, w in parts if s is not None]
        if not parts:
            return None
        frame = pd.concat([s.rename(i) for i, (s, _) in enumerate(parts)], axis=1).fillna(0.0)
        w = np.asarray([w for _, w in parts], dtype=float)
        return pd.Series(frame.to_numpy() @ (w / w.sum()), index=frame.index, name="importance")
