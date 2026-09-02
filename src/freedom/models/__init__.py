"""Model interface and registry.

v1 models (docs/design.md §7): baselines `zero`, `base_rate`, `historical_mean`,
`hist_abs_mean`, `vol_scaled`, `sign_of_reaction`, `always_extends`, `surprise_sign`;
`linear` (ridge for the return, logistic for the direction, standardised, strong L2);
`lightgbm` (num_leaves <= 7, min_data_in_leaf >= 20, feature/bagging fraction, early stopping
on an inner split, deterministic); `ensemble` (mean of members' p_up and r_hat).
Baselines that need per-name history read the f_history_* features; they must not touch the
target store directly. Every model also exposes `predict_magnitude` (default |r_hat|). No per-model interval method: the
10/90 % band comes from out-of-sample residual percentiles computed by eval and stored next to
the trained model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

REGISTRY: dict[str, type[BaseModel]] = {}


def register(name: str):
    def deco(cls: type[BaseModel]) -> type[BaseModel]:
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


class BaseModel(ABC):
    """Fit on features X (f_* columns) and targets; predict for the headline checkpoint.
    Models must be deterministic given `seed` and must tolerate NaN features (impute or use a
    NaN-aware learner) and constant columns."""

    name: str = "base"

    def __init__(self, *, seed: int = 7, **params):
        self.seed = seed
        self.params = params
        self.feature_names_: list[str] = []
        self.residual_q_: tuple[float, float] | None = None  # (q10, q90) of OOS residuals, set by eval

    @abstractmethod
    def fit(self, X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series) -> BaseModel: ...

    @abstractmethod
    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_return(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_magnitude(self, X: pd.DataFrame) -> np.ndarray:
        return np.abs(self.predict_return(X))

    def feature_importance(self) -> pd.Series | None:
        return None

    def save(self, path: Path) -> None:
        """Pickle-free persistence is not required; joblib is acceptable. Must round-trip
        residual_q_ and feature_names_."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> BaseModel:
        raise NotImplementedError


def make_model(name: str, *, seed: int = 7, **params) -> BaseModel:
    return REGISTRY[name](seed=seed, **params)


def available_models() -> list[str]:
    return sorted(REGISTRY)
