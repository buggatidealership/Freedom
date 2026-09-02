"""Model interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
    """Fit on features X (f_* columns) and targets; predict for the headline checkpoint."""

    name: str = "base"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def fit(self, X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series) -> BaseModel: ...

    @abstractmethod
    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_return(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        r = self.predict_return(X)
        return r, r

    def feature_importance(self) -> pd.Series | None:
        return None

    def save(self, path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> BaseModel:
        raise NotImplementedError


def make_model(name: str, **params) -> BaseModel:
    return REGISTRY[name](**params)
