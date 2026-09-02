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

Conventions shared by every model
---------------------------------
* Features are the `f_*` columns of X (schemas.D.feature_prefix), including the `__missing`
  companions; every other column (event_id, t0, ...) is ignored. The column list seen in `fit`
  is stored in `feature_names_`; at prediction time X is re-indexed to it, so extra columns are
  dropped and absent columns become NaN (then imputed / handled natively). Non-finite values
  (NaN, +/-inf) are treated as missing by every model.
* `y_return` is the headline log return (NaN where the event has no 24h target); `y_direction`
  is +1 / -1 (0 or NaN = unlabelled). Rows are aligned positionally with X. The return head
  trains on rows with a finite return, the direction head on rows with a +1/-1 label.
* `p_up` is a probability clipped to [PROBA_EPS, 1 - PROBA_EPS]; `predict_return` is in
  log-return units; both return float ndarrays of len(X) (empty for an empty X).
* Every fit records the training base rates (`up_rate_`, `mean_return_`, `mean_abs_return_`);
  a learner with fewer than MIN_TRAIN_ROWS usable rows for a head falls back to those constants
  for that head and emits `SmallSampleWarning`, so tiny folds degrade to `base_rate`.
* `save(path)` / `load(path)` use joblib; `path` may be a directory (the model is written as
  `<dir>/model.joblib`, leaving room for eval's model.json next to it) or a file. The whole
  object round-trips, in particular `feature_names_` and `residual_q_`.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..schemas import D

log = logging.getLogger(__name__)

REGISTRY: dict[str, type[BaseModel]] = {}
ALIASES: dict[str, str] = {"ridge": "linear", "logistic": "linear"}  # design §7 names -> v1 model

PROBA_EPS = 1e-3  # p_up is clipped to [PROBA_EPS, 1 - PROBA_EPS]
MIN_TRAIN_ROWS = 30  # a head with fewer usable rows falls back to the base rate
MODEL_FILENAME = "model.joblib"


class SmallSampleWarning(UserWarning):
    """A learner had too few usable training rows and fell back to base-rate behaviour."""


def register(name: str):
    def deco(cls: type[BaseModel]) -> type[BaseModel]:
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


# ---- small shared helpers (also used by the model modules) ------------------------------------
def feature_columns(X: pd.DataFrame) -> list[str]:
    """The feature columns of a dataset frame: every column starting with schemas.D.feature_prefix."""
    return [c for c in X.columns if str(c).startswith(D.feature_prefix)]


def clip_proba(p) -> np.ndarray:
    """Clip probabilities into the open interval (0, 1) so log-loss is always finite."""
    return np.clip(np.asarray(p, dtype=float), PROBA_EPS, 1.0 - PROBA_EPS)


def sign_proba(signal) -> np.ndarray:
    """0.5 + 0.25 * sign(signal): 0.75 up, 0.25 down, 0.5 for a zero signal, NaN for NaN."""
    return 0.5 + 0.25 * np.sign(np.asarray(signal, dtype=float))


def target_arrays(y_return, y_direction) -> tuple[np.ndarray, np.ndarray]:
    """Coerce the targets to float arrays: returns (NaN where missing) and directions in
    {-1, 0, +1, NaN}. A missing `y_direction` is derived from the sign of the return."""
    r = pd.to_numeric(pd.Series(np.asarray(y_return, dtype=object).ravel()), errors="coerce")
    r = r.to_numpy(dtype=float)
    if y_direction is None:
        d = np.sign(r)
    else:
        d = pd.to_numeric(pd.Series(np.asarray(y_direction, dtype=object).ravel()), errors="coerce")
        d = np.sign(d.to_numpy(dtype=float))
    if len(d) != len(r):
        raise ValueError(f"y_return has {len(r)} rows but y_direction has {len(d)}")
    return r, d


def _model_file(path: Path, *, for_write: bool = False) -> Path:
    p = Path(path)
    if p.is_dir() or (not p.exists() and p.suffix == ""):
        if for_write:
            p.mkdir(parents=True, exist_ok=True)
        return p / MODEL_FILENAME
    if for_write:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


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
        self.is_fitted_: bool = False
        self.n_train_: int = 0  # rows with a finite return target
        self.n_direction_: int = 0  # rows with a +1 / -1 direction label
        self.up_rate_: float = 0.5  # training-window up rate (clipped into (0, 1))
        self.mean_return_: float = 0.0  # training mean of y_return
        self.mean_abs_return_: float = 0.0  # training mean of |y_return|

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, seed={self.seed}, params={self.params})"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series) -> BaseModel: ...

    @abstractmethod
    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_return(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_magnitude(self, X: pd.DataFrame) -> np.ndarray:
        return np.abs(self.predict_return(X))

    def predict_band(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(r_lo, r_hi) = r_hat + residual_q_; NaN arrays until eval has set residual_q_."""
        r_hat = np.asarray(self.predict_return(X), dtype=float)
        if self.residual_q_ is None:
            nan = np.full(len(r_hat), np.nan)
            return nan, nan.copy()
        q10, q90 = self.residual_q_
        return r_hat + float(q10), r_hat + float(q90)

    def feature_importance(self) -> pd.Series | None:
        return None

    # ---- helpers for subclasses -------------------------------------------------------------
    def _fit_base_rates(self, X: pd.DataFrame, y_return, y_direction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Record the feature list and the training base rates. Returns (M, r, d): the float
        feature matrix aligned to feature_names_, the return target (NaN where missing) and
        the direction in {-1, 0, +1, NaN}."""
        self.feature_names_ = feature_columns(X)
        r, d = target_arrays(y_return, y_direction)
        if len(r) != len(X):
            raise ValueError(f"X has {len(X)} rows but the targets have {len(r)}")
        ok = np.isfinite(r)
        self.n_train_ = int(ok.sum())
        self.mean_return_ = float(r[ok].mean()) if self.n_train_ else 0.0
        self.mean_abs_return_ = float(np.abs(r[ok]).mean()) if self.n_train_ else 0.0
        lab = d[np.isfinite(d) & (d != 0)]
        self.n_direction_ = int(len(lab))
        self.up_rate_ = float(clip_proba((lab > 0).mean())) if len(lab) else 0.5
        self.is_fitted_ = True
        return self._matrix(X), r, d

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Float matrix of X aligned to feature_names_ (absent columns -> NaN, extras dropped);
        +/-inf becomes NaN so it is imputed / split on like a missing value."""
        cols = self.feature_names_
        if not cols:
            return np.empty((len(X), 0), dtype=float)
        M = X.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(M), M, np.nan)

    @staticmethod
    def _column(X: pd.DataFrame, name: str) -> np.ndarray:
        """One feature as a float array; all-NaN when the column is absent."""
        if name not in X.columns:
            return np.full(len(X), np.nan)
        return pd.to_numeric(X[name], errors="coerce").to_numpy(dtype=float)

    def _base_proba(self, n: int) -> np.ndarray:
        return np.full(n, self.up_rate_, dtype=float)

    def _base_return(self, n: int) -> np.ndarray:
        return np.full(n, self.mean_return_, dtype=float)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"model {self.name!r} is not fitted")

    def _small_sample(self, head: str, n: int) -> None:
        msg = (f"{self.name}: only {n} usable rows for the {head} head (< {MIN_TRAIN_ROWS}); "
               "falling back to the training base rate")
        log.warning(msg)
        warnings.warn(msg, SmallSampleWarning, stacklevel=3)

    # ---- persistence ----------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Pickle-free persistence is not required; joblib is acceptable. Must round-trip
        residual_q_ and feature_names_."""
        joblib.dump(self, _model_file(path, for_write=True))

    @classmethod
    def load(cls, path: Path) -> BaseModel:
        obj = joblib.load(_model_file(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} holds a {type(obj).__name__}, not a {cls.__name__}")
        return obj


def make_model(name: str, *, seed: int = 7, **params) -> BaseModel:
    key = ALIASES.get(name, name)
    if key not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(available_models())}")
    return REGISTRY[key](seed=seed, **params)


def available_models() -> list[str]:
    return sorted(REGISTRY)


# Importing the model modules registers the v1 models; keep this after the definitions above.
from . import baselines, ensemble, lgbm, linear  # noqa: E402, F401
