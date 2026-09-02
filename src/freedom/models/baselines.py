"""Baseline models (docs/design.md §7). Every comparison in the reports is against the best
baseline per metric, so these are deliberately simple and their semantics are fixed here.

All baselines learn only the training base rates (`up_rate_`, `mean_return_`,
`mean_abs_return_`) plus, for `vol_scaled` and `always_extends`, one scalar from the training
distribution. Rows that lack the feature a baseline reads get the base-rate prediction
("pooled fallback"); no baseline touches the target store, per-name history arrives through the
history features.

| name             | p_up                                   | r_hat                                  | magnitude              |
|------------------|----------------------------------------|----------------------------------------|------------------------|
| zero             | 0.5                                    | 0                                      | 0                      |
| base_rate        | training up rate                       | training mean r                        | |r_hat|                |
| historical_mean  | 0.5 +/- 0.25 * sign(name mean r)       | name's past mean r (pooled fallback)   | |r_hat|                |
| hist_abs_mean    | training up rate                       | 0                                      | name's past mean |r|   |
| vol_scaled       | training up rate                       | sigma_h * mean(z)                      | sigma_h * mean(|z|)    |
| sign_of_reaction | 0.5 +/- 0.25 * sign(r_k)               | r_k (the reaction holds)               | |r_hat|                |
| always_extends   | 0.5 +/- 0.25 * sign(r_k)               | r_k + sign(r_k) * mean |r_24h - r_k|   | |r_hat|                |
| surprise_sign    | 0.5 +/- 0.25 * sign(EPS surprise)      | sign(surprise) * training mean |r|     | |r_hat|                |

Feature lookup: each baseline has a `features` preference tuple (F_* constants below) and per
row uses the first candidate that is present and finite. The tuples list both naming
conventions seen for the same quantity — `f_<group>_<key>` and the flat `f_<key>` that
`features.build_features` writes as D.feature_prefix + key — so either dataset works unchanged;
pass `features=` to override.

`vol_scaled`: sigma_h = rvol_20d * sqrt(horizon_hours / 24), i.e. the 20-day realised vol of
daily log returns (one trading day, close to close, which already spans an overnight) scaled
by the square root of the horizon in days; z = r_24h / sigma_h over the training rows supplies
the mean, the mean absolute value and the quantiles. Because everything is calibrated on the
training z, a different constant in the horizon scaling would cancel out; the convention only
matters for interpreting sigma_h.

`sign_of_reaction` / `always_extends` read the reaction at the decision time: r_now when the
reaction group emits it, else the longest reaction window present in the row (r_60m, r_30m,
r_15m, r_5m, r_1m), which is r_k at decision time post_k. At pre decision times no reaction
exists and both return the base rate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import BaseModel, register, sign_proba

F_HISTORY_MEAN = ("f_history_mean_r24h", "f_hist_r24_mean")
F_HISTORY_ABS_MEAN = ("f_history_mean_abs_r24h", "f_hist_abs_r24_mean")
F_RVOL_20D = ("f_pre_price_rvol_20d", "f_rvol_20d")
F_REACTION = ("f_reaction_r_now", "f_reaction_r_60m", "f_reaction_r_30m", "f_reaction_r_15m",
              "f_reaction_r_5m", "f_reaction_r_1m",
              "f_r_now", "f_r_60m", "f_r_30m", "f_r_15m", "f_r_5m", "f_r_1m")
F_EPS_SURPRISE = ("f_surprise_eps_surprise_pct", "f_eps_surprise")


def _names(features: str | Sequence[str]) -> tuple[str, ...]:
    return (features,) if isinstance(features, str) else tuple(features)


class _Baseline(BaseModel):
    """Shared fit: record the feature list and the training base rates."""

    def fit(self, X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series) -> BaseModel:
        self._fit_base_rates(X, y_return, y_direction)
        return self

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._base_proba(len(X))

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._base_return(len(X))

    def _first_available(self, X: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
        """Per row, the first finite value over `names` (NaN when none is present)."""
        out = np.full(len(X), np.nan)
        for name in names:
            if name not in X.columns:
                continue
            col = self._column(X, name)
            fill = np.isnan(out) & np.isfinite(col)
            out[fill] = col[fill]
        return out


class _FeatureBaseline(_Baseline):
    """A baseline that reads one quantity through a preference tuple of column names."""

    default_features: tuple[str, ...] = ()

    def __init__(self, *, seed: int = 7, features: str | Sequence[str] | None = None, **params):
        names = self.default_features if features is None else _names(features)
        super().__init__(seed=seed, features=names, **params)
        self.features = names

    def _signal(self, X: pd.DataFrame) -> np.ndarray:
        return self._first_available(X, self.features)


@register("zero")
class Zero(_Baseline):
    """r = 0, p = 0.5. Needs no fitting, but fit() still records the feature list."""

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.5)

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


@register("base_rate")
class BaseRate(_Baseline):
    """p = training-window up rate, r = training mean of y_return."""


@register("historical_mean")
class HistoricalMean(_FeatureBaseline):
    """r = the name's past mean r_24h (history feature), pooled fallback = training mean;
    p = 0.5 +/- 0.25 * sign of that mean where the name has history, else the base rate."""

    default_features = F_HISTORY_MEAN

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        s = self._signal(X)
        return np.where(np.isfinite(s), s, self.mean_return_)

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        s = self._signal(X)
        return np.where(np.isfinite(s), sign_proba(s), self.up_rate_)


@register("hist_abs_mean")
class HistAbsMean(_FeatureBaseline):
    """Magnitude baseline: |r_24h| = the name's past mean |r_24h| (history feature), pooled
    fallback = training mean |r|. No directional view: r = 0, p = base rate."""

    default_features = F_HISTORY_ABS_MEAN

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.zeros(len(X))

    def predict_magnitude(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        s = self._signal(X)
        return np.where(np.isfinite(s) & (s >= 0), s, self.mean_abs_return_)


@register("vol_scaled")
class VolScaled(_FeatureBaseline):
    """sigma_h from the 20-day realised vol scaled to the horizon; return, magnitude and
    quantiles from the training distribution of z = r_24h / sigma_h (see module docstring).
    Rows without a positive rvol use the pooled sigma (training mean sigma_h)."""

    default_features = F_RVOL_20D
    QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)

    def __init__(self, *, seed: int = 7, features: str | Sequence[str] | None = None,
                 horizon_hours: float = 24.0, **params):
        super().__init__(seed=seed, features=features, horizon_hours=horizon_hours, **params)
        self.horizon_hours = float(horizon_hours)
        self.pooled_sigma_: float = np.nan
        self.z_mean_: float = 0.0
        self.z_abs_mean_: float = 0.0
        self.z_quantiles_: dict[float, float] = {}

    def _sigma_raw(self, X: pd.DataFrame) -> np.ndarray:
        rvol = self._signal(X)
        rvol = np.where(np.isfinite(rvol) & (rvol > 0), rvol, np.nan)
        return rvol * np.sqrt(self.horizon_hours / 24.0)

    def sigma_h(self, X: pd.DataFrame) -> np.ndarray:
        """Horizon volatility per row with the pooled fallback applied."""
        self._check_fitted()
        s = self._sigma_raw(X)
        return np.where(np.isfinite(s), s, self.pooled_sigma_)

    def fit(self, X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series) -> BaseModel:
        _, r, _ = self._fit_base_rates(X, y_return, y_direction)
        sig = self._sigma_raw(X)
        ok_s = np.isfinite(sig)
        ok_r = np.isfinite(r)
        if ok_s.any():
            self.pooled_sigma_ = float(sig[ok_s].mean())
        elif ok_r.sum() > 1 and float(r[ok_r].std()) > 0:
            self.pooled_sigma_ = float(r[ok_r].std())
        else:
            self.pooled_sigma_ = 1.0
        both = ok_s & ok_r
        z = r[both] / sig[both] if both.any() else r[ok_r] / self.pooled_sigma_
        if len(z) == 0:
            z = np.zeros(1)
        self.z_mean_ = float(z.mean())
        self.z_abs_mean_ = float(np.abs(z).mean())
        self.z_quantiles_ = {q: float(np.quantile(z, q)) for q in self.QUANTILES}
        return self

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        return self.sigma_h(X) * self.z_mean_

    def predict_magnitude(self, X: pd.DataFrame) -> np.ndarray:
        return self.sigma_h(X) * self.z_abs_mean_

    def predict_quantile(self, X: pd.DataFrame, q: float) -> np.ndarray:
        """sigma_h * the training quantile of z (q in VolScaled.QUANTILES)."""
        if q not in self.z_quantiles_:
            raise KeyError(f"quantile {q} not stored; available: {sorted(self.z_quantiles_)}")
        return self.sigma_h(X) * self.z_quantiles_[q]


class _SignBaseline(_FeatureBaseline):
    """p_up = 0.5 + 0.25 * sign(signal); rows without a signal get the base rate."""

    def _return_from_signal(self, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        s = self._signal(X)
        return np.where(np.isfinite(s), sign_proba(s), self.up_rate_)

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        s = self._signal(X)
        ok = np.isfinite(s)
        r = self._return_from_signal(np.where(ok, s, 0.0))
        return np.where(ok, r, self.mean_return_)


@register("sign_of_reaction")
class SignOfReaction(_SignBaseline):
    """Post decision times: direction = sign(r_k), r_hat = r_k (the early reaction holds)."""

    default_features = F_REACTION

    def _return_from_signal(self, s: np.ndarray) -> np.ndarray:
        return s


@register("always_extends")
class AlwaysExtends(SignOfReaction):
    """continuation = +1: the early reaction extends by the typical further move,
    r_hat = r_k + sign(r_k) * mean |r_24h - r_k| over the training rows (pooled fallback:
    training mean |r_24h|). Direction = sign(r_k) as for sign_of_reaction."""

    def __init__(self, *, seed: int = 7, features: str | Sequence[str] | None = None, **params):
        super().__init__(seed=seed, features=features, **params)
        self.extension_: float = 0.0

    def fit(self, X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series) -> BaseModel:
        _, r, _ = self._fit_base_rates(X, y_return, y_direction)
        rk = self._signal(X)
        both = np.isfinite(r) & np.isfinite(rk)
        self.extension_ = float(np.abs(r[both] - rk[both]).mean()) if both.any() else self.mean_abs_return_
        return self

    def _return_from_signal(self, s: np.ndarray) -> np.ndarray:
        return s + np.sign(s) * self.extension_


@register("surprise_sign")
class SurpriseSign(_SignBaseline):
    """Direction = sign of the EPS surprise; r_hat = that sign times the training mean |r|."""

    default_features = F_EPS_SURPRISE

    def _return_from_signal(self, s: np.ndarray) -> np.ndarray:
        return np.sign(s) * self.mean_abs_return_
