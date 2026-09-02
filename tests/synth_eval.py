"""Synthetic dataset, price paths, funding and stand-in models for the eval tests.

Everything is generated from a fixed seed and nothing touches the network. The stand-in
models (`zero`, `base_rate`, `hist_abs_mean`, `linear`) are registered only when the models
package does not already provide them, so these tests keep working once the real models land.
"""

from __future__ import annotations

import json
import warnings
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from freedom import models as models_mod
from freedom.eval.folds import season_start
from freedom.schemas import CHECKPOINTS, DECISION_TIMES, C, D, E, T, season_of

PERP_LISTING = pd.Timestamp("2025-11-12", tz="UTC")
SEASONS = ["2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]
SIGNAL = 0.02  # r_24h = SIGNAL * f_a + NOISE * eps
NOISE = 0.03


def make_dataset(*, n_names: int = 40, seasons: list[str] | None = None,
                 decision_times: tuple[str, ...] = ("pre_5m", "post_30m"), seed: int = 0) -> pd.DataFrame:
    """features.build_dataset-shaped frame: one row per (event, decision_time)."""
    rng = np.random.default_rng(seed)
    seasons = seasons or SEASONS
    rows: list[dict] = []
    for s in seasons:
        start = season_start(s)
        for i in range(n_names):
            name = f"N{i:02d}"
            day = start + pd.Timedelta(days=int(rng.integers(15, 80)))
            while day.weekday() >= 5:
                day += pd.Timedelta(days=1)
            amc = rng.random() < 0.8
            minute = int(rng.integers(20 * 60 + 5, 21 * 60 + 30)) if amc else int(rng.integers(11 * 60, 12 * 60))
            t0 = day + pd.Timedelta(minutes=minute)
            fa, fb, fc = rng.normal(size=3)
            r24 = SIGNAL * fa + NOISE * rng.normal()
            r30 = 0.3 * r24 + 0.01 * rng.normal()
            u = rng.random()
            if u < 0.85:
                src, conf = "sec_8k", 0.95
            elif u < 0.95:
                src, conf = "detected", 0.8
            else:
                src, conf = "calendar_flag", 0.3
            missing = rng.random() < 0.03
            kind = "equity_us" if rng.random() < 0.8 else "equity_fpi"
            has_perp = bool(t0 >= PERP_LISTING)
            event_id = f"{name}:{t0.year}-{t0.month:02d}"
            for d in decision_times:
                offset = DECISION_TIMES[d]
                row: dict = {
                    D.event_id: event_id, D.decision_time: d, D.as_of: t0 + pd.Timedelta(minutes=offset),
                    "f_a": fa, "f_b": fb, "f_c": fc, "f_a__missing": False,
                    "f_r_30m": r30 if offset >= 30 else np.nan, "f_r_30m__missing": offset < 30,
                }
                for cp in CHECKPOINTS:
                    frac = {"5m": 0.1, "15m": 0.2, "30m": 0.3, "60m": 0.4, "2h": 0.5, "next_open": 0.7,
                            "next_open_30m": 0.75, "next_close": 0.9, "24h": 1.0}[cp]
                    val = np.nan if missing else (r24 if cp == "24h" else (r30 if cp == "30m" else frac * r24))
                    row[T.r(cp)] = val
                    row[T.ar(cp)] = val - 0.001 if not missing else np.nan
                row[T.direction] = np.nan if missing else float(np.sign(r24))
                row[T.magnitude] = np.nan if missing else abs(r24)
                cont = np.nan if missing or abs(r30) < 0.0025 else float(np.sign(r30) * np.sign(r24 - r30))
                row[T.continuation_30m] = cont
                row[T.continuation_15m] = cont
                row.update({
                    E.t0: t0, E.t0_source: src, E.t0_confidence: conf, E.kind: kind,
                    E.timing: "AMC" if amc else "BMO", E.has_perp_at_t0: has_perp, E.market: f"xyz:{name}",
                    "season": season_of(t0), T.price_source: "hl_archive" if has_perp else "fmp_intraday",
                    "target_missing": missing,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def make_bars(t0: pd.Timestamp, r24: float, *, interval_min: int = 5, seed: int = 0, p0: float = 100.0,
              market: str = "xyz:TEST", before_h: float = 2.0, after_h: float = 28.0) -> pd.DataFrame:
    """Fine bars around t0 whose log price walks linearly from p0 at t0 to p0 * exp(r24) at t0 + 24h."""
    rng = np.random.default_rng(seed)
    step = pd.Timedelta(minutes=interval_min)
    start = (t0 - pd.Timedelta(hours=before_h)).floor(f"{interval_min}min")
    n = int(pd.Timedelta(hours=before_h + after_h) / step)
    t = pd.date_range(start, periods=n, freq=step)
    t_end = t + step
    frac = np.clip(((t_end - t0) / pd.Timedelta(hours=24)).to_numpy(dtype=float), 0.0, 1.0)
    close = p0 * np.exp(r24 * frac + 0.0005 * rng.normal(size=n))
    open_ = np.concatenate([[p0], close[:-1]])
    high = np.maximum(open_, close) * 1.0005
    low = np.minimum(open_, close) * 0.9995
    return pd.DataFrame({
        C.market: market, C.interval: f"{interval_min}m", C.t: t, C.t_end: t_end, C.open: open_, C.high: high,
        C.low: low, C.close: close, C.volume: 1000.0, C.n_trades: 10, C.source: "hl_archive",
    })


def make_paths(dataset: pd.DataFrame, *, interval_min: int = 5) -> Callable[[str], pd.DataFrame | None]:
    """paths(event_id) -> synthetic bars consistent with the event's r_24h; None when the
    event's targets are missing (no path)."""
    first = dataset.drop_duplicates(D.event_id).set_index(D.event_id)
    cache: dict[str, pd.DataFrame | None] = {}

    def paths(event_id: str) -> pd.DataFrame | None:
        if event_id in cache:
            return cache[event_id]
        if event_id not in first.index:
            return None
        ev = first.loc[event_id]
        r24 = ev[T.r("24h")]
        if pd.isna(r24):
            cache[event_id] = None
            return None
        bars = make_bars(pd.Timestamp(ev[E.t0]), float(r24), interval_min=interval_min,
                         seed=zlib.crc32(event_id.encode()), market=str(ev[E.market]))
        cache[event_id] = bars
        return bars

    return paths


def make_funding(rate: float = 1e-5, start: str = "2025-11-01", end: str = "2026-10-01",
                 drop_hours: list[pd.Timestamp] | None = None) -> Callable[[str], pd.DataFrame]:
    hours = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="h")
    if drop_hours:
        hours = hours[~hours.isin(drop_hours)]

    def funding(market: str) -> pd.DataFrame:
        return pd.DataFrame({"market": market, "t": hours, "funding_rate": rate, "premium": 0.0})

    return funding


# ---- stand-in models --------------------------------------------------------------------------------
class _Fake(models_mod.BaseModel):
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "fake_model.json").write_text(json.dumps(
            {"name": self.name, "residual_q_": self.residual_q_, "feature_names_": self.feature_names_}))

    @classmethod
    def load(cls, path: Path):
        meta = json.loads((path / "fake_model.json").read_text())
        m = cls()
        m.residual_q_ = tuple(meta["residual_q_"]) if meta["residual_q_"] else None
        m.feature_names_ = meta["feature_names_"]
        return m


class ZeroModel(_Fake):
    def fit(self, X, y_return, y_direction):
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba_up(self, X):
        return np.full(len(X), 0.5)

    def predict_return(self, X):
        return np.zeros(len(X))


class BaseRateModel(_Fake):
    def fit(self, X, y_return, y_direction):
        self.feature_names_ = list(X.columns)
        self.p_ = float(np.mean(np.asarray(y_direction) > 0))
        self.r_ = float(np.mean(y_return))
        return self

    def predict_proba_up(self, X):
        return np.full(len(X), self.p_)

    def predict_return(self, X):
        return np.full(len(X), self.r_)


class HistAbsMeanModel(_Fake):
    """The magnitude baseline's shape: base-rate direction, no return forecast (r_hat = 0)
    and the training mean |r| as the magnitude forecast."""

    def fit(self, X, y_return, y_direction):
        self.feature_names_ = list(X.columns)
        self.p_ = float(np.mean(np.asarray(y_direction) > 0))
        self.abs_ = float(np.mean(np.abs(np.asarray(y_return, dtype=float))))
        return self

    def predict_proba_up(self, X):
        return np.full(len(X), self.p_)

    def predict_return(self, X):
        return np.zeros(len(X))

    def predict_magnitude(self, X):
        return np.full(len(X), self.abs_)


class LinearModel(_Fake):
    """Closed-form ridge for the return and a strongly regularised logistic for the direction,
    on standardised features with NaN imputed to the training mean."""

    def _z(self, X):
        A = X.to_numpy(dtype=float) if len(X.columns) else np.zeros((len(X), 0))
        A = np.where(np.isnan(A), self.mu_, A)
        return (A - self.mu_) / self.sd_

    def fit(self, X, y_return, y_direction):
        self.feature_names_ = list(X.columns)
        A = X.to_numpy(dtype=float) if len(X.columns) else np.zeros((len(X), 0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (pre-release rows)
            self.mu_ = np.nan_to_num(np.nanmean(A, axis=0)) if A.shape[1] else np.zeros(0)
            self.sd_ = np.nan_to_num(np.nanstd(A, axis=0)) + 1e-9 if A.shape[1] else np.ones(0)
        Z = self._z(X)
        y = np.asarray(y_return, dtype=float)
        self.b_ = float(y.mean())
        k = Z.shape[1]
        alpha = float(self.params.get("alpha", 1.0))
        self.w_ = np.linalg.solve(Z.T @ Z + alpha * np.eye(k), Z.T @ (y - self.b_)) if k else np.zeros(0)
        up = (np.asarray(y_direction) > 0).astype(int)
        self.clf_ = None
        self.p_const_ = float(up.mean())
        if k and 0 < up.sum() < len(up):
            from sklearn.linear_model import LogisticRegression

            self.clf_ = LogisticRegression(C=0.5, random_state=self.seed).fit(Z, up)
        return self

    def predict_proba_up(self, X):
        if self.clf_ is None:
            return np.full(len(X), self.p_const_)
        return self.clf_.predict_proba(self._z(X))[:, 1]

    def predict_return(self, X):
        return self._z(X) @ self.w_ + self.b_


FAKES = {"zero": ZeroModel, "base_rate": BaseRateModel, "hist_abs_mean": HistAbsMeanModel, "linear": LinearModel}


def register_fakes() -> None:
    for name, cls in FAKES.items():
        if name not in models_mod.REGISTRY:
            models_mod.register(name)(cls)
