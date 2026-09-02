"""Walk-forward evaluation, metrics, cost-aware simulation, bootstrap and reports.

Decisions from review (docs/design.md §8): pinned holdout season excluded from all folds and
scored only by `evaluate(..., final=True)` which appends to settings.holdout_log_path; fills at
the OPEN of the first bar starting at or after the signal instant (never the signal bar's
close) with fill_lag recorded and trades beyond max_fill_lag_minutes dropped; execution cost
per leg = slippage_floor_bps + slippage_range_coeff * range_bps(execution bar) + taker fee;
funding accrued only with archived coverage (funding_source recorded); equal_split capital
rule with a gross exposure cap; metrics reported for has_perp_at_t0 (headline) and all events,
stratified by t0_source, kind, timing; run_id = <UTC ts>-<sha256(dataset)[:8]>.
Every cell reports n and the minimum detectable improvement over the best baseline; the
summary labels a cell "inconclusive" unless the bootstrap interval excludes it. Sizing
variants: fixed, by_confidence, by_magnitude, magnitude_gate. `final=True` refuses while any
holdout-season event has t0 + 24h in the future or targets pending.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ..config import Settings


@dataclass(frozen=True)
class Fold:
    fold: int
    train_idx: pd.Index
    test_idx: pd.Index
    test_season: str  # e.g. "2026Q2"


def walk_forward_folds(events: pd.DataFrame, *, min_train: int, embargo_days: int,
                       holdout_season: str | None) -> tuple[list[Fold], Fold | None]:
    """Expanding-window folds by season (schemas.season_of(t0)). Test fold = one season;
    train = all events with t0 < season_start - embargo_days. The holdout season (if it exists
    in `events`) is returned separately and never appears in the fold list. `events` must
    carry event_id and t0; folds index into it. Seasons with fewer than min_train prior events
    are skipped."""
    raise NotImplementedError


def classification_metrics(p_up, y_dir) -> dict[str, float]:
    """accuracy, balanced_accuracy, brier, log_loss, n; y_dir in {-1, +1}; zero labels dropped."""
    raise NotImplementedError


def regression_metrics(r_hat, r_true) -> dict[str, float]:
    """mae, rmse, spearman_ic, n."""
    raise NotImplementedError


def calibration_table(p_up, y_dir, bins: int = 10) -> pd.DataFrame:
    raise NotImplementedError


def residual_band(r_hat, r_true, lo: float = 0.1, hi: float = 0.9) -> tuple[float, float]:
    """Percentiles of r_true - r_hat over out-of-sample predictions."""
    raise NotImplementedError


def fill_price(bars: pd.DataFrame, when: pd.Timestamp, *, max_lag: pd.Timedelta) -> tuple[float, pd.Timestamp, float] | None:
    """(open, bar_start, range_bps) of the first bar with t >= when, or None if that bar
    starts more than max_lag after `when` or does not exist."""
    raise NotImplementedError


def min_detectable_improvement(n: int, metric: str, base_value: float, *, alpha: float = 0.05,
                               power: float = 0.8) -> float:
    """Smallest improvement in `metric` ('brier' or 'accuracy') over `base_value` detectable at
    sample size n with a two-sided test at `alpha` and the given power (normal approximation)."""
    raise NotImplementedError


def simulate(predictions: pd.DataFrame, paths: Callable[[str], pd.DataFrame | None], *,
             settings: Settings, funding: Callable[[str], pd.DataFrame | None] | None = None,
             sizing: str = "fixed", threshold: float = 0.0, target_vol: float = 0.03) -> pd.DataFrame:
    """Per-event trades from a predictions frame (schemas.P columns + t0, decision_time,
    has_perp_at_t0, market). `paths(event_id)` returns the event's fine bars; `funding(market)`
    returns archived hourly funding. Returns one row per prediction with side, entry/exit fill,
    fill_lag_min, cost_bps, funding_bps, funding_source, gross_return, net_return, traded."""
    raise NotImplementedError


def portfolio_metrics(trades: pd.DataFrame, *, gross_exposure_cap: float = 1.0) -> dict[str, float]:
    """equal_split capital rule over overlapping [entry, exit] intervals; daily PnL series keyed
    by UTC exit date; sharpe_like, max_drawdown, turnover, n_trades, n_untraded."""
    raise NotImplementedError


def bootstrap_ci(values: pd.Series, stat: Callable[[pd.Series], float], *, n: int = 2000,
                 block: pd.Series | None = None, seed: int = 7) -> tuple[float, float, float]:
    """(point, lo95, hi95); block bootstrap by season when `block` is given."""
    raise NotImplementedError


def evaluate(settings: Settings, dataset: pd.DataFrame, *, model_names: list[str],
             decision_times: list[str], final: bool = False, run_id: str | None = None,
             target: str = "r_24h") -> dict:
    """Walk-forward for each (model, decision_time) on folds that exclude the holdout season;
    with final=True additionally scores the holdout once and logs it. Writes
    reports/<run_id>/{summary.json, predictions.parquet, trades.parquet, leaderboard.md} and
    returns the summary dict (metrics per model/decision_time/subset, trading sim, bootstrap
    intervals, paired comparison vs best baseline, residual bands, provenance)."""
    raise NotImplementedError


def train_final(settings: Settings, dataset: pd.DataFrame, *, model_name: str, decision_time: str,
                target: str = "r_24h") -> object:
    """Fit on all non-holdout events that pass the headline filters (min_t0_confidence,
    has_perp_at_t0 when enough events exist), attach the residual band from a walk-forward pass,
    save under settings.models_dir/<decision_time>/<model_name>/ with model.json (decision_time,
    dataset_sha256, git sha, config hash, trained_at, n_events, filters, holdout reference) and
    return the model."""
    raise NotImplementedError
