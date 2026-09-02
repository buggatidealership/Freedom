"""Walk-forward evaluation, metrics, cost-aware simulation, bootstrap and reports."""

from __future__ import annotations

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
                       holdout_last_season: bool = True) -> tuple[list[Fold], Fold | None]:
    """Expanding-window folds by earnings season (calendar quarter of t0). Test fold = one
    season; train = all events with t0 < season_start - embargo. The final season is returned
    separately as the holdout and never appears in the fold list when holdout_last_season."""
    raise NotImplementedError


def classification_metrics(p_up, y_dir) -> dict[str, float]:
    raise NotImplementedError


def regression_metrics(r_hat, r_true) -> dict[str, float]:
    raise NotImplementedError


def simulate(predictions: pd.DataFrame, *, taker_fee_bps: float, slippage_bps: float,
             funding: pd.DataFrame | None, sizing: str = "fixed") -> pd.DataFrame:
    """Per-event PnL: side = sign(p_up - 0.5) (or 0 when |p_up-0.5| < threshold), entry at the
    decision-time price, exit at t0+24h, costs = 2 * (fee + slippage), funding accrued hourly.
    Returns per-event rows with net_return and the summary via summarize_pnl."""
    raise NotImplementedError


def summarize_pnl(sim: pd.DataFrame) -> dict[str, float]:
    raise NotImplementedError


def bootstrap_ci(values: pd.Series, stat, *, n: int = 2000, block: pd.Series | None = None,
                 seed: int = 7) -> tuple[float, float, float]:
    """(point, lo95, hi95); block bootstrap by season when `block` is given."""
    raise NotImplementedError


def evaluate(settings: Settings, dataset: pd.DataFrame, *, model_names: list[str],
             decision_times: list[str], run_id: str | None = None) -> dict:
    """Runs walk-forward for each (model, decision_time), writes reports/<run_id>/ with
    summary.json, predictions.parquet, leaderboard.md; returns the summary dict."""
    raise NotImplementedError
