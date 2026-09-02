"""Price paths and targets around each event. Output uses schemas.T columns."""

from __future__ import annotations

import pandas as pd

from ..config import Settings


def checkpoint_times(t0: pd.Timestamp, horizon_hours: int = 24) -> dict[str, pd.Timestamp]:
    """Map schemas.CHECKPOINTS -> UTC instants. next_open/next_close use the XNYS calendar;
    next_open_30m = next_open + 30 min; 24h = t0 + horizon_hours."""
    raise NotImplementedError


def price_at(bars: pd.DataFrame, when: pd.Timestamp, *, strictly_before: bool = False) -> tuple[float, pd.Timestamp] | None:
    """Close of the last bar with t_end <= when (or t_end < when when strictly_before).
    Bars are half-open [t, t_end); a bar containing `when` is never used. Returns
    (price, bar_end_time) or None."""
    raise NotImplementedError


def build_price_path(settings: Settings, event: pd.Series, *, market_bars: pd.DataFrame | None,
                     equity_bars: pd.DataFrame | None) -> pd.DataFrame:
    """Merged 1-minute (or coarser) bars covering [t0 - 2 days, t0 + horizon + 1 day] with a
    source column: Hyperliquid archive/live where available, else FMP extended-hours."""
    raise NotImplementedError


def compute_targets(event: pd.Series, path: pd.DataFrame, market_path: pd.DataFrame | None,
                    *, horizon_hours: int = 24) -> pd.Series:
    """p0 = price strictly before t0; r_<cp> = ln(p_cp / p0); ar_<cp> = r_<cp> - r_<cp>(market);
    labels direction/magnitude/continuation. Missing checkpoints stay NaN."""
    raise NotImplementedError


def build_targets(settings: Settings, events: pd.DataFrame, *, write: bool = True) -> pd.DataFrame:
    raise NotImplementedError
