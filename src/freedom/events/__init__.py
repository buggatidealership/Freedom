"""Earnings event table and release-time resolver. Output uses schemas.E columns."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import Settings


@dataclass(frozen=True)
class ResolvedT0:
    t0: pd.Timestamp  # UTC
    confidence: float  # 0..1
    source: str  # schemas.T0Source
    detail: str = ""


def resolve_release_time(
    *,
    report_date_ny: pd.Timestamp,
    sec_filings: pd.DataFrame | None,
    intraday: pd.DataFrame | None,
    calendar_flag: str | None,
    manual: pd.Timestamp | None = None,
) -> ResolvedT0:
    """Priority: manual > SEC 8-K item 2.02 acceptance (refined by detection when both agree
    within 15 min) > detection from 1-minute extended-hours bars > calendar flag mapped to
    16:05 / 07:00 America/New_York. Confidence: manual 1.0, sec_8k 0.95, detected 0.8,
    calendar_flag 0.5; SEC and detection disagreeing by > 15 min lowers confidence to 0.7 and
    records the disagreement in `detail`."""
    raise NotImplementedError


def detect_release_from_bars(bars: pd.DataFrame, report_date_ny: pd.Timestamp,
                             *, vol_z_threshold: float = 6.0, abs_ret_threshold: float = 0.01,
                             baseline_days: int = 10) -> pd.Timestamp | None:
    """First 1-minute bar on the report date (New York) whose volume z-score against the same
    clock-minute over the previous `baseline_days` sessions exceeds `vol_z_threshold` and whose
    |return| over the bar exceeds `abs_ret_threshold`. Returns the bar START time (UTC)."""
    raise NotImplementedError


def build_events(settings: Settings, *, underlyings: list[str] | None = None,
                 since: pd.Timestamp | None = None, write: bool = True) -> pd.DataFrame:
    """Assemble events from FMP earnings history (primary), SEC filings (timing), Alpha Vantage
    and Nasdaq (optional cross-checks). Never silently resolves a date disagreement > 1 day:
    such rows get flags='date_conflict' and confidence 0."""
    raise NotImplementedError


def load_events(settings: Settings) -> pd.DataFrame:
    raise NotImplementedError


def upcoming_events(settings: Settings, days: int = 14) -> pd.DataFrame:
    """Future events for the event universe from the FMP calendar."""
    raise NotImplementedError
