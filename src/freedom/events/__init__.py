"""Earnings event table and release-time resolver. Output uses schemas.E columns.

Decisions from review (docs/design.md §2, §5):
* detection may only move an 8-K time EARLIER (within 15 min); a later detection never changes
  t0 and is recorded as `t0_lag_s`; 8-K confidence does not depend on detection;
* a detection on the very first bar of the extended session is flagged `detection_first_bar`
  and downgraded to calendar-flag confidence;
* calendar-flag defaults: AMC -> 16:05, BMO -> 07:00 America/New_York, confidence 0.5;
* event_id = f"{underlying}:{fiscal_period}" with fiscal_period = fiscal quarter-end month;
* vendor estimates for past events are `estimate_source = fmp_final` (not point-in-time);
  archived consensus snapshots (data/archive/consensus/*.parquet) win when they exist and
  give `estimate_source = consensus_snapshot` with `estimate_snapshot_time`;
* budget exhaustion is a checkpoint: rows already resolved are written, the rest are marked
  `pending = True`, the command exits non-zero, and a rerun resumes from the cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import Settings

CONFIDENCE = {"manual": 1.0, "sec_8k": 0.95, "detected": 0.8, "calendar_flag": 0.5}
DETECTION_WINDOW = pd.Timedelta(minutes=15)


@dataclass(frozen=True)
class ResolvedT0:
    t0: pd.Timestamp  # UTC
    confidence: float  # 0..1
    source: str  # schemas.T0Source value
    detail: str = ""
    t0_lag_s: float | None = None  # acceptance - detected start, seconds (8-K events only)
    flags: tuple[str, ...] = ()


def resolve_release_time(
    *,
    report_date_ny: pd.Timestamp,
    sec_filings: pd.DataFrame | None,
    intraday: pd.DataFrame | None,
    calendar_flag: str | None,
    manual: pd.Timestamp | None = None,
) -> ResolvedT0:
    """Priority: manual > SEC 8-K item 2.02 acceptance on the report date (or the next calendar
    day before 04:00 NY for late acceptances) > detection from 1-minute extended-hours bars >
    calendar flag. When both an 8-K time and a detection exist: t0 = min(acceptance, detected
    bar start) if detected is within DETECTION_WINDOW before acceptance, else t0 = acceptance;
    t0_lag_s = acceptance - detected_start is recorded either way. 6-K rows are never a time
    source. Returns a ResolvedT0 with the CONFIDENCE of its source; `detection_first_bar`
    lowers a detected source to calendar-flag confidence."""
    raise NotImplementedError


def detect_release_from_bars(bars: pd.DataFrame, report_date_ny: pd.Timestamp,
                             *, vol_z_threshold: float = 6.0, abs_ret_threshold: float = 0.01,
                             baseline_days: int = 10) -> tuple[pd.Timestamp, bool] | None:
    """First 1-minute bar on the report date (New York) whose volume z-score against the same
    clock-minute over the previous `baseline_days` sessions exceeds `vol_z_threshold` and whose
    |return| over the bar exceeds `abs_ret_threshold`. Returns (bar START time UTC,
    is_first_bar_of_session) or None. Bars must be schemas.C rows with tz-aware UTC t/t_end."""
    raise NotImplementedError


def fiscal_period_for(report_date_ny: pd.Timestamp, *, sec_eps_facts: pd.DataFrame | None,
                      av_rows: pd.DataFrame | None) -> tuple[str, str, bool]:
    """(fiscal_period 'YYYY-MM', source, derived_flag). SEC companyfacts period_end nearest
    before the report date (within 120 days) for US filers, Alpha Vantage fiscalDateEnding for
    FPIs, else the calendar quarter end preceding the report date (derived_flag=True)."""
    raise NotImplementedError


def build_events(settings: Settings, *, underlyings: list[str] | None = None,
                 since: pd.Timestamp | None = None, write: bool = True) -> pd.DataFrame:
    """Assemble events from FMP earnings history (primary), SEC filings (timing and fiscal
    period), archived consensus snapshots, and optional Nasdaq / Alpha Vantage cross-checks.
    Nasdaq/AV rows match the FMP row by nearest report date within +/-10 days; a matched pair
    whose dates differ by more than one day gets flags += 'date_conflict' and confidence 0.
    Events are processed newest-first; on BudgetExhausted the completed rows are written, the
    remaining ones are written with pending=True, and the function raises after writing."""
    raise NotImplementedError


def load_events(settings: Settings) -> pd.DataFrame:
    raise NotImplementedError


def expected_release_clock(events: pd.DataFrame, underlying: str) -> tuple[str, str] | None:
    """(HH:MM America/New_York, source) = the issuer's median acceptance clock over its past
    sec_8k events, or None when it has none (caller falls back to the calendar-flag default).
    Used by `freedom predict` for pre-release decision times (docs/design.md §10)."""
    raise NotImplementedError


def detect_release_live(bars: pd.DataFrame, expected_date_ny: pd.Timestamp, **kw) -> pd.Timestamp | None:
    """Live wrapper around detect_release_from_bars for bars ending at or before now; returns the
    detected bar start or None. Same thresholds as the historical detector."""
    raise NotImplementedError


def upcoming_events(settings: Settings, days: int = 14) -> pd.DataFrame:
    """Future events for the event universe from the FMP calendar, with the consensus taken from
    the newest archived snapshot when present."""
    raise NotImplementedError


def snapshot_consensus(settings: Settings, *, days: int = 14) -> pd.DataFrame:
    """Archiver hook: fetch the FMP earnings calendar (one request) and the Nasdaq calendar for
    the next `days` days and append to data/archive/consensus/<UTC date>.parquet with columns
    snapshot_time, symbol, report_date_ny, eps_estimate, rev_estimate, n_estimates, vendor,
    vendor_last_updated. Returns the rows written."""
    raise NotImplementedError
