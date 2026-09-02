"""Column names, enumerations and frame contracts shared by every module.

Rule: any DataFrame that crosses a module boundary uses these names. Timestamps are
tz-aware UTC pandas Timestamps unless the name ends in `_ny` (America/New_York).
"""

from __future__ import annotations

from enum import StrEnum

UTC = "UTC"
NY = "America/New_York"


class Kind(StrEnum):
    equity_us = "equity_us"  # US domestic filer: 8-K item 2.02 gives release time
    equity_fpi = "equity_fpi"  # foreign private issuer with US-listed line (6-K filer)
    equity_intl = "equity_intl"  # listed only outside the US; no US data sources; excluded
    etf = "etf"
    index = "index"
    commodity = "commodity"
    fx = "fx"
    crypto = "crypto"
    preipo = "preipo"
    rate = "rate"
    other = "other"


EVENT_KINDS = {Kind.equity_us, Kind.equity_fpi}


class Timing(StrEnum):
    amc = "AMC"  # after regular close, before next open
    bmo = "BMO"  # before regular open on a session day
    rth = "RTH"  # during regular trading hours
    closed = "CLOSED"  # weekend / holiday


class T0Source(StrEnum):
    sec_8k = "sec_8k"
    detected = "detected"
    calendar_flag = "calendar_flag"
    manual = "manual"


class PriceSource(StrEnum):
    hl_archive = "hl_archive"  # archived Hyperliquid candles (1m/5m)
    hl_live = "hl_live"  # Hyperliquid candleSnapshot (recent only)
    fmp_intraday = "fmp_intraday"  # underlying extended-hours bars


# ---- universe.parquet ------------------------------------------------------------------------
class U:
    market = "market"  # e.g. "xyz:NVDA"
    dex = "dex"
    symbol = "symbol"  # "NVDA"
    kind = "kind"
    underlying = "underlying"  # US ticker of the underlying equity, or None
    cik = "cik"
    name = "name"
    exclude_reason = "exclude_reason"
    verify = "verify"  # classification uncertain, human should confirm
    listing_start = "listing_start"
    max_leverage = "max_leverage"
    growth_mode = "growth_mode"
    deployer_fee_scale = "deployer_fee_scale"
    median_notional_30d = "median_notional_30d"
    is_primary = "is_primary"  # primary market for this underlying
    in_event_universe = "in_event_universe"


# ---- candles (archive and live) -------------------------------------------------------------
class C:
    market = "market"
    interval = "interval"
    t = "t"  # bar start, UTC
    t_end = "t_end"  # bar end (exclusive), UTC
    open = "open"
    high = "high"
    low = "low"
    close = "close"
    volume = "volume"
    n_trades = "n_trades"
    source = "source"


# ---- events.parquet --------------------------------------------------------------------------
class E:
    event_id = "event_id"  # f"{underlying}:{fiscal_period}" with fiscal_period = YYYY-MM quarter end
    underlying = "underlying"
    market = "market"  # perp market used for prices (primary, or alternate if primary unlisted at t0)
    cik = "cik"
    kind = "kind"  # equity_us | equity_fpi
    fiscal_period = "fiscal_period"  # "YYYY-MM" fiscal quarter-end month
    # sec_facts | alphavantage (the period's own filing) | sec_facts_projected |
    # alphavantage_projected (latest known period end + whole quarters: fresh / upcoming
    # events, stable before and after the filing) | derived (calendar quarter, no facts)
    fiscal_period_source = "fiscal_period_source"
    report_date_ny = "report_date_ny"  # calendar date in New York
    t0 = "t0"
    t0_confidence = "t0_confidence"  # 0..1
    t0_source = "t0_source"
    timing = "timing"
    eps_actual = "eps_actual"
    eps_estimate = "eps_estimate"
    eps_surprise_pct = "eps_surprise_pct"
    rev_actual = "rev_actual"
    rev_estimate = "rev_estimate"
    rev_surprise_pct = "rev_surprise_pct"
    n_estimates = "n_estimates"
    # consensus_snapshot | fmp_final | nasdaq_final (vendor's final value, past events) |
    # fmp_calendar (vendor's current, not-yet-final value on upcoming rows)
    estimate_source = "estimate_source"
    estimate_snapshot_time = "estimate_snapshot_time"  # UTC time the consensus was captured, or NaT
    sources_used = "sources_used"
    has_perp_at_t0 = "has_perp_at_t0"  # t0 >= earliest listing_start over all markets of the underlying
    listing_start = "listing_start"
    pending = "pending"  # True when data fetching stopped (budget) before this row was completed
    # ';'-joined: date_conflict, fiscal_period_derived, detection_first_bar, upcoming,
    # corporate_action (a split ex-date inside [t0 - 60 d, t0 + horizon], docs/design.md §2), ...
    flags = "flags"
    # UTC instant of 00:00 America/New_York on the split ex-date nearest t0 when the
    # corporate_action flag is set, else NaT; targets NaN the headline label when it falls in
    # [p0_time, t0 + horizon]
    ca_ex_date = "corporate_action_ex_date"


# ---- targets ----------------------------------------------------------------------------------
CHECKPOINTS: list[str] = [
    "5m", "15m", "30m", "60m", "2h", "next_open", "next_open_30m", "next_close", "24h",
]
HEADLINE_CHECKPOINT = "24h"


class T:
    event_id = "event_id"
    p0 = "p0"
    p0_time = "p0_time"  # end of the bar that supplied p0
    p0_staleness_min = "p0_staleness_min"  # (t0 - buffer) - p0_time, minutes
    price_source = "price_source"  # schemas.PriceSource
    price_interval = "price_interval"  # 1m | 5m
    price_market = "price_market"  # perp market or underlying ticker the path came from
    horizon_actual_h = "horizon_actual_h"  # hours between p0_time and the bar used for 24h
    h24_in_closure = "h24_in_closure"  # XNYS not in session at t0 + 24h
    # per checkpoint: r_<cp>, ar_<cp>, p_<cp>, t_<cp>, s_<cp> (staleness in minutes)
    direction = "direction_24h"  # +1 / -1 / 0
    magnitude = "magnitude_24h"  # |r_24h|
    # continuation_k = sign(r_k) * sign(r_24h - r_k): +1 the early reaction extended, -1 it
    # reversed; NaN when |r_k| is inside the dead band (CONTINUATION_DEAD_BAND)
    continuation_15m = "continuation_15m"
    continuation_30m = "continuation_30m"
    continuation = continuation_30m  # headline continuation label

    @staticmethod
    def r(cp: str) -> str:
        return f"r_{cp}"

    @staticmethod
    def ar(cp: str) -> str:
        return f"ar_{cp}"

    @staticmethod
    def p(cp: str) -> str:
        return f"p_{cp}"

    @staticmethod
    def t(cp: str) -> str:
        return f"t_{cp}"

    @staticmethod
    def s(cp: str) -> str:
        return f"s_{cp}"


CONTINUATION_DEAD_BAND = 0.0025  # 25 bp: below this the early reaction has no sign to extend


# ---- decision times -----------------------------------------------------------------------------
# name -> offset in minutes relative to t0 (negative = before release)
DECISION_TIMES: dict[str, int] = {
    "pre_5m": -5,
    "post_1m": 1,
    "post_15m": 15,
    "post_30m": 30,
    "post_60m": 60,
}
DEFAULT_DECISION_TIME = "post_30m"


# ---- dataset / predictions ---------------------------------------------------------------------
class D:
    event_id = "event_id"
    decision_time = "decision_time"
    as_of = "as_of"
    feature_prefix = "f_"  # every feature column starts with this
    missing_suffix = "__missing"


class P:
    event_id = "event_id"
    decision_time = "decision_time"
    model = "model"
    fold = "fold"
    test_season = "test_season"
    p_up = "p_up"
    r_hat = "r_hat"
    magnitude_hat = "magnitude_hat"  # predicted |r_24h| (model.predict_magnitude; default |r_hat|)
    r_lo = "r_lo"  # r_hat + 10th percentile of out-of-sample residuals
    r_hi = "r_hi"  # r_hat + 90th percentile
    r_true = "r_true"
    direction_true = "direction_true"


SCHEMA_VERSION = 3  # bump when any artifact's columns change; written into every parquet's metadata
# 3 (2026-09-02): estimate_source joins dataset.parquet and predictions.parquet, trades.parquet gains
# `headline`; a dataset stamped 2 still evaluates (eval.runner.attach_estimate_source fills the
# column from the events calendar, else reports it as 'unavailable').


def season_of(ts) -> str:
    """Earnings season label = calendar quarter of t0 in UTC, e.g. '2026Q3'."""
    import pandas as pd

    t = pd.Timestamp(ts)
    return f"{t.year}Q{(t.month - 1) // 3 + 1}"
