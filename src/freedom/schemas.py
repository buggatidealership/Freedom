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
    event_id = "event_id"  # f"{underlying}:{fiscal_period_end}"
    underlying = "underlying"
    market = "market"  # primary perp market or None
    cik = "cik"
    fiscal_period_end = "fiscal_period_end"
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
    sources_used = "sources_used"
    has_perp_at_t0 = "has_perp_at_t0"
    flags = "flags"


# ---- targets ----------------------------------------------------------------------------------
CHECKPOINTS: list[str] = [
    "5m", "15m", "30m", "60m", "2h", "next_open", "next_open_30m", "next_close", "24h",
]
HEADLINE_CHECKPOINT = "24h"


class T:
    event_id = "event_id"
    p0 = "p0"
    p0_time = "p0_time"
    price_source = "price_source"
    # per checkpoint: r_<cp>, ar_<cp>, p_<cp>, t_<cp>
    direction = "direction_24h"  # +1 / -1 / 0
    magnitude = "magnitude_24h"  # |r_24h|
    continuation = "continuation_24h"  # sign(r_24h - r_30m)

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
    p_up = "p_up"
    r_hat = "r_hat"
    r_q10 = "r_q10"
    r_q90 = "r_q90"
    r_true = "r_true"
