"""The v1 feature groups (docs/design.md §6): pure functions of a FeatureContext.

Rules every group here follows:

* Inputs are cut with `cut(frame, instant)`: only bars whose END time (t_end) is at or before
  the instant survive, so a bar that starts before the instant but ends after it is never
  used (the first look-ahead trap). Prices "at" an instant are `targets.price_at`, the same
  rule the targets use (close of the last bar with t_end <= instant). Daily bars of the
  underlying and the equity proxies go through `cut_daily`: a bar labelled by a New York
  session date ends at that session's XNYS close (`session_ends`), not at the next midnight
  the loaders stamp as t_end, so an after-close release sees the release-day session and an
  in-session release does not.
* The "pre" groups (calendar, pre_price, history, market, perp_state) are anchored at
  `pre_cut(ctx) = min(as_of, t0 - p0 buffer)`: the last pre-release instant the targets module
  also treats as pre-release. A pre feature therefore has one value for every post decision
  time and never contains the reaction; the reaction group alone reads bars after t0, and only
  up to as_of. Like the targets, the reaction reads 1m/5m bars only (targets.FINE_INTERVALS).
* Anything about other events comes from `ctx.history` (built by `history_view`) and nowhere
  else; the event's own targets are never an input (the second look-ahead trap).
* Every other event attribute is point-in-time as well: the perp listing age is None when the
  listing is after the anchor (the loaders fill listing_start for events released before the
  perp existed). The one documented exception is max_leverage, see PERP_KEYS.
* A missing input gives None for the feature, never an exception. Calendar lookups outside
  the exchange calendar's range also give None.

Output keys are short snake_case without the f_ prefix; `build_features` adds the prefix and
the __missing companions.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from ..schemas import NY, UTC, C, E, T, Timing
from ..targets import FINE_INTERVALS, INTERVAL_TD, max_staleness, p0_buffer_for, price_at
from ..timeutil import (
    classify_timing,
    is_rth,
    next_close_after,
    next_open_after,
    to_ny,
    to_utc,
    xnys,
)
from . import FeatureContext, feature_group

log = logging.getLogger(__name__)

REACTION_HORIZONS_MIN: tuple[int, ...] = (1, 5, 15, 30, 60)
FUNDING_SETTLE_LAG = pd.Timedelta(minutes=1)  # a rate settled at hour t is known moments after t
MIN_52W_BARS = 200  # a "52-week" high/low needs most of a year of daily bars
MIN_HISTORY_FOR_Z = 4  # standardising a surprise needs this many past surprises
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)

# ctx.extra keys the loader fills (all optional):
X_N_EVENTS_SAME_DAY = "n_events_same_day"  # universe events on the same New York date
X_VIX_DAILY = "vix_daily"  # schemas.C daily bars of the VIX proxy
X_SECTOR_DAILY = "sector_daily"  # schemas.C daily bars of the sector ETF proxy
X_PERP_DAILY = "perp_daily"  # schemas.C 1d candles of the event's perp market
X_FUNDING = "funding"  # market, t (settlement hour), funding_rate, premium
X_MAX_LEVERAGE = "max_leverage"  # the market's CURRENT leverage cap (not point-in-time, see PERP_KEYS)
X_LISTING_START = "listing_start"  # fallback when the event row has none


# ---- helpers -------------------------------------------------------------------------------------
def value(row: pd.Series | dict | None, key: str):
    """Scalar from an event row; None when the key is absent or the value is NA."""
    if row is None:
        return None
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return None
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def fnum(v) -> float | None:
    """float(v) or None for None/NaN/non-numeric."""
    if v is None or isinstance(v, str):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def event_t0(ctx: FeatureContext) -> pd.Timestamp | None:
    v = value(ctx.event, E.t0)
    if v is None:
        return None
    try:
        return to_utc(v, assume_tz=UTC)
    except (TypeError, ValueError):
        return None


def as_of_of(ctx: FeatureContext) -> pd.Timestamp:
    return to_utc(ctx.as_of, assume_tz=UTC)


def pre_cut(ctx: FeatureContext) -> pd.Timestamp:
    """Anchor of the pre-release groups: min(as_of, t0 - p0 buffer)."""
    as_of = as_of_of(ctx)
    t0 = event_t0(ctx)
    if t0 is None:
        return as_of
    return min(as_of, t0 - p0_buffer_for(ctx.event))


def _cut_by(frame: pd.DataFrame, ends: pd.Series, instant: pd.Timestamp, col: str) -> pd.DataFrame | None:
    keep = (ends <= to_utc(instant, assume_tz=UTC)).to_numpy()
    if not keep.any():
        return None
    out = frame.loc[keep]
    if not ends[keep].is_monotonic_increasing:
        out = out.sort_values(col, kind="mergesort")
    return out


def cut(frame: pd.DataFrame | None, instant: pd.Timestamp, col: str = C.t_end) -> pd.DataFrame | None:
    """Rows whose `col` (a bar END time by default) is <= instant, sorted by it; None when
    nothing survives. This (and cut_daily for daily bars) is the only way bars enter a feature."""
    if frame is None or len(frame) == 0 or col not in frame.columns:
        return None
    return _cut_by(frame, pd.to_datetime(frame[col], utc=True, errors="coerce"), instant, col)


def session_ends(frame: pd.DataFrame) -> pd.Series:
    """Effective end time of each daily bar. The FMP/Nasdaq loaders label a session's bar with
    t = 00:00 America/New_York of the session date and t_end = the next New York midnight, but
    its close, high, low and volume are known at the XNYS close of that session (16:00 ET,
    earlier on half days): such a bar ends at that close here. Any other bar (perp 1d candles
    start at UTC midnight and are complete only at t_end) and any date the calendar does not
    know as a session keep their t_end."""
    ends = pd.to_datetime(frame[C.t_end], utc=True, errors="coerce")
    if C.t not in frame.columns:
        return ends
    starts = pd.to_datetime(frame[C.t], utc=True, errors="coerce")
    ny = starts.dt.tz_convert(NY)
    labelled = starts.notna() & (ny.dt.hour == 0) & (ny.dt.minute == 0)
    if not labelled.any():
        return ends
    try:
        closes = xnys().closes
    except Exception:  # no calendar available: the conservative calendar-day end stands
        return ends
    session_close = pd.to_datetime(ny.dt.tz_localize(None).dt.normalize().map(closes), utc=True,
                                   errors="coerce")
    use = labelled & session_close.notna() & (session_close < ends)
    return ends.where(~use, session_close)


def cut_daily(frame: pd.DataFrame | None, instant: pd.Timestamp) -> pd.DataFrame | None:
    """cut() for daily bars, ending each session bar at its XNYS close (session_ends)."""
    if frame is None or len(frame) == 0 or C.t_end not in frame.columns:
        return None
    return _cut_by(frame, session_ends(frame), instant, C.t_end)


def between(frame: pd.DataFrame | None, lo: pd.Timestamp, hi: pd.Timestamp, col: str = C.t_end) -> pd.DataFrame | None:
    """Rows with lo < col <= hi (None when empty)."""
    c = cut(frame, hi, col)
    if c is None:
        return None
    ends = pd.to_datetime(c[col], utc=True, errors="coerce")
    keep = (ends > to_utc(lo, assume_tz=UTC)).to_numpy()
    if not keep.any():
        return None
    return c.loc[keep]


def closes(frame: pd.DataFrame | None, col: str = C.close) -> np.ndarray:
    if frame is None or len(frame) == 0 or col not in frame.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype="float64")


def log_ret(num: float | None, den: float | None) -> float | None:
    num, den = fnum(num), fnum(den)
    if num is None or den is None or num <= 0 or den <= 0:
        return None
    return math.log(num / den)


def ret_n(c: np.ndarray, n: int) -> float | None:
    """ln(c[-1] / c[-1-n]) from an array of closes."""
    if len(c) <= n:
        return None
    return log_ret(c[-1], c[-1 - n])


def realised_vol(c: np.ndarray, n: int) -> float | None:
    """Sample std of the last `n` daily log returns (needs n + 1 closes)."""
    if len(c) < n + 1 or n < 2:
        return None
    tail = c[-(n + 1):]
    if np.any(~np.isfinite(tail)) or np.any(tail <= 0):
        return None
    r = np.diff(np.log(tail))
    return float(np.std(r, ddof=1))


def px(bars: pd.DataFrame | None, when: pd.Timestamp) -> tuple[float, pd.Timestamp] | None:
    """targets.price_at on already-cut bars; None for empty inputs or bars without a close."""
    if bars is None or len(bars) == 0 or C.close not in bars.columns:
        return None
    return price_at(bars, when)


def interval_label(bars: pd.DataFrame) -> str:
    """'1m' | '5m' | ... from the interval column, else from the median bar length (targets
    module convention: anything unknown counts as 1h)."""
    if C.interval in bars.columns:
        s = bars[C.interval].dropna()
        if len(s):
            return str(s.iloc[0])
    if C.t in bars.columns and C.t_end in bars.columns:
        d = (pd.to_datetime(bars[C.t_end], utc=True) - pd.to_datetime(bars[C.t], utc=True)).median()
        for k, v in INTERVAL_TD.items():
            if v == d:
                return k
    return "1h"


def previous_close_before(ts: pd.Timestamp) -> pd.Timestamp | None:
    """Last XNYS regular-session close strictly before `ts` (None outside the calendar range)."""
    try:
        return pd.Timestamp(xnys().previous_close(to_utc(ts, assume_tz=UTC))).tz_convert(UTC)
    except Exception:  # MinuteOutOfBounds and friends: the calendar does not cover the instant
        return None


def holiday_adjacent(t0: pd.Timestamp) -> float:
    """1.0 when a weekday XNYS holiday touches the release: the New York date of t0 is itself
    a weekday that is not a session, or a weekday holiday lies between that date and the
    previous or the next session. Weekends alone do not count (weekday/friday carry them).
    Raises outside the calendar range (callers wrap it in `safe`)."""
    cal = xnys()
    day = to_ny(t0).normalize().tz_localize(None)
    nxt = pd.Timestamp(cal.date_to_session(day + DAY, "next"))
    prv = pd.Timestamp(cal.date_to_session(day - DAY, "previous"))
    d64 = np.datetime64(day.date())
    skipped = int(np.busday_count(d64 + 1, np.datetime64(nxt.date())))  # weekdays in (day, nxt)
    skipped += int(np.busday_count(np.datetime64(prv.date()) + 1, d64))  # weekdays in (prv, day)
    own = day.weekday() < 5 and not cal.is_session(day)
    return float(skipped > 0 or own)


def safe(fn, *args):
    """Call a calendar helper; None when the calendar cannot answer."""
    try:
        return fn(*args)
    except Exception:
        return None


def none_dict(keys: tuple[str, ...] | list[str]) -> dict[str, float | None]:
    return dict.fromkeys(keys)


# ---- calendar ------------------------------------------------------------------------------------
CALENDAR_KEYS = ("amc", "bmo", "rth", "weekday", "friday", "hour_ny", "hours_to_next_open",
                 "hours_to_next_close", "h24_closed", "holiday_adjacent", "days_since_last_event",
                 "n_events_same_day")


@feature_group("calendar", admissible=("pre", "post"))
def calendar(ctx: FeatureContext) -> dict[str, float | None]:
    """AMC/BMO, weekday, clock, distance to the next session boundaries, whether XNYS is closed
    at t0 + horizon, an explicit weekday-holiday adjacency flag, days since this name's last
    event (history only), same-day event count."""
    out = none_dict(CALENDAR_KEYS)
    t0 = event_t0(ctx)
    if t0 is None:
        return out
    timing = value(ctx.event, E.timing)
    timing = str(timing) if timing else safe(classify_timing, t0)
    if timing is not None:
        out["amc"] = float(timing == Timing.amc)
        out["bmo"] = float(timing == Timing.bmo)
        out["rth"] = float(timing == Timing.rth)
    ny = to_ny(t0)
    out["weekday"] = float(ny.weekday())
    out["friday"] = float(ny.weekday() == 4)
    out["hour_ny"] = ny.hour + ny.minute / 60 + ny.second / 3600
    nxt_open = safe(next_open_after, t0)
    if nxt_open is not None:
        out["hours_to_next_open"] = (nxt_open - t0) / HOUR
    nxt_close = safe(next_close_after, t0)
    if nxt_close is not None:
        out["hours_to_next_close"] = (nxt_close - t0) / HOUR
    open_at_h = safe(is_rth, t0 + pd.Timedelta(hours=ctx.horizon_hours))
    if open_at_h is not None:
        out["h24_closed"] = float(not open_at_h)
    out["holiday_adjacent"] = safe(holiday_adjacent, t0)
    h = ctx.history
    if h is not None and len(h) and E.t0 in h.columns:
        last = pd.to_datetime(h[E.t0], utc=True, errors="coerce").max()
        if pd.notna(last):
            out["days_since_last_event"] = (t0 - last) / DAY
    out["n_events_same_day"] = fnum(ctx.extra.get(X_N_EVENTS_SAME_DAY))
    return out


# ---- pre_price -----------------------------------------------------------------------------------
PRE_PRICE_KEYS = ("ret_1d", "ret_5d", "ret_20d", "ret_60d", "rvol_20d", "dist_52w_high",
                  "dist_52w_low", "dvol_5d_ratio", "drift_60m", "drift_30m", "gap_since_close",
                  "ext_vol_ratio", "vol_30m_ratio")


def _drift(fine: pd.DataFrame | None, at: pd.Timestamp, minutes: int) -> float | None:
    now = px(fine, at)
    then = px(fine, at - pd.Timedelta(minutes=minutes))
    if now is None or then is None or then[1] >= now[1]:
        return None
    return log_ret(now[0], then[0])


def _mean_volume(frame: pd.DataFrame | None, min_bars: int = 1) -> float | None:
    """Mean volume per bar of a (possibly None) bar window with at least `min_bars` bars."""
    if frame is None or len(frame) < min_bars:
        return None
    v = closes(frame, C.volume)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) >= min_bars else None


@feature_group("pre_price", admissible=("pre", "post"))
def pre_price(ctx: FeatureContext) -> dict[str, float | None]:
    """Daily-bar returns, realised vol, 52-week distances, recent volume, plus the fine-bar
    drift in the last 60/30 minutes, the gap since the last regular close and extended-hours
    volume versus the same window one day earlier. All at pre_cut(ctx); daily bars count from
    their session close (cut_daily), so an after-close release sees the release-day session."""
    out = none_dict(PRE_PRICE_KEYS)
    at = pre_cut(ctx)
    d = cut_daily(ctx.daily, at)
    c = closes(d)
    out["ret_1d"], out["ret_5d"] = ret_n(c, 1), ret_n(c, 5)
    out["ret_20d"], out["ret_60d"] = ret_n(c, 20), ret_n(c, 60)
    out["rvol_20d"] = realised_vol(c, 20)
    if d is not None and len(c) >= MIN_52W_BARS and c[-1] > 0:
        hi = closes(d, C.high)[-252:]
        lo = closes(d, C.low)[-252:]
        if np.isfinite(hi).any() and np.nanmax(hi) > 0:
            out["dist_52w_high"] = math.log(c[-1] / float(np.nanmax(hi)))
        if np.isfinite(lo).any() and np.nanmin(lo) > 0:
            out["dist_52w_low"] = math.log(c[-1] / float(np.nanmin(lo)))
    v = closes(d, C.volume)
    if len(v) >= 20 and np.isfinite(v[-60:]).any() and np.isfinite(v[-5:]).any():
        base = float(np.nanmean(v[-60:]))
        recent = float(np.nanmean(v[-5:]))
        if base > 0:
            out["dvol_5d_ratio"] = recent / base

    fine = cut(ctx.bars, at)
    out["drift_60m"] = _drift(fine, at, 60)
    out["drift_30m"] = _drift(fine, at, 30)
    prev_close = previous_close_before(at)
    if fine is not None and prev_close is not None:
        now = px(fine, at)
        ref = px(fine, prev_close)
        if now is not None and ref is not None and ref[1] < now[1]:
            out["gap_since_close"] = log_ret(now[0], ref[0])
        # volume per bar since the last regular close vs the 24 hours before that close (the
        # window the event loaders always cover; a same-clock window one day earlier is not)
        vol_ext = _mean_volume(between(fine, prev_close, at))
        vol_base = _mean_volume(between(fine, prev_close - DAY, prev_close), min_bars=12)
        if vol_ext is not None and vol_base is not None and vol_base > 0:
            out["ext_vol_ratio"] = vol_ext / vol_base
    if fine is not None:
        recent = between(fine, at - pd.Timedelta(minutes=30), at)
        base = between(fine, at - DAY, at - pd.Timedelta(minutes=30))
        if recent is not None and base is not None and len(base) >= 12:
            vr = closes(recent, C.volume)
            vb = closes(base, C.volume)
            vr, vb = vr[np.isfinite(vr)], vb[np.isfinite(vb)]
            mb = float(np.mean(vb)) if len(vb) else 0.0
            if mb > 0 and len(vr):
                out["vol_30m_ratio"] = float(np.mean(vr)) / mb
    return out


# ---- history -------------------------------------------------------------------------------------
HISTORY_KEYS = ("hist_n", "hist_r24_mean", "hist_r24_std", "hist_r24_skew", "hist_abs_r24_mean",
                "hist_up_rate", "hist_cont_rate", "hist_last1_r24", "hist_last2_r24",
                "hist_last3_r24", "hist_last4_r24", "hist_surprise_beta", "hist_eps_surprise_mean")


@feature_group("history", admissible=("pre", "post"))
def history(ctx: FeatureContext) -> dict[str, float | None]:
    """This name's past reactions from ctx.history only: mean/std/skew of r_24h, mean |r_24h|,
    up rate, continuation hit rate, the last four r_24h, and the slope of r_24h on the EPS
    surprise. None when no history is available; hist_n = 0 for an empty history."""
    out = none_dict(HISTORY_KEYS)
    h = ctx.history
    if h is None:
        return out
    if len(h) == 0:
        out["hist_n"] = 0.0
        return out
    h = h.copy()
    h[E.t0] = pd.to_datetime(h[E.t0], utc=True, errors="coerce") if E.t0 in h.columns else pd.NaT
    h = h.sort_values(E.t0, kind="mergesort")
    r24_col = T.r("24h")
    r = pd.to_numeric(h[r24_col], errors="coerce") if r24_col in h.columns else pd.Series(dtype="float64")
    valid = r.dropna()
    n = int(len(valid))
    out["hist_n"] = float(n)
    if n >= 1:
        out["hist_r24_mean"] = float(valid.mean())
        out["hist_abs_r24_mean"] = float(valid.abs().mean())
        out["hist_up_rate"] = float((valid > 0).mean())
        last = valid.to_numpy()[::-1]
        for i in range(4):
            if i < len(last):
                out[f"hist_last{i + 1}_r24"] = float(last[i])
    if n >= 2:
        out["hist_r24_std"] = float(valid.std(ddof=1))
    if n >= 3:
        out["hist_r24_skew"] = fnum(valid.skew())
    if T.continuation_30m in h.columns:
        cont = pd.to_numeric(h[T.continuation_30m], errors="coerce").dropna()
        if len(cont):
            out["hist_cont_rate"] = float((cont > 0).mean())
    if E.eps_surprise_pct in h.columns:
        s = pd.to_numeric(h[E.eps_surprise_pct], errors="coerce")
        if s.notna().any():
            out["hist_eps_surprise_mean"] = float(s.dropna().mean())
        pair = pd.DataFrame({"x": s, "y": r if len(r) == len(h) else np.nan}).dropna()
        if len(pair) >= MIN_HISTORY_FOR_Z:
            x = pair["x"].to_numpy(dtype="float64")
            y = pair["y"].to_numpy(dtype="float64")
            vx = float(np.var(x, ddof=1))
            if vx > 0:
                out["hist_surprise_beta"] = float(np.cov(x, y, ddof=1)[0, 1] / vx)
    return out


# ---- market --------------------------------------------------------------------------------------
MARKET_KEYS = ("mkt_ret_1d", "mkt_ret_5d", "mkt_ret_20d", "mkt_rvol_20d", "vix_level",
               "vix_chg_5d", "sector_ret_1d", "sector_ret_5d", "mkt_drift_60m")


@feature_group("market", admissible=("pre", "post"))
def market(ctx: FeatureContext) -> dict[str, float | None]:
    """Benchmark (xyz:SP500 or SPY) daily returns and vol, VIX level and 5-day change,
    sector-proxy returns, and the benchmark's last-hour drift from its fine bars, at pre_cut.
    Daily bars count from their session close (cut_daily); perp 1d candles from their t_end."""
    out = none_dict(MARKET_KEYS)
    at = pre_cut(ctx)
    c = closes(cut_daily(ctx.market_daily, at))
    out["mkt_ret_1d"], out["mkt_ret_5d"], out["mkt_ret_20d"] = ret_n(c, 1), ret_n(c, 5), ret_n(c, 20)
    out["mkt_rvol_20d"] = realised_vol(c, 20)
    v = closes(cut_daily(ctx.extra.get(X_VIX_DAILY), at))
    if len(v) and v[-1] > 0:
        out["vix_level"] = float(v[-1])
        out["vix_chg_5d"] = ret_n(v, 5)
    s = closes(cut_daily(ctx.extra.get(X_SECTOR_DAILY), at))
    out["sector_ret_1d"], out["sector_ret_5d"] = ret_n(s, 1), ret_n(s, 5)
    out["mkt_drift_60m"] = _drift(cut(ctx.market_bars, at), at, 60)
    return out


# ---- perp_state ----------------------------------------------------------------------------------
PERP_KEYS = ("funding_rate", "funding_mean_24h", "premium", "oi_notional", "oi_chg_24h",
             "day_ntl_vlm", "perp_vol_30d", "max_leverage", "listing_age_d")
# max_leverage is NOT point-in-time: Hyperliquid publishes no history of leverage caps, so the
# loaders supply the cap in universe.parquet / the dex meta at build time, which may differ
# from the cap in force at t0. It is carried as a slowly-changing market attribute; drop it
# from a model's feature list when a cap change inside the sample would make it an era proxy.


def _settled_funding(fund: pd.DataFrame | None, at: pd.Timestamp) -> pd.DataFrame | None:
    """Funding rows whose settlement (hour t, known a moment later) is at or before `at`."""
    if fund is None or len(fund) == 0 or "t" not in fund.columns:
        return None
    return cut(fund, at - FUNDING_SETTLE_LAG, col="t")


@feature_group("perp_state", admissible=("pre", "post"))
def perp_state(ctx: FeatureContext) -> dict[str, float | None]:
    """Funding (last settled rate and its 24h mean), premium, open interest and its 24h change,
    day notional volume from the ctx snapshots, 30-day median daily notional from perp 1d
    candles, leverage cap (the current one, see PERP_KEYS) and listing age. Everything at
    pre_cut; None without a perp, and the listing age is None unless the listing is known at
    pre_cut (has_perp_at_t0 not False and listing_start <= pre_cut)."""
    out = none_dict(PERP_KEYS)
    at = pre_cut(ctx)
    fund = _settled_funding(ctx.extra.get(X_FUNDING), at)
    if fund is not None and "funding_rate" in fund.columns:
        fr = pd.to_numeric(fund["funding_rate"], errors="coerce")
        if fr.notna().any():
            out["funding_rate"] = fnum(fr.dropna().iloc[-1])
            t = pd.to_datetime(fund["t"], utc=True, errors="coerce")
            last_day = fr[(t > at - DAY).to_numpy()].dropna()
            if len(last_day):
                out["funding_mean_24h"] = float(last_day.mean())
    snaps = cut(ctx.perp_ctx, at, col="t")
    if snaps is not None:
        last = snaps.iloc[-1]
        out["premium"] = fnum(value(last, "premium"))
        oi = fnum(value(last, "open_interest"))
        mark = fnum(value(last, "mark_px")) or fnum(value(last, "oracle_px"))
        if oi is not None and mark is not None:
            out["oi_notional"] = oi * mark
        out["day_ntl_vlm"] = fnum(value(last, "day_ntl_vlm"))
        earlier = cut(snaps, at - DAY, col="t")
        if earlier is not None and oi is not None and oi > 0:
            prev = earlier.iloc[-1]
            prev_t = pd.to_datetime(value(prev, "t"), utc=True)
            prev_oi = fnum(value(prev, "open_interest"))
            if prev_oi is not None and prev_oi > 0 and at - prev_t <= 2 * DAY:
                out["oi_chg_24h"] = math.log(oi / prev_oi)
    if out["premium"] is None and fund is not None and "premium" in fund.columns:
        pr = pd.to_numeric(fund["premium"], errors="coerce").dropna()
        if len(pr):
            out["premium"] = fnum(pr.iloc[-1])
    pdaily = cut(ctx.extra.get(X_PERP_DAILY), at)
    if pdaily is not None and C.volume in pdaily.columns and C.close in pdaily.columns:
        vol = closes(pdaily, C.volume)[-30:]
        cl = closes(pdaily, C.close)[-30:]
        if len(vol) and len(vol) == len(cl):
            notional = vol * cl
            notional = notional[np.isfinite(notional)]
            if len(notional) >= 5:
                out["perp_vol_30d"] = float(np.median(notional))
    out["max_leverage"] = fnum(ctx.extra.get(X_MAX_LEVERAGE))
    listing = value(ctx.event, E.listing_start)
    if listing is None:
        listing = ctx.extra.get(X_LISTING_START)
    t0 = event_t0(ctx)
    has_perp = value(ctx.event, E.has_perp_at_t0)
    if listing is not None and t0 is not None and (has_perp is None or bool(has_perp)):
        try:
            ls = to_utc(listing, assume_tz=UTC)
        except (TypeError, ValueError):
            ls = None
        # a listing after the anchor is not known at the anchor: the events module fills
        # listing_start with the underlying's earliest listing even for earlier releases
        if ls is not None and pd.notna(ls) and ls <= at:
            out["listing_age_d"] = (t0 - ls) / DAY
    return out


# ---- surprise (post only) ------------------------------------------------------------------------
SURPRISE_KEYS = ("eps_surprise", "rev_surprise", "eps_beat", "eps_surprise_abs", "sign_agree",
                 "eps_surprise_z", "rev_surprise_z", "n_estimates")


def _surprise_pct(actual, estimate, given) -> float | None:
    g = fnum(given)
    if g is not None:
        return g
    a, e = fnum(actual), fnum(estimate)
    if a is None or e is None or e == 0:
        return None
    return (a - e) / abs(e) * 100.0


def _zscore(x: float | None, hist: pd.DataFrame | None, col: str) -> float | None:
    if x is None or hist is None or len(hist) == 0 or col not in hist.columns:
        return None
    s = pd.to_numeric(hist[col], errors="coerce").dropna()
    if len(s) < MIN_HISTORY_FOR_Z:
        return None
    sd = float(s.std(ddof=1))
    if not (sd > 0):
        return None
    return (x - float(s.mean())) / sd


@feature_group("surprise", admissible=("post",))
def surprise(ctx: FeatureContext) -> dict[str, float | None]:
    """EPS and revenue surprise (%), beat flag, |EPS surprise|, sign agreement, both surprises
    standardised against this name's past surprises (ctx.history), and the estimate count."""
    out = none_dict(SURPRISE_KEYS)
    ev = ctx.event
    eps = _surprise_pct(value(ev, E.eps_actual), value(ev, E.eps_estimate), value(ev, E.eps_surprise_pct))
    rev = _surprise_pct(value(ev, E.rev_actual), value(ev, E.rev_estimate), value(ev, E.rev_surprise_pct))
    out["eps_surprise"], out["rev_surprise"] = eps, rev
    if eps is not None:
        out["eps_beat"] = float(eps > 0)
        out["eps_surprise_abs"] = abs(eps)
    if eps is not None and rev is not None:
        out["sign_agree"] = float(np.sign(eps) * np.sign(rev))
    out["eps_surprise_z"] = _zscore(eps, ctx.history, E.eps_surprise_pct)
    out["rev_surprise_z"] = _zscore(rev, ctx.history, E.rev_surprise_pct)
    out["n_estimates"] = fnum(value(ev, E.n_estimates))
    return out


# ---- reaction (post only) ------------------------------------------------------------------------
REACTION_KEYS = tuple(f"r_{k}m" for k in REACTION_HORIZONS_MIN) + (
    "r_now", "abs_r_now", "path_max", "path_min", "path_range", "vol_z", "vol_ratio", "ar_now",
    "premium_post")


def _valid_hit(hit, p0_time: pd.Timestamp, when: pd.Timestamp, stale: pd.Timedelta) -> float | None:
    """Close of a checkpoint bar under the targets' validity rule: the bar must end after the
    P0 bar and no more than `stale` before the instant."""
    if hit is None or hit[1] <= p0_time or when - hit[1] > stale:
        return None
    return float(hit[0])


@feature_group("reaction", admissible=("post",))
def reaction(ctx: FeatureContext) -> dict[str, float | None]:
    """Early reaction from bars ending at or before as_of: r_k for k in {1,5,15,30,60} minutes
    (only those with t0 + k <= as_of, same P0 and validity rules as targets.compute_targets),
    the return at as_of, the post-release high/low path (bars ending after t0), volume z-score
    against the bars up to the P0 bar, the abnormal return versus the benchmark fine bars and
    the perp premium after t0. Like the targets, only 1m/5m bars resolve any of this: on 1h or
    coarser bars every key is None."""
    out = none_dict(REACTION_KEYS)
    t0 = event_t0(ctx)
    if t0 is None:
        return out
    as_of = as_of_of(ctx)
    fine = cut(ctx.bars, as_of)
    if fine is None or interval_label(fine) not in FINE_INTERVALS:
        return out
    buffer = p0_buffer_for(ctx.event)
    stale = max_staleness(interval_label(fine))
    ref = px(fine, t0 - buffer)
    if ref is None or not ref[0] > 0:
        return out
    p0, p0_time = float(ref[0]), ref[1]
    for k in REACTION_HORIZONS_MIN:
        when = t0 + pd.Timedelta(minutes=k)
        if when > as_of:
            continue
        p = _valid_hit(px(fine, when), p0_time, when, stale)
        out[f"r_{k}m"] = log_ret(p, p0)
    p_now = _valid_hit(px(fine, as_of), p0_time, as_of, stale)
    r_now = log_ret(p_now, p0)
    out["r_now"] = r_now
    out["abs_r_now"] = abs(r_now) if r_now is not None else None
    ends = pd.to_datetime(fine[C.t_end], utc=True, errors="coerce")
    # the path is what traded after the release: bars ending after t0. Bars between the P0 bar
    # and t0 (the 8-K buffer) belong to neither the path nor the pre-release volume baseline.
    post = fine.loc[(ends > max(p0_time, t0)).to_numpy()]
    pre = fine.loc[(ends <= p0_time).to_numpy()]
    if len(post):
        hi, lo = closes(post, C.high), closes(post, C.low)
        if np.isfinite(hi).any() and np.nanmax(hi) > 0:
            out["path_max"] = math.log(float(np.nanmax(hi)) / p0)
        if np.isfinite(lo).any() and np.nanmin(lo) > 0:
            out["path_min"] = math.log(float(np.nanmin(lo)) / p0)
        if out["path_max"] is not None and out["path_min"] is not None:
            out["path_range"] = out["path_max"] - out["path_min"]
        vpost, vpre = closes(post, C.volume), closes(pre, C.volume)
        vpre = vpre[np.isfinite(vpre)]
        vpost = vpost[np.isfinite(vpost)]
        if len(vpre) >= 5 and len(vpost):
            mean_pre, sd_pre = float(np.mean(vpre)), float(np.std(vpre, ddof=1))
            mean_post = float(np.mean(vpost))
            if sd_pre > 0:
                out["vol_z"] = (mean_post - mean_pre) / sd_pre
            if mean_pre > 0:
                out["vol_ratio"] = mean_post / mean_pre
    mb = cut(ctx.market_bars, as_of)
    if mb is not None and r_now is not None:
        mref = px(mb, t0 - buffer)
        mnow = px(mb, as_of)
        if mref is not None and mnow is not None and mnow[1] > mref[1] and mref[0] > 0 \
                and as_of - mnow[1] <= max_staleness(interval_label(mb)):
            out["ar_now"] = r_now - math.log(mnow[0] / mref[0])
    snaps = between(ctx.perp_ctx, t0, as_of, col="t")
    if snaps is not None:
        out["premium_post"] = fnum(value(snaps.iloc[-1], "premium"))
    if out["premium_post"] is None:
        fund = ctx.extra.get(X_FUNDING)
        if fund is not None and len(fund) and "t" in fund.columns and "premium" in fund.columns:
            settled = between(fund, t0 - FUNDING_SETTLE_LAG, as_of - FUNDING_SETTLE_LAG, col="t")
            if settled is not None:
                pr = pd.to_numeric(settled["premium"], errors="coerce").dropna()
                if len(pr):
                    out["premium_post"] = fnum(pr.iloc[-1])
    return out


GROUP_KEYS: dict[str, tuple[str, ...]] = {
    "calendar": CALENDAR_KEYS, "pre_price": PRE_PRICE_KEYS, "history": HISTORY_KEYS,
    "market": MARKET_KEYS, "perp_state": PERP_KEYS, "surprise": SURPRISE_KEYS,
    "reaction": REACTION_KEYS,
}
