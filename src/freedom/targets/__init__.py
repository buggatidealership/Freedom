"""Price paths and targets around each event. Output uses schemas.T columns.

Bar convention everywhere: half-open [t, t_end). A price "at" instant `when` is the close of the
last bar whose t_end <= when, so a bar that contains `when` is never used. The reference price
p0 additionally backs off by a source-dependent buffer: `Settings.p0_buffer_minutes_sec_8k`
(default 3 minutes) when t0 comes from an 8-K acceptance time (measured acceptance-minus-wire
lags of 25-134 s), 0 otherwise.

Validity rules (from review): a checkpoint is NaN unless the bar used ends after the p0 bar
and its staleness (t0+h minus bar end) is within max(2 x interval, 5 min). Only 1m/5m bars
resolve prices; 1h or coarser candles are never used for p0, checkpoints or fills. One price
source per event, never mixed inside the window.

Corporate actions (docs/design.md §2): when the event carries a split ex-date
(schemas.E.ca_ex_date, set by the events builder from the FMP splits calendar) inside
[p0_time, t0 + horizon], the headline +24h checkpoint and the labels derived from it are NaN.
FMP bars are split-adjusted, so the proxy path keeps its intermediate checkpoints; a perp path
is not adjusted, so every checkpoint whose bar ends at or after the ex-date is NaN too (the
design's "used only if measured continuous" condition has no measurement yet, so the
conservative branch is the only one implemented).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from ..config import Settings
from ..schemas import (
    CHECKPOINTS,
    CONTINUATION_DEAD_BAND,
    HEADLINE_CHECKPOINT,
    NY,
    C,
    E,
    PriceSource,
    T,
)
from ..timeutil import is_rth, next_close_after, next_open_after, to_utc

log = logging.getLogger(__name__)

INTERVAL_TD = {"1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
               "15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1)}
FINE_INTERVALS = ("1m", "5m")  # the only intervals allowed to resolve p0, checkpoints and fills
P0_BUFFER_MINUTES_SEC_8K = 3.0  # default of Settings.p0_buffer_minutes_sec_8k; other sources: none
P0_BUFFER_SOURCES = ("sec_8k",)  # t0 sources whose P0 backs off by the buffer
PERP_SOURCES = (PriceSource.hl_archive.value, PriceSource.hl_live.value)  # unadjusted across splits
MIN_STALENESS = pd.Timedelta(minutes=5)


def max_staleness(interval: str) -> pd.Timedelta:
    """A checkpoint bar may end at most max(2 x interval, 5 min) before the checkpoint instant."""
    td = INTERVAL_TD.get(interval, pd.Timedelta(hours=1))
    return max(2 * td, MIN_STALENESS)


def p0_buffer_for(event: pd.Series, sec_8k_minutes: float = P0_BUFFER_MINUTES_SEC_8K) -> pd.Timedelta:
    """P0 backs off by `sec_8k_minutes` (Settings.p0_buffer_minutes_sec_8k) when t0 comes from
    an 8-K acceptance time, by nothing otherwise."""
    src = event.get(E.t0_source) if hasattr(event, "get") else None
    if str(src) in P0_BUFFER_SOURCES:
        return pd.Timedelta(minutes=float(sec_8k_minutes))
    return pd.Timedelta(0)


def corporate_action_ex(event: pd.Series) -> pd.Timestamp | None:
    """UTC instant of the split ex-date carried by the event (schemas.E.ca_ex_date), or None.
    A bare date is read as 00:00 America/New_York on that day."""
    v = event.get(E.ca_ex_date) if hasattr(event, "get") else None
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    try:
        t = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(t):
        return None
    return to_utc(t, assume_tz=NY)


def checkpoint_times(t0: pd.Timestamp, horizon_hours: int = 24) -> dict[str, pd.Timestamp]:
    """Map schemas.CHECKPOINTS -> UTC instants. next_open/next_close use the XNYS calendar;
    next_open_30m = next_open + 30 min; 24h = t0 + horizon_hours."""
    t0 = to_utc(t0)
    nxt_open = next_open_after(t0)
    out = {
        "5m": t0 + pd.Timedelta(minutes=5),
        "15m": t0 + pd.Timedelta(minutes=15),
        "30m": t0 + pd.Timedelta(minutes=30),
        "60m": t0 + pd.Timedelta(minutes=60),
        "2h": t0 + pd.Timedelta(hours=2),
        "next_open": nxt_open,
        "next_open_30m": nxt_open + pd.Timedelta(minutes=30),
        "next_close": next_close_after(t0),
        "24h": t0 + pd.Timedelta(hours=horizon_hours),
    }
    assert set(out) == set(CHECKPOINTS)
    return out


def _sorted_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if C.t_end not in bars.columns:
        raise ValueError("bars need a t_end column (half-open bars)")
    b = bars.sort_values(C.t_end)
    if b[C.t_end].dt.tz is None:
        raise ValueError("bars must be tz-aware UTC")
    return b


def price_at(bars: pd.DataFrame, when: pd.Timestamp, *, strictly_before: bool = False) -> tuple[float, pd.Timestamp] | None:
    """Close of the last bar with t_end <= when (or t_end < when when strictly_before).
    Bars are half-open [t, t_end); a bar containing `when` is never used. Returns
    (price, bar_end_time) or None."""
    if bars is None or len(bars) == 0:
        return None
    when = to_utc(when)
    b = _sorted_bars(bars)
    ends = b[C.t_end].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()
    side = "left" if strictly_before else "right"
    idx = int(np.searchsorted(ends, np.int64(when.value), side=side)) - 1
    if idx < 0:
        return None
    row = b.iloc[idx]
    return float(row[C.close]), pd.Timestamp(row[C.t_end])


def _interval_of(bars: pd.DataFrame) -> str:
    if C.interval in bars.columns and bars[C.interval].notna().any():
        return str(bars[C.interval].dropna().iloc[0])
    d = (bars[C.t_end] - bars[C.t]).median()
    for k, v in INTERVAL_TD.items():
        if v == d:
            return k
    return "1h"


def build_price_path(settings: Settings, event: pd.Series, *, market_bars: pd.DataFrame | None,
                     equity_bars: pd.DataFrame | None) -> pd.DataFrame:
    """Choose ONE source for the event window so the path never mixes price bases:
    the perp (market_bars) when 1m/5m candles cover [t0 - 1h, t0 + horizon], else the
    underlying's 1-minute extended-hours bars. Coarser candles are never used. Returns bars
    sorted by t with a `source` column; an empty frame with schemas.C columns when neither
    fine source covers the window (targets then stay NaN)."""
    t0 = to_utc(event[E.t0])
    lo, hi = t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=settings.horizon_hours)
    cols = [C.market, C.interval, C.t, C.t_end, C.open, C.high, C.low, C.close, C.volume, C.n_trades, C.source]

    def covers(b: pd.DataFrame | None) -> bool:
        if b is None or len(b) == 0 or _interval_of(b) not in FINE_INTERVALS:
            return False
        return bool(b[C.t].min() <= lo and b[C.t_end].max() >= hi - pd.Timedelta(hours=1))

    if covers(market_bars):
        out = market_bars
    elif covers(equity_bars):
        out = equity_bars
    else:
        return pd.DataFrame(columns=cols)
    out = out.sort_values(C.t).drop_duplicates(C.t, keep="last").reset_index(drop=True)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def compute_targets(event: pd.Series, path: pd.DataFrame, market_path: pd.DataFrame | None,
                    *, horizon_hours: int = 24, p0_buffer: pd.Timedelta | None = None) -> pd.Series:
    """p0 = close of the last bar with t_end <= t0 - buffer; r_<cp> = ln(p_cp / p0);
    ar_<cp> = r_<cp> - r_<cp>(market); labels direction/magnitude and
    continuation_k = sign(r_k) * sign(r_24h - r_k) for k in {15m, 30m} (NaN inside the dead band).
    Every checkpoint records the bar end used (t_<cp>) and staleness in minutes (s_<cp>);
    invalid checkpoints stay NaN. An event without t0 (a pending row) yields the all-NaN row.
    A split ex-date (E.ca_ex_date) inside [p0_time, t0 + horizon] NaNs the headline checkpoint
    and its labels; on a perp path every checkpoint from the ex-date on (module docstring)."""
    if p0_buffer is None:
        p0_buffer = p0_buffer_for(event)
    t0_raw = event.get(E.t0) if hasattr(event, "get") else event[E.t0]
    t0 = None if t0_raw is None or pd.isna(t0_raw) else to_utc(t0_raw)
    out: dict[str, object] = {T.event_id: event[E.event_id], T.p0: np.nan, T.p0_time: pd.NaT,
                              T.p0_staleness_min: np.nan, T.price_source: None, T.price_interval: None,
                              T.price_market: None, T.horizon_actual_h: np.nan,
                              T.h24_in_closure: (not is_rth(t0 + pd.Timedelta(hours=horizon_hours)))
                              if t0 is not None else pd.NA}
    for cp in CHECKPOINTS:
        out[T.r(cp)] = np.nan
        out[T.ar(cp)] = np.nan
        out[T.p(cp)] = np.nan
        out[T.t(cp)] = pd.NaT
        out[T.s(cp)] = np.nan
    out[T.direction] = np.nan
    out[T.magnitude] = np.nan
    out[T.continuation_15m] = np.nan
    out[T.continuation_30m] = np.nan
    if t0 is None or path is None or len(path) == 0:
        return pd.Series(out)

    interval = _interval_of(path)
    if interval not in FINE_INTERVALS:
        log.warning("%s: path interval %s is too coarse for targets", event[E.event_id], interval)
        return pd.Series(out)
    out[T.price_interval] = interval
    out[T.price_source] = str(path[C.source].dropna().iloc[0]) if path[C.source].notna().any() else None
    out[T.price_market] = str(path[C.market].dropna().iloc[0]) if C.market in path and path[C.market].notna().any() else None
    stale = max_staleness(interval)

    ref = price_at(path, t0 - p0_buffer)
    if ref is None:
        return pd.Series(out)
    p0, p0_time = ref
    if not (p0 > 0):
        return pd.Series(out)
    out[T.p0], out[T.p0_time] = p0, p0_time
    out[T.p0_staleness_min] = ((t0 - p0_buffer) - p0_time) / pd.Timedelta(minutes=1)

    mref = price_at(market_path, t0 - p0_buffer) if market_path is not None and len(market_path) else None
    m_stale = max_staleness(_interval_of(market_path)) if mref is not None else stale
    cps = checkpoint_times(t0, horizon_hours)
    for cp, when in cps.items():
        hit = price_at(path, when)
        if hit is None or hit[1] <= p0_time or when - hit[1] > stale:
            continue  # no post-release bar, or the last bar is too far before the checkpoint
        p, t_end = hit
        r = math.log(p / p0)
        out[T.p(cp)], out[T.t(cp)], out[T.r(cp)] = p, t_end, r
        out[T.s(cp)] = (when - t_end) / pd.Timedelta(minutes=1)
        if mref is not None:
            mh = price_at(market_path, when)
            if mh is not None and mh[1] > mref[1] and when - mh[1] <= m_stale:
                out[T.ar(cp)] = r - math.log(mh[0] / mref[0])
    ex = corporate_action_ex(event)
    if ex is not None and p0_time <= ex <= t0 + pd.Timedelta(hours=horizon_hours):
        # design §2: an ex-date inside [P0, t0 + horizon] leaves no headline label. The FMP
        # proxy is split-adjusted, so its intermediate checkpoints stand; a perp path is not,
        # so nothing measured at or after the ex-date is comparable with p0.
        void = [HEADLINE_CHECKPOINT]
        if out[T.price_source] in PERP_SOURCES:
            void += [cp for cp in CHECKPOINTS if cp != HEADLINE_CHECKPOINT
                     and not pd.isna(out[T.t(cp)]) and out[T.t(cp)] >= ex]
        for cp in void:
            out[T.r(cp)], out[T.ar(cp)], out[T.p(cp)], out[T.t(cp)], out[T.s(cp)] = np.nan, np.nan, np.nan, pd.NaT, np.nan
        log.info("%s: split ex-date %s inside the target window; %s left NaN",
                 event[E.event_id], ex.tz_convert(NY).date(), ", ".join(void))
    r24 = out[T.r("24h")]
    if not (isinstance(r24, float) and math.isnan(r24)):
        out[T.horizon_actual_h] = (out[T.t("24h")] - p0_time) / pd.Timedelta(hours=1)
        out[T.direction] = float(np.sign(r24))
        out[T.magnitude] = abs(r24)
        for k, col in (("15m", T.continuation_15m), ("30m", T.continuation_30m)):
            rk = out[T.r(k)]
            if isinstance(rk, float) and not math.isnan(rk) and abs(rk) >= CONTINUATION_DEAD_BAND:
                out[col] = float(np.sign(rk) * np.sign(r24 - rk))
    return pd.Series(out)


def build_targets(settings: Settings, events: pd.DataFrame, *, write: bool = True,
                  benchmark_market: str = "xyz:SP500", benchmark_equity: str = "SPY") -> pd.DataFrame:
    """Compute targets for every event. Perp bars come from the archive, then the live
    candleSnapshot window; otherwise the underlying's FMP 1-minute extended-hours bars.
    Pending rows (no t0: `freedom events` stopped at the budget before them) get the all-NaN
    row without any provider request, so the table keeps one row per event_id."""
    from ..data.base import ProviderUnavailable
    from ..data.hyperliquid import HyperliquidClient
    from .loaders import load_event_bars

    settings.ensure_dirs()
    hl = HyperliquidClient(settings)
    fmp = None
    try:
        from ..data.fmp import FMPClient

        fmp = FMPClient(settings)
    except ProviderUnavailable as exc:
        log.warning("FMP unavailable (%s): events before perp listing will have no targets", exc)
    horizon = settings.horizon_hours
    rows = []
    for _, ev in events.iterrows():
        buffer = p0_buffer_for(ev, settings.p0_buffer_minutes_sec_8k)
        t0, pending = ev.get(E.t0), ev.get(E.pending, False)
        pending = pending is not None and not pd.isna(pending) and bool(pending)
        if pending or t0 is None or pd.isna(t0):
            log.warning("%s: no t0 (pending), targets left NaN", ev.get(E.event_id))
            rows.append(compute_targets(ev, pd.DataFrame(), None, horizon_hours=horizon, p0_buffer=buffer))
            continue
        try:
            path, mpath = load_event_bars(settings, ev, hl=hl, fmp=fmp,
                                          benchmark_market=benchmark_market, benchmark_equity=benchmark_equity)
            rows.append(compute_targets(ev, path, mpath, horizon_hours=horizon, p0_buffer=buffer))
        except Exception as exc:  # one bad event must not kill the run
            log.warning("targets failed for %s: %s", ev.get(E.event_id), exc)
            empty = compute_targets(ev, pd.DataFrame(), None, horizon_hours=horizon, p0_buffer=buffer)
            rows.append(empty)
    out = pd.DataFrame(rows)
    if write:
        out.to_parquet(settings.targets_path, index=False)
    return out
