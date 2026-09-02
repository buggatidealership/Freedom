"""Price paths and targets around each event. Output uses schemas.T columns.

Bar convention everywhere: half-open [t, t_end). A price "at" instant `when` is the close of the
last bar whose t_end <= when, so a bar that contains `when` is never used. The reference price
p0 additionally backs off by `p0_buffer` (default 1 minute) because the 8-K acceptance time can
trail the newswire by up to a minute.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from ..config import Settings
from ..schemas import CHECKPOINTS, C, E, T
from ..timeutil import next_close_after, next_open_after, to_utc

log = logging.getLogger(__name__)

INTERVAL_TD = {"1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
               "15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1)}
P0_BUFFER = pd.Timedelta(minutes=1)
# a checkpoint price is accepted only if the bar used ends within this staleness of the checkpoint
MAX_STALENESS = {"1m": pd.Timedelta(minutes=10), "5m": pd.Timedelta(minutes=15),
                 "15m": pd.Timedelta(minutes=45), "1h": pd.Timedelta(hours=2)}


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
    the perp (market_bars) when it covers [t0 - 1h, t0 + horizon] at 15m or finer, else the
    underlying's extended-hours bars. Returns bars sorted by t with a `source` column; an empty
    frame with schemas.C columns when neither source covers the window."""
    t0 = to_utc(event[E.t0])
    lo, hi = t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=settings.horizon_hours)
    cols = [C.market, C.interval, C.t, C.t_end, C.open, C.high, C.low, C.close, C.volume, C.n_trades, C.source]

    def covers(b: pd.DataFrame | None, fine_only: bool) -> bool:
        if b is None or len(b) == 0:
            return False
        if fine_only and _interval_of(b) not in ("1m", "5m", "15m"):
            return False
        return bool(b[C.t].min() <= lo and b[C.t_end].max() >= hi - pd.Timedelta(hours=1))

    if covers(market_bars, fine_only=True):
        out = market_bars
    elif covers(equity_bars, fine_only=False):
        out = equity_bars
    elif covers(market_bars, fine_only=False):  # coarse perp bars beat nothing
        out = market_bars
    else:
        return pd.DataFrame(columns=cols)
    out = out.sort_values(C.t).drop_duplicates(C.t, keep="last").reset_index(drop=True)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def compute_targets(event: pd.Series, path: pd.DataFrame, market_path: pd.DataFrame | None,
                    *, horizon_hours: int = 24, p0_buffer: pd.Timedelta | None = None) -> pd.Series:
    """p0 = price strictly before t0; r_<cp> = ln(p_cp / p0); ar_<cp> = r_<cp> - r_<cp>(market);
    labels direction/magnitude/continuation. Missing checkpoints stay NaN."""
    if p0_buffer is None:
        p0_buffer = P0_BUFFER
    t0 = to_utc(event[E.t0])
    out: dict[str, object] = {T.event_id: event[E.event_id], T.p0: np.nan, T.p0_time: pd.NaT,
                              T.price_source: None, "path_interval": None}
    for cp in CHECKPOINTS:
        out[T.r(cp)] = np.nan
        out[T.ar(cp)] = np.nan
        out[T.p(cp)] = np.nan
        out[T.t(cp)] = pd.NaT
    out[T.direction] = np.nan
    out[T.magnitude] = np.nan
    out[T.continuation] = np.nan
    if path is None or len(path) == 0:
        return pd.Series(out)

    interval = _interval_of(path)
    out["path_interval"] = interval
    out[T.price_source] = str(path[C.source].dropna().iloc[0]) if path[C.source].notna().any() else None
    stale = MAX_STALENESS.get(interval, pd.Timedelta(hours=2))

    ref = price_at(path, t0 - p0_buffer)
    if ref is None:
        return pd.Series(out)
    p0, p0_time = ref
    if not (p0 > 0):
        return pd.Series(out)
    out[T.p0], out[T.p0_time] = p0, p0_time

    mref = price_at(market_path, t0 - p0_buffer) if market_path is not None and len(market_path) else None
    cps = checkpoint_times(t0, horizon_hours)
    for cp, when in cps.items():
        hit = price_at(path, when)
        if hit is None or hit[1] <= p0_time or when - hit[1] > stale:
            continue  # no post-release bar, or the last bar is too far before the checkpoint
        p, t_end = hit
        r = math.log(p / p0)
        out[T.p(cp)], out[T.t(cp)], out[T.r(cp)] = p, t_end, r
        if mref is not None:
            mh = price_at(market_path, when)
            if mh is not None and mh[1] > mref[1] and when - mh[1] <= MAX_STALENESS.get(_interval_of(market_path), stale):
                out[T.ar(cp)] = r - math.log(mh[0] / mref[0])
    r24, r30 = out[T.r("24h")], out[T.r("30m")]
    if not (isinstance(r24, float) and math.isnan(r24)):
        out[T.direction] = float(np.sign(r24))
        out[T.magnitude] = abs(r24)
        if not (isinstance(r30, float) and math.isnan(r30)):
            out[T.continuation] = float(np.sign(r24 - r30))
    return pd.Series(out)


def build_targets(settings: Settings, events: pd.DataFrame, *, write: bool = True,
                  benchmark_market: str = "xyz:SP500", benchmark_equity: str = "SPY") -> pd.DataFrame:
    """Compute targets for every event. Perp bars come from the archive, then the live
    candleSnapshot window; otherwise the underlying's FMP 1-minute extended-hours bars."""
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
    rows = []
    for _, ev in events.iterrows():
        try:
            path, mpath = load_event_bars(settings, ev, hl=hl, fmp=fmp,
                                          benchmark_market=benchmark_market, benchmark_equity=benchmark_equity)
            rows.append(compute_targets(ev, path, mpath, horizon_hours=settings.horizon_hours))
        except Exception as exc:  # one bad event must not kill the run
            log.warning("targets failed for %s: %s", ev.get(E.event_id), exc)
            empty = compute_targets(ev, pd.DataFrame(), None, horizon_hours=settings.horizon_hours)
            rows.append(empty)
    out = pd.DataFrame(rows)
    if write:
        out.to_parquet(settings.data_dir / "targets.parquet", index=False)
    return out
