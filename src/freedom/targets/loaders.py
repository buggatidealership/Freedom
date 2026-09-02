"""Fetch the bars needed for one event's price path from the archive, live candles or FMP."""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Settings
from ..schemas import C, E, PriceSource
from ..timeutil import to_utc
from . import build_price_path

log = logging.getLogger(__name__)


def _perp_bars(settings: Settings, hl, market: str, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame | None:
    """Archive (1m, then 5m) if it covers the window, else live 1m/5m candles when the
    5000-candle window still reaches back to `lo`, else None. Coarser candles are never used."""
    from ..data.archive import load_archive

    for interval in ("1m", "5m"):
        try:
            b = load_archive(settings, market, interval, lo, hi)
        except FileNotFoundError:
            b = None
        if b is not None and len(b) and b[C.t].min() <= lo + pd.Timedelta(minutes=30) and b[C.t_end].max() >= hi - pd.Timedelta(hours=1):
            b = b.copy()
            b[C.source] = PriceSource.hl_archive.value
            return b
    now = pd.Timestamp.now(tz="UTC")
    for interval, span in (("1m", pd.Timedelta(minutes=5000)), ("5m", pd.Timedelta(minutes=5 * 5000))):
        if now - lo < span * 0.95:
            b = hl.candles(market, interval, lo, hi)
            if b is not None and len(b):
                b = b.copy()
                b[C.source] = PriceSource.hl_live.value
                return b
    return None


def _equity_bars(fmp, symbol: str, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame | None:
    if fmp is None:
        return None
    # [t0 - 1 day, t0 + horizon + 2h] in New York calendar days: three days for every release
    # clock, one FMP request (1-minute responses are capped at three sessions), and the same
    # window the events resolver requests so the cache is shared.
    b = fmp.intraday(symbol, "1min", lo.tz_convert("America/New_York").normalize(),
                     hi.tz_convert("America/New_York").normalize(), extended=True)
    if b is None or len(b) == 0:
        return None
    b = b.copy()
    b[C.source] = PriceSource.fmp_intraday.value
    return b


def load_event_bars(settings: Settings, event: pd.Series, *, hl, fmp, benchmark_market: str,
                    benchmark_equity: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    t0 = to_utc(event[E.t0])
    lo = t0 - pd.Timedelta(days=1)
    hi = t0 + pd.Timedelta(hours=settings.horizon_hours) + pd.Timedelta(hours=2)
    market = event.get(E.market)
    perp = _perp_bars(settings, hl, market, lo, hi) if isinstance(market, str) and market else None
    equity = _equity_bars(fmp, event[E.underlying], lo, hi)
    path = build_price_path(settings, event, market_bars=perp, equity_bars=equity)
    if len(path) == 0:
        return path, None
    src = path[C.source].iloc[0]
    if src in (PriceSource.hl_archive.value, PriceSource.hl_live.value):
        mpath = _perp_bars(settings, hl, benchmark_market, lo, hi)
    else:
        mpath = _equity_bars(fmp, benchmark_equity, lo, hi)
    if mpath is not None and len(mpath):
        mpath = build_price_path(settings, event, market_bars=mpath if src != PriceSource.fmp_intraday.value else None,
                                 equity_bars=mpath if src == PriceSource.fmp_intraday.value else None)
    return path, (mpath if mpath is not None and len(mpath) else None)
