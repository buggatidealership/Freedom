"""Hyperliquid info-endpoint client (no key). See docs/data-sources.md for measured limits.

Weights: every info request costs 20 except the cheap ones (allMids, l2Book: 2); candleSnapshot
adds 1 per 60 candles returned. The limiter is shared across all methods of one client.

Measured server semantics (2026-09-02) that the code below relies on:
* `candleSnapshot` returns bars with `t` (start, epoch ms) and `T = t + interval - 1 ms`
  (inclusive end). `endTime` is inclusive of the bar that *starts* at it, and an unaligned
  `startTime` returns the bar that *contains* it, so every page is requested with
  `endTime = end - 1` and post-filtered to `start <= t < end`.
* Only the most recent 5000 candles per (market, interval) exist; older windows return `[]`.
* `fundingHistory` returns at most 500 entries per request, oldest first.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from ..config import Settings
from ..data.base import DiskCache, HttpClient, TokenBucket, utcnow
from ..schemas import UTC, C, PriceSource
from ..timeutil import to_utc

INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w", "1M"]
INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
               "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "8h": 28_800_000,
               "12h": 43_200_000, "1d": 86_400_000}
MAX_CANDLES_PER_REQUEST = 5000

INFO_WEIGHT = 20  # base weight of every documented info request
CANDLES_PER_WEIGHT = 60  # candleSnapshot: +1 weight per 60 candles
FUNDING_ITEMS_PER_WEIGHT = 20  # fundingHistory: +1 weight per 20 items
FUNDING_PAGE_SIZE = 500  # measured: a 40-day request returned exactly 500 entries
FUNDING_INTERVAL_MS = 3_600_000  # funding settles hourly on HIP-3 perps

# Calendar-length intervals: nominal spans for paging; t_end comes from the server's `T + 1`.
_CALENDAR_INTERVAL_MS = {"3d": 3 * 86_400_000, "1w": 7 * 86_400_000, "1M": 31 * 86_400_000}

CANDLE_COLUMNS: list[str] = [C.market, C.interval, C.t, C.t_end, C.open, C.high, C.low, C.close,
                             C.volume, C.n_trades, C.source]
FUNDING_COLUMNS: list[str] = ["market", "t", "funding_rate", "premium"]
MARKET_COLUMNS: list[str] = ["market", "dex", "symbol", "max_leverage", "growth_mode",
                             "deployer_fee_scale", "only_isolated"]


# ---- helpers -------------------------------------------------------------------------------------
def to_ms(ts: pd.Timestamp | str) -> int:
    """Epoch milliseconds of a timestamp (naive inputs are taken as UTC)."""
    return int(to_utc(ts, assume_tz=UTC).value // 1_000_000)


def from_ms(ms: pd.Series | Iterable[int]) -> pd.Series:
    return pd.to_datetime(pd.Series(ms, dtype="int64"), unit="ms", utc=True)


def interval_ms(interval: str) -> int:
    """Bar length in ms; calendar intervals (3d, 1w, 1M) get a nominal length used for paging."""
    if interval in INTERVAL_MS:
        return INTERVAL_MS[interval]
    if interval in _CALENDAR_INTERVAL_MS:
        return _CALENDAR_INTERVAL_MS[interval]
    raise ValueError(f"unknown Hyperliquid interval {interval!r}; expected one of {INTERVALS}")


def candle_weight(n_candles: int) -> int:
    """Rate-limit weight of one candleSnapshot request returning `n_candles`."""
    return INFO_WEIGHT + math.ceil(n_candles / CANDLES_PER_WEIGHT)


def funding_weight(n_items: int) -> int:
    return INFO_WEIGHT + math.ceil(n_items / FUNDING_ITEMS_PER_WEIGHT)


def empty_candles() -> pd.DataFrame:
    """Zero-row frame with the schemas.C columns and the dtypes non-empty frames carry."""
    return pd.DataFrame({
        C.market: pd.Series(dtype=str), C.interval: pd.Series(dtype=str),
        C.t: pd.Series(dtype="datetime64[ns, UTC]"), C.t_end: pd.Series(dtype="datetime64[ns, UTC]"),
        C.open: pd.Series(dtype="float64"), C.high: pd.Series(dtype="float64"),
        C.low: pd.Series(dtype="float64"), C.close: pd.Series(dtype="float64"),
        C.volume: pd.Series(dtype="float64"), C.n_trades: pd.Series(dtype="int64"),
        C.source: pd.Series(dtype=str),
    })


def empty_funding() -> pd.DataFrame:
    return pd.DataFrame({
        "market": pd.Series(dtype=str), "t": pd.Series(dtype="datetime64[ns, UTC]"),
        "funding_rate": pd.Series(dtype="float64"), "premium": pd.Series(dtype="float64"),
    })


def candles_frame(market: str, interval: str, raw: Iterable[dict], start_ms: int, end_ms: int,
                  *, source: str = PriceSource.hl_live.value) -> pd.DataFrame:
    """Raw candleSnapshot records -> schemas.C frame restricted to start <= t < end,
    deduplicated on t (last occurrence wins) and sorted by t."""
    rows = [r for r in raw if start_ms <= int(r["t"]) < end_ms]
    if not rows:
        return empty_candles()
    df = pd.DataFrame(rows)
    df["t"] = df["t"].astype("int64")
    df = df.drop_duplicates(subset="t", keep="last").sort_values("t", kind="mergesort")
    df = df.reset_index(drop=True)
    t = from_ms(df["t"])
    if interval in INTERVAL_MS:
        t_end = t + pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    else:  # calendar-length bars: the server's inclusive close + 1 ms
        t_end = from_ms(df["T"].astype("int64") + 1)
    out = pd.DataFrame({
        C.market: market, C.interval: interval, C.t: t, C.t_end: t_end,
        C.open: df["o"].astype("float64"), C.high: df["h"].astype("float64"),
        C.low: df["l"].astype("float64"), C.close: df["c"].astype("float64"),
        C.volume: df["v"].astype("float64"), C.n_trades: df["n"].astype("int64"),
        C.source: source,
    })
    return out[CANDLE_COLUMNS]


def funding_frame(market: str, raw: Iterable[dict], start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = [r for r in raw if start_ms <= int(r["time"]) < end_ms]
    if not rows:
        return empty_funding()
    df = pd.DataFrame(rows)
    df["time"] = df["time"].astype("int64")
    df = df.drop_duplicates(subset="time", keep="last").sort_values("time", kind="mergesort")
    df = df.reset_index(drop=True)
    out = pd.DataFrame({
        "market": market, "t": from_ms(df["time"]),
        "funding_rate": df["fundingRate"].astype("float64"),
        "premium": df["premium"].astype("float64"),
    })
    return out[FUNDING_COLUMNS]


class HyperliquidClient:
    def __init__(self, settings: Settings, cache: DiskCache | None = None):
        self.settings = settings
        self.http = HttpClient(
            provider="hyperliquid",
            cache=cache or DiskCache(settings.cache_dir),
            limiter=TokenBucket(settings.hyperliquid_weight_per_minute),
            timeout=settings.http_timeout_seconds,
        )
        self.url = f"{settings.hyperliquid_api_url}/info"

    def _info(self, body: dict, *, cache_ttl: int | None, weight: float = INFO_WEIGHT):
        return self.http.post_json(self.url, body, cache_ttl=cache_ttl, weight=weight)

    # ---- metadata ----------------------------------------------------------------------------
    def perp_dexs(self, *, cache_ttl: int | None = 3600) -> list[dict]:
        """Raw `perpDexs` response with the leading null (main dex) removed."""
        raw = self._info({"type": "perpDexs"}, cache_ttl=cache_ttl)
        return [d for d in (raw or []) if d is not None]

    def meta(self, dex: str, *, cache_ttl: int | None = 3600) -> dict:
        """Raw `meta` for a dex; `universe` entries carry `isDelisted`, `maxLeverage`, ..."""
        return self._info({"type": "meta", "dex": dex}, cache_ttl=cache_ttl)

    def meta_and_asset_ctxs(self, dex: str) -> tuple[dict, list[dict]]:
        """Live (uncached) `metaAndAssetCtxs`: funding, openInterest, premium, oraclePx, markPx, midPx."""
        raw = self._info({"type": "metaAndAssetCtxs", "dex": dex}, cache_ttl=None)
        meta, ctxs = raw
        return meta, list(ctxs)

    def all_markets(self) -> pd.DataFrame:
        """One row per non-delisted market across all dexes:
        columns market, dex, symbol, max_leverage, growth_mode, deployer_fee_scale, only_isolated."""
        rows = []
        for d in self.perp_dexs():
            dex = d["name"]
            for asset in self.meta(dex).get("universe", []):
                if asset.get("isDelisted", False):
                    continue
                name = asset["name"]
                symbol = name.split(":", 1)[1] if ":" in name else name
                fee_scale = asset.get("deployerFeeScale")
                rows.append({
                    "market": name,
                    "dex": dex,
                    "symbol": symbol,
                    "max_leverage": int(asset["maxLeverage"]),
                    "growth_mode": asset.get("growthMode") == "enabled",
                    "deployer_fee_scale": float(fee_scale) if fee_scale is not None else math.nan,
                    "only_isolated": bool(asset.get("onlyIsolated", False)),
                })
        df = pd.DataFrame(rows, columns=MARKET_COLUMNS)
        return df.astype({"max_leverage": "int64", "growth_mode": bool,
                          "deployer_fee_scale": "float64", "only_isolated": bool})

    # ---- candles -----------------------------------------------------------------------------
    def _candle_page(self, market: str, interval: str, win_start: int, win_end: int, step: int,
                     cache_ttl: int | None) -> list[dict]:
        """One candleSnapshot request for start <= t < win_end (at most 5000 bars).
        The limiter is charged before the response is known, so the weight uses the number
        of bars the window can hold; it is exact for full pages and conservative otherwise."""
        n_expected = min(MAX_CANDLES_PER_REQUEST, max(1, math.ceil((win_end - win_start) / step)))
        body = {"type": "candleSnapshot",
                "req": {"coin": market, "interval": interval,
                        "startTime": win_start, "endTime": win_end - 1}}
        raw = self._info(body, cache_ttl=cache_ttl, weight=candle_weight(n_expected))
        return list(raw or [])

    def candles(self, market: str, interval: str, start: pd.Timestamp, end: pd.Timestamp,
                *, cache_ttl: int | None = None) -> pd.DataFrame:
        """Candles in [start, end) as a frame with schemas.C columns (t, t_end tz-aware UTC,
        floats, n_trades int, source='hl_live'). Pages by startTime when more than
        MAX_CANDLES_PER_REQUEST would be needed; never returns duplicates; sorted by t.
        Weight per call = 20 + ceil(n_returned / 60). Only the most recent 5000 candles per
        interval exist server-side; older requests return empty frames, not errors."""
        step = interval_ms(interval)
        start_ms, end_ms = to_ms(start), to_ms(end)
        pages: list[dict] = []
        # Page backwards from `end`: the newest window is the one the server can serve, and the
        # first empty page means everything older is beyond the 5000-candle horizon.
        win_end = end_ms
        while win_end > start_ms:
            win_start = max(start_ms, win_end - MAX_CANDLES_PER_REQUEST * step)
            page = self._candle_page(market, interval, win_start, win_end, step, cache_ttl)
            if not page:
                break
            pages.extend(page)
            win_end = win_start
        return candles_frame(market, interval, pages, start_ms, end_ms)

    def listing_start(self, market: str) -> pd.Timestamp | None:
        """Start time of the first daily candle, or None if the market has no candles."""
        today = pd.Timestamp(utcnow()).normalize()
        df = self.candles(market, "1d", pd.Timestamp("2020-01-01", tz=UTC),
                          today + pd.Timedelta(days=1),
                          cache_ttl=self.settings.cache_ttl_seconds)
        if df.empty:
            return None
        return pd.Timestamp(df[C.t].iloc[0])

    # ---- funding -----------------------------------------------------------------------------
    def funding_history(self, market: str, start: pd.Timestamp, end: pd.Timestamp | None = None,
                        *, cache_ttl: int | None = None) -> pd.DataFrame:
        """Hourly funding: columns market, t (UTC), funding_rate, premium. Pages by time."""
        start_ms = to_ms(start)
        end_ms = to_ms(end) if end is not None else to_ms(pd.Timestamp(utcnow()))
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            n_expected = min(FUNDING_PAGE_SIZE,
                             max(1, math.ceil((end_ms - cursor) / FUNDING_INTERVAL_MS)))
            body = {"type": "fundingHistory", "coin": market,
                    "startTime": cursor, "endTime": end_ms}
            page = list(self._info(body, cache_ttl=cache_ttl, weight=funding_weight(n_expected))
                        or [])
            if not page:
                break
            rows.extend(page)
            last = max(int(r["time"]) for r in page)
            if len(page) < FUNDING_PAGE_SIZE or last + 1 >= end_ms:
                break
            cursor = last + 1
        return funding_frame(market, rows, start_ms, end_ms)
