"""Financial Modeling Prep client (key required: FMP_API_KEY). Daily budget enforced.

Endpoints, all under ``settings.fmp_base_url`` and authenticated with the ``apikey`` query
parameter (see docs/data-sources.md for what was measured):

* ``stable/earnings?symbol=&limit=`` and ``stable/earnings-calendar?from=&to=``
* ``stable/historical-chart/{1min|5min}?symbol=&from=&to=&extended=true`` (newest-first, naive
  America/New_York timestamps, pre-market and after-hours bars 04:00-19:55 ET)
* ``stable/historical-price-eod/full?symbol=&from=&to=``
* ``stable/profile?symbol=`` and ``stable/aftermarket-trade?symbol=``
* ``stable/splits?symbol=`` (split ex-dates; intraday and EOD bars are already split-adjusted,
  the calendar only flags events whose window straddles an ex-date)

Conventions:

* every timestamp leaving this module is a tz-aware UTC pandas Timestamp; calendar dates
  (``report_date_ny``, ``last_updated``) are plain ``datetime.date`` objects;
* the API key is added to the request only: it never enters cache keys (``cache_params`` is
  passed without it), httpx log lines (a redacting filter is installed) or exception messages;
* intraday requests are chunked into fixed ``MAX_INTRADAY_DAYS_PER_REQUEST``-day windows
  counted from ``start_day`` so a rerun with the same arguments hits the disk cache;
* cache policy for price ranges: a chunk that ends before today (New York) is cached as
  immutable because completed sessions do not change, except that an *empty* payload is only
  trusted for ``settings.cache_ttl_seconds`` (a weekend/holiday is legitimately empty, but a
  transient empty 200 or a symbol whose history is backfilled later must not stick for ever);
  a chunk that touches today is a live, in-progress session and is cached for
  ``settings.live_cache_ttl_seconds`` only, so ``freedom predict`` never sees a stale partial day.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any

import httpx
import pandas as pd

from ..config import Settings
from ..data.base import DailyBudget, DiskCache, HttpClient, ProviderUnavailable
from ..schemas import NY, UTC, C, E, PriceSource, U
from ..timeutil import to_utc

MAX_INTRADAY_DAYS_PER_REQUEST = 5
# Measured 2026-09-02: a 1-minute request covering five sessions returned only the latest three
# (2880 bars); 5-minute requests returned all five. Chunk 1-minute windows at three calendar
# days so no session is silently dropped.
MAX_CALENDAR_DAYS_PER_REQUEST = 30  # stable/earnings-calendar truncates long windows (see earnings_calendar)
MAX_INTRADAY_DAYS_BY_INTERVAL = {"1min": 3}
DAILY_SOURCE = "fmp_daily"  # daily bars are not intraday; PriceSource only names path sources
IMMUTABLE_TTL_SECONDS = 10 * 365 * 24 * 3600  # completed sessions never change

EARNINGS_COLUMNS: list[str] = [
    U.symbol, E.report_date_ny, E.eps_actual, E.eps_estimate, E.rev_actual, E.rev_estimate,
    "last_updated",
]
CANDLE_COLUMNS: list[str] = [
    C.market, C.interval, C.t, C.t_end, C.open, C.high, C.low, C.close, C.volume, C.n_trades,
    C.source,
]
SPLIT_COLUMNS: list[str] = ["symbol", "ex_date", "numerator", "denominator"]

# FMP path segment -> (harness interval label as used by the archive, bar length)
_INTERVALS: dict[str, tuple[str, pd.Timedelta]] = {
    "1min": ("1m", pd.Timedelta(minutes=1)),
    "5min": ("5m", pd.Timedelta(minutes=5)),
    "15min": ("15m", pd.Timedelta(minutes=15)),
    "30min": ("30m", pd.Timedelta(minutes=30)),
    "1hour": ("1h", pd.Timedelta(hours=1)),
    "4hour": ("4h", pd.Timedelta(hours=4)),
}
_INTERVAL_ALIASES: dict[str, str] = {label: api for api, (label, _) in _INTERVALS.items()}

_APIKEY_RE = re.compile(r"(apikey=)[^&\s'\"]+", re.IGNORECASE)


class FMPError(RuntimeError):
    """FMP answered with an error (HTTP status or an error payload). Never carries the key."""


# ---- key hygiene ------------------------------------------------------------------------------
def _redact(text: str) -> str:
    """Replace the value of any ``apikey=`` query parameter in ``text`` with ``***``."""
    return _APIKEY_RE.sub(r"\1***", text)


class _RedactApiKey(logging.Filter):
    """httpx logs every request URL at INFO level; scrub the key before the record is formatted."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str) and "apikey" in record.msg.lower():
            record.msg = _redact(record.msg)
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(
                _redact(str(a)) if "apikey" in str(a).lower() else a for a in args
            )
        elif isinstance(args, dict):
            record.args = {
                k: (_redact(str(v)) if "apikey" in str(v).lower() else v) for k, v in args.items()
            }
        return True


def _install_log_redaction() -> None:
    for name in ("httpx", "httpcore"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, _RedactApiKey) for f in logger.filters):
            logger.addFilter(_RedactApiKey())


# ---- small conversions ---------------------------------------------------------------------------
def _norm_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not s:
        raise ValueError("symbol must be a non-empty ticker")
    return s


def _as_ny_date(ts: Any) -> date:
    """Calendar date in New York for a Timestamp/str/date; tz-aware inputs are converted first."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(NY)
    return t.date()


def _today_ny() -> date:
    return pd.Timestamp.now(tz=NY).date()


def _day_chunks(first: date, last: date, max_days: int) -> list[tuple[date, date]]:
    """Consecutive inclusive [a, b] windows of at most ``max_days`` days covering [first, last].

    Anchored at ``first`` so the same call always produces the same requests (cache hits)."""
    out: list[tuple[date, date]] = []
    a = first
    while a <= last:
        b = min(a + timedelta(days=max_days - 1), last)
        out.append((a, b))
        a = b + timedelta(days=1)
    return out


def _resolve_interval(interval: str) -> tuple[str, str, pd.Timedelta]:
    """Accept FMP names ('5min') or harness labels ('5m'); return (api_name, label, bar length)."""
    key = str(interval).strip().lower()
    api = _INTERVAL_ALIASES.get(key, key)
    if api not in _INTERVALS:
        raise ValueError(
            f"unsupported FMP interval {interval!r}; use one of {sorted(_INTERVALS)} "
            f"or {sorted(_INTERVAL_ALIASES)}"
        )
    label, step = _INTERVALS[api]
    return api, label, step


def _ny_naive_to_utc(values: Any) -> pd.DatetimeIndex:
    """Vectorised twin of ``timeutil.to_utc(..., assume_tz=NY)``: naive New York -> UTC (ns).

    Ambiguous fall-back instants raise (as ``to_utc`` does) instead of guessing; they cannot occur
    inside FMP's 04:00-19:55 ET extended session, but the contract is enforced regardless."""
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        raise ValueError("expected naive America/New_York timestamps from FMP")
    local = idx.tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward")
    if local.isna().any():
        bad = list(idx[local.isna()][:3])
        raise ValueError(f"ambiguous DST timestamp(s) from FMP in {NY}: {bad}")
    return pd.DatetimeIndex(local.tz_convert(UTC)).as_unit("ns")


def _to_dates(values: pd.Series) -> Any:
    """Object array of ``datetime.date`` (NaT where missing/unparseable)."""
    return pd.to_datetime(values, errors="coerce").dt.date.to_numpy()


def _epoch_to_utc(value: Any) -> pd.Timestamp | None:
    """Epoch seconds or milliseconds (FMP uses ms) -> tz-aware UTC Timestamp."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    v = int(value)
    unit = "ms" if abs(v) >= 100_000_000_000 else "s"
    return pd.Timestamp(v, unit=unit, tz=UTC)


def _is_empty_payload(payload: Any) -> bool:
    """``None``, ``[]`` and ``{}`` carry no records (``_records`` maps all three to ``[]``)."""
    return payload is None or (isinstance(payload, list | dict) and not payload)


# ---- frame builders -----------------------------------------------------------------------------
def _empty_candles() -> pd.DataFrame:
    return pd.DataFrame({
        C.market: pd.Series(dtype="str"),
        C.interval: pd.Series(dtype="str"),
        C.t: pd.Series(dtype="datetime64[ns, UTC]"),
        C.t_end: pd.Series(dtype="datetime64[ns, UTC]"),
        C.open: pd.Series(dtype="float64"),
        C.high: pd.Series(dtype="float64"),
        C.low: pd.Series(dtype="float64"),
        C.close: pd.Series(dtype="float64"),
        C.volume: pd.Series(dtype="float64"),
        C.n_trades: pd.Series(dtype="Int64"),
        C.source: pd.Series(dtype="str"),
    })[CANDLE_COLUMNS]


def _candles(market: str, interval: str, t: pd.DatetimeIndex, t_end: pd.DatetimeIndex,
             raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Assemble a schemas.C frame; ``raw`` carries FMP's open/high/low/close/volume columns."""
    n = len(raw)
    raw = raw.reindex(columns=["open", "high", "low", "close", "volume"])

    def col(name: str) -> Any:
        return pd.to_numeric(raw[name], errors="coerce").astype("float64").to_numpy()

    return pd.DataFrame({
        C.market: [market] * n,
        C.interval: [interval] * n,
        C.t: pd.DatetimeIndex(t),
        C.t_end: pd.DatetimeIndex(t_end),
        C.open: col("open"),
        C.high: col("high"),
        C.low: col("low"),
        C.close: col("close"),
        C.volume: col("volume"),
        C.n_trades: pd.array([pd.NA] * n, dtype="Int64"),  # FMP does not report trade counts
        C.source: [source] * n,
    })[CANDLE_COLUMNS]


def _finish_candles(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate chunks, sort oldest-first, drop duplicate bar starts, reset the index."""
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty_candles()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(C.t, kind="mergesort").drop_duplicates(subset=[C.t], keep="first")
    return df.reset_index(drop=True)


def _intraday_frame(records: list[dict], symbol: str, label: str, step: pd.Timedelta,
                    first: date, last: date) -> pd.DataFrame:
    if not records:
        return _empty_candles()
    raw = pd.DataFrame.from_records(records)
    if "date" not in raw.columns:
        raise FMPError("FMP intraday payload has no 'date' field")
    naive = pd.DatetimeIndex(pd.to_datetime(raw["date"]))
    # Defensive: keep only bars inside the requested New York calendar days.
    keep = (naive >= pd.Timestamp(first)) & (naive < pd.Timestamp(last) + pd.Timedelta(days=1))
    raw, naive = raw[keep], naive[keep]
    t = _ny_naive_to_utc(naive)
    return _candles(symbol, label, t, t + step, raw, PriceSource.fmp_intraday)


def _earnings_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame({
            U.symbol: pd.Series(dtype="str"),
            E.report_date_ny: pd.Series(dtype="object"),
            E.eps_actual: pd.Series(dtype="float64"),
            E.eps_estimate: pd.Series(dtype="float64"),
            E.rev_actual: pd.Series(dtype="float64"),
            E.rev_estimate: pd.Series(dtype="float64"),
            "last_updated": pd.Series(dtype="object"),
        })[EARNINGS_COLUMNS]
    raw = pd.DataFrame.from_records(records).reindex(
        columns=["symbol", "date", "epsActual", "epsEstimated", "revenueActual",
                 "revenueEstimated", "lastUpdated"]
    )

    def num(name: str) -> Any:
        return pd.to_numeric(raw[name], errors="coerce").astype("float64").to_numpy()

    df = pd.DataFrame({
        U.symbol: ["" if s is None else str(s).strip().upper() for s in raw["symbol"]],
        E.report_date_ny: _to_dates(raw["date"]),
        E.eps_actual: num("epsActual"),
        E.eps_estimate: num("epsEstimated"),
        E.rev_actual: num("revenueActual"),
        E.rev_estimate: num("revenueEstimated"),
        "last_updated": _to_dates(raw["lastUpdated"]),
    })[EARNINGS_COLUMNS]
    df = df[df[E.report_date_ny].notna()]
    df = df.sort_values([U.symbol, E.report_date_ny, "last_updated"], kind="mergesort",
                        na_position="first")
    df = df.drop_duplicates(subset=[U.symbol, E.report_date_ny], keep="last")
    return df.reset_index(drop=True)


def _splits_frame(records: list[dict]) -> pd.DataFrame:
    """SPLIT_COLUMNS from ``stable/splits`` records: ``ex_date`` is the New York calendar date
    of the split (``date``), numerator/denominator the ratio (10:1 forward -> 10, 1). Rows
    without a parseable date are dropped; sorted oldest-first, one row per (symbol, ex_date)."""
    if not records:
        return pd.DataFrame({
            "symbol": pd.Series(dtype="str"),
            "ex_date": pd.Series(dtype="object"),
            "numerator": pd.Series(dtype="float64"),
            "denominator": pd.Series(dtype="float64"),
        })[SPLIT_COLUMNS]
    raw = pd.DataFrame.from_records(records).reindex(columns=["symbol", "date", "numerator", "denominator"])
    df = pd.DataFrame({
        "symbol": ["" if s is None else str(s).strip().upper() for s in raw["symbol"]],
        "ex_date": _to_dates(raw["date"]),
        "numerator": pd.to_numeric(raw["numerator"], errors="coerce").astype("float64").to_numpy(),
        "denominator": pd.to_numeric(raw["denominator"], errors="coerce").astype("float64").to_numpy(),
    })[SPLIT_COLUMNS]
    df = df[df["ex_date"].notna()]
    df = df.sort_values(["symbol", "ex_date"], kind="mergesort").drop_duplicates(subset=["symbol", "ex_date"], keep="last")
    return df.reset_index(drop=True)


# ---- client ---------------------------------------------------------------------------------------
class FMPClient:
    def __init__(self, settings: Settings, cache: DiskCache | None = None):
        if not settings.fmp_api_key:
            raise ProviderUnavailable("FMP_API_KEY is not set; see .env.example")
        self.settings = settings
        self.http = HttpClient(
            provider="fmp",
            cache=cache or DiskCache(settings.cache_dir),
            budget=DailyBudget("fmp", settings.fmp_daily_budget, settings.data_dir),
            timeout=settings.http_timeout_seconds,
        )
        self.base = settings.fmp_base_url
        _install_log_redaction()

    # ---- plumbing ----------------------------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any], *, cache_ttl: int) -> Any:
        """GET ``base/path``. The key goes on the wire only; cache keys use ``params`` as given.

        ``cache_ttl=IMMUTABLE_TTL_SECONDS`` is honoured for non-empty payloads only: an empty
        payload is re-read with ``settings.cache_ttl_seconds`` so it is fetched again once that
        shorter TTL has expired instead of being pinned for ten years."""
        url = f"{self.base.rstrip('/')}/{path}"
        payload = self._fetch(url, path, params, cache_ttl)
        if cache_ttl == IMMUTABLE_TTL_SECONDS and _is_empty_payload(payload):
            payload = self._fetch(url, path, params, int(self.settings.cache_ttl_seconds))
        return payload

    def _fetch(self, url: str, path: str, params: dict[str, Any], cache_ttl: int) -> Any:
        # The key is spliced into the request inline so that no local variable ever holds it:
        # a traceback that dumps frame locals (rich's show_locals, a debugger, an error
        # reporter) would otherwise print it at the raise sites below.
        try:
            return self.http.get_json(
                url, {**params, "apikey": self.settings.fmp_api_key},
                cache_ttl=cache_ttl, cache_params=params,
            )
        except httpx.HTTPStatusError as exc:
            # str(exc) embeds the full URL (with the key): keep only the status and a redacted
            # body, then leave the except block so ``exc`` is unbound before anything is raised.
            status = exc.response.status_code
            body = _redact(exc.response.text[:200].replace("\n", " "))
        if status in (401, 402, 403):
            raise ProviderUnavailable(
                f"FMP rejected {path} (HTTP {status}): {body} -- check FMP_API_KEY and "
                "whether the plan includes this endpoint"
            ) from None
        raise FMPError(f"FMP {path} failed (HTTP {status}): {body}") from None

    @staticmethod
    def _records(payload: Any, path: str) -> list[dict]:
        """Normalise a payload to a list of dicts; error payloads become FMPError."""
        if payload is None:
            return []
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        if isinstance(payload, dict):
            if not payload:
                return []
            msg = payload.get("Error Message") or payload.get("message") or payload.get("error")
            if msg:
                raise FMPError(f"FMP {path}: {_redact(str(msg))}")
            return [payload]
        raise FMPError(f"FMP {path}: unexpected payload type {type(payload).__name__}")

    def _ttl(self, last_day: date, override: int | None) -> int:
        """Cache TTL for a price range ending on ``last_day`` (New York calendar day).

        Completed sessions are immutable; a range that reaches today is still being written
        and must be refreshed on the live cadence, never for ``cache_ttl_seconds`` (a week)."""
        if override is not None:
            return int(override)
        if last_day < _today_ny():
            return IMMUTABLE_TTL_SECONDS
        return int(self.settings.live_cache_ttl_seconds)

    # ---- earnings ----------------------------------------------------------------------------
    def earnings_history(self, symbol: str, *, limit: int = 60) -> pd.DataFrame:
        """`stable/earnings`: columns symbol, report_date_ny (date), eps_actual, eps_estimate,
        rev_actual, rev_estimate, last_updated. Rows with a null eps_actual are future events.
        Sorted oldest-first; one row per (symbol, report_date_ny)."""
        symbol = _norm_symbol(symbol)
        params = {"symbol": symbol, "limit": int(limit)}
        payload = self._get("stable/earnings", params, cache_ttl=self.settings.cache_ttl_seconds)
        return _earnings_frame(self._records(payload, "stable/earnings"))

    def earnings_calendar(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """`stable/earnings-calendar`: same columns as earnings_history for all symbols.
        `start`/`end` are inclusive New York calendar days (tz-aware inputs are converted)."""
        first, last = _as_ny_date(start), _as_ny_date(end)
        if last < first:
            raise ValueError(f"end {last} is before start {first}")
        # Measured 2026-09-05: a 100-day window answered with the last month only (the
        # September and October events were missing), so the range is asked in windows of at
        # most MAX_CALENDAR_DAYS_PER_REQUEST days, one request each, anchored at `first`.
        frames = []
        for a, b in _day_chunks(first, last, MAX_CALENDAR_DAYS_PER_REQUEST):
            params = {"from": a.isoformat(), "to": b.isoformat()}
            payload = self._get("stable/earnings-calendar", params,
                                cache_ttl=self.settings.cache_ttl_seconds)
            frames.append(_earnings_frame(self._records(payload, "stable/earnings-calendar")))
        out = pd.concat(frames, ignore_index=True) if frames else _earnings_frame([])
        return out.drop_duplicates().reset_index(drop=True)

    # ---- prices ------------------------------------------------------------------------------
    def intraday(self, symbol: str, interval: str, start_day: pd.Timestamp, end_day: pd.Timestamp,
                 *, extended: bool = True, cache_ttl: int | None = None) -> pd.DataFrame:
        """`stable/historical-chart/{1min|5min}` with extended=true. Returns schemas.C columns.
        FMP timestamps are naive America/New_York; convert with timeutil.to_utc(assume_tz=NY).
        Bars are labelled by start time; t_end = t + interval. source='fmp_intraday'.
        The API accepts at most a few days per request; chunk by day range.
        `start_day`/`end_day` are inclusive New York calendar days; the `interval` column carries
        the harness label ('1m', '5m'); `interval` accepts '1min'/'5min' or '1m'/'5m'.
        `cache_ttl=None` means: immutable for chunks ending before today (NY), else
        settings.live_cache_ttl_seconds (the session is still in progress)."""
        api_interval, label, step = _resolve_interval(interval)
        symbol = _norm_symbol(symbol)
        first, last = _as_ny_date(start_day), _as_ny_date(end_day)
        if last < first:
            raise ValueError(f"end_day {last} is before start_day {first}")
        path = f"stable/historical-chart/{api_interval}"
        frames: list[pd.DataFrame] = []
        today = _today_ny()
        for a, b in _day_chunks(first, last, MAX_INTRADAY_DAYS_BY_INTERVAL.get(interval, MAX_INTRADAY_DAYS_PER_REQUEST)):
            if a > today:
                # a chunk entirely in the future has no bars: FMP would answer with the latest
                # sessions (which the day filter discards) at the cost of a budgeted request
                # that the live TTL re-spends on every run
                frames.append(_empty_candles())
                continue
            params = {
                "symbol": symbol,
                "from": a.isoformat(),
                "to": b.isoformat(),
                "extended": "true" if extended else "false",
            }
            payload = self._get(path, params, cache_ttl=self._ttl(b, cache_ttl))
            frames.append(_intraday_frame(self._records(payload, path), symbol, label, step, a, b))
        return _finish_candles(frames)

    def daily(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
              *, cache_ttl: int | None = None) -> pd.DataFrame:
        """`stable/historical-price-eod/full`: schemas.C columns at interval '1d', t = session
        date at 00:00 America/New_York converted to UTC. t_end = the next New York midnight in
        UTC (half-open calendar day). source='fmp_daily'; n_trades is not reported (NA).
        `cache_ttl` follows the same policy as `intraday`: None means immutable when `end` is
        before today (NY), else settings.live_cache_ttl_seconds."""
        symbol = _norm_symbol(symbol)
        first, last = _as_ny_date(start), _as_ny_date(end)
        if last < first:
            raise ValueError(f"end {last} is before start {first}")
        path = "stable/historical-price-eod/full"
        params = {"symbol": symbol, "from": first.isoformat(), "to": last.isoformat()}
        payload = self._get(path, params, cache_ttl=self._ttl(last, cache_ttl))
        records = self._records(payload, path)
        if not records:
            return _empty_candles()
        raw = pd.DataFrame.from_records(records)
        if "date" not in raw.columns:
            raise FMPError("FMP daily payload has no 'date' field")
        days = pd.DatetimeIndex(pd.to_datetime(raw["date"])).normalize()
        keep = (days >= pd.Timestamp(first)) & (days <= pd.Timestamp(last))
        raw, days = raw[keep], days[keep]
        t = _ny_naive_to_utc(days)
        t_end = _ny_naive_to_utc(days + pd.Timedelta(days=1))
        return _finish_candles([_candles(symbol, "1d", t, t_end, raw, DAILY_SOURCE)])

    # ---- corporate actions -------------------------------------------------------------------
    def splits(self, symbol: str) -> pd.DataFrame:
        """`stable/splits`: SPLIT_COLUMNS (symbol, ex_date as a New York ``datetime.date``,
        numerator, denominator), oldest-first. Cached for ``settings.cache_ttl_seconds`` like
        the earnings history: one request per underlying per week, not per event."""
        symbol = _norm_symbol(symbol)
        payload = self._get("stable/splits", {"symbol": symbol}, cache_ttl=self.settings.cache_ttl_seconds)
        return _splits_frame(self._records(payload, "stable/splits"))

    # ---- reference / live --------------------------------------------------------------------
    def profile(self, symbol: str) -> dict:
        """`stable/profile` first element (sector, industry, marketCap, cik, exchange).
        Empty dict when FMP knows nothing about the symbol."""
        symbol = _norm_symbol(symbol)
        payload = self._get("stable/profile", {"symbol": symbol},
                            cache_ttl=self.settings.cache_ttl_seconds)
        records = self._records(payload, "stable/profile")
        return dict(records[0]) if records else {}

    def aftermarket_trade(self, symbol: str) -> dict | None:
        """Latest after-hours trade: {price, size, t}. Uncached (live_cache_ttl).
        `t` is a tz-aware UTC Timestamp (FMP reports epoch milliseconds). None when there is no
        usable trade (empty payload, missing price or time)."""
        symbol = _norm_symbol(symbol)
        payload = self._get("stable/aftermarket-trade", {"symbol": symbol},
                            cache_ttl=self.settings.live_cache_ttl_seconds)
        records = self._records(payload, "stable/aftermarket-trade")
        if not records:
            return None
        rec = records[0]
        price = rec.get("price")
        if price is None:
            return None
        t = _epoch_to_utc(rec.get("timestamp"))
        if t is None and rec.get("date"):
            t = to_utc(str(rec["date"]), assume_tz=NY)
        if t is None:
            return None
        size = rec.get("tradeSize", rec.get("size"))
        return {
            "price": float(price),
            "size": float(size) if size is not None else float("nan"),
            "t": t,
        }
