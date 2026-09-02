"""Nasdaq public API (no key; needs a browser-like User-Agent). Cross-check source only.

Measured 2026-09-02 (docs/data-sources.md, fixtures under tests/fixtures/nasdaq/):

* ``GET {BASE_URL}/calendar/earnings?date=YYYY-MM-DD`` -> ``{"data": {"asOf", "headers",
  "rows"}, "message", "status"}``. Row fields: ``symbol, name, eps ('$0.65'), epsForecast
  ('$0.60' or ''), surprise ('8.33'), noOfEsts ('13'), fiscalQuarterEnding ('Jul/2024'),
  time ('time-not-supplied' | 'pre-market' | 'after-hours' ...)``. ``data.rows`` is ``null``
  on days without events. ``time`` is kept verbatim as ``time_flag``.
* ``GET {BASE_URL}/quote/{sym}/historical?assetclass=stocks&fromdate=&todate=&limit=9999`` ->
  ``{"data": {"symbol", "totalRecords", "tradesTable": {"rows": [...]}}}``, newest first,
  rows ``date ('09/01/2026'), close ('$217.44'), volume ('109,756,200'), open, high, low``.

Both endpoints wrap failures as HTTP 200 with ``status.rCode != 200``; that is raised as a
``ValueError``. Display strings are parsed by :func:`parse_number`. Every timestamp leaving
this module is tz-aware UTC; calendar dates are New York dates.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd

from ..config import Settings
from ..data.base import DiskCache, HttpClient, TokenBucket
from ..schemas import NY, UTC, C

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
BASE_URL = "https://api.nasdaq.com/api"
HISTORICAL_LIMIT = 9999
ONE_DAY = pd.Timedelta(days=1)
IMMUTABLE_TTL = 30 * 24 * 3600  # responses for dates strictly in the past never change

CALENDAR_COLUMNS = [
    "symbol", "name", "eps_actual", "eps_estimate", "surprise_pct", "n_estimates",
    "fiscal_quarter_ending", "time_flag", "report_date_ny",
]
DAILY_COLUMNS = [
    C.market, C.interval, C.t, C.t_end, C.open, C.high, C.low, C.close, C.volume, C.n_trades,
    C.source,
]

_NA_STRINGS = frozenset({"", "-", "--", "n/a", "na", "none", "null", "nan"})


def parse_number(value: object) -> float:
    """Parse Nasdaq display strings: ``'$3.07'`` -> 3.07, ``'-4.62'`` -> -4.62,
    ``'($0.12)'`` / ``'-$0.12'`` -> -0.12, ``'109,756,200'`` -> 109756200.0, ``'8.33%'`` -> 8.33.
    ``None``, blanks and ``'N/A'`` become NaN; so does anything else that is not a number."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.lower() in _NA_STRINGS:
        return math.nan
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        v = float(s)
    except ValueError:
        return math.nan
    return -v if negative else v


def ny_date(ts: pd.Timestamp | date | str) -> date:
    """Calendar date in New York: tz-aware inputs are converted, naive ones are taken as NY."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(NY)
    return t.date()


def _today_ny() -> date:
    return pd.Timestamp.now(tz=NY).date()


def _int_or_none(value: object) -> int | None:
    n = parse_number(value)
    return None if math.isnan(n) else int(n)


def _check_status(payload: Any, what: str) -> dict:
    """Return ``payload['data']`` (``{}`` when null) or raise on a non-200 Nasdaq status code."""
    if not isinstance(payload, dict):
        raise ValueError(f"Nasdaq {what}: unexpected payload of type {type(payload).__name__}")
    status = payload.get("status") or {}
    code = status.get("rCode", 200)
    if code != 200:
        msgs = status.get("bCodeMessage") or []
        if isinstance(msgs, list):
            detail = "; ".join(str(m.get("errorMessage", m)) if isinstance(m, dict) else str(m)
                               for m in msgs)
        else:
            detail = str(msgs)
        detail = detail or str(payload.get("message") or "")
        raise ValueError(f"Nasdaq {what}: rCode {code} {detail}".rstrip())
    return payload.get("data") or {}


def _empty_calendar() -> pd.DataFrame:
    # `dtype=str` is the version-default string dtype (object on pandas 2, str on pandas 3), so
    # empty frames concat cleanly with populated ones; date columns hold datetime.date objects.
    return pd.DataFrame({
        "symbol": pd.Series(dtype=str),
        "name": pd.Series(dtype=str),
        "eps_actual": pd.Series(dtype="float64"),
        "eps_estimate": pd.Series(dtype="float64"),
        "surprise_pct": pd.Series(dtype="float64"),
        "n_estimates": pd.Series(dtype="Int64"),
        "fiscal_quarter_ending": pd.Series(dtype=str),
        "time_flag": pd.Series(dtype=str),
        "report_date_ny": pd.Series(dtype=object),
    })


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame({
        C.market: pd.Series(dtype=str),
        C.interval: pd.Series(dtype=str),
        C.t: pd.Series(dtype="datetime64[ns, UTC]"),
        C.t_end: pd.Series(dtype="datetime64[ns, UTC]"),
        C.open: pd.Series(dtype="float64"),
        C.high: pd.Series(dtype="float64"),
        C.low: pd.Series(dtype="float64"),
        C.close: pd.Series(dtype="float64"),
        C.volume: pd.Series(dtype="float64"),
        C.n_trades: pd.Series(dtype="Int64"),
        C.source: pd.Series(dtype=str),
    })


def _numeric_column(raw: pd.DataFrame, name: str) -> pd.Series:
    if name in raw.columns:
        return raw[name].map(parse_number).astype("float64")
    return pd.Series(math.nan, index=raw.index, dtype="float64")


class NasdaqClient:
    def __init__(self, settings: Settings, cache: DiskCache | None = None):
        self.settings = settings
        self.http = HttpClient(
            provider="nasdaq",
            cache=cache or DiskCache(settings.cache_dir),
            limiter=TokenBucket(settings.nasdaq_requests_per_minute),
            timeout=settings.http_timeout_seconds,
            default_headers={"User-Agent": UA, "Accept": "application/json"},
        )

    def _ttl_for(self, last_day: date) -> int:
        """Past dates are immutable; anything touching today or the future uses the live TTL."""
        if last_day < _today_ny():
            return IMMUTABLE_TTL
        return self.settings.live_cache_ttl_seconds

    def earnings_calendar(self, day: pd.Timestamp) -> pd.DataFrame:
        """Calendar for one New York date: symbol, name, eps_actual, eps_estimate,
        surprise_pct, n_estimates, fiscal_quarter_ending, time_flag."""
        d = ny_date(day)
        payload = self.http.get_json(
            f"{BASE_URL}/calendar/earnings", {"date": d.isoformat()}, cache_ttl=self._ttl_for(d)
        )
        data = _check_status(payload, f"calendar {d}")
        rows = data.get("rows") or []
        recs = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            sym = str(r.get("symbol") or "").strip()
            if not sym:
                continue
            recs.append({
                "symbol": sym,
                "name": r.get("name") or None,
                "eps_actual": parse_number(r.get("eps")),
                "eps_estimate": parse_number(r.get("epsForecast")),
                "surprise_pct": parse_number(r.get("surprise")),
                "n_estimates": _int_or_none(r.get("noOfEsts")),
                "fiscal_quarter_ending": r.get("fiscalQuarterEnding") or None,
                "time_flag": r.get("time"),
                "report_date_ny": d,
            })
        if not recs:
            return _empty_calendar()
        df = pd.DataFrame.from_records(recs, columns=CALENDAR_COLUMNS)
        df["n_estimates"] = pd.array(df["n_estimates"].tolist(), dtype="Int64")
        df = df.sort_values("symbol", kind="stable").drop_duplicates("symbol", keep="first")
        return df.reset_index(drop=True)

    def daily(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Daily bars as schemas.C columns (interval '1d').

        Sessions from the New York date of `start` through that of `end` inclusive. `t` is the
        session date at 00:00 America/New_York converted to UTC, `t_end = t + 1 day`,
        `n_trades` is not provided by Nasdaq (all NA), `source` is 'nasdaq'."""
        s, e = ny_date(start), ny_date(end)
        if e < s:
            raise ValueError(f"end {e} is before start {s}")
        sym = symbol.strip().upper()
        params = {
            "assetclass": "stocks", "fromdate": s.isoformat(), "todate": e.isoformat(),
            "limit": HISTORICAL_LIMIT,
        }
        payload = self.http.get_json(
            f"{BASE_URL}/quote/{sym}/historical", params, cache_ttl=self._ttl_for(e)
        )
        data = _check_status(payload, f"historical {sym}")
        rows = (data.get("tradesTable") or {}).get("rows") or []
        rows = [r for r in rows if isinstance(r, dict)]
        if not rows:
            return _empty_daily()
        raw = pd.DataFrame.from_records(rows)
        if "date" not in raw.columns:
            return _empty_daily()
        days = pd.to_datetime(raw["date"], format="%m/%d/%Y", errors="coerce")
        keep = days.notna() & (days >= pd.Timestamp(s)) & (days <= pd.Timestamp(e))
        raw = raw.loc[keep].reset_index(drop=True)
        days = days.loc[keep].reset_index(drop=True)
        if raw.empty:
            return _empty_daily()
        t = days.dt.tz_localize(NY).dt.tz_convert(UTC).dt.as_unit("ns")
        out = pd.DataFrame({
            C.market: sym,
            C.interval: "1d",
            C.t: t,
            C.t_end: t + ONE_DAY,
            C.open: _numeric_column(raw, "open"),
            C.high: _numeric_column(raw, "high"),
            C.low: _numeric_column(raw, "low"),
            C.close: _numeric_column(raw, "close"),
            C.volume: _numeric_column(raw, "volume"),
            C.n_trades: pd.array([pd.NA] * len(raw), dtype="Int64"),
            C.source: "nasdaq",
        })
        out = out.sort_values(C.t, kind="stable").drop_duplicates(C.t, keep="last")
        return out.reset_index(drop=True)[DAILY_COLUMNS]
