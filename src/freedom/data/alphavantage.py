"""Alpha Vantage client (key optional: ALPHAVANTAGE_API_KEY). Free tier: 25 requests/day.

Used only for the EARNINGS endpoint's reportTime flag (pre-market / post-market) as a
release-timing fallback for foreign private issuers. Everything is cached for 30 days.

Measured 2026-09-02 (tests/fixtures/alphavantage/earnings_NVDA.json): ``GET {BASE_URL}?
function=EARNINGS&symbol=NVDA&apikey=...`` -> ``{"symbol", "annualEarnings",
"quarterlyEarnings": [{"fiscalDateEnding", "reportedDate", "reportedEPS", "estimatedEPS",
"surprise", "surprisePercentage", "reportTime"}]}``, newest first. Missing numbers are the
string ``'None'``; ``reportTime`` may be absent. Quota and key problems arrive as HTTP 200 with
an ``Information`` / ``Note`` / ``Error Message`` body; those are never left in the cache.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd

from ..config import Settings
from ..data.base import (
    BudgetExhausted,
    DailyBudget,
    DiskCache,
    HttpClient,
    ProviderUnavailable,
    cache_key,
)

BASE_URL = "https://www.alphavantage.co/query"
CACHE_TTL = 30 * 24 * 3600
COLUMNS = [
    "fiscal_period_end", "report_date_ny", "eps_actual", "eps_estimate", "surprise_pct",
    "report_time", "symbol",
]
REPORT_TIMES = frozenset({"pre-market", "post-market"})

_NA_STRINGS = frozenset({"", "-", "n/a", "none", "null", "nan"})
_ERROR_KEYS = ("Error Message", "Information", "Note")


def _num(value: object) -> float:
    """Alpha Vantage numbers are strings; ``'None'`` and blanks become NaN."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.lower() in _NA_STRINGS:
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def _date(value: object) -> date | None:
    if value is None:
        return None
    t = pd.to_datetime(str(value).strip(), errors="coerce")
    return None if pd.isna(t) else t.date()


def _report_time(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    return s if s in REPORT_TIMES else None


def _empty() -> pd.DataFrame:
    # Date columns hold datetime.date objects; `dtype=str` is the version-default string dtype.
    return pd.DataFrame({
        "fiscal_period_end": pd.Series(dtype=object),
        "report_date_ny": pd.Series(dtype=object),
        "eps_actual": pd.Series(dtype="float64"),
        "eps_estimate": pd.Series(dtype="float64"),
        "surprise_pct": pd.Series(dtype="float64"),
        "report_time": pd.Series(dtype=str),
        "symbol": pd.Series(dtype=str),
    })


class AlphaVantageClient:
    def __init__(self, settings: Settings, cache: DiskCache | None = None):
        if not settings.alphavantage_api_key:
            raise ProviderUnavailable("ALPHAVANTAGE_API_KEY is not set; see .env.example")
        self.settings = settings
        self.http = HttpClient(
            provider="alphavantage",
            cache=cache or DiskCache(settings.cache_dir),
            budget=DailyBudget("alphavantage", settings.alphavantage_daily_budget, settings.data_dir),
            timeout=settings.http_timeout_seconds,
        )

    def earnings(self, symbol: str) -> pd.DataFrame:
        """quarterlyEarnings: fiscal_period_end, report_date_ny, eps_actual, eps_estimate,
        surprise_pct, report_time ('pre-market' | 'post-market' | None).

        Dates are datetime.date objects (New York calendar dates); numbers are float64 with
        NaN for 'None'; a missing or unrecognised reportTime is missing (None/NaN, test with
        pd.isna). Oldest fiscal period first, one row per fiscal period, plus a `symbol` column."""
        sym = symbol.strip().upper()
        cache_params = {"function": "EARNINGS", "symbol": sym}
        # The key is excluded from the cache identity so rotating it does not invalidate the cache.
        payload = self.http.get_json(
            BASE_URL,
            {**cache_params, "apikey": self.settings.alphavantage_api_key},
            cache_ttl=CACHE_TTL,
            cache_params=cache_params,
        )
        self._raise_if_error(payload, cache_params)
        recs = []
        for r in payload.get("quarterlyEarnings") or []:
            if not isinstance(r, dict):
                continue
            period_end = _date(r.get("fiscalDateEnding"))
            if period_end is None:
                continue
            recs.append({
                "fiscal_period_end": period_end,
                "report_date_ny": _date(r.get("reportedDate")),
                "eps_actual": _num(r.get("reportedEPS")),
                "eps_estimate": _num(r.get("estimatedEPS")),
                "surprise_pct": _num(r.get("surprisePercentage")),
                "report_time": _report_time(r.get("reportTime")),
                "symbol": sym,
            })
        if not recs:
            return _empty()
        df = pd.DataFrame.from_records(recs, columns=COLUMNS)
        df = df.sort_values("fiscal_period_end", kind="stable")
        return df.drop_duplicates("fiscal_period_end", keep="last").reset_index(drop=True)

    # ---- error handling ------------------------------------------------------------------
    def _raise_if_error(self, payload: Any, cache_params: dict) -> None:
        """Alpha Vantage reports quota, key and request errors as HTTP 200 JSON. Those must
        not be served from the 30-day cache, so the entry is evicted before raising."""
        if not isinstance(payload, dict):
            self._evict(cache_params)
            raise ProviderUnavailable(
                f"Alpha Vantage: unexpected payload of type {type(payload).__name__}"
            )
        if "quarterlyEarnings" in payload:
            return
        for key in _ERROR_KEYS:
            if key not in payload:
                continue
            self._evict(cache_params)
            msg = str(payload[key])
            low = msg.lower()
            if "limit" in low or "per day" in low or "per minute" in low:
                raise BudgetExhausted(
                    f"Alpha Vantage: {msg} Cached responses are still served; wait until "
                    "tomorrow (UTC) or use another source."
                )
            if key == "Error Message":
                raise ValueError(f"Alpha Vantage: {msg}")
            raise ProviderUnavailable(f"Alpha Vantage: {msg}")
        # An unknown symbol answers `{}`: legitimately empty, and safe to keep cached.

    def _evict(self, cache_params: dict) -> None:
        # DiskCache has no delete; this mirrors its key and path layout for the one entry.
        key = cache_key(self.http.provider, f"GET {BASE_URL}", cache_params)
        self.http.cache._path(self.http.provider, key).unlink(missing_ok=True)
