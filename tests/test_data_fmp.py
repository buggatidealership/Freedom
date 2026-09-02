"""Offline tests for FMPClient: fixtures under tests/fixtures/fmp, no network.

Two mocking layers are used on purpose: monkeypatching HttpClient.get_json (fast, checks the
request the client builds) and respx (goes through the real HttpClient so caching, the daily
budget and key hygiene are exercised end to end).
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from freedom.config import Settings
from freedom.data import fmp as fmp_mod
from freedom.data.base import BudgetExhausted, HttpClient, ProviderUnavailable, cache_key
from freedom.data.fmp import (
    CANDLE_COLUMNS,
    EARNINGS_COLUMNS,
    IMMUTABLE_TTL_SECONDS,
    MAX_INTRADAY_DAYS_PER_REQUEST,
    FMPClient,
    FMPError,
)
from freedom.schemas import C, E, PriceSource, U

FMP_FIXTURES = Path(__file__).parent / "fixtures" / "fmp"
BASE = "https://financialmodelingprep.com"
FIXTURE_BY_PATH = {
    "stable/earnings": "earnings_NVDA.json",
    "stable/earnings-calendar": "earnings-calendar_20260901_10.json",
    "stable/historical-chart/5min": "historical-chart_5min_NVDA_20260826_27_extended.json",
    "stable/historical-chart/1min": "historical-chart_1min_NVDA_20260826_extended.json",
    "stable/historical-price-eod/full": "historical-price-eod_NVDA_20260601_0901.json",
    "stable/profile": "profile_NVDA.json",
    "stable/aftermarket-trade": "aftermarket-trade_NVDA.json",
}


def load(name: str):
    return json.loads((FMP_FIXTURES / name).read_text())


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


class FakeHttp:
    """Replaces HttpClient.get_json: records every call, serves fixtures (or overrides)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses: dict[str, object] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeHttp:
        fake = self

        def get_json(http_self, url, params=None, *, cache_ttl, weight=1.0, headers=None,
                     cache_params=None):
            fake.calls.append({"url": url, "params": dict(params or {}), "cache_ttl": cache_ttl,
                               "cache_params": cache_params})
            path = url.removeprefix(BASE + "/")
            if path in fake.responses:
                r = fake.responses[path]
                return r(params) if callable(r) else r
            return load(FIXTURE_BY_PATH[path])

        def post_json(http_self, *args, **kwargs):
            raise AssertionError("FMPClient must never POST")

        monkeypatch.setattr(HttpClient, "get_json", get_json)
        monkeypatch.setattr(HttpClient, "post_json", post_json)
        return self


@pytest.fixture
def fake_http(monkeypatch) -> FakeHttp:
    return FakeHttp().install(monkeypatch)


@pytest.fixture
def client(settings, fake_http) -> FMPClient:
    return FMPClient(settings)


def assert_candle_contract(df: pd.DataFrame, step: pd.Timedelta, market: str, interval: str,
                           source: str) -> None:
    assert list(df.columns) == CANDLE_COLUMNS
    assert str(df[C.t].dt.tz) == "UTC" and str(df[C.t_end].dt.tz) == "UTC"
    assert df[C.t].is_monotonic_increasing and df[C.t].is_unique
    assert (df[C.t_end] == df[C.t] + step).all()
    assert (df[C.market] == market).all() and (df[C.interval] == interval).all()
    assert (df[C.source] == source).all()
    for col in (C.open, C.high, C.low, C.close, C.volume):
        assert df[col].dtype == "float64", col
    assert df[C.n_trades].isna().all()
    assert list(df.index) == list(range(len(df)))


# ---- construction ---------------------------------------------------------------------------------
def test_requires_api_key(tmp_path):
    s = Settings(data_dir=tmp_path, fmp_api_key=None, _env_file=None)
    with pytest.raises(ProviderUnavailable, match="FMP_API_KEY"):
        FMPClient(s)


# ---- earnings ------------------------------------------------------------------------------------
def test_earnings_history_columns_values_and_request(client, fake_http):
    df = client.earnings_history("nvda")
    assert list(df.columns) == EARNINGS_COLUMNS
    assert len(df) == 40
    dates = df[E.report_date_ny].tolist()
    assert all(isinstance(d, date) for d in dates)
    assert dates == sorted(dates), "oldest first"
    assert df[U.symbol].eq("NVDA").all()
    for col in (E.eps_actual, E.eps_estimate, E.rev_actual, E.rev_estimate):
        assert df[col].dtype == "float64"
    row = df[df[E.report_date_ny] == date(2026, 8, 26)].iloc[0]
    assert row[E.eps_actual] == 2.22 and row[E.eps_estimate] == 2.09
    assert row[E.rev_actual] == 96221000000.0 and row[E.rev_estimate] == 92270940000.0
    assert row["last_updated"] == date(2026, 9, 2)
    future = df[df[E.report_date_ny] == date(2026, 11, 18)].iloc[0]
    assert pd.isna(future[E.eps_actual]) and future[E.eps_estimate] == 2.47

    (call,) = fake_http.calls
    assert call["url"] == f"{BASE}/stable/earnings"
    assert call["params"] == {"symbol": "NVDA", "limit": 60, "apikey": "test"}
    assert call["cache_params"] == {"symbol": "NVDA", "limit": 60}
    assert call["cache_ttl"] == client.settings.cache_ttl_seconds


def test_earnings_calendar(client, fake_http):
    df = client.earnings_calendar(utc("2026-09-01 12:00"), pd.Timestamp("2026-09-10"))
    assert list(df.columns) == EARNINGS_COLUMNS
    assert len(df) == 868
    assert df[E.report_date_ny].map(lambda d: date(2026, 9, 1) <= d <= date(2026, 9, 10)).all()
    keys = list(zip(df[U.symbol], df[E.report_date_ny], strict=True))
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    (call,) = fake_http.calls
    assert call["url"] == f"{BASE}/stable/earnings-calendar"
    assert call["cache_params"] == {"from": "2026-09-01", "to": "2026-09-10"}
    assert "apikey" not in call["cache_params"] and call["params"]["apikey"] == "test"


def test_earnings_calendar_uses_new_york_date_for_tz_aware_bounds(client, fake_http):
    # 2026-09-02 02:00 UTC is still 2026-09-01 in New York.
    client.earnings_calendar(utc("2026-09-02 02:00"), utc("2026-09-11 02:00"))
    assert fake_http.calls[0]["cache_params"] == {"from": "2026-09-01", "to": "2026-09-10"}
    with pytest.raises(ValueError):
        client.earnings_calendar(pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-01"))


def test_earnings_dedup_keeps_latest_update_and_empty_payload(client, fake_http):
    fake_http.responses["stable/earnings"] = [
        {"symbol": "nvda", "date": "2026-08-26", "epsActual": None, "epsEstimated": 2.0,
         "revenueActual": None, "revenueEstimated": 9e10, "lastUpdated": "2026-08-01"},
        {"symbol": "NVDA", "date": "2026-08-26", "epsActual": 2.22, "epsEstimated": 2.09,
         "revenueActual": 96221000000, "revenueEstimated": 92270940000,
         "lastUpdated": "2026-09-02"},
        {"symbol": "NVDA", "date": None, "epsActual": 1.0, "epsEstimated": 1.0,
         "revenueActual": 1, "revenueEstimated": 1, "lastUpdated": "2026-09-02"},
    ]
    df = client.earnings_history("NVDA", limit=5)
    assert len(df) == 1
    assert df.iloc[0][E.eps_actual] == 2.22 and df.iloc[0]["last_updated"] == date(2026, 9, 2)
    assert fake_http.calls[-1]["cache_params"] == {"symbol": "NVDA", "limit": 5}

    fake_http.responses["stable/earnings"] = []
    empty = client.earnings_history("ZZZZ")
    assert list(empty.columns) == EARNINGS_COLUMNS and empty.empty


def test_error_payload_raises(client, fake_http):
    fake_http.responses["stable/earnings"] = {"Error Message": "Invalid query apikey=test"}
    with pytest.raises(FMPError) as info:
        client.earnings_history("NVDA")
    assert "apikey=test" not in str(info.value) and "apikey=***" in str(info.value)


# ---- intraday ------------------------------------------------------------------------------------
def test_intraday_5min_contract(client, fake_http):
    df = client.intraday("NVDA", "5min", pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-27"))
    assert_candle_contract(df, pd.Timedelta(minutes=5), "NVDA", "5m", PriceSource.fmp_intraday)
    assert len(df) == 384
    assert df[C.t].dtype == "datetime64[ns, UTC]"
    assert df[C.t].iloc[0] == utc("2026-08-26 08:00")  # 04:00 New York, EDT
    assert df[C.t].iloc[-1] == utc("2026-08-27 23:55")  # 19:55 New York
    assert df[C.t_end].iloc[-1] == utc("2026-08-28 00:00")
    first = df.iloc[0]
    assert (first[C.open], first[C.high], first[C.low], first[C.close], first[C.volume]) == (
        213.72, 213.78, 213.4, 213.73, 7226.0)
    last = df.iloc[-1]
    assert last[C.close] == 226.15 and last[C.volume] == 28869.0

    (call,) = fake_http.calls
    assert call["url"] == f"{BASE}/stable/historical-chart/5min"
    assert call["cache_params"] == {"symbol": "NVDA", "from": "2026-08-26", "to": "2026-08-27",
                                    "extended": "true"}
    assert call["params"] == {**call["cache_params"], "apikey": "test"}


def test_intraday_1min_and_interval_alias(client, fake_http):
    df = client.intraday("NVDA", "1m", pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-26"))
    assert_candle_contract(df, pd.Timedelta(minutes=1), "NVDA", "1m", PriceSource.fmp_intraday)
    assert len(df) == 960
    assert df[C.t].iloc[0] == utc("2026-08-26 08:00") and df[C.t].iloc[-1] == utc("2026-08-26 23:59")
    assert fake_http.calls[0]["url"].endswith("/stable/historical-chart/1min")
    assert fake_http.calls[0]["cache_params"]["from"] == "2026-08-26"
    assert fake_http.calls[0]["cache_params"]["to"] == "2026-08-26"


def test_intraday_chunks_are_deterministic_and_deduplicated(client, fake_http):
    # 13 calendar days -> three chunks of at most MAX_INTRADAY_DAYS_PER_REQUEST days.
    assert MAX_INTRADAY_DAYS_PER_REQUEST == 5
    expected = [("2026-08-20", "2026-08-24"), ("2026-08-25", "2026-08-29"),
                ("2026-08-30", "2026-09-01")]
    df = client.intraday("NVDA", "5min", pd.Timestamp("2026-08-20"), pd.Timestamp("2026-09-01"))
    windows = [(c["cache_params"]["from"], c["cache_params"]["to"]) for c in fake_http.calls]
    assert windows == expected
    # The fake served the same two days for every chunk: duplicates must collapse.
    assert len(df) == 384 and df[C.t].is_unique and df[C.t].is_monotonic_increasing

    fake_http.calls.clear()
    client.intraday("NVDA", "5min", pd.Timestamp("2026-08-20"), pd.Timestamp("2026-09-01"))
    assert [(c["cache_params"]["from"], c["cache_params"]["to"]) for c in fake_http.calls] == expected
    assert all("apikey" not in c["cache_params"] for c in fake_http.calls)


def test_intraday_drops_bars_outside_requested_days(client, fake_http):
    df = client.intraday("NVDA", "5min", pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-26"))
    assert len(df) == 192
    assert df[C.t].iloc[-1] == utc("2026-08-26 23:55")


def test_intraday_localises_as_new_york_across_dst(client, fake_http):
    bar = {"open": 1.0, "low": 1.0, "high": 1.0, "close": 1.0, "volume": 1}
    fake_http.responses["stable/historical-chart/1min"] = [
        {"date": "2026-03-09 04:00:00", **bar},  # EDT (UTC-4) after the spring-forward
        {"date": "2026-03-06 04:00:00", **bar},  # EST (UTC-5)
    ]
    df = client.intraday("NVDA", "1min", pd.Timestamp("2026-03-06"), pd.Timestamp("2026-03-09"))
    assert df[C.t].tolist() == [utc("2026-03-06 09:00"), utc("2026-03-09 08:00")]
    assert df[C.t_end].tolist() == [utc("2026-03-06 09:01"), utc("2026-03-09 08:01")]


def test_intraday_empty_payload_and_argument_errors(client, fake_http):
    fake_http.responses["stable/historical-chart/5min"] = []
    df = client.intraday("NVDA", "5min", pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-27"))
    assert list(df.columns) == CANDLE_COLUMNS and df.empty
    assert df[C.t].dtype == "datetime64[ns, UTC]" and df[C.volume].dtype == "float64"
    with pytest.raises(ValueError):
        client.intraday("NVDA", "2min", pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-27"))
    with pytest.raises(ValueError):
        client.intraday("NVDA", "5min", pd.Timestamp("2026-08-27"), pd.Timestamp("2026-08-26"))
    assert fake_http.calls[-1]["cache_params"]["symbol"] == "NVDA"


def test_intraday_extended_flag_and_cache_ttl_policy(client, fake_http, monkeypatch):
    monkeypatch.setattr(fmp_mod, "_today_ny", lambda: date(2026, 9, 2))
    client.intraday("NVDA", "5min", pd.Timestamp("2026-08-30"), pd.Timestamp("2026-09-02"),
                    extended=False)
    (call,) = fake_http.calls
    assert call["cache_params"]["extended"] == "false"
    assert call["cache_ttl"] == client.settings.cache_ttl_seconds  # chunk touches today

    fake_http.calls.clear()
    client.intraday("NVDA", "5min", pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-27"))
    assert fake_http.calls[0]["cache_ttl"] == IMMUTABLE_TTL_SECONDS  # completed sessions

    fake_http.calls.clear()
    client.intraday("NVDA", "5min", pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-27"),
                    cache_ttl=30)
    assert fake_http.calls[0]["cache_ttl"] == 30


# ---- daily ---------------------------------------------------------------------------------------
def test_daily_contract(client, fake_http, monkeypatch):
    monkeypatch.setattr(fmp_mod, "_today_ny", lambda: date(2026, 9, 2))
    df = client.daily("NVDA", pd.Timestamp("2026-06-01"), pd.Timestamp("2026-09-01"))
    assert_candle_contract(df, pd.Timedelta(days=1), "NVDA", "1d", fmp_mod.DAILY_SOURCE)
    assert len(df) == 65
    assert df[C.t].iloc[0] == utc("2026-06-01 04:00")  # midnight New York (EDT)
    assert df[C.t].iloc[-1] == utc("2026-09-01 04:00")
    assert df[C.t_end].iloc[-1] == utc("2026-09-02 04:00")
    last = df.iloc[-1]
    assert (last[C.open], last[C.close], last[C.volume]) == (216.75, 217.44, 109756184.0)
    (call,) = fake_http.calls
    assert call["url"] == f"{BASE}/stable/historical-price-eod/full"
    assert call["cache_params"] == {"symbol": "NVDA", "from": "2026-06-01", "to": "2026-09-01"}
    assert call["cache_ttl"] == IMMUTABLE_TTL_SECONDS


def test_daily_filters_to_requested_range(client, fake_http):
    df = client.daily("NVDA", pd.Timestamp("2026-08-24"), pd.Timestamp("2026-08-28"))
    assert df[C.t].tolist() == [utc(f"2026-08-{d} 04:00") for d in (24, 25, 26, 27, 28)]


# ---- profile / live ----------------------------------------------------------------------------
def test_profile(client, fake_http):
    p = client.profile("NVDA")
    assert p["sector"] == "Technology" and p["industry"] == "Semiconductors"
    assert p["cik"] == "0001045810" and p["exchange"] == "NASDAQ" and p["marketCap"] > 0
    assert fake_http.calls[0]["cache_params"] == {"symbol": "NVDA"}
    fake_http.responses["stable/profile"] = []
    assert client.profile("ZZZZ") == {}


def test_aftermarket_trade(client, fake_http):
    trade = client.aftermarket_trade("NVDA")
    assert trade == {"price": 216.60001, "size": 1.0,
                     "t": pd.Timestamp(1788343608000, unit="ms", tz="UTC")}
    assert trade["t"].tzinfo is not None and str(trade["t"].tz) == "UTC"
    assert isinstance(trade["price"], float) and isinstance(trade["size"], float)
    (call,) = fake_http.calls
    assert call["url"] == f"{BASE}/stable/aftermarket-trade"
    assert call["cache_ttl"] == client.settings.live_cache_ttl_seconds
    assert call["cache_params"] == {"symbol": "NVDA"}

    fake_http.responses["stable/aftermarket-trade"] = []
    assert client.aftermarket_trade("NVDA") is None
    fake_http.responses["stable/aftermarket-trade"] = [{"symbol": "NVDA", "price": 1.0}]
    assert client.aftermarket_trade("NVDA") is None  # no time -> not admissible


# ---- through the real HttpClient (respx) --------------------------------------------------------
@respx.mock
def test_cache_key_excludes_api_key_and_reruns_are_free(settings):
    payload = load("historical-chart_5min_NVDA_20260826_27_extended.json")
    route = respx.get(f"{BASE}/stable/historical-chart/5min").mock(
        return_value=httpx.Response(200, json=payload))
    client = FMPClient(settings)
    args = ("NVDA", "5min", pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-27"))
    df1 = client.intraday(*args)
    df2 = client.intraday(*args)
    assert route.call_count == 1, "second identical call must be served from the disk cache"
    pd.testing.assert_frame_equal(df1, df2)
    assert len(df1) == 384 and df1[C.t].iloc[0] == utc("2026-08-26 08:00")

    sent = route.calls[0].request.url
    assert sent.params["apikey"] == "test" and sent.params["extended"] == "true"
    assert sent.params["from"] == "2026-08-26" and sent.params["to"] == "2026-08-27"

    params_without_key = {"symbol": "NVDA", "from": "2026-08-26", "to": "2026-08-27",
                          "extended": "true"}
    key = cache_key("fmp", f"GET {BASE}/stable/historical-chart/5min", params_without_key)
    path = settings.cache_dir / "fmp" / key[:2] / f"{key}.json.gz"
    assert path.exists(), "cache entry must be addressable without the API key"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert "apikey" not in f.read()
    assert client.http.budget.used_today() == 1


@respx.mock
def test_daily_budget_is_enforced_but_cache_still_served(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", fmp_api_key="test", fmp_daily_budget=1,
                        _env_file=None)
    settings.ensure_dirs()
    respx.get(f"{BASE}/stable/profile").mock(
        return_value=httpx.Response(200, json=load("profile_NVDA.json")))
    client = FMPClient(settings)
    assert client.profile("NVDA")["symbol"] == "NVDA"
    assert client.profile("NVDA")["symbol"] == "NVDA"  # cached, no budget consumed
    with pytest.raises(BudgetExhausted):
        client.profile("AAPL")


@respx.mock
def test_http_errors_never_expose_the_key(settings):
    respx.get(f"{BASE}/stable/profile").mock(
        return_value=httpx.Response(401, json={"Error Message": "Invalid API KEY"}))
    respx.get(f"{BASE}/stable/earnings").mock(return_value=httpx.Response(404, text="nope"))
    client = FMPClient(settings)
    with pytest.raises(ProviderUnavailable) as info:
        client.profile("NVDA")
    assert "401" in str(info.value) and "test" not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__
    with pytest.raises(FMPError) as info2:
        client.earnings_history("NVDA")
    assert "404" in str(info2.value) and "apikey" not in str(info2.value)


def test_httpx_log_lines_are_redacted(settings, fake_http, caplog):
    FMPClient(settings)  # installs the filter
    with caplog.at_level(logging.INFO, logger="httpx"):
        logging.getLogger("httpx").info(
            'HTTP Request: %s %s "%s"', "GET",
            httpx.URL(f"{BASE}/stable/profile?symbol=NVDA&apikey=SECRET"), "HTTP/1.1 200 OK")
    assert "SECRET" not in caplog.text and "apikey=***" in caplog.text
    assert fmp_mod._redact("a?apikey=SECRET&x=1 apikey=Z") == "a?apikey=***&x=1 apikey=***"


# ---- helpers -------------------------------------------------------------------------------------
def test_day_chunks():
    chunks = fmp_mod._day_chunks(date(2026, 8, 1), date(2026, 8, 13), 5)
    assert chunks == [(date(2026, 8, 1), date(2026, 8, 5)), (date(2026, 8, 6), date(2026, 8, 10)),
                      (date(2026, 8, 11), date(2026, 8, 13))]
    assert fmp_mod._day_chunks(date(2026, 8, 1), date(2026, 8, 1), 5) == [
        (date(2026, 8, 1), date(2026, 8, 1))]


def test_epoch_to_utc_accepts_seconds_and_milliseconds():
    assert fmp_mod._epoch_to_utc(1788343608000) == utc("2026-09-02 10:06:48")
    assert fmp_mod._epoch_to_utc(1788343608) == utc("2026-09-02 10:06:48")
    assert fmp_mod._epoch_to_utc(None) is None
