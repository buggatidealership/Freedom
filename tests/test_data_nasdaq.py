"""Offline tests for NasdaqClient against the JSON fixtures under tests/fixtures/nasdaq/.

All HTTP goes through respx (assert_all_mocked is the default), so no network is touched.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from freedom import schemas
from freedom.data import nasdaq
from freedom.data.nasdaq import (
    BASE_URL,
    CALENDAR_COLUMNS,
    DAILY_COLUMNS,
    IMMUTABLE_TTL,
    UA,
    NasdaqClient,
    ny_date,
    parse_number,
)
from freedom.schemas import C
from freedom.timeutil import to_utc

CALENDAR_URL = f"{BASE_URL}/calendar/earnings"
HISTORICAL_URL = f"{BASE_URL}/quote/NVDA/historical"
OK_STATUS = {"rCode": 200, "bCodeMessage": None, "developerMessage": None}


def _load(fixtures_dir: Path, name: str) -> dict:
    return json.loads((fixtures_dir / "nasdaq" / name).read_text())


def _assert_utc(series: pd.Series) -> None:
    assert str(series.dtype) == "datetime64[ns, UTC]"
    assert all(ts.tzinfo is not None and ts.utcoffset().total_seconds() == 0 for ts in series)


# ---- helpers ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$3.07", 3.07),
        ("-4.62", -4.62),
        ("$0.65", 0.65),
        ("-$0.12", -0.12),
        ("$-0.12", -0.12),
        ("($0.12)", -0.12),
        ("109,756,200", 109756200.0),
        ("$5,320,798,100,000", 5320798100000.0),
        ("8.33%", 8.33),
        ("13", 13.0),
        (13, 13.0),
        (0.5, 0.5),
    ],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", " ", "N/A", "n/a", "-", "--", "None", None, "abc", "$"])
def test_parse_number_missing(raw):
    assert math.isnan(parse_number(raw))


def test_ny_date_from_naive_and_aware():
    assert ny_date(pd.Timestamp("2024-08-28")) == date(2024, 8, 28)
    assert ny_date("2024-08-28 23:30") == date(2024, 8, 28)
    # 03:00 UTC on the 28th is still the evening of the 27th in New York.
    assert ny_date(to_utc("2024-08-28 03:00", assume_tz=schemas.UTC)) == date(2024, 8, 27)
    assert ny_date(to_utc("2024-08-28 16:05", assume_tz=schemas.NY)) == date(2024, 8, 28)


# ---- earnings_calendar -----------------------------------------------------------------------
@respx.mock
def test_earnings_calendar_parses_fixture(settings, fixtures_dir):
    payload = _load(fixtures_dir, "calendar_20240828.json")
    route = respx.get(CALENDAR_URL).mock(return_value=httpx.Response(200, json=payload))

    df = NasdaqClient(settings).earnings_calendar(pd.Timestamp("2024-08-28"))

    assert route.call_count == 1
    req = route.calls.last.request
    assert req.url.params["date"] == "2024-08-28"
    assert req.headers["user-agent"] == UA
    assert req.headers["accept"] == "application/json"

    assert list(df.columns) == CALENDAR_COLUMNS
    assert len(df) == 41
    assert df["symbol"].is_unique
    assert df["symbol"].is_monotonic_increasing
    assert df.index.equals(pd.RangeIndex(len(df)))

    nvda = df.set_index("symbol").loc["NVDA"]
    assert nvda["name"] == "NVIDIA Corporation"
    assert nvda["eps_actual"] == pytest.approx(0.65)
    assert nvda["eps_estimate"] == pytest.approx(0.60)
    assert nvda["surprise_pct"] == pytest.approx(8.33)
    assert nvda["n_estimates"] == 13
    assert nvda["fiscal_quarter_ending"] == "Jul/2024"
    assert nvda["time_flag"] == "time-not-supplied"  # raw Nasdaq string, not normalised
    assert nvda["report_date_ny"] == date(2024, 8, 28)

    assert str(df["n_estimates"].dtype) == "Int64"
    for col in ("eps_actual", "eps_estimate", "surprise_pct"):
        assert df[col].dtype == "float64"
    # A blank consensus forecast is NaN, not 0 and not an error.
    blank = df[df["eps_estimate"].isna()]
    assert len(blank) >= 1
    raw_blank = {r["symbol"] for r in payload["data"]["rows"] if r["epsForecast"] == ""}
    assert set(blank["symbol"]) == raw_blank
    # Negative surprises survive as negative floats; Nasdaq writes negative EPS as '($0.14)'.
    assert (df["surprise_pct"] < 0).sum() == sum(
        1 for r in payload["data"]["rows"] if r["surprise"].startswith("-")
    )
    n_paren = sum(1 for r in payload["data"]["rows"] if r["eps"].startswith("("))
    assert n_paren == 10 and (df["eps_actual"] < 0).sum() == 10
    assert df.set_index("symbol").loc["APLD", "eps_actual"] == pytest.approx(
        parse_number(next(r["eps"] for r in payload["data"]["rows"] if r["symbol"] == "APLD"))
    )


@respx.mock
def test_earnings_calendar_uses_new_york_date_of_aware_timestamp(settings, fixtures_dir):
    payload = _load(fixtures_dir, "calendar_20240828.json")
    route = respx.get(CALENDAR_URL).mock(return_value=httpx.Response(200, json=payload))
    # 03:00 UTC on 2024-08-28 is 23:00 New York on 2024-08-27.
    NasdaqClient(settings).earnings_calendar(to_utc("2024-08-28 03:00", assume_tz=schemas.UTC))
    assert route.calls.last.request.url.params["date"] == "2024-08-27"


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"asOf": None, "headers": {}, "rows": None}, "message": None, "status": OK_STATUS},
        {"data": None, "message": None, "status": OK_STATUS},
        {"data": {"asOf": None, "headers": {}, "rows": []}, "message": None, "status": OK_STATUS},
    ],
)
def test_earnings_calendar_empty_day(settings, payload):
    respx.get(CALENDAR_URL).mock(return_value=httpx.Response(200, json=payload))
    df = NasdaqClient(settings).earnings_calendar(pd.Timestamp("2024-08-31"))
    assert list(df.columns) == CALENDAR_COLUMNS
    assert df.empty
    assert str(df["n_estimates"].dtype) == "Int64"
    assert df["eps_actual"].dtype == "float64"


@respx.mock
def test_earnings_calendar_parses_negative_and_duplicate_rows(settings):
    rows = [
        {"eps": "($0.12)", "surprise": "-171.43", "time": "after-hours", "symbol": "ZZZ",
         "name": "Z Corp", "marketCap": "", "fiscalQuarterEnding": "Jun/2024",
         "epsForecast": "-$0.05", "noOfEsts": ""},
        {"eps": "$1.00", "surprise": "0", "time": "pre-market", "symbol": "AAA",
         "name": "A Corp", "marketCap": "$1", "fiscalQuarterEnding": "Jul/2024",
         "epsForecast": "$1.00", "noOfEsts": "2"},
        {"eps": "$9.99", "surprise": "0", "time": "pre-market", "symbol": "AAA",
         "name": "A Corp dup", "marketCap": "$1", "fiscalQuarterEnding": "Jul/2024",
         "epsForecast": "$1.00", "noOfEsts": "2"},
        {"eps": "", "surprise": "", "time": None, "symbol": "", "name": "no symbol",
         "marketCap": "", "fiscalQuarterEnding": "", "epsForecast": "", "noOfEsts": ""},
    ]
    payload = {"data": {"asOf": None, "headers": {}, "rows": rows}, "message": None,
               "status": OK_STATUS}
    respx.get(CALENDAR_URL).mock(return_value=httpx.Response(200, json=payload))
    df = NasdaqClient(settings).earnings_calendar(pd.Timestamp("2024-08-28"))
    assert list(df["symbol"]) == ["AAA", "ZZZ"]  # sorted, deduplicated (first kept), blank dropped
    aaa, zzz = df.iloc[0], df.iloc[1]
    assert aaa["eps_actual"] == pytest.approx(1.00)
    assert zzz["eps_actual"] == pytest.approx(-0.12)
    assert zzz["eps_estimate"] == pytest.approx(-0.05)
    assert zzz["surprise_pct"] == pytest.approx(-171.43)
    assert pd.isna(zzz["n_estimates"])
    assert aaa["n_estimates"] == 2
    assert zzz["time_flag"] == "after-hours"


@respx.mock
def test_earnings_calendar_raises_on_nasdaq_error_status(settings):
    payload = {"data": None, "message": None,
               "status": {"rCode": 400, "bCodeMessage": [{"code": 1001,
                                                          "errorMessage": "Invalid date"}],
                          "developerMessage": None}}
    respx.get(CALENDAR_URL).mock(return_value=httpx.Response(200, json=payload))
    with pytest.raises(ValueError, match="rCode 400 Invalid date"):
        NasdaqClient(settings).earnings_calendar(pd.Timestamp("2024-08-28"))


@respx.mock
def test_earnings_calendar_is_served_from_disk_cache(settings, fixtures_dir):
    payload = _load(fixtures_dir, "calendar_20240828.json")
    route = respx.get(CALENDAR_URL).mock(return_value=httpx.Response(200, json=payload))
    client = NasdaqClient(settings)
    first = client.earnings_calendar(pd.Timestamp("2024-08-28"))
    second = NasdaqClient(settings).earnings_calendar(pd.Timestamp("2024-08-28"))
    assert route.call_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_cache_ttl_depends_on_whether_the_date_is_in_the_past(settings, monkeypatch):
    monkeypatch.setattr(nasdaq, "_today_ny", lambda: date(2026, 9, 2))
    client = NasdaqClient(settings)
    seen: list[int | None] = []

    def fake_get_json(url, params=None, *, cache_ttl, **kw):
        seen.append(cache_ttl)
        return {"data": {"rows": None}, "message": None, "status": OK_STATUS}

    monkeypatch.setattr(client.http, "get_json", fake_get_json)
    client.earnings_calendar(pd.Timestamp("2026-09-01"))
    client.earnings_calendar(pd.Timestamp("2026-09-02"))
    client.earnings_calendar(pd.Timestamp("2026-09-03"))
    assert seen == [IMMUTABLE_TTL, settings.live_cache_ttl_seconds,
                    settings.live_cache_ttl_seconds]


# ---- daily -----------------------------------------------------------------------------------
@respx.mock
def test_daily_parses_fixture(settings, fixtures_dir):
    payload = _load(fixtures_dir, "historical_NVDA_20260824_0901.json")
    route = respx.get(HISTORICAL_URL).mock(return_value=httpx.Response(200, json=payload))

    df = NasdaqClient(settings).daily("nvda", pd.Timestamp("2026-08-24"), pd.Timestamp("2026-09-01"))

    assert route.call_count == 1
    req = route.calls.last.request
    assert req.headers["user-agent"] == UA
    assert dict(req.url.params) == {"assetclass": "stocks", "fromdate": "2026-08-24",
                                    "todate": "2026-09-01", "limit": "9999"}

    assert list(df.columns) == DAILY_COLUMNS
    assert list(df.columns) == [C.market, C.interval, C.t, C.t_end, C.open, C.high, C.low,
                                C.close, C.volume, C.n_trades, C.source]
    assert len(df) == 7  # fixture is newest-first; output is oldest-first
    assert df.index.equals(pd.RangeIndex(7))
    _assert_utc(df[C.t])
    _assert_utc(df[C.t_end])
    assert df[C.t].is_monotonic_increasing and df[C.t].is_unique
    # Half-open bars: t_end == t + 1 day.
    assert (df[C.t_end] - df[C.t] == pd.Timedelta("1D")).all()
    # 2026-08-24 00:00 America/New_York (EDT) is 04:00 UTC.
    assert df[C.t].iloc[0] == to_utc("2026-08-24 00:00", assume_tz=schemas.NY)
    assert df[C.t].iloc[0] == pd.Timestamp("2026-08-24 04:00", tz="UTC")
    assert df[C.t].iloc[-1] == pd.Timestamp("2026-09-01 04:00", tz="UTC")

    last = df.iloc[-1]
    assert last[C.market] == "NVDA"
    assert last[C.interval] == "1d"
    assert last[C.source] == "nasdaq"
    assert last[C.open] == pytest.approx(216.75)
    assert last[C.high] == pytest.approx(220.41)
    assert last[C.low] == pytest.approx(215.10)
    assert last[C.close] == pytest.approx(217.44)
    assert last[C.volume] == pytest.approx(109_756_200)
    assert pd.isna(last[C.n_trades])
    assert df.iloc[1][C.open] == pytest.approx(211.025)  # '$211.025' on 08/25
    assert (df[C.market] == "NVDA").all() and (df[C.interval] == "1d").all()
    for col in (C.open, C.high, C.low, C.close, C.volume):
        assert df[col].dtype == "float64"
    assert str(df[C.n_trades].dtype) == "Int64"


@respx.mock
def test_daily_filters_to_requested_range_inclusive(settings, fixtures_dir):
    payload = _load(fixtures_dir, "historical_NVDA_20260824_0901.json")
    respx.get(HISTORICAL_URL).mock(return_value=httpx.Response(200, json=payload))
    df = NasdaqClient(settings).daily(
        "NVDA", to_utc("2026-08-26 12:00", assume_tz=schemas.UTC),
        to_utc("2026-08-31 23:59", assume_tz=schemas.NY),
    )
    assert [ts.tz_convert(schemas.NY).date() for ts in df[C.t]] == [
        date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31)]


@respx.mock
def test_daily_dedups_sorts_and_handles_dst(settings):
    rows = [
        {"date": "07/15/2026", "close": "$2.00", "volume": "1,000", "open": "$1.90",
         "high": "$2.10", "low": "$1.80"},
        {"date": "01/15/2026", "close": "$1.00", "volume": "2,000", "open": "$0.90",
         "high": "$1.10", "low": "$0.80"},
        {"date": "01/15/2026", "close": "$1.50", "volume": "2,500", "open": "$0.95",
         "high": "$1.60", "low": "$0.85"},
        {"date": "not a date", "close": "$9.00", "volume": "N/A", "open": "", "high": "",
         "low": ""},
    ]
    payload = {"data": {"symbol": "NVDA", "totalRecords": 4,
                        "tradesTable": {"asOf": None, "headers": {}, "rows": rows}},
               "message": None, "status": OK_STATUS}
    respx.get(HISTORICAL_URL).mock(return_value=httpx.Response(200, json=payload))
    df = NasdaqClient(settings).daily("NVDA", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))
    assert len(df) == 2
    # Winter midnight in New York is 05:00 UTC, summer midnight is 04:00 UTC.
    assert df[C.t].iloc[0] == pd.Timestamp("2026-01-15 05:00", tz="UTC")
    assert df[C.t].iloc[1] == pd.Timestamp("2026-07-15 04:00", tz="UTC")
    assert (df[C.t_end] == df[C.t] + pd.Timedelta("1D")).all()
    assert df[C.close].iloc[0] == pytest.approx(1.50)  # duplicate date: last row wins
    assert df[C.volume].iloc[0] == pytest.approx(2500)


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"symbol": "NVDA", "totalRecords": 0,
                  "tradesTable": {"asOf": None, "headers": {}, "rows": None}},
         "message": None, "status": OK_STATUS},
        {"data": {"symbol": "NVDA", "totalRecords": 0, "tradesTable": None},
         "message": None, "status": OK_STATUS},
        {"data": None, "message": "No data found", "status": OK_STATUS},
    ],
)
def test_daily_empty(settings, payload):
    respx.get(HISTORICAL_URL).mock(return_value=httpx.Response(200, json=payload))
    df = NasdaqClient(settings).daily("NVDA", pd.Timestamp("2026-08-29"), pd.Timestamp("2026-08-30"))
    assert list(df.columns) == DAILY_COLUMNS
    assert df.empty
    assert str(df[C.t].dtype) == "datetime64[ns, UTC]"
    assert str(df[C.t_end].dtype) == "datetime64[ns, UTC]"
    assert str(df[C.n_trades].dtype) == "Int64"


def test_daily_rejects_reversed_range(settings):
    with pytest.raises(ValueError):
        NasdaqClient(settings).daily("NVDA", pd.Timestamp("2026-09-01"), pd.Timestamp("2026-08-01"))


@respx.mock
def test_daily_raises_on_nasdaq_error_status(settings):
    payload = {"data": None, "message": None,
               "status": {"rCode": 400, "bCodeMessage": [{"code": 1001,
                                                          "errorMessage": "Symbol not found"}],
                          "developerMessage": None}}
    respx.get(HISTORICAL_URL).mock(return_value=httpx.Response(200, json=payload))
    with pytest.raises(ValueError, match="Symbol not found"):
        NasdaqClient(settings).daily("NVDA", pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01"))
