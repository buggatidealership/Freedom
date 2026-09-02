"""Offline tests for AlphaVantageClient against tests/fixtures/alphavantage/earnings_NVDA.json.

Alpha Vantage is never called live from tests (25 requests/day); respx mocks every request.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from freedom.config import Settings
from freedom.data.alphavantage import BASE_URL, CACHE_TTL, COLUMNS, AlphaVantageClient
from freedom.data.base import BudgetExhausted, HttpClient, ProviderUnavailable

RATE_LIMIT_BODY = {
    "Information": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 "
                   "requests per day. Please subscribe to any of the premium plans to instantly "
                   "remove all daily rate limits."
}


@pytest.fixture
def av_settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports",
                 alphavantage_api_key="test-key", _env_file=None)
    s.ensure_dirs()
    return s


def _fixture(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "alphavantage" / "earnings_NVDA.json").read_text())


def _mock_earnings(payload) -> respx.Route:
    return respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=payload))


def test_missing_key_raises_provider_unavailable(settings, monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    assert settings.alphavantage_api_key is None
    with pytest.raises(ProviderUnavailable, match="ALPHAVANTAGE_API_KEY"):
        AlphaVantageClient(settings)


@respx.mock
def test_earnings_parses_fixture(av_settings, fixtures_dir):
    payload = _fixture(fixtures_dir)
    route = _mock_earnings(payload)

    df = AlphaVantageClient(av_settings).earnings("nvda")

    assert route.call_count == 1
    req = route.calls.last.request
    assert dict(req.url.params) == {"function": "EARNINGS", "symbol": "NVDA", "apikey": "test-key"}

    assert list(df.columns) == COLUMNS
    assert list(df.columns[:6]) == ["fiscal_period_end", "report_date_ny", "eps_actual",
                                    "eps_estimate", "surprise_pct", "report_time"]
    assert len(df) == 110
    assert df.index.equals(pd.RangeIndex(110))
    assert (df["symbol"] == "NVDA").all()
    # Oldest first, unique fiscal periods (the API answers newest first).
    fpe = df["fiscal_period_end"]
    assert fpe.is_unique and list(fpe) == sorted(fpe)
    assert all(isinstance(d, date) for d in fpe)
    assert all(isinstance(d, date) for d in df["report_date_ny"])

    latest = df.iloc[-1]
    assert latest["fiscal_period_end"] == date(2026, 7, 31)
    assert latest["report_date_ny"] == date(2026, 8, 26)
    assert latest["eps_actual"] == pytest.approx(2.22)
    assert latest["eps_estimate"] == pytest.approx(2.09)
    assert latest["surprise_pct"] == pytest.approx(6.2201)
    assert latest["report_time"] == "post-market"

    oldest = df.iloc[0]
    assert oldest["fiscal_period_end"] == date(1999, 4, 30)
    assert oldest["report_time"] == "pre-market"

    for col in ("eps_actual", "eps_estimate", "surprise_pct"):
        assert df[col].dtype == "float64"
    assert set(df["report_time"].dropna()) == {"pre-market", "post-market"}
    assert df["report_time"].value_counts().to_dict() == {"pre-market": 62, "post-market": 48}

    # 'None' strings become NaN (2006-07-31 has estimatedEPS 'None' and surprisePercentage 'None').
    row = df.set_index("fiscal_period_end").loc[date(2006, 7, 31)]
    assert row["eps_actual"] == pytest.approx(0.0038)
    assert pd.isna(row["eps_estimate"])
    assert pd.isna(row["surprise_pct"])


@respx.mock
def test_earnings_handles_missing_report_time_and_none_strings(av_settings):
    payload = {
        "symbol": "FOO",
        "annualEarnings": [],
        "quarterlyEarnings": [
            {"fiscalDateEnding": "2026-06-30", "reportedDate": "2026-08-05",
             "reportedEPS": "None", "estimatedEPS": "1.10", "surprise": "None",
             "surprisePercentage": "None"},
            {"fiscalDateEnding": "2026-03-31", "reportedDate": "2026-05-06",
             "reportedEPS": "0.9", "estimatedEPS": "0.8", "surprise": "0.1",
             "surprisePercentage": "12.5", "reportTime": "POST-MARKET"},
            {"fiscalDateEnding": "2025-12-31", "reportedDate": "2026-02-04",
             "reportedEPS": "0.5", "estimatedEPS": "None", "surprise": "None",
             "surprisePercentage": "None", "reportTime": "unknown"},
            {"fiscalDateEnding": "2025-12-31", "reportedDate": "2026-02-04",
             "reportedEPS": "0.55", "estimatedEPS": "0.5", "surprise": "0.05",
             "surprisePercentage": "10", "reportTime": "pre-market"},
            {"fiscalDateEnding": "None", "reportedDate": "None", "reportedEPS": "None",
             "estimatedEPS": "None", "surprise": "None", "surprisePercentage": "None"},
        ],
    }
    _mock_earnings(payload)
    df = AlphaVantageClient(av_settings).earnings("FOO")
    assert list(df.columns) == COLUMNS
    assert list(df["fiscal_period_end"]) == [date(2025, 12, 31), date(2026, 3, 31),
                                             date(2026, 6, 30)]
    by = df.set_index("fiscal_period_end")
    assert pd.isna(by.loc[date(2026, 6, 30), "report_time"])  # reportTime absent -> missing
    assert pd.isna(by.loc[date(2026, 6, 30), "eps_actual"])
    assert by.loc[date(2026, 6, 30), "eps_estimate"] == pytest.approx(1.10)
    assert by.loc[date(2026, 3, 31), "report_time"] == "post-market"  # normalised case
    # Duplicate fiscal period: the later row wins; unknown flags become None.
    assert by.loc[date(2025, 12, 31), "eps_actual"] == pytest.approx(0.55)
    assert by.loc[date(2025, 12, 31), "report_time"] == "pre-market"


@respx.mock
@pytest.mark.parametrize("payload", [{}, {"symbol": "XXXX", "quarterlyEarnings": []},
                                     {"symbol": "XXXX", "quarterlyEarnings": None}])
def test_earnings_empty_for_unknown_symbol(av_settings, payload):
    _mock_earnings(payload)
    df = AlphaVantageClient(av_settings).earnings("XXXX")
    assert list(df.columns) == COLUMNS
    assert df.empty
    assert df["eps_actual"].dtype == "float64"


@respx.mock
def test_earnings_is_cached_for_30_days_and_consumes_budget_once(av_settings, fixtures_dir,
                                                                 monkeypatch):
    route = _mock_earnings(_fixture(fixtures_dir))
    seen_ttl: list[int | None] = []
    original = HttpClient.get_json

    def spy(self, url, params=None, *, cache_ttl, **kw):
        seen_ttl.append(cache_ttl)
        return original(self, url, params, cache_ttl=cache_ttl, **kw)

    monkeypatch.setattr(HttpClient, "get_json", spy)
    client = AlphaVantageClient(av_settings)
    first = client.earnings("NVDA")
    second = AlphaVantageClient(av_settings).earnings("NVDA")  # fresh client, same disk cache
    assert route.call_count == 1
    assert seen_ttl == [CACHE_TTL, CACHE_TTL]
    assert CACHE_TTL == 30 * 24 * 3600
    assert client.http.budget is not None and client.http.budget.used_today() == 1
    pd.testing.assert_frame_equal(first, second)


@respx.mock
def test_cache_identity_ignores_the_api_key(av_settings, fixtures_dir):
    route = _mock_earnings(_fixture(fixtures_dir))
    AlphaVantageClient(av_settings).earnings("NVDA")
    rotated = Settings(data_dir=av_settings.data_dir, alphavantage_api_key="other-key",
                       _env_file=None)
    AlphaVantageClient(rotated).earnings("NVDA")
    assert route.call_count == 1


@respx.mock
def test_budget_exhausted_before_any_request(tmp_path):
    s = Settings(data_dir=tmp_path / "data", alphavantage_api_key="test-key",
                 alphavantage_daily_budget=0, _env_file=None)
    s.ensure_dirs()
    route = _mock_earnings({"symbol": "NVDA", "quarterlyEarnings": []})
    with pytest.raises(BudgetExhausted, match="alphavantage"):
        AlphaVantageClient(s).earnings("NVDA")
    assert route.call_count == 0


@respx.mock
def test_rate_limit_body_raises_and_is_not_cached(av_settings, fixtures_dir):
    route = respx.get(BASE_URL).mock(side_effect=[
        httpx.Response(200, json=RATE_LIMIT_BODY),
        httpx.Response(200, json=_fixture(fixtures_dir)),
    ])
    client = AlphaVantageClient(av_settings)
    with pytest.raises(BudgetExhausted, match="25 requests per day"):
        client.earnings("NVDA")
    # The error body must not poison the 30-day cache: the next call goes back to the network.
    df = client.earnings("NVDA")
    assert route.call_count == 2
    assert len(df) == 110


@respx.mock
def test_error_message_body_raises_value_error(av_settings):
    _mock_earnings({"Error Message": "Invalid API call. Please retry or visit the documentation."})
    with pytest.raises(ValueError, match="Invalid API call"):
        AlphaVantageClient(av_settings).earnings("NVDA")


@respx.mock
def test_note_body_raises_provider_unavailable(av_settings):
    _mock_earnings({"Note": "Something went wrong upstream."})
    with pytest.raises(ProviderUnavailable, match="Something went wrong"):
        AlphaVantageClient(av_settings).earnings("NVDA")
