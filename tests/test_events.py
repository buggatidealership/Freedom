"""Offline tests for the event table and release-time resolver (tests/fixtures, no network).

`FakeHttp` replaces `HttpClient.get_json` and serves the SEC, FMP, Nasdaq and Alpha Vantage
fixtures by URL, so the real clients' parsing runs end to end; `HttpClient.post_json`
(Hyperliquid) is never needed here except in the archiver-hook test, which uses
`tests.fakes.FakeHyperliquidInfo` for it.
"""

from __future__ import annotations

import json
import re
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freedom import events as ev_mod
from freedom.config import Settings
from freedom.data import fmp as fmp_mod
from freedom.data import sec as sec_mod
from freedom.data.alphavantage import BASE_URL as AV_URL
from freedom.data.archive import archive_markets
from freedom.data.base import BudgetExhausted, HttpClient
from freedom.data.nasdaq import BASE_URL as NASDAQ_URL
from freedom.data.sec import COMPANYFACTS_URL, SUBMISSIONS_URL, TICKERS_URL
from freedom.events import (
    CONFIDENCE,
    CONFIDENCE_UNKNOWN,
    EVENT_COLUMNS,
    SNAPSHOT_COLUMNS,
    UPCOMING_COLUMNS,
    ReleaseClock,
    ResolvedT0,
    build_events,
    consensus_path,
    corporate_action_near,
    detect_release_from_bars,
    detect_release_live,
    expected_release_clock,
    expected_t0_for,
    fiscal_period_for,
    load_events,
    project_quarter_end,
    release_clock_for,
    resolve_release_time,
    snapshot_consensus,
    upcoming_events,
)
from freedom.schemas import SCHEMA_VERSION, C, E, U
from tests.fakes import NVDA as XYZ_NVDA
from tests.fakes import FakeHyperliquidInfo

FIX = Path(__file__).parent / "fixtures"
FMP_BASE = "https://financialmodelingprep.com"
NVDA_CIK, AAPL_CIK, TSM_CIK = 1045810, 320193, 1046179
T_NVDA_ACC = pd.Timestamp("2026-08-26 20:21:19", tz="UTC")
T_NVDA_DET = pd.Timestamp("2026-08-26 20:20:00", tz="UTC")
TODAY = date(2026, 9, 2)
REPORT_DAY = pd.Timestamp("2026-08-26")

AAPL_8K = ["2026-07-30T20:30:28.000Z", "2026-04-30T20:30:41.000Z", "2026-01-29T21:30:33.000Z"]
AAPL_EARNINGS = [
    {"symbol": "AAPL", "date": "2026-07-30", "epsActual": 1.57, "epsEstimated": 1.43,
     "revenueActual": 94036000000, "revenueEstimated": 89300000000, "lastUpdated": "2026-09-01"},
    {"symbol": "AAPL", "date": "2026-04-30", "epsActual": 1.65, "epsEstimated": 1.62,
     "revenueActual": 95359000000, "revenueEstimated": 94500000000, "lastUpdated": "2026-09-01"},
    {"symbol": "AAPL", "date": "2026-01-29", "epsActual": 2.40, "epsEstimated": 2.35,
     "revenueActual": 124300000000, "revenueEstimated": 124000000000, "lastUpdated": "2026-09-01"},
]


def load(provider: str, name: str):
    return json.loads((FIX / provider / name).read_text())


def nasdaq_row(symbol: str, eps: str = "$2.20", forecast: str = "$2.05", n: str = "32",
               time_flag: str = "time-not-supplied") -> dict:
    return {"symbol": symbol, "name": symbol, "eps": eps, "epsForecast": forecast,
            "surprise": "7.3", "noOfEsts": n, "fiscalQuarterEnding": "Jul/2026", "time": time_flag}


def submissions_page(rows: list[tuple[str, str, str]]) -> dict:
    """Columnar EDGAR page from (form, acceptanceDateTime, items) triples."""
    return {
        "accessionNumber": [f"0000320193-26-{i:06d}" for i in range(len(rows))],
        "filingDate": [r[1][:10] for r in rows],
        "acceptanceDateTime": [r[1] for r in rows],
        "form": [r[0] for r in rows],
        "items": [r[2] for r in rows],
        "primaryDocument": ["doc.htm"] * len(rows),
        "primaryDocDescription": ["8-K"] * len(rows),
    }


class FakeHttp:
    """HttpClient.get_json stand-in dispatching on URL. Attributes are knobs for the tests."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.earnings: dict[str, list[dict]] = {"NVDA": load("fmp", "earnings_NVDA.json"),
                                                "AAPL": AAPL_EARNINGS}
        self.calendar = load("fmp", "earnings-calendar_20260901_10.json")
        self.intraday: dict[tuple[str, str], list[dict]] = {
            ("NVDA", "2026-08-25"): load("fmp", "historical-chart_1min_NVDA_20260826_extended.json"),
        }
        self.intraday_budget: int | None = None  # BudgetExhausted after this many 1min requests
        self.n_intraday = 0
        self.splits: dict[str, list[dict]] = {"NFLX": load("fmp", "splits_NFLX.json")}  # symbol -> stable/splits rows
        self.splits_errors: set[str] = set()  # symbols whose splits request answers an error payload
        self.intraday_errors: set[str] = set()  # symbols whose 1min request answers an error payload
        self.earnings_budget_exhausted = False  # every earnings-history request hits the budget
        self.tickers_extra: dict[str, int] = {}  # ticker -> CIK appended to the SEC ticker map
        self.nasdaq: dict[str, list[dict]] = {}  # iso date -> rows
        self.submissions: dict[int, dict] = {
            NVDA_CIK: load("sec", "submissions_CIK0001045810.json"),
            TSM_CIK: load("sec", "submissions_CIK0001046179.json"),
            AAPL_CIK: {"filings": {"recent": submissions_page(
                [("8-K", t, "2.02,9.01") for t in AAPL_8K] + [("8-K", "2026-03-02T21:05:00.000Z", "5.02")]
            ), "files": []}},
        }
        self.facts: dict[int, dict] = {NVDA_CIK: load("sec", "companyfacts_CIK0001045810_trimmed.json")}
        self.av_symbols: set[str] = set()  # symbols served the NVDA Alpha Vantage fixture

    def install(self, monkeypatch) -> FakeHttp:
        fake = self

        def get_json(http_self, url, params=None, *, cache_ttl=None, weight=1.0, headers=None,
                     cache_params=None):
            return fake.get_json(http_self.provider, url, dict(params or {}))

        monkeypatch.setattr(HttpClient, "get_json", get_json)
        return self

    def get_json(self, provider: str, url: str, params: dict):
        self.calls.append({"provider": provider, "url": url, "params": params})
        if url == TICKERS_URL:
            payload = load("sec", "company_tickers_head.json")
            for i, (ticker, cik) in enumerate(self.tickers_extra.items()):
                payload[str(len(payload) + i)] = {"cik_str": cik, "ticker": ticker, "title": ticker}
            return payload
        if url.startswith(SUBMISSIONS_URL):
            name = url[len(SUBMISSIONS_URL):]
            if "-submissions-" in name:
                if name == "CIK0001045810-submissions-001.json":
                    return load("sec", "submissions_CIK0001045810-001_trimmed.json")
                return submissions_page([])
            cik = int(name[3:13])
            return self.submissions.get(cik, {"filings": {"recent": submissions_page([]), "files": []}})
        if url.startswith(COMPANYFACTS_URL):
            return self.facts.get(int(url[len(COMPANYFACTS_URL) + 3:][:10]), {})
        if url.startswith(FMP_BASE):
            path = url[len(FMP_BASE) + 1:]
            if path == "stable/earnings-calendar":
                return self.calendar
            if path == "stable/earnings":
                if self.earnings_budget_exhausted:
                    raise BudgetExhausted("fmp: daily budget of 240 requests exhausted (240 used).")
                return self.earnings.get(params["symbol"], [])
            if path == "stable/historical-chart/1min":
                self.n_intraday += 1
                if self.intraday_budget is not None and self.n_intraday > self.intraday_budget:
                    raise BudgetExhausted("fmp: daily budget of 240 requests exhausted (240 used).")
                if params["symbol"] in self.intraday_errors:
                    return {"Error Message": "Invalid symbol"}
                return self.intraday.get((params["symbol"], params["from"]), [])
            if path == "stable/splits":
                if params["symbol"] in self.splits_errors:
                    return {"Error Message": "Invalid symbol"}
                return self.splits.get(params["symbol"], [])
            raise AssertionError(f"unexpected FMP path {path}")
        if url.startswith(NASDAQ_URL):
            rows = self.nasdaq.get(params["date"])
            return {"data": {"asOf": None, "headers": {}, "rows": rows}, "message": None,
                    "status": {"rCode": 200, "bCodeMessage": None, "developerMessage": None}}
        if url == AV_URL:
            if params["symbol"] in self.av_symbols:
                return load("alphavantage", "earnings_NVDA.json")
            return {}
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def fake(monkeypatch) -> FakeHttp:
    monkeypatch.setattr(ev_mod, "_today_ny", lambda: TODAY)
    return FakeHttp().install(monkeypatch)


def write_universe(settings: Settings, *, para_nvda_listing: str | None = "2025-06-01") -> None:
    rows = [
        {U.market: "xyz:NVDA", U.dex: "xyz", U.symbol: "NVDA", U.kind: "equity_us", U.underlying: "NVDA",
         U.cik: NVDA_CIK, U.listing_start: "2025-11-12", U.is_primary: True, U.in_event_universe: True},
        {U.market: "xyz:AAPL", U.dex: "xyz", U.symbol: "AAPL", U.kind: "equity_us", U.underlying: "AAPL",
         U.cik: AAPL_CIK, U.listing_start: "2025-11-21", U.is_primary: True, U.in_event_universe: True},
        {U.market: "xyz:TSM", U.dex: "xyz", U.symbol: "TSM", U.kind: "equity_fpi", U.underlying: "TSM",
         U.cik: TSM_CIK, U.listing_start: "2025-11-12", U.is_primary: True, U.in_event_universe: True},
        {U.market: "xyz:GOLD", U.dex: "xyz", U.symbol: "GOLD", U.kind: "commodity", U.underlying: None,
         U.cik: None, U.listing_start: "2025-11-12", U.is_primary: False, U.in_event_universe: False},
    ]
    if para_nvda_listing is not None:
        rows.append({U.market: "para:NVDA", U.dex: "para", U.symbol: "NVDA", U.kind: "equity_us",
                     U.underlying: "NVDA", U.cik: NVDA_CIK, U.listing_start: para_nvda_listing,
                     U.is_primary: False, U.in_event_universe: False})
    u = pd.DataFrame(rows)
    u[U.listing_start] = pd.to_datetime(u[U.listing_start], utc=True)
    u[U.cik] = u[U.cik].astype("Int64")
    u.to_parquet(settings.universe_path, index=False)


def nvda_bars() -> pd.DataFrame:
    raw = load("fmp", "historical-chart_1min_NVDA_20260826_extended.json")
    return fmp_mod._intraday_frame(raw, "NVDA", "1m", pd.Timedelta(minutes=1),
                                   date(2026, 8, 25), date(2026, 8, 28))


def nvda_filings() -> pd.DataFrame:
    page = load("sec", "submissions_CIK0001045810.json")["filings"]["recent"]
    return sec_mod._submissions_page_to_frame(page)


def tsm_filings() -> pd.DataFrame:
    page = load("sec", "submissions_CIK0001046179.json")["filings"]["recent"]
    return sec_mod._submissions_page_to_frame(page)


def filings_at(*accepted: str) -> pd.DataFrame:
    return sec_mod._submissions_page_to_frame(submissions_page([("8-K", a, "2.02,9.01") for a in accepted]))


def synthetic_bars(days: list[str], *, spike_ny: str | None = None, spike_day: str | None = None,
                   spike_volume: float = 400_000.0, spike_ret: float = 0.03) -> pd.DataFrame:
    """Flat 1-minute extended-hours bars (04:00-19:59 NY) for `days`, volume ~5000 with a mild
    deterministic ripple, plus one spike bar at `spike_ny` (HH:MM) on `spike_day`."""
    frames = []
    rng = np.random.default_rng(0)
    for d in days:
        idx = pd.date_range(f"{d} 04:00", f"{d} 19:59", freq="1min", tz="America/New_York")
        n = len(idx)
        vol = 5000.0 + rng.integers(-1500, 1500, n)
        clock = idx.hour * 60 + idx.minute
        vol = np.where((clock >= 570) & (clock < 960), vol * 20, vol)  # regular hours are busier
        px = np.full(n, 100.0)
        opn, close = px.copy(), px.copy()
        if spike_ny is not None and d == spike_day:
            hh, mm = (int(x) for x in spike_ny.split(":"))
            k = int(np.flatnonzero(clock == hh * 60 + mm)[0])
            vol[k] = spike_volume
            close[k] = opn[k] * (1 + spike_ret)
            opn[k + 1:] = close[k]
            close[k + 1:] = close[k]
        t = idx.tz_convert("UTC")
        frames.append(pd.DataFrame({
            C.market: "SYN", C.interval: "1m", C.t: t, C.t_end: t + pd.Timedelta(minutes=1),
            C.open: opn, C.high: np.maximum(opn, close), C.low: np.minimum(opn, close),
            C.close: close, C.volume: vol, C.n_trades: pd.array([pd.NA] * n, dtype="Int64"),
            C.source: "fmp_intraday",
        }))
    return pd.concat(frames, ignore_index=True)


def ny(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


# ---- detector -----------------------------------------------------------------------------------
def test_detection_on_nvda_fixture_finds_the_release_bar():
    bars = nvda_bars()
    hit = detect_release_from_bars(bars, REPORT_DAY)
    assert hit is not None
    start, first_bar = hit
    assert start in (T_NVDA_DET, T_NVDA_DET + pd.Timedelta(minutes=1))
    assert str(start.tz) == "UTC" and first_bar is False
    # nothing fires before the release on the report day, and no bars -> no detection
    pre = bars[bars[C.t] < T_NVDA_DET]
    assert detect_release_from_bars(pre, REPORT_DAY) is None
    assert detect_release_from_bars(bars.iloc[:0], REPORT_DAY) is None
    assert detect_release_from_bars(bars, pd.Timestamp("2026-08-27")) is None


def test_detection_uses_prior_sessions_and_first_bar_flag():
    bars = synthetic_bars(["2026-08-25", "2026-08-26"], spike_ny="16:20", spike_day="2026-08-26")
    hit = detect_release_from_bars(bars, REPORT_DAY)
    assert hit == (ny("2026-08-26 16:20"), False)
    # a spike on the very first bar of the extended session is detected and flagged
    first = synthetic_bars(["2026-08-25", "2026-08-26"], spike_ny="04:00", spike_day="2026-08-26")
    assert detect_release_from_bars(first, REPORT_DAY) == (ny("2026-08-26 04:00"), True)
    # a volume spike without a price move is not a release
    quiet = synthetic_bars(["2026-08-25", "2026-08-26"], spike_ny="16:20", spike_day="2026-08-26",
                           spike_ret=0.002)
    assert detect_release_from_bars(quiet, REPORT_DAY) is None
    # not_before restricts the candidates but keeps the baseline
    assert detect_release_from_bars(bars, REPORT_DAY, not_before=ny("2026-08-26 16:21")) is None


def test_detect_release_live_only_uses_closed_bars():
    bars = nvda_bars()
    assert detect_release_live(bars, REPORT_DAY, now=T_NVDA_DET + pd.Timedelta(seconds=30)) is None
    assert detect_release_live(bars, REPORT_DAY, now=T_NVDA_DET + pd.Timedelta(minutes=1)) == T_NVDA_DET


# ---- resolver -----------------------------------------------------------------------------------
def test_resolver_priority_and_earlier_only_rule():
    bars = nvda_bars()
    filings = nvda_filings()
    manual = pd.Timestamp("2026-08-26 20:19:00", tz="UTC")
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=filings, intraday=bars,
                             calendar_flag="post-market", manual=manual)
    assert (r.t0, r.source, r.confidence) == (manual, "manual", 1.0)

    # 8-K + detection 79 s earlier: t0 moves earlier, confidence stays that of the 8-K
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=filings, intraday=bars,
                             calendar_flag=None)
    assert isinstance(r, ResolvedT0)
    assert r.source == "sec_8k" and r.confidence == CONFIDENCE["sec_8k"]
    assert r.t0 in (T_NVDA_DET, T_NVDA_DET + pd.Timedelta(minutes=1))
    assert r.t0_lag_s is not None and r.t0_lag_s >= 0
    assert r.t0_lag_s == pytest.approx((T_NVDA_ACC - r.t0).total_seconds())
    assert str(r.t0.tz) == "UTC" and r.flags == ()

    # a detection later than the acceptance never changes t0 but is logged as the lag
    early_8k = filings_at("2026-08-26T20:10:00.000Z")
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=early_8k, intraday=bars,
                             calendar_flag=None)
    assert r.t0 == pd.Timestamp("2026-08-26 20:10:00", tz="UTC") and r.source == "sec_8k"
    assert r.t0_lag_s == pytest.approx(-600.0)

    # a detection more than 15 minutes before the acceptance is not the release either
    late_8k = filings_at("2026-08-26T20:40:00.000Z")
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=late_8k, intraday=bars,
                             calendar_flag=None)
    assert r.t0 == pd.Timestamp("2026-08-26 20:40:00", tz="UTC") and r.confidence == CONFIDENCE["sec_8k"]
    # the 20:20 release bar is outside the window; anything detected is a post-acceptance
    # reaction (negative lag) and never moves t0
    assert r.t0_lag_s is None or r.t0_lag_s <= 0

    # without an 8-K the detection is the source; 6-K rows are never a time source
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=tsm_filings(), intraday=bars,
                             calendar_flag="post-market")
    assert r.source == "detected" and r.confidence == CONFIDENCE["detected"] and r.t0 == T_NVDA_DET

    # calendar flag defaults: AMC 16:05, BMO 07:00 New York; unknown flag -> low confidence
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=None, intraday=None,
                             calendar_flag="post-market")
    assert r.t0 == ny("2026-08-26 16:05") and r.source == "calendar_flag" and r.confidence == 0.5
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=None, intraday=None,
                             calendar_flag="pre-market")
    assert r.t0 == ny("2026-08-26 07:00")
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=None, intraday=None,
                             calendar_flag="time-not-supplied")
    assert r.confidence == CONFIDENCE_UNKNOWN and "timing_unknown" in r.flags
    # Nasdaq's prefixed flags are read like the vendor-neutral ones
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=None, intraday=None,
                             calendar_flag="time-after-hours")
    assert r.t0 == ny("2026-08-26 16:05") and r.confidence == CONFIDENCE["calendar_flag"]

    # an 8-K on another date is not attributed to this report date
    r = resolve_release_time(report_date_ny=pd.Timestamp("2026-08-20"), sec_filings=filings,
                             intraday=None, calendar_flag="post-market")
    assert r.source == "calendar_flag"
    # ... but a late acceptance just after midnight New York is
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=filings_at("2026-08-27T02:30:00.000Z"),
                             intraday=None, calendar_flag=None)
    assert r.source == "sec_8k" and r.t0 == pd.Timestamp("2026-08-27 02:30:00", tz="UTC")


def test_first_bar_detection_is_flagged_and_downgraded():
    bars = synthetic_bars(["2026-08-25", "2026-08-26"], spike_ny="04:00", spike_day="2026-08-26")
    r = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=None, intraday=bars,
                             calendar_flag="pre-market")
    assert r.source == "detected" and r.t0 == ny("2026-08-26 04:00")
    assert "detection_first_bar" in r.flags
    assert r.confidence == CONFIDENCE["calendar_flag"]
    # a detection that is not on the first bar keeps the detected confidence
    later = synthetic_bars(["2026-08-25", "2026-08-26"], spike_ny="07:00", spike_day="2026-08-26")
    r2 = resolve_release_time(report_date_ny=REPORT_DAY, sec_filings=None, intraday=later,
                              calendar_flag="pre-market")
    assert r2.flags == () and r2.confidence == CONFIDENCE["detected"]


# ---- fiscal period -------------------------------------------------------------------------------
def test_fiscal_period_from_sec_facts_alphavantage_and_derived(settings, fake):
    facts = sec_mod.SECClient(settings).company_facts_eps(NVDA_CIK)
    assert fiscal_period_for(REPORT_DAY, sec_eps_facts=facts, av_rows=None) == ("2026-07", "sec_facts", False)
    assert fiscal_period_for(pd.Timestamp("2026-02-25"), sec_eps_facts=facts, av_rows=None)[0] == "2026-01"
    # Alpha Vantage fiscalDateEnding for foreign filers, matched by report date within 10 days
    av = pd.DataFrame({"fiscal_period_end": [date(2026, 7, 31), date(2026, 4, 30)],
                       "report_date_ny": [date(2026, 8, 28), date(2026, 5, 20)]})
    assert fiscal_period_for(REPORT_DAY, sec_eps_facts=None, av_rows=av) == ("2026-07", "alphavantage", False)
    assert fiscal_period_for(REPORT_DAY, sec_eps_facts=facts.iloc[:0], av_rows=av)[1] == "alphavantage"
    # calendar quarter end preceding the report date when nothing else is known
    assert fiscal_period_for(REPORT_DAY, sec_eps_facts=None, av_rows=None) == ("2026-06", "derived", True)
    assert fiscal_period_for(pd.Timestamp("2026-01-15"), sec_eps_facts=None, av_rows=None)[0] == "2025-12"
    assert fiscal_period_for(pd.Timestamp("2026-06-30"), sec_eps_facts=None, av_rows=None)[0] == "2026-03"
    # a period end that is too old (> 120 days) does not qualify as the period itself; with any
    # facts on file the issuer's quarter end is projected from the latest one in whole quarters
    old = facts[facts["period_end"] < pd.Timestamp("2026-01-01", tz="UTC")]
    assert fiscal_period_for(REPORT_DAY, sec_eps_facts=old, av_rows=None) == ("2026-07", "sec_facts_projected", False)
    # so the id is stable: the Aug-2026 event before its 10-Q landed gets the id the 10-Q gives
    before_10q = facts[facts["period_end"] < pd.Timestamp("2026-07-01", tz="UTC")]
    assert fiscal_period_for(REPORT_DAY, sec_eps_facts=before_10q, av_rows=None) == ("2026-07", "sec_facts_projected", False)
    # ... and the upcoming Nov-2026 event keeps its id before and after the Q3 10-Q is on file
    nov = pd.Timestamp("2026-11-18")
    assert fiscal_period_for(nov, sec_eps_facts=facts, av_rows=None) == ("2026-10", "sec_facts_projected", False)
    assert fiscal_period_for(nov, sec_eps_facts=before_10q, av_rows=None)[0] == "2026-10"
    q3 = pd.DataFrame({"period_end": [pd.Timestamp("2026-10-25", tz="UTC")],
                       "filed": [pd.Timestamp("2026-11-19", tz="UTC")]})
    assert fiscal_period_for(nov, sec_eps_facts=pd.concat([facts, q3]), av_rows=None) == ("2026-10", "sec_facts", False)
    # the projection also runs backwards for an old event whose own facts are missing
    recent = facts[facts["period_end"] >= pd.Timestamp("2026-01-01", tz="UTC")]
    assert fiscal_period_for(pd.Timestamp("2024-02-21"), sec_eps_facts=recent, av_rows=None) == ("2024-01", "sec_facts_projected", False)
    # Alpha Vantage periods project the same way when no row matches the report date
    assert fiscal_period_for(pd.Timestamp("2026-11-12"), sec_eps_facts=None, av_rows=av) == ("2026-10", "alphavantage_projected", False)
    assert project_quarter_end(date(2026, 4, 26), date(2026, 8, 26)) == date(2026, 7, 26)
    assert project_quarter_end(date(2026, 7, 26), date(2026, 2, 25)) == date(2026, 1, 26)
    assert project_quarter_end(date(2025, 12, 31), date(2026, 10, 1)) == date(2026, 9, 30)
    assert project_quarter_end(date(2025, 12, 31), date(2026, 9, 30)) == date(2026, 6, 30)


# ---- build_events ---------------------------------------------------------------------------------
def test_build_events_end_to_end_from_fixtures(settings, fake):
    write_universe(settings)
    fake.nasdaq["2026-08-27"] = [nasdaq_row("NVDA")]  # one day off: matched, not a conflict
    df = build_events(settings, underlyings=["NVDA", "AAPL"], since=pd.Timestamp("2025-08-01"))
    assert list(df.columns) == EVENT_COLUMNS
    assert df.attrs["schema_version"] == SCHEMA_VERSION
    assert set(df[E.underlying]) == {"NVDA", "AAPL"}
    assert not df[E.pending].any()
    assert str(df[E.t0].dtype) == "datetime64[ns, UTC]"
    assert df[E.event_id].map(lambda s: re.fullmatch(r"[A-Z.]+:\d{4}-\d{2}", s) is not None).all()
    assert df[E.event_id].is_unique
    assert df[E.report_date_ny].map(lambda d: isinstance(d, date)).all()

    by = df.set_index(E.event_id)
    q2 = by.loc["NVDA:2026-07"]
    assert q2[E.fiscal_period] == "2026-07" and q2[E.fiscal_period_source] == "sec_facts"
    assert q2[E.report_date_ny] == date(2026, 8, 26)
    assert q2[E.t0] in (T_NVDA_DET, T_NVDA_DET + pd.Timedelta(minutes=1))
    assert q2[E.t0_source] == "sec_8k" and q2[E.t0_confidence] == CONFIDENCE["sec_8k"]
    assert q2["t0_acceptance"] == T_NVDA_ACC and q2["t0_lag_s"] >= 0
    assert q2[E.timing] == "AMC"
    assert q2[E.eps_actual] == 2.22 and q2[E.eps_estimate] == 2.09
    assert q2[E.eps_surprise_pct] == pytest.approx((2.22 - 2.09) / 2.09 * 100)
    assert q2[E.rev_surprise_pct] == pytest.approx((96221000000 - 92270940000) / 92270940000 * 100)
    assert q2[E.estimate_source] == "fmp_final" and pd.isna(q2[E.estimate_snapshot_time])
    assert q2[E.n_estimates] == 32  # from the matched Nasdaq row
    assert "date_conflict" not in q2[E.flags]
    for src in ("fmp", "sec", "sec_8k", "fmp_intraday", "nasdaq"):
        assert src in q2[E.sources_used].split(";")
    assert q2[E.market] == "xyz:NVDA" and bool(q2[E.has_perp_at_t0]) is True
    assert q2[E.cik] == NVDA_CIK and q2[E.kind] == "equity_us"
    assert q2[E.listing_start] == pd.Timestamp("2025-06-01", tz="UTC")  # earliest over all markets

    # AAPL: 8-K time, no bars served -> acceptance stands; fiscal period derived (no facts)
    q3 = by.loc["AAPL:2026-06"]
    assert q3[E.t0] == pd.Timestamp("2026-07-30 20:30:28", tz="UTC") and q3[E.t0_source] == "sec_8k"
    assert "fiscal_period_derived" in q3[E.flags].split(";") and "no_intraday" in q3[E.flags].split(";")
    assert np.isnan(q3["t0_lag_s"])

    # markets: primary when listed at t0, else the earliest-listed alternate, else primary/False
    aug25 = by.loc["NVDA:2025-07"]
    assert aug25[E.report_date_ny] == date(2025, 8, 27)
    assert aug25[E.market] == "para:NVDA" and bool(aug25[E.has_perp_at_t0]) is True
    nov = by.loc["NVDA:2025-10"]
    assert nov[E.market] == "xyz:NVDA" and bool(nov[E.has_perp_at_t0]) is True
    # a future event is kept with the calendar default and flagged upcoming; its fiscal period
    # is projected from the issuer's latest filed period (the Q3 10-Q is not on file yet)
    up = by.loc["NVDA:2026-10"]
    assert "upcoming" in up[E.flags].split(";") and pd.isna(up[E.eps_actual])
    assert up[E.t0_source] == "calendar_flag" and up[E.t0] == ny("2026-11-18 16:05")
    assert "timing_from_history" in up[E.flags].split(";")
    assert up[E.fiscal_period_source] == "sec_facts_projected" and "fiscal_period_derived" not in up[E.flags]
    assert up[E.estimate_source] == "fmp_calendar" and up[E.eps_estimate] == 2.47

    # intraday requests use exactly the loaders window: report date - 1 .. + 2, one chunk
    intraday_calls = [c for c in fake.calls if c["url"].endswith("historical-chart/1min")]
    nvda_q2 = [c for c in intraday_calls if c["params"]["symbol"] == "NVDA" and c["params"]["from"] == "2026-08-25"]
    assert len(nvda_q2) == 1 and nvda_q2[0]["params"]["to"] == "2026-08-27"
    assert nvda_q2[0]["params"]["extended"] == "true"
    # no Alpha Vantage request for US filers (no key in these settings anyway)
    assert not any(c["provider"] == "alphavantage" for c in fake.calls)

    # the table round-trips with its schema version and UTC timestamps
    back = load_events(settings)
    assert back.attrs["schema_version"] == SCHEMA_VERSION
    assert str(back[E.t0].dtype) == "datetime64[ns, UTC]"
    assert len(back) == len(df) and set(back[E.event_id]) == set(df[E.event_id])
    assert expected_release_clock(back, "NVDA") == ("16:21", "median of 5 sec_8k acceptances")
    assert expected_release_clock(back, "TSM") is None


def test_build_events_flags_date_conflicts(settings, fake):
    write_universe(settings)
    # Nasdaq lists NVDA three days after FMP's date; AAPL's synthetic event makes that date fetched
    fake.earnings["AAPL"] = [dict(AAPL_EARNINGS[0], date="2026-08-29")]
    fake.nasdaq["2026-08-29"] = [nasdaq_row("NVDA"), nasdaq_row("AAPL")]
    df = build_events(settings, underlyings=["NVDA", "AAPL"], since=pd.Timestamp("2026-08-01"))
    q2 = df.set_index(E.event_id).loc["NVDA:2026-07"]
    assert "date_conflict" in q2[E.flags].split(";")
    assert q2[E.t0_confidence] == 0.0 and q2[E.t0_source] == "sec_8k"
    assert q2[E.t0] in (T_NVDA_DET, T_NVDA_DET + pd.Timedelta(minutes=1))  # the time is still kept


def test_consensus_snapshot_precedence(settings, fake):
    write_universe(settings)
    snaps = pd.DataFrame({
        "snapshot_time": pd.to_datetime(["2026-08-20 12:00", "2026-08-25 12:00", "2026-08-25 12:00",
                                         "2026-08-26 21:00"], utc=True),
        "symbol": ["NVDA"] * 4,
        "report_date_ny": [date(2026, 8, 26)] * 4,
        "eps_estimate": [2.00, 2.05, 2.06, 2.30],
        "rev_estimate": [9.0e10, 9.1e10, np.nan, 9.5e10],
        "n_estimates": pd.array([pd.NA, pd.NA, 35, 40], dtype="Int64"),
        "vendor": ["fmp", "fmp", "nasdaq", "fmp"],
        "vendor_last_updated": pd.to_datetime(["2026-08-19", "2026-08-24", None, "2026-08-26"], utc=True),
    })[SNAPSHOT_COLUMNS]
    settings.consensus_archive_dir.mkdir(parents=True, exist_ok=True)
    snaps.to_parquet(settings.consensus_archive_dir / "2026-08-26.parquet", index=False)
    df = build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-05-01"))
    q2 = df.set_index(E.event_id).loc["NVDA:2026-07"]
    # the latest snapshot before t0 wins (2026-08-25, FMP row), the post-release one is ignored
    assert q2[E.estimate_source] == "consensus_snapshot"
    assert q2[E.estimate_snapshot_time] == pd.Timestamp("2026-08-25 12:00", tz="UTC")
    assert q2[E.eps_estimate] == 2.05 and q2[E.rev_estimate] == 9.1e10
    assert q2[E.n_estimates] == 35  # from the Nasdaq row of the same snapshot
    assert q2[E.eps_surprise_pct] == pytest.approx((2.22 - 2.05) / 2.05 * 100)
    assert "consensus_snapshot" in q2[E.sources_used].split(";")
    # an older event without a snapshot keeps the vendor's final value
    q1 = df.set_index(E.event_id).loc["NVDA:2026-04"]
    assert q1[E.estimate_source] == "fmp_final" and q1[E.eps_estimate] == 1.76


def test_budget_exhaustion_writes_completed_and_pending_rows(settings, fake):
    write_universe(settings)
    fake.intraday_budget = 1  # the first intraday request succeeds, the second hits the budget
    with pytest.raises(BudgetExhausted, match="pending"):
        build_events(settings, underlyings=["NVDA", "AAPL"], since=pd.Timestamp("2026-01-01"))
    df = load_events(settings)
    assert df.attrs["schema_version"] == SCHEMA_VERSION
    done, pending = df[~df[E.pending]], df[df[E.pending]]
    # newest first: the upcoming NVDA event (no request) and NVDA 2026-08-26 completed
    assert set(done[E.event_id]) == {"NVDA:2026-10", "NVDA:2026-07"}
    assert set(pending[E.event_id]) == {"AAPL:2026-06", "NVDA:2026-04", "AAPL:2026-03", "NVDA:2026-01", "AAPL:2025-12"}
    assert pending[E.t0].isna().all() and pending[E.t0_source].isna().all()
    assert pending[E.flags].map(lambda f: "pending" in f.split(";")).all()
    assert not pending[E.has_perp_at_t0].any()
    assert (pending[E.eps_actual].notna()).all()  # FMP data already known is kept
    assert df[E.event_id].is_unique
    # a rerun with budget resolves everything and replaces the pending rows
    fake.intraday_budget = None
    df2 = build_events(settings, underlyings=["NVDA", "AAPL"], since=pd.Timestamp("2026-01-01"))
    assert not df2[E.pending].any() and set(df2[E.event_id]) == set(df[E.event_id])
    assert not load_events(settings)[E.pending].any()


def test_budget_hit_rerun_never_downgrades_completed_rows(settings, fake):
    write_universe(settings)
    full = build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-01-01"))
    assert len(full) == 4 and not full[E.pending].any()
    # a week later the empty-payload cache entries have expired and the very first intraday
    # request hits the budget: only the upcoming row (no request) resolves in this run
    fake.intraday_budget = 0
    with pytest.raises(BudgetExhausted):
        build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-01-01"))
    again = load_events(settings)
    assert set(again[E.event_id]) == set(full[E.event_id]) and again[E.event_id].is_unique
    assert not again[E.pending].any()
    a = full.set_index(E.event_id).sort_index()
    b = again.set_index(E.event_id).sort_index()
    for col in (E.t0, E.t0_source, E.t0_confidence, E.report_date_ny, E.eps_actual, E.flags):
        assert a[col].equals(b[col]), col
    # when not even the earnings history can be fetched, the placeholder joins the completed
    # rows instead of replacing them
    fake.intraday_budget = None
    fake.earnings_budget_exhausted = True
    with pytest.raises(BudgetExhausted):
        build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-01-01"))
    third = load_events(settings)
    assert set(third[~third[E.pending]][E.event_id]) == set(full[E.event_id])
    assert list(third[third[E.pending]][E.event_id]) == ["NVDA:pending"]
    # the next run with budget drops the placeholder again
    fake.earnings_budget_exhausted = False
    build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-01-01"))
    assert set(load_events(settings)[E.event_id]) == set(full[E.event_id])
    # a pending row for an event the table never completed is still written
    fake.intraday_budget = 0
    with pytest.raises(BudgetExhausted):
        build_events(settings, underlyings=["NVDA", "AAPL"], since=pd.Timestamp("2026-01-01"))
    mixed = load_events(settings)
    assert set(mixed[~mixed[E.pending]][E.event_id]) >= set(full[E.event_id])
    assert set(mixed[mixed[E.pending]][E.underlying]) == {"AAPL"}


def test_budget_exhaustion_without_write_reports_nothing_written(settings, fake):
    write_universe(settings)
    fake.intraday_budget = 0
    with pytest.raises(BudgetExhausted, match=r"nothing written \(write=False\)") as info:
        build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-01-01"), write=False)
    assert "written as pending" not in str(info.value)
    assert not settings.events_path.exists()


def test_fmp_error_on_one_intraday_request_does_not_abort_the_build(settings, fake):
    write_universe(settings)
    fake.intraday_errors.add("AAPL")
    df = build_events(settings, underlyings=["NVDA", "AAPL"], since=pd.Timestamp("2026-01-01"))
    assert settings.events_path.exists() and not df[E.pending].any()
    by = df.set_index(E.event_id)
    aapl = by.loc["AAPL:2026-06"]
    assert {"intraday_error", "no_intraday"} <= set(aapl[E.flags].split(";"))
    assert aapl[E.t0_source] == "sec_8k"  # the 8-K time stands without bars
    assert "fmp_intraday" not in aapl[E.sources_used].split(";")
    nvda = by.loc["NVDA:2026-07"]
    assert "intraday_error" not in nvda[E.flags].split(";")
    assert "fmp_intraday" in nvda[E.sources_used].split(";")


def test_corporate_action_flag_from_the_splits_calendar(settings, fake):
    write_universe(settings)
    # a 10:1 split whose ex-date is the morning after the NVDA Q2 release (t0 2026-08-26 20:20 UTC)
    fake.splits["NVDA"] = [{"symbol": "NVDA", "date": "2026-08-27", "numerator": 10, "denominator": 1,
                            "splitType": "forward"}]
    df = build_events(settings, underlyings=["NVDA", "AAPL"], since=pd.Timestamp("2026-01-01"))
    assert list(df.columns) == EVENT_COLUMNS and str(df[E.ca_ex_date].dtype) == "datetime64[ns, UTC]"
    by = df.set_index(E.event_id)
    q2 = by.loc["NVDA:2026-07"]
    assert "corporate_action" in q2[E.flags].split(";")
    assert q2[E.ca_ex_date] == ny("2026-08-27 00:00")
    # the same ex-date is outside [t0 - 60 d, t0 + 24 h] of the May event and of the upcoming one
    for eid in ("NVDA:2026-04", "NVDA:2026-10", "AAPL:2026-06"):
        assert "corporate_action" not in by.loc[eid, E.flags].split(";"), eid
        assert pd.isna(by.loc[eid, E.ca_ex_date]), eid
    # one splits request per underlying, not per event
    splits_calls = [c for c in fake.calls if c["url"].endswith("stable/splits")]
    assert sorted(c["params"]["symbol"] for c in splits_calls) == ["AAPL", "NVDA"]
    back = load_events(settings).set_index(E.event_id)
    assert back.loc["NVDA:2026-07", E.ca_ex_date] == ny("2026-08-27 00:00")
    assert pd.isna(back.loc["AAPL:2026-06", E.ca_ex_date])
    # an error payload on the splits request is not a checkpoint: flagged, the build goes on
    fake.splits_errors.add("AAPL")
    df2 = build_events(settings, underlyings=["AAPL"], since=pd.Timestamp("2026-01-01"))
    assert not df2[E.pending].any()
    assert "splits_error" in df2.set_index(E.event_id).loc["AAPL:2026-06", E.flags].split(";")


def test_corporate_action_near_picks_the_ex_date_nearest_t0():
    splits = fmp_mod._splits_frame(load("fmp", "splits_NFLX.json"))
    ex = ny("2025-11-17 00:00")
    assert corporate_action_near(splits, pd.Timestamp("2025-11-18 21:05", tz="UTC"), 24) == ex
    assert corporate_action_near(splits, pd.Timestamp("2025-11-16 21:05", tz="UTC"), 24) == ex  # ex-date inside t0 + 24h
    assert corporate_action_near(splits, pd.Timestamp("2026-01-15 21:05", tz="UTC"), 24) == ex  # 59 days after
    assert corporate_action_near(splits, pd.Timestamp("2026-01-20 21:05", tz="UTC"), 24) is None  # 64 days after
    assert corporate_action_near(splits, pd.Timestamp("2025-11-15 21:05", tz="UTC"), 24) is None  # 27 h before
    assert corporate_action_near(None, pd.Timestamp("2025-11-18", tz="UTC"), 24) is None
    two = fmp_mod._splits_frame(load("fmp", "splits_NFLX.json") + [
        {"symbol": "NFLX", "date": "2025-10-01", "numerator": 2, "denominator": 1}])
    assert corporate_action_near(two, pd.Timestamp("2025-10-20", tz="UTC"), 24) == ny("2025-10-01 00:00")


def test_budget_exhaustion_before_any_history_yields_placeholders(settings, fake):
    write_universe(settings)

    def boom(http_self, url, params=None, **kw):
        raise BudgetExhausted("fmp: daily budget of 240 requests exhausted (240 used).")

    original = HttpClient.get_json
    HttpClient.get_json = boom
    try:
        with pytest.raises(BudgetExhausted):
            build_events(settings, underlyings=["NVDA"])
    finally:
        HttpClient.get_json = original
    df = load_events(settings)
    assert len(df) == 1 and bool(df.loc[0, E.pending]) and df.loc[0, E.event_id] == "NVDA:pending"
    assert "earnings_history_pending" in df.loc[0, E.flags]


def test_fpi_uses_alphavantage_flag_and_fiscal_period(tmp_path, fake):
    (tmp_path / "configs").mkdir()  # no release clock for TSM here: the calendar flag is the subject
    s = Settings(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports",
                 configs_dir=tmp_path / "configs",
                 fmp_api_key="test", alphavantage_api_key="test-key", _env_file=None)
    s.ensure_dirs()
    write_universe(s)
    fake.av_symbols.add("TSM")
    fake.earnings["TSM"] = [dict(r, symbol="TSM") for r in load("fmp", "earnings_NVDA.json")[1:3]]
    df = build_events(s, underlyings=["TSM", "NVDA"], since=pd.Timestamp("2026-05-01"))
    tsm = df[df[E.underlying] == "TSM"].set_index(E.event_id)
    row = tsm.loc["TSM:2026-07"]
    # 6-K only -> no SEC time; no bars served -> calendar flag from Alpha Vantage
    assert row[E.t0_source] == "calendar_flag" and row[E.t0] == ny("2026-08-26 16:05")
    assert row[E.t0_confidence] == 0.5
    assert row[E.fiscal_period_source] == "alphavantage"
    assert "alphavantage" in row[E.sources_used].split(";")
    assert row[E.kind] == "equity_fpi"
    av_calls = [c for c in fake.calls if c["provider"] == "alphavantage"]
    assert [c["params"]["symbol"] for c in av_calls] == ["TSM"]  # never for the US filer


def test_manual_override_wins(settings, fake):
    write_universe(settings)
    settings.configs_dir = settings.data_dir / "configs"
    settings.configs_dir.mkdir()
    (settings.configs_dir / "t0_overrides.yaml").write_text('"NVDA:2026-07": "2026-08-26T20:19:30Z"\n')
    df = build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-08-01"))
    q2 = df.set_index(E.event_id).loc["NVDA:2026-07"]
    assert q2[E.t0_source] == "manual" and q2[E.t0_confidence] == 1.0
    assert q2[E.t0] == pd.Timestamp("2026-08-26 20:19:30", tz="UTC")


def test_subset_run_merges_into_the_existing_table(settings, fake):
    write_universe(settings)
    build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-01-01"))
    build_events(settings, underlyings=["AAPL"], since=pd.Timestamp("2026-01-01"))
    both = load_events(settings)
    assert set(both[E.underlying]) == {"NVDA", "AAPL"} and both[E.event_id].is_unique
    # narrowing `since` for one name keeps that name's older rows
    build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-08-01"))
    again = load_events(settings)
    assert set(again[E.event_id]) == set(both[E.event_id])
    # a run without `since` rebuilds the name completely (older rows come from FMP history)
    full = build_events(settings, underlyings=["NVDA"])
    assert len(full) == 40 and set(load_events(settings)[E.underlying]) == {"NVDA", "AAPL"}


def test_missing_universe_requires_explicit_underlyings(settings, fake):
    with pytest.raises(FileNotFoundError):
        build_events(settings)
    fake.tickers_extra["TSM"] = TSM_CIK
    fake.earnings["TSM"] = [dict(r, symbol="TSM") for r in load("fmp", "earnings_NVDA.json")[1:3]]
    df = build_events(settings, underlyings=["NVDA", "TSM"], since=pd.Timestamp("2026-08-01"))
    q2 = df.set_index(E.event_id).loc["NVDA:2026-07"]
    assert q2[E.cik] == NVDA_CIK and q2[E.t0_source] == "sec_8k" and q2[E.kind] == "equity_us"
    assert pd.isna(q2[E.market]) and bool(q2[E.has_perp_at_t0]) is False
    assert "no_universe" in q2[E.flags].split(";")
    # the kind is guessed from the filings: 6-K only -> equity_fpi
    tsm = df[df[E.underlying] == "TSM"]
    assert len(tsm) == 1 and (tsm[E.kind] == "equity_fpi").all() and (tsm[E.cik] == TSM_CIK).all()


# ---- upcoming and consensus snapshots ---------------------------------------------------------
def test_snapshot_consensus_appends_fmp_and_nasdaq_rows(settings, fake):
    fake.nasdaq["2026-09-10"] = [nasdaq_row("NVDA", eps="", forecast="$2.47", n="38")]
    now = pd.Timestamp("2026-09-02 08:00", tz="UTC")
    written = snapshot_consensus(settings, days=10, now=now)
    assert list(written.columns) == SNAPSHOT_COLUMNS
    assert (written["snapshot_time"] == now).all() and str(written["snapshot_time"].dtype) == "datetime64[ns, UTC]"
    assert set(written["vendor"]) == {"fmp", "nasdaq"}
    nq = written[written["vendor"] == "nasdaq"].iloc[0]
    assert nq["symbol"] == "NVDA" and nq["eps_estimate"] == 2.47 and nq["n_estimates"] == 38
    assert nq["report_date_ny"] == date(2026, 9, 10) and pd.isna(nq["vendor_last_updated"])
    fmp_rows = written[written["vendor"] == "fmp"]
    assert len(fmp_rows) == 868 and fmp_rows["n_estimates"].isna().all()
    assert fmp_rows["vendor_last_updated"].notna().all()
    path = consensus_path(settings, date(2026, 9, 2))
    assert path.exists() and len(pd.read_parquet(path)) == len(written)
    # a second snapshot the same day accumulates; identical rows are not duplicated
    snapshot_consensus(settings, days=10, now=now)
    assert len(pd.read_parquet(path)) == len(written)
    snapshot_consensus(settings, days=10, now=now + pd.Timedelta(hours=12))
    assert len(pd.read_parquet(path)) == 2 * len(written)
    cal_calls = [c for c in fake.calls if c["url"].endswith("earnings-calendar")]
    assert cal_calls[0]["params"]["from"] == "2026-09-02" and cal_calls[0]["params"]["to"] == "2026-09-12"
    nq_calls = [c for c in fake.calls if c["url"].startswith(NASDAQ_URL)]
    assert {c["params"]["date"] for c in nq_calls} == {(date(2026, 9, 2) + pd.Timedelta(days=i)).isoformat()
                                                        for i in range(11)}


def test_upcoming_events_prefers_archived_consensus(settings, fake):
    write_universe(settings)
    fake.calendar = [
        {"symbol": "NVDA", "date": "2026-09-10", "epsActual": None, "epsEstimated": 2.47,
         "revenueActual": None, "revenueEstimated": 1.0e11, "lastUpdated": "2026-09-02"},
        {"symbol": "TSM", "date": "2026-09-11", "epsActual": None, "epsEstimated": 2.9,
         "revenueActual": None, "revenueEstimated": 3.0e10, "lastUpdated": "2026-09-02"},
        {"symbol": "ZZZ", "date": "2026-09-11", "epsActual": None, "epsEstimated": 0.1,
         "revenueActual": None, "revenueEstimated": 1.0, "lastUpdated": "2026-09-02"},
    ]
    build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-01-01"))
    snapshot_consensus(settings, days=14, now=pd.Timestamp("2026-09-01 08:00", tz="UTC"))
    fake.calendar[0]["epsEstimated"] = 2.50  # the vendor moved after the snapshot
    up = upcoming_events(settings, days=14)
    assert list(up[E.underlying]) == ["NVDA", "TSM"]  # ZZZ is not in the event universe
    nvda = up.set_index(E.underlying).loc["NVDA"]
    assert nvda[E.estimate_source] == "consensus_snapshot" and nvda[E.eps_estimate] == 2.47
    assert nvda[E.estimate_snapshot_time] == pd.Timestamp("2026-09-01 08:00", tz="UTC")
    assert nvda["expected_t0"] == ny("2026-09-10 16:21") and nvda["expected_t0_source"].startswith("median")
    tsm = up.set_index(E.underlying).loc["TSM"]
    # TSM: the shipped release clock (14:00 Asia/Taipei), not the AMC default
    assert tsm["expected_t0"] == pd.Timestamp("2026-09-11 06:00", tz="UTC") and tsm[E.market] == "xyz:TSM"
    assert tsm["expected_t0_source"] == "issuer release clock 14:00 Asia/Taipei (configs/release_clock_overrides.yaml)"


def test_upcoming_expected_t0_fallbacks(settings, fake):
    write_universe(settings)
    settings.configs_dir = settings.data_dir / "configs"  # no release clock: the fallbacks are the subject
    settings.configs_dir.mkdir()

    def cal_row(sym: str, day: str) -> dict:
        return {"symbol": sym, "date": day, "epsActual": None, "epsEstimated": 1.5,
                "revenueActual": None, "revenueEstimated": 9.0e10, "lastUpdated": "2026-09-02"}

    fake.calendar = [cal_row("TSM", "2026-09-11"), cal_row("AAPL", "2026-09-15")]
    # without an events table: the Nasdaq calendar's time flag, else the AMC default
    fake.nasdaq["2026-09-11"] = [nasdaq_row("TSM", time_flag="time-pre-market")]
    up = upcoming_events(settings, days=14).set_index(E.underlying)
    assert up.loc["TSM", "expected_t0"] == ny("2026-09-11 07:00")
    assert up.loc["TSM", "expected_t0_source"] == "nasdaq flag 'time-pre-market' (BMO default)"
    assert up.loc["AAPL", "expected_t0"] == ny("2026-09-15 16:05")
    assert up.loc["AAPL", "expected_t0_source"] == "calendar default (AMC)"
    assert (up[E.estimate_source] == "fmp_calendar").all()
    # an upcoming row in events.parquet whose calendar flag came from the filing history (a
    # morning filer) beats the Nasdaq flag; a row carrying only the timing_unknown default does not
    fake.submissions[AAPL_CIK] = {"filings": {"recent": submissions_page(
        [("8-K", t, "2.02,9.01") for t in ("2026-04-30T11:30:00.000Z", "2026-01-29T11:30:00.000Z")]
    ), "files": []}}
    fake.earnings["AAPL"] = [cal_row("AAPL", "2026-09-15")]
    fake.earnings["TSM"] = [cal_row("TSM", "2026-09-11")]
    build_events(settings, underlyings=["AAPL", "TSM"], since=pd.Timestamp("2026-09-01"))
    table = load_events(settings).set_index(E.underlying)
    assert "timing_from_history" in table.loc["AAPL", E.flags].split(";")
    assert table.loc["AAPL", E.estimate_source] == "fmp_calendar"
    assert "timing_unknown" in table.loc["TSM", E.flags].split(";")
    up = upcoming_events(settings, days=14).set_index(E.underlying)
    assert up.loc["AAPL", "expected_t0"] == ny("2026-09-15 07:00")
    assert up.loc["AAPL", "expected_t0_source"] == "events table: calendar_flag (timing_from_history)"
    assert up.loc["TSM", "expected_t0"] == ny("2026-09-11 07:00")
    assert up.loc["TSM", "expected_t0_source"].startswith("nasdaq flag")
    # a manual override recorded on the upcoming row wins over everything
    (settings.configs_dir / "t0_overrides.yaml").write_text('"TSM:2026-09-11": "2026-09-11T12:30:00Z"\n')
    build_events(settings, underlyings=["TSM"], since=pd.Timestamp("2026-09-01"))
    up = upcoming_events(settings, days=14).set_index(E.underlying)
    assert up.loc["TSM", "expected_t0"] == pd.Timestamp("2026-09-11 12:30", tz="UTC")
    # the yaml is read directly by upcoming_events (same instant, and no rebuild would be needed)
    assert up.loc["TSM", "expected_t0_source"] == "manual override (configs/t0_overrides.yaml)"
    assert load_events(settings).set_index(E.underlying).loc["TSM", E.t0_source] == "manual"


def _sec_8k_row(event_id: str, day: str, acceptance_utc: str, **over) -> dict:
    acc = pd.Timestamp(acceptance_utc, tz="UTC")
    row = {E.event_id: event_id, E.underlying: "NVDA", E.market: XYZ_NVDA, E.report_date_ny: date.fromisoformat(day),
           E.t0: acc, "t0_acceptance": acc, E.t0_source: "sec_8k", E.timing: "AMC", E.pending: False, E.flags: ""}
    row.update(over)
    return row


def test_expected_release_clock_before_gate_and_expected_t0_chain():
    # New York clocks: 16:30 (Feb), 16:20 (May), the event's own 16:50 (Aug), a later 16:40 (Nov)
    ev = pd.DataFrame([_sec_8k_row("NVDA:2026-01", "2026-02-25", "2026-02-25 21:30:00"),
                       _sec_8k_row("NVDA:2026-04", "2026-05-28", "2026-05-28 20:20:00"),
                       _sec_8k_row("NVDA:2026-07", "2026-08-26", "2026-08-26 20:50:00"),
                       _sec_8k_row("NVDA:2026-10", "2026-11-18", "2026-11-18 21:40:00")])
    assert expected_release_clock(ev, "NVDA") == ("16:35", "median of 4 sec_8k acceptances")
    before = pd.Timestamp("2026-08-26 20:00", tz="UTC")  # the pre_5m decision clock of the August event
    assert expected_release_clock(ev, "NVDA", before=before) == ("16:25", "median of 2 sec_8k acceptances")
    assert expected_release_clock(ev, "NVDA", before=pd.Timestamp("2026-01-01", tz="UTC")) is None
    assert expected_release_clock(ev.drop(columns=[E.t0_source]), "NVDA") is None
    # the gate flows through expected_t0_for
    d = date(2026, 8, 26)
    assert expected_t0_for(ev, "NVDA", d, before=before) == (ny("2026-08-26 16:25"), "median of 2 sec_8k acceptances")
    # a resolved row's own t0 never seeds the expectation: the event's AMC/BMO class, else the AMC default
    own = ev.iloc[[2]]
    assert expected_t0_for(own, "NVDA", d, before=before, timing="BMO") == (ny("2026-08-26 07:00"),
                                                                            "calendar-flag default for BMO")
    assert expected_t0_for(own, "NVDA", d, before=before) == (ny("2026-08-26 16:05"), "calendar default (AMC)")
    assert expected_t0_for(own, "NVDA", d, "time-pre-market", before=before, timing="AMC") == (
        ny("2026-08-26 07:00"), "nasdaq flag 'time-pre-market' (BMO default)")
    # a manual override on the row wins over everything; a calendar flag on it beats the defaults
    manual = ev.assign(**{E.t0_source: ["sec_8k", "sec_8k", "manual", "sec_8k"]})
    assert expected_t0_for(manual, "NVDA", d, before=before) == (pd.Timestamp("2026-08-26 20:50", tz="UTC"),
                                                                 "events table: manual")
    flagged = own.assign(**{E.t0_source: "calendar_flag", E.t0: ny("2026-08-26 07:00"), E.flags: "upcoming"})
    assert expected_t0_for(flagged, "NVDA", d, before=before) == (ny("2026-08-26 07:00"), "events table: calendar_flag")


# ---- issuer release clock ------------------------------------------------------------------------
def test_release_clock_overrides_parse_zones_and_follow_dst(settings):
    """The shipped configs/release_clock_overrides.yaml parses; a clock converts through its zone
    (ASML 07:00 Amsterdam is 06:00 UTC in January and 05:00 UTC in July; Taipei has no DST);
    malformed entries and unknown zones are skipped."""
    clocks = ev_mod._load_release_clock_overrides(settings)
    assert clocks["ASML"] == ReleaseClock(time(7, 0), "Europe/Amsterdam")
    assert str(clocks["ASML"]) == "07:00 Europe/Amsterdam"
    assert clocks["TSM"] == ReleaseClock(time(14, 0), "Asia/Taipei")
    assert clocks["ASML"].t0_on(date(2026, 1, 28)) == pd.Timestamp("2026-01-28 06:00", tz="UTC")
    assert clocks["ASML"].t0_on(date(2026, 7, 15)) == pd.Timestamp("2026-07-15 05:00", tz="UTC")
    assert clocks["ASML"].t0_on(pd.Timestamp("2026-10-14")) == pd.Timestamp("2026-10-14 05:00", tz="UTC")
    assert clocks["TSM"].t0_on(date(2026, 1, 15)) == pd.Timestamp("2026-01-15 06:00", tz="UTC")
    assert clocks["TSM"].t0_on(date(2026, 7, 16)) == pd.Timestamp("2026-07-16 06:00", tz="UTC")
    assert release_clock_for(settings, "tsm") == clocks["TSM"] and release_clock_for(settings, "NVDA") is None
    settings.configs_dir = settings.data_dir / "configs"
    settings.configs_dir.mkdir()
    assert ev_mod._load_release_clock_overrides(settings) == {}  # no file
    (settings.configs_dir / "release_clock_overrides.yaml").write_text(
        'asml: "07:00 Europe/Amsterdam"\nBAD1: "7am Europe/Amsterdam"\nBAD2: "07:00 Mars/Olympus"\n'
        'BAD3: 700\nBAD4: "07:00"\n')
    assert ev_mod._load_release_clock_overrides(settings) == {"ASML": ReleaseClock(time(7, 0), "Europe/Amsterdam")}


def test_resolver_issuer_clock_beats_detection_and_flags_but_not_an_8k(settings):
    clock = release_clock_for(settings, "ASML")
    day = pd.Timestamp("2026-01-28")
    # the proxy has no overnight bars: the 04:00 ET first-bar spike is the artefact the clock beats
    bars = synthetic_bars(["2026-01-27", "2026-01-28"], spike_ny="04:00", spike_day="2026-01-28")
    r = resolve_release_time(report_date_ny=day, sec_filings=tsm_filings(), intraday=bars,
                             calendar_flag="pre-market", issuer_clock=clock)
    assert r.t0 == pd.Timestamp("2026-01-28 06:00", tz="UTC") and r.source == "issuer_clock"
    assert r.confidence == CONFIDENCE["issuer_clock"] == 0.7 and r.flags == () and r.t0_lag_s is None
    assert r.detail.startswith("issuer release clock 07:00 Europe/Amsterdam -> 2026-01-28T06:00:00Z")
    assert "detection 2026-01-28T09:00:00Z on the first bar of the session not used" in r.detail
    # July: the same clock is an hour earlier in UTC; no bars and no flag still give the clock
    r = resolve_release_time(report_date_ny=pd.Timestamp("2026-07-15"), sec_filings=None, intraday=None,
                             calendar_flag=None, issuer_clock=clock)
    assert r.t0 == pd.Timestamp("2026-07-15 05:00", tz="UTC")
    assert r.detail == "issuer release clock 07:00 Europe/Amsterdam -> 2026-07-15T05:00:00Z"
    # an 8-K acceptance on the day outranks the clock; a manual override outranks both
    r = resolve_release_time(report_date_ny=day, sec_filings=filings_at("2026-01-28T06:02:00.000Z"),
                             intraday=None, calendar_flag=None, issuer_clock=clock)
    assert r.source == "sec_8k" and r.t0 == pd.Timestamp("2026-01-28 06:02", tz="UTC")
    manual = pd.Timestamp("2026-01-28 06:01", tz="UTC")
    r = resolve_release_time(report_date_ny=day, sec_filings=None, intraday=None, calendar_flag=None,
                             manual=manual, issuer_clock=clock)
    assert r.source == "manual" and r.t0 == manual


def test_issuer_clock_drives_the_builder_and_the_upcoming_schedule(settings, fake):
    """TSM (6-K filer: no 8-K; no bars served): every event gets t0 = 14:00 Asia/Taipei = 06:00
    UTC on its report date from the shipped configs/release_clock_overrides.yaml; `freedom
    upcoming` reads the same file, and the live stratum key is expected_issuer_clock."""
    from freedom.live import expected_t0_source_key

    write_universe(settings)

    def cal_row(sym: str, day: str, **over) -> dict:
        row = {"symbol": sym, "date": day, "epsActual": None, "epsEstimated": 2.9,
               "revenueActual": None, "revenueEstimated": 3.0e10, "lastUpdated": "2026-09-02"}
        row.update(over)
        return row

    fake.earnings["TSM"] = [cal_row("TSM", "2026-04-16", epsActual=2.47), cal_row("TSM", "2026-09-11"),
                            cal_row("TSM", "2026-10-15")]
    fake.nasdaq["2026-04-16"] = [nasdaq_row("TSM", time_flag="time-pre-market")]
    df = build_events(settings, underlyings=["TSM"], since=pd.Timestamp("2026-04-01")).set_index(E.report_date_ny)
    past, upcoming = df.loc[date(2026, 4, 16)], df.loc[date(2026, 9, 11)]
    for row, day in ((past, "2026-04-16"), (upcoming, "2026-09-11")):
        assert row[E.t0] == pd.Timestamp(f"{day} 06:00", tz="UTC") and row[E.t0_source] == "issuer_clock"
        assert row[E.t0_confidence] == 0.7 and row[E.timing] == "BMO"
        assert "issuer_clock" in row[E.sources_used].split(";")
        assert row["t0_detail"].startswith("issuer release clock 14:00 Asia/Taipei -> ")
        flags = row[E.flags].split(";")
        assert "timing_from_history" not in flags and "timing_unknown" not in flags
    assert "upcoming" in upcoming[E.flags].split(";") and "upcoming" not in past[E.flags].split(";")
    assert past[E.n_estimates] == 32  # the Nasdaq row is still the consensus cross-check
    # configs/t0_overrides.yaml still pins Oct 15 2026 by hand, and a manual override outranks the clock
    pinned = df.loc[date(2026, 10, 15)]
    assert pinned[E.t0_source] == "manual" and pinned[E.t0] == pd.Timestamp("2026-10-15 06:00", tz="UTC")

    fake.calendar = [cal_row("TSM", "2026-09-11")]
    up = upcoming_events(settings, days=14).set_index(E.underlying)
    assert up.loc["TSM", E.event_id] == upcoming[E.event_id]
    assert up.loc["TSM", "expected_t0"] == pd.Timestamp("2026-09-11 06:00", tz="UTC")
    assert up.loc["TSM", "expected_t0_source"] == "issuer release clock 14:00 Asia/Taipei (configs/release_clock_overrides.yaml)"
    assert expected_t0_source_key(up.loc["TSM", "expected_t0_source"]) == "expected_issuer_clock"
    # a table row resolved by the clock seeds the expectation too, under its own label
    table = load_events(settings)
    assert expected_t0_for(table, "TSM", date(2026, 9, 11)) == (pd.Timestamp("2026-09-11 06:00", tz="UTC"),
                                                                "events table: issuer_clock")
    assert expected_t0_source_key("events table: issuer_clock") == "expected_issuer_clock"


def test_expected_t0_issuer_clock_ranks_below_the_median_and_above_the_flags():
    clock = ReleaseClock(time(14, 0), "Asia/Taipei")
    d = date(2026, 9, 11)
    hit = (pd.Timestamp("2026-09-11 06:00", tz="UTC"),
           "issuer release clock 14:00 Asia/Taipei (configs/release_clock_overrides.yaml)")
    # the clock beats the Nasdaq flag and a table calendar flag ...
    assert expected_t0_for(None, "TSM", d, "time-pre-market", issuer_clock=clock) == hit
    flagged = pd.DataFrame([_sec_8k_row("TSM:2026-06", "2026-09-11", "2026-09-11 11:00:00",
                                        **{E.underlying: "TSM", E.t0_source: "calendar_flag", E.flags: "upcoming"})])
    assert expected_t0_for(flagged, "TSM", d, issuer_clock=clock) == hit
    # ... and loses to a manual row and to the issuer's measured 8-K acceptance clock
    manual = flagged.assign(**{E.t0_source: "manual"})
    assert expected_t0_for(manual, "TSM", d, issuer_clock=clock) == (pd.Timestamp("2026-09-11 11:00", tz="UTC"),
                                                                     "events table: manual")
    acc = pd.DataFrame([_sec_8k_row("TSM:2026-03", "2026-04-16", "2026-04-16 12:00:00", **{E.underlying: "TSM"})])
    assert expected_t0_for(acc, "TSM", d, issuer_clock=clock) == (ny("2026-09-11 08:00"),
                                                                  "median of 1 sec_8k acceptances")


def test_upcoming_events_carry_the_events_table_id(settings, fake):
    write_universe(settings)

    def cal_row(sym: str, day: str) -> dict:
        return {"symbol": sym, "date": day, "epsActual": None, "epsEstimated": 2.47,
                "revenueActual": None, "revenueEstimated": 1.0e11, "lastUpdated": "2026-09-02"}

    fake.calendar = [cal_row("NVDA", "2026-11-18"), cal_row("TSM", "2026-09-11")]
    build_events(settings, underlyings=["NVDA"], since=pd.Timestamp("2026-01-01"))
    up = upcoming_events(settings, days=90)
    assert list(up.columns) == UPCOMING_COLUMNS and up.columns[0] == E.event_id
    by = up.set_index(E.underlying)
    # the table's projected fiscal quarter (off-calendar fiscal year), not the calendar quarter's NVDA:2026-09
    assert by.loc["NVDA", E.event_id] == "NVDA:2026-10" and pd.isna(by.loc["TSM", E.event_id])
    from freedom.live import with_event_ids

    assert with_event_ids(up)[E.event_id].tolist() == ["TSM:2026-06", "NVDA:2026-10"]


def test_archiver_records_a_consensus_summary_row(settings, fake, monkeypatch):
    hl = FakeHyperliquidInfo()
    monkeypatch.setattr(HttpClient, "post_json", lambda http, url, body, **kw: hl.post_json(url, body, **kw))
    now = pd.Timestamp("2026-08-29 00:30", tz="UTC")
    # without an event universe the archiver does not snapshot consensus (nothing to snapshot for)
    summary = archive_markets(settings, [XYZ_NVDA], ["1h"], now=now)
    assert "consensus" not in set(summary["interval"])
    pd.DataFrame({U.market: [XYZ_NVDA], U.dex: ["xyz"]}).to_parquet(settings.universe_path, index=False)
    summary = archive_markets(settings, [XYZ_NVDA], ["1h"], now=now)
    assert "consensus" not in set(summary["interval"])
    write_universe(settings)
    summary = archive_markets(settings, [XYZ_NVDA], ["1h"], now=now)
    row = summary[summary["interval"] == "consensus"].iloc[0]
    assert row["market"] == "consensus" and pd.isna(row["error"])
    assert row["rows_added"] == 868 and row["rows_total"] == 868
    assert row["first_t"] == now and row["last_t"] == now
    assert consensus_path(settings, date(2026, 8, 29)).exists()
    # a provider that cannot run is recorded, not fatal
    settings.fmp_api_key = None
    summary = archive_markets(settings, [XYZ_NVDA], ["1h"], now=now)
    row = summary[summary["interval"] == "consensus"].iloc[0]
    assert str(row["error"]).startswith("ProviderUnavailable") and row["rows_added"] == 0
    assert row["rows_total"] == 868



def test_detection_never_moves_an_after_close_filing_into_the_session():
    """MU 2026-06-24: 8-K accepted 20:02:01Z; the 15:59 ET closing-auction bar spikes."""
    import numpy as np

    from freedom.events import resolve_release_time
    from freedom.schemas import C

    day = pd.Timestamp("2026-06-24")
    ts_ = pd.date_range("2026-06-24 04:00", "2026-06-24 19:59", freq="1min", tz="America/New_York").tz_convert("UTC")
    bars = pd.DataFrame({C.t: ts_, C.t_end: ts_ + pd.Timedelta(minutes=1), C.open: 100.0, C.high: 100.5,
                         C.low: 99.5, C.close: 100.0, C.volume: 1000.0, C.n_trades: 10, C.interval: "1m",
                         C.market: "MU", C.source: "fmp_intraday"})
    ny = bars[C.t].dt.tz_convert("America/New_York")
    auction = (ny.dt.hour == 15) & (ny.dt.minute == 59)
    bars.loc[auction, [C.volume, C.close]] = [80000.0, 102.0]  # closing cross: big volume, 2 % print
    release = (ny.dt.hour == 16) & (ny.dt.minute == 2)
    bars.loc[release, [C.volume, C.close]] = [60000.0, 96.0]
    bars[C.close] = bars[C.close].astype(float)
    filings = pd.DataFrame({"form": ["8-K"], "items": ["2.02,9.01"],
                            "accepted": [pd.Timestamp("2026-06-24 20:02:01", tz="UTC")],
                            "accession": ["0000723125-26-000013"], "filing_date": [pd.Timestamp("2026-06-24", tz="UTC")]})
    r = resolve_release_time(report_date_ny=day, sec_filings=filings, intraday=bars, calendar_flag=None)
    assert r.source == "sec_8k"
    assert r.t0 >= pd.Timestamp("2026-06-24 20:00:00", tz="UTC"), r.detail
    assert "closing auction" in r.detail or r.t0 == pd.Timestamp("2026-06-24 20:02:00", tz="UTC")
    assert not np.isnan(r.confidence)



def test_merge_existing_keeps_rows_of_an_older_additive_schema(settings):
    """An events.parquet written before a column was added is merged, not replaced."""
    from freedom.events import EVENT_COLUMNS, _merge_existing, _write_events

    old_cols = [c for c in EVENT_COLUMNS if c != E.ca_ex_date]
    old = pd.DataFrame([{c: None for c in old_cols}]).astype(object)
    old.loc[0, E.event_id] = "TSLA:2026-06"
    old.loc[0, E.underlying] = "TSLA"
    old.loc[0, E.pending] = False
    old.loc[0, E.flags] = ""
    old.loc[0, E.t0] = pd.Timestamp("2026-07-22 20:35", tz="UTC")
    old.loc[0, E.report_date_ny] = pd.Timestamp("2026-07-22").date()
    settings.ensure_dirs()
    _write_events(settings, old[old_cols])
    new = pd.DataFrame([{c: None for c in EVENT_COLUMNS}]).astype(object)
    new.loc[0, E.event_id] = "NVDA:2026-07"
    new.loc[0, E.underlying] = "NVDA"
    new.loc[0, E.pending] = False
    new.loc[0, E.flags] = ""
    new.loc[0, E.t0] = pd.Timestamp("2026-08-26 20:20", tz="UTC")
    new.loc[0, E.report_date_ny] = pd.Timestamp("2026-08-26").date()
    merged = _merge_existing(settings, new, processed={"NVDA"}, since=None)
    assert set(merged[E.event_id]) == {"TSLA:2026-06", "NVDA:2026-07"}
    assert E.ca_ex_date in merged.columns


def test_report_date_override_moves_the_event_and_the_upcoming_schedule(settings, fake):
    """configs/report_date_overrides.yaml corrects the vendor's day before the 8-K search, the
    Nasdaq lookup, the event id and the upcoming schedule; the vendor's own row on the corrected
    day does not make a second event; a malformed entry is ignored."""
    from freedom.events import _load_report_date_overrides, corrected_report_date

    write_universe(settings)
    settings.configs_dir = settings.data_dir / "configs"
    settings.configs_dir.mkdir()
    (settings.configs_dir / "report_date_overrides.yaml").write_text(
        'tsm:2026-09-11: 2026-09-14   # issuer moved the date\n"BAD": "2026-09-14"\n"TSM:nonsense": "2026-09-14"\n')
    over = _load_report_date_overrides(settings)
    assert over == {"TSM:2026-09-11": date(2026, 9, 14)}
    assert corrected_report_date(over, "TSM", date(2026, 9, 11)) == (date(2026, 9, 14), True)
    assert corrected_report_date(over, "TSM", date(2026, 9, 12)) == (date(2026, 9, 12), False)

    def cal_row(sym: str, day: str) -> dict:
        return {"symbol": sym, "date": day, "epsActual": None, "epsEstimated": 1.5,
                "revenueActual": None, "revenueEstimated": 9.0e10, "lastUpdated": "2026-09-02"}

    fake.calendar = [cal_row("TSM", "2026-09-11")]
    fake.nasdaq["2026-09-14"] = [nasdaq_row("TSM", time_flag="time-pre-market")]
    up = upcoming_events(settings, days=14).set_index(E.underlying)
    assert up.loc["TSM", E.report_date_ny] == date(2026, 9, 14)
    assert up.loc["TSM", "expected_t0"] == ny("2026-09-14 07:00")  # the Nasdaq flag of the corrected day
    assert up.loc["TSM", "expected_t0_source"] == ("nasdaq flag 'time-pre-market' (BMO default); "
                                                   "report date overridden (vendor said 2026-09-11)")
    # the events table: one row on the corrected day, flagged, even when the vendor lists both days
    fake.earnings["TSM"] = [cal_row("TSM", "2026-09-11"), cal_row("TSM", "2026-09-14")]
    build_events(settings, underlyings=["TSM"], since=pd.Timestamp("2026-09-01"))
    table = load_events(settings)
    tsm = table[table[E.underlying] == "TSM"]
    assert len(tsm) == 1 and tsm.iloc[0][E.report_date_ny] == date(2026, 9, 14)
    assert "report_date_override" in tsm.iloc[0][E.flags].split(";")
    up = upcoming_events(settings, days=14).set_index(E.underlying)
    assert up.loc["TSM", E.event_id] == tsm.iloc[0][E.event_id]  # the upcoming row finds its table row


def test_t0_override_reaches_the_upcoming_schedule_without_a_table_rebuild(settings, fake):
    """A foreign issuer releasing overnight: the t0 override in configs/t0_overrides.yaml sets the
    upcoming row's expected_t0 directly (keyed by UNDERLYING:date, or by the table's event id),
    and the live stratum key reads it as a manual schedule."""
    from freedom.live import expected_t0_source_key

    write_universe(settings)
    settings.configs_dir = settings.data_dir / "configs"
    settings.configs_dir.mkdir()

    def cal_row(sym: str, day: str) -> dict:
        return {"symbol": sym, "date": day, "epsActual": None, "epsEstimated": 1.5,
                "revenueActual": None, "revenueEstimated": 9.0e10, "lastUpdated": "2026-09-02"}

    fake.calendar = [cal_row("TSM", "2026-09-11"), cal_row("AAPL", "2026-09-15")]
    (settings.configs_dir / "t0_overrides.yaml").write_text('"tsm:2026-09-11": "2026-09-11T06:00:00Z"\n')
    up = upcoming_events(settings, days=14).set_index(E.underlying)
    assert up.loc["TSM", "expected_t0"] == pd.Timestamp("2026-09-11 06:00", tz="UTC")
    assert up.loc["TSM", "expected_t0_source"] == "manual override (configs/t0_overrides.yaml)"
    assert expected_t0_source_key(up.loc["TSM", "expected_t0_source"]) == "expected_manual"
    assert up.loc["AAPL", "expected_t0"] == ny("2026-09-15 16:05")  # untouched
    # keyed by the table's event id once the row exists, and composed with a report-date override
    (settings.configs_dir / "report_date_overrides.yaml").write_text("TSM:2026-09-11: 2026-09-14\n")
    fake.earnings["TSM"] = [cal_row("TSM", "2026-09-11")]
    build_events(settings, underlyings=["TSM"], since=pd.Timestamp("2026-09-01"))
    eid = load_events(settings).set_index(E.underlying).loc["TSM", E.event_id]
    (settings.configs_dir / "t0_overrides.yaml").write_text(f'"{eid}": "2026-09-14T06:00:00Z"\n')
    up = upcoming_events(settings, days=14).set_index(E.underlying)
    assert up.loc["TSM", E.event_id] == eid and up.loc["TSM", E.report_date_ny] == date(2026, 9, 14)
    assert up.loc["TSM", "expected_t0"] == pd.Timestamp("2026-09-14 06:00", tz="UTC")
    assert up.loc["TSM", "expected_t0_source"].startswith("manual override") and "vendor said 2026-09-11" in up.loc["TSM", "expected_t0_source"]
