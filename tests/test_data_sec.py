"""Offline tests for the SEC EDGAR client against the committed fixtures.

Every HTTP request is intercepted by respx (unmatched requests raise), so nothing here touches
the network. Fixtures were captured live on 2026-09-02; the trimmed ones keep the server's
exact shape with fewer rows.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import respx
from httpx import Response

from freedom.data import sec as secmod
from freedom.data.base import ProviderUnavailable
from freedom.data.sec import (
    ARCHIVES_URL,
    COMPANYFACTS_URL,
    SUBMISSIONS_URL,
    TICKERS_URL,
    TTL_FACTS,
    TTL_SUBMISSIONS,
    TTL_TICKERS,
    SECClient,
    edgar_date,
    find_exhibit_path,
    html_to_text,
    normalise_accession,
    split_items,
)

FIX = Path(__file__).parent / "fixtures" / "sec"
NVDA, TSM = 1045810, 1046179
NVDA_Q2 = "0001045810-26-000073"
NVDA_FOLDER = f"{ARCHIVES_URL}1045810/000104581026000073/"
T_NVDA_Q2 = pd.Timestamp("2026-08-26 20:21:19", tz="UTC")


def load_json(name: str):
    return json.loads((FIX / name).read_text())


def load_text(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def html(body: str) -> Response:
    return Response(200, text=body, headers={"content-type": "text/html; charset=UTF-8"})


@pytest.fixture
def edgar():
    """The measured EDGAR surface, served from fixtures. Routes are exposed for call counting."""
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as m:
        m.get(TICKERS_URL, name="tickers").respond(json=load_json("company_tickers_head.json"))
        m.get(f"{SUBMISSIONS_URL}CIK0001045810.json", name="nvda_recent").respond(
            json=load_json("submissions_CIK0001045810.json")
        )
        m.get(f"{SUBMISSIONS_URL}CIK0001045810-submissions-001.json", name="nvda_older").respond(
            json=load_json("submissions_CIK0001045810-001_trimmed.json")
        )
        m.get(f"{SUBMISSIONS_URL}CIK0001046179.json", name="tsm_recent").respond(
            json=load_json("submissions_CIK0001046179.json")
        )
        m.get(f"{SUBMISSIONS_URL}CIK0001046179-submissions-001.json", name="tsm_older").respond(
            json={"accessionNumber": [], "form": [], "filingDate": [], "acceptanceDateTime": []}
        )
        m.get(f"{COMPANYFACTS_URL}CIK0001045810.json", name="nvda_facts").respond(
            json=load_json("companyfacts_CIK0001045810_trimmed.json")
        )
        m.get(f"{NVDA_FOLDER}{NVDA_Q2}-index.htm", name="nvda_index").mock(
            return_value=html(load_text("0001045810-26-000073-index.htm"))
        )
        m.get(f"{NVDA_FOLDER}q2fy27pr.htm", name="nvda_ex991").mock(
            return_value=html(load_text("q2fy27pr_trimmed.htm"))
        )
        yield m


@pytest.fixture
def sec(settings) -> SECClient:
    return SECClient(settings)


def assert_utc(series: pd.Series) -> None:
    assert str(series.dtype) == "datetime64[ns, UTC]", series.dtype
    assert series.dt.tz is not None and str(series.dt.tz) == "UTC"


# ---- helpers ----------------------------------------------------------------------------------
def test_normalise_accession_accepts_both_forms():
    assert normalise_accession("0001045810-26-000073") == NVDA_Q2
    assert normalise_accession("000104581026000073") == NVDA_Q2
    with pytest.raises(ValueError):
        normalise_accession("nvda-20260826.htm")


def test_split_items_handles_blanks_and_garbage():
    assert split_items("2.02,9.01") == ["2.02", "9.01"]
    assert split_items("") == [] and split_items(None) == []
    assert split_items(" 2.02 , ,9.01") == ["2.02", "9.01"]
    assert "2.02" not in split_items("2023-05-31 17:00:00")  # a real oddity in the NVDA feed


# ---- tickers ----------------------------------------------------------------------------------
def test_ticker_map_columns_types_and_cache(edgar, sec):
    df = sec.ticker_map()
    assert list(df.columns) == ["ticker", "cik", "title"]
    assert str(df["cik"].dtype) == "int64"
    assert int(df.set_index("ticker").loc["NVDA", "cik"]) == NVDA
    assert df.set_index("ticker").loc["ASML", "cik"] == 937966  # the head fixture has 400 rows
    assert df["ticker"].is_unique
    assert edgar["tickers"].call_count == 1
    assert edgar["tickers"].calls.last.request.headers["user-agent"] == sec.settings.sec_user_agent
    sec.ticker_map()
    assert edgar["tickers"].call_count == 1, "second call must be served from the disk cache"


# ---- submissions ------------------------------------------------------------------------------
def test_submissions_concatenates_recent_and_older_pages(edgar, sec):
    df = sec.submissions(NVDA)
    assert list(df.columns) == [
        "accession", "form", "filing_date", "accepted", "items", "primary_doc", "description",
    ]
    n_recent = len(load_json("submissions_CIK0001045810.json")["filings"]["recent"]["accessionNumber"])
    n_older = len(load_json("submissions_CIK0001045810-001_trimmed.json")["accessionNumber"])
    assert len(df) == n_recent + n_older == 1002 + 93
    assert df["accession"].is_unique
    assert_utc(df["accepted"])
    assert_utc(df["filing_date"])
    assert df["accepted"].is_monotonic_increasing
    assert df["accepted"].notna().all()
    # the older page reaches back to 1998; the recent page ends 2026-08-31
    assert df["accepted"].iloc[0] == pd.Timestamp("1998-03-06 05:00:00", tz="UTC")
    assert df["accepted"].iloc[-1] == pd.Timestamp("2026-08-31 20:32:07", tz="UTC")
    row = df.set_index("accession").loc[NVDA_Q2]
    assert row["form"] == "8-K"
    assert row["accepted"] == T_NVDA_Q2
    assert row["filing_date"].date() == date(2026, 8, 26)
    assert row["items"] == "2.02,9.01"
    assert row["primary_doc"] == "nvda-20260826.htm"
    assert row["description"] == "8-K"
    assert df["items"].map(type).eq(str).all()
    assert edgar["nvda_recent"].call_count == 1 and edgar["nvda_older"].call_count == 1
    for route in ("nvda_recent", "nvda_older"):
        assert edgar[route].calls.last.request.headers["user-agent"] == sec.settings.sec_user_agent


def test_submissions_include_older_false_skips_paging(edgar, sec):
    df = sec.submissions(NVDA, include_older=False)
    assert len(df) == 1002
    assert edgar["nvda_older"].call_count == 0
    assert df["accepted"].is_monotonic_increasing


def test_submissions_dedups_overlap_between_pages_recent_wins(settings):
    cik = 999
    recent = {
        "accessionNumber": ["0000000999-26-000002", "0000000999-26-000001"],
        "filingDate": ["2026-08-02", "2026-08-01"],
        "acceptanceDateTime": ["2026-08-02T21:00:00.000Z", "2026-08-01T20:00:00.000Z"],
        "form": ["8-K", "8-K"],
        "items": ["2.02,9.01", "5.02"],
        "primaryDocument": ["b.htm", "a.htm"],
        "primaryDocDescription": ["8-K", "8-K"],
    }
    older = {
        "accessionNumber": ["0000000999-26-000001", "0000000999-25-000001"],
        "filingDate": ["2026-08-01", "2025-08-01"],
        "acceptanceDateTime": ["2026-08-01T20:00:00.000Z", ""],  # blank acceptance in old data
        "form": ["8-K", "8-K"],
        "items": ["5.02 stale", "2.02"],
        "primaryDocument": ["a-old.htm", "z.htm"],
        "primaryDocDescription": ["8-K", ""],
    }
    root = {"cik": "0000000999", "filings": {"recent": recent,
                                              "files": [{"name": "CIK0000000999-submissions-001.json"}]}}
    with respx.mock(assert_all_mocked=True) as m:
        m.get(f"{SUBMISSIONS_URL}CIK0000000999.json").respond(json=root)
        m.get(f"{SUBMISSIONS_URL}CIK0000000999-submissions-001.json").respond(json=older)
        df = SECClient(settings).submissions(cik)
    assert list(df["accession"]) == [
        "0000000999-26-000001", "0000000999-26-000002", "0000000999-25-000001",
    ]  # sorted by accepted, NaT last
    assert df.set_index("accession").loc["0000000999-26-000001", "primary_doc"] == "a.htm"
    assert pd.isna(df["accepted"].iloc[-1])
    assert_utc(df["accepted"])


def test_submissions_empty_recent_has_schema(settings):
    with respx.mock(assert_all_mocked=True) as m:
        m.get(f"{SUBMISSIONS_URL}CIK0000000001.json").respond(json={"filings": {"recent": {}, "files": []}})
        df = SECClient(settings).submissions(1)
    assert df.empty
    assert list(df.columns) == ["accession", "form", "filing_date", "accepted", "items",
                                "primary_doc", "description"]
    assert_utc(df["accepted"])


# ---- earnings filings -------------------------------------------------------------------------
def test_earnings_filings_nvda_8k_item_202(edgar, sec):
    df = sec.earnings_filings(NVDA)
    assert list(df.columns) == ["accession", "form", "filing_date", "accepted", "items",
                                "primary_doc", "description"]
    assert set(df["form"]) == {"8-K"}
    assert df["items"].map(lambda s: "2.02" in split_items(s)).all()
    assert df["accepted"].is_monotonic_increasing
    assert_utc(df["accepted"])
    # the live check target: Q2 FY27 release
    hit = df[df["accepted"] == T_NVDA_Q2]
    assert len(hit) == 1 and hit.iloc[0]["accession"] == NVDA_Q2
    assert "2.02" in split_items(hit.iloc[0]["items"])
    # expected count from both fixture pages
    recent = load_json("submissions_CIK0001045810.json")["filings"]["recent"]
    older = load_json("submissions_CIK0001045810-001_trimmed.json")
    expected = sum(
        1
        for page in (recent, older)
        for f, it in zip(page["form"], page["items"], strict=True)
        if f == "8-K" and "2.02" in split_items(it)
    )
    assert len(df) == expected
    # non-earnings 8-Ks and amendments are out
    assert "0001045810-26-000069" not in set(df["accession"])  # 1.01,2.03,7.01
    assert not df["form"].str.endswith("/A").any()
    # older page rows made it through
    assert (df["accepted"] < pd.Timestamp("2010-01-01", tz="UTC")).any()


def test_earnings_filings_tsm_returns_all_6k_rows(edgar, sec):
    df = sec.earnings_filings(TSM)
    recent = load_json("submissions_CIK0001046179.json")["filings"]["recent"]
    assert set(df["form"]) == {"6-K"}
    assert len(df) == recent["form"].count("6-K") == 712
    assert (df["items"] == "").all()
    assert df["accepted"].is_monotonic_increasing
    assert_utc(df["accepted"])
    row = df.set_index("accession").loc["0001046179-26-000541"]
    assert row["accepted"] == pd.Timestamp("2026-08-14 10:02:33", tz="UTC")
    assert row["primary_doc"] == "tsm-fsx20260814x6k.htm"


# ---- company facts ----------------------------------------------------------------------------
def test_company_facts_eps_columns_values_and_order(edgar, sec):
    df = sec.company_facts_eps(NVDA)
    assert list(df.columns)[:5] == ["period_end", "value", "fp", "form", "filed"]
    assert list(df.columns)[5:] == ["period_start", "fy", "accession", "frame"]
    assert_utc(df["period_end"])
    assert_utc(df["filed"])
    assert_utc(df["period_start"])
    assert str(df["value"].dtype) == "float64"
    assert df["period_end"].is_monotonic_increasing
    assert len(df) == 309
    q2 = df[df["period_end"] == pd.Timestamp("2026-07-26", tz="UTC")].set_index("period_start")
    quarter = q2.loc[pd.Timestamp("2026-04-27", tz="UTC")]
    ytd = q2.loc[pd.Timestamp("2026-01-26", tz="UTC")]
    assert quarter["value"] == 2.46 and ytd["value"] == 4.85
    assert quarter["fp"] == "Q2" and quarter["form"] == "10-Q"
    assert quarter["filed"].date() == date(2026, 8, 26)
    assert quarter["accession"] == "0001045810-26-000075" and quarter["frame"] == "CY2026Q2"
    assert int(quarter["fy"]) == 2027
    assert pd.isna(ytd["frame"])
    assert edgar["nvda_facts"].calls.last.request.headers["user-agent"] == sec.settings.sec_user_agent


def test_company_facts_eps_missing_concept_is_empty(settings):
    with respx.mock(assert_all_mocked=True) as m:
        m.get(f"{COMPANYFACTS_URL}CIK0000000001.json").respond(
            json={"cik": 1, "facts": {"us-gaap": {"Revenues": {"units": {"USD": []}}}}}
        )
        df = SECClient(settings).company_facts_eps(1)
    assert df.empty and list(df.columns)[:5] == ["period_end", "value", "fp", "form", "filed"]


# ---- press release ----------------------------------------------------------------------------
def test_press_release_text_fetches_ex991_and_strips_html(edgar, sec):
    text = sec.press_release_text(NVDA, NVDA_Q2)
    assert text is not None
    assert edgar["nvda_index"].call_count == 1
    assert edgar["nvda_ex991"].call_count == 1, "must fetch the EX-99.1 document, not the 8-K"
    assert edgar["nvda_ex991"].calls.last.request.headers["user-agent"] == sec.settings.sec_user_agent
    lines = text.split("\n")
    assert lines[0] == "NVIDIA Announces Financial Results for Second Quarter Fiscal 2027"
    assert "• Revenue of $96.2 billion, up 106% from a year ago" in text  # bullet + inline font
    assert "Q2 Fiscal 2027 Summary" in text
    assert "<" not in text and "font-family" not in text
    assert "q2fy27pr.htm" not in text and "<TYPE>" not in text  # SGML wrapper stripped
    assert "\xa0" not in text and "\n\n\n" not in text
    assert all(ln == ln.strip() for ln in lines)
    # the accession may also be given without dashes; both hit the cache now
    again = sec.press_release_text(NVDA, "000104581026000073")
    assert again == text
    assert edgar["nvda_index"].call_count == 1 and edgar["nvda_ex991"].call_count == 1


def test_press_release_text_none_when_no_exhibit_or_missing_filing(settings):
    index_without_exhibit = """
    <table class="tableFile"><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
    <tr><td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/1/000000000126000001/a.htm">a.htm</a></td><td>8-K</td><td>1</td></tr>
    </table>"""
    folder = f"{ARCHIVES_URL}1/000000000126000001/"
    with respx.mock(assert_all_mocked=True) as m:
        m.get(f"{folder}0000000001-26-000001-index.htm").mock(return_value=html(index_without_exhibit))
        m.get(f"{ARCHIVES_URL}1/000000000126000002/0000000001-26-000002-index.htm").respond(404)
        client = SECClient(settings)
        assert client.press_release_text(1, "0000000001-26-000001") is None
        assert client.press_release_text(1, "0000000001-26-000002") is None


def test_press_release_text_pdf_only_exhibit_is_none(settings):
    index_pdf = """
    <table class="tableFile">
    <tr><td>2</td><td>PRESS RELEASE</td><td><a href="/Archives/edgar/data/1/000000000126000003/pr.pdf">pr.pdf</a></td><td>EX-99.1</td><td>1</td></tr>
    </table>"""
    with respx.mock(assert_all_mocked=True) as m:
        m.get(f"{ARCHIVES_URL}1/000000000126000003/0000000001-26-000003-index.htm").mock(
            return_value=html(index_pdf)
        )
        assert SECClient(settings).press_release_text(1, "0000000001-26-000003") is None


def index_url(cik: int, accession: str) -> str:
    return f"{ARCHIVES_URL}{cik}/{accession.replace('-', '')}/{accession}-index.htm"


def test_press_release_text_404_is_remembered_and_rechecked_after_ttl(settings, monkeypatch):
    client = SECClient(settings)
    with respx.mock(assert_all_mocked=True) as m:
        route = m.get(index_url(1, "0000000001-26-000004")).respond(404)
        assert client.press_release_text(1, "0000000001-26-000004") is None
        assert client.press_release_text(1, "0000000001-26-000004") is None
        assert route.call_count == 1, "a genuine 404 is cached; do not re-hit EDGAR"
    monkeypatch.setattr(secmod, "TTL_MISSING", -1)  # a marker written from now on is already stale
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as m:
        route = m.get(index_url(1, "0000000001-26-000004")).respond(404)
        assert client.press_release_text(1, "0000000001-26-000004") is None
        assert route.call_count == 0, "the marker written under the default TTL is still fresh"
        route = m.get(index_url(1, "0000000001-26-000005")).respond(404)
        assert client.press_release_text(1, "0000000001-26-000005") is None
        assert client.press_release_text(1, "0000000001-26-000005") is None
        assert route.call_count == 2, "an expired marker is refetched"


FORBIDDEN = Response(
    403,
    text="<html><body><h1>Undeclared Automated Tool</h1><p>Request Rate Threshold Exceeded"
         "</p></body></html>",
    headers={"content-type": "text/html"},
)


def test_press_release_text_403_is_a_policy_block_not_a_missing_document(settings):
    """EDGAR uses 403 for rejected User-Agents and throttling, never for absent documents: a
    throttled run must abort, not report 'no press release' for every filing."""
    folder = f"{ARCHIVES_URL}1/000000000126000006/"
    index_html = f"""
    <table class="tableFile">
    <tr><td>2</td><td>PRESS RELEASE</td><td><a href="{folder}pr.htm">pr.htm</a></td><td>EX-99.1</td><td>1</td></tr>
    </table>"""
    with respx.mock(assert_all_mocked=True) as m:
        index = m.get(f"{folder}0000000001-26-000006-index.htm").mock(return_value=FORBIDDEN)
        client = SECClient(settings)
        with pytest.raises(ProviderUnavailable) as info:
            client.press_release_text(1, "0000000001-26-000006")
        msg = str(info.value)
        assert "403" in msg and "Undeclared Automated Tool" in msg
        assert "FREEDOM_SEC_USER_AGENT" in msg and "FREEDOM_SEC_REQUESTS_PER_SECOND" in msg
        assert index.call_count == 1, "403 is not retried"
        # nothing was cached for the blocked URL: a fixed User-Agent gets a fresh request
        index.mock(return_value=html(index_html))
        exhibit = m.get(f"{folder}pr.htm").mock(return_value=FORBIDDEN)
        with pytest.raises(ProviderUnavailable):
            client.press_release_text(1, "0000000001-26-000006")
        assert index.call_count == 2 and exhibit.call_count == 1
        exhibit.mock(return_value=html("<p>All good</p>"))
        assert client.press_release_text(1, "0000000001-26-000006") == "All good"


def test_json_endpoints_403_raise_provider_unavailable(settings):
    with respx.mock(assert_all_mocked=True) as m:
        m.get(f"{SUBMISSIONS_URL}CIK0000000001.json").mock(return_value=FORBIDDEN)
        with pytest.raises(ProviderUnavailable, match="FREEDOM_SEC_USER_AGENT"):
            SECClient(settings).submissions(1)


def test_press_release_text_resolves_relative_exhibit_href(settings):
    folder = f"{ARCHIVES_URL}1/000000000126000007/"
    index_rel = """
    <table class="tableFile">
    <tr><td>2</td><td>PRESS RELEASE</td><td><a href="pr.htm">pr.htm</a></td><td>EX-99.1</td><td>1</td></tr>
    </table>"""
    with respx.mock(assert_all_mocked=True) as m:
        m.get(f"{folder}0000000001-26-000007-index.htm").mock(return_value=html(index_rel))
        exhibit = m.get(f"{folder}pr.htm").mock(return_value=html("<p>Relative</p>"))
        assert SECClient(settings).press_release_text(1, "0000000001-26-000007") == "Relative"
        assert exhibit.call_count == 1


def test_find_exhibit_path_prefers_exact_type_then_ex99_family():
    idx = load_text("0001045810-26-000073-index.htm")
    assert find_exhibit_path(idx) == "/Archives/edgar/data/1045810/000104581026000073/q2fy27pr.htm"
    assert find_exhibit_path(idx, "EX-99.2") == (
        "/Archives/edgar/data/1045810/000104581026000073/q2fy27cfocommentary.htm"
    )
    # iXBRL viewer links are unwrapped to the raw document path
    assert find_exhibit_path(idx, "8-K") == "/Archives/edgar/data/1045810/000104581026000073/nvda-20260826.htm"
    assert find_exhibit_path(idx, "EX-42") is None
    family = """<table><tr><td>2</td><td>x</td><td><a href="/Archives/e/1/2/ex99.htm">ex99.htm</a></td><td>EX-99</td><td>1</td></tr></table>"""
    assert find_exhibit_path(family) == "/Archives/e/1/2/ex99.htm"


def test_html_to_text_units():
    raw = (
        "<DOCUMENT>\n<TYPE>EX-99.1\n<SEQUENCE>2\n<FILENAME>pr.htm\n<TEXT>\n"
        "<html><head><title>Document</title><style>p{color:red}</style></head><body>"
        "<script>alert(1)</script><p>Acme &amp; Co reports&nbsp;record&#8217;s</p>"
        "<table><tr><td>Revenue</td><td></td><td>$</td><td>1,234</td></tr>"
        "<tr><td>EPS</td><td>0.50</td></tr></table><div><br/><br/></div><p>  Ends  </p>"
        "</body></html>\n</TEXT>\n</DOCUMENT>\n"
    )
    assert html_to_text(raw) == "Acme & Co reports record’s\n\nRevenue $ 1,234\nEPS 0.50\n\nEnds"
    assert html_to_text("plain text\n\n\n\nmore") == "plain text\n\nmore"
    assert html_to_text("") == ""


# ---- calendar dates ---------------------------------------------------------------------------
def test_edgar_date_reads_utc_midnight_columns_as_calendar_days(edgar, sec):
    df = sec.submissions(NVDA, include_older=False).set_index("accession")
    dates = edgar_date(df["filing_date"])
    assert dates.loc[NVDA_Q2] == date(2026, 8, 26)
    assert dates.map(lambda d: isinstance(d, date)).all()
    # the trap the helper exists for: converting the zone first shifts the day
    shifted = df["filing_date"].dt.tz_convert("America/New_York").dt.date
    assert shifted.loc[NVDA_Q2] == date(2026, 8, 25)
    blanks = edgar_date(pd.Series(pd.to_datetime(["2026-08-26", None], utc=True)))
    assert blanks.iloc[0] == date(2026, 8, 26) and pd.isna(blanks.iloc[1])


# ---- cache policy -----------------------------------------------------------------------------
def test_cache_ttls_per_endpoint(edgar, sec, monkeypatch):
    seen: dict[str, int | None] = {}
    orig_json, orig_text = secmod._SecHttp.get_json, secmod._SecHttp.get_text

    def spy_json(self, url, params=None, *, cache_ttl, **kw):
        seen[url] = cache_ttl
        return orig_json(self, url, params, cache_ttl=cache_ttl, **kw)

    def spy_text(self, url, *, cache_ttl):
        seen[url] = cache_ttl
        return orig_text(self, url, cache_ttl=cache_ttl)

    monkeypatch.setattr(secmod._SecHttp, "get_json", spy_json)
    monkeypatch.setattr(secmod._SecHttp, "get_text", spy_text)
    sec.ticker_map()
    sec.submissions(NVDA)
    sec.company_facts_eps(NVDA)
    sec.press_release_text(NVDA, NVDA_Q2)
    day = 24 * 3600
    assert seen[TICKERS_URL] == TTL_TICKERS == 7 * day
    assert seen[f"{SUBMISSIONS_URL}CIK0001045810.json"] == TTL_SUBMISSIONS == 1 * day
    assert seen[f"{SUBMISSIONS_URL}CIK0001045810-submissions-001.json"] == TTL_SUBMISSIONS
    assert seen[f"{COMPANYFACTS_URL}CIK0001045810.json"] == TTL_FACTS == 7 * day
    for url in (f"{NVDA_FOLDER}{NVDA_Q2}-index.htm", f"{NVDA_FOLDER}q2fy27pr.htm"):
        assert seen[url] is not None and seen[url] >= 10 * 365 * day, "documents cache forever"
