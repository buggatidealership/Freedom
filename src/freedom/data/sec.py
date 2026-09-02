"""SEC EDGAR client (no key; identify yourself with a User-Agent). Fair-access limit 10 req/s.

Measured server facts (2026-09-02, see docs/data-sources.md):

* ``https://www.sec.gov/files/company_tickers.json`` is a dict keyed by row number:
  ``{"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}, ...}``.
* ``https://data.sec.gov/submissions/CIK##########.json``: ``filings.recent`` is *columnar*
  (parallel lists per field, at most ~1000 filings); ``filings.files`` lists older pages, each
  served at ``https://data.sec.gov/submissions/<name>`` as the same columnar dict at top level.
  ``acceptanceDateTime`` is an ISO instant with a ``Z`` suffix (``2026-08-26T20:21:19.000Z``);
  8-K ``items`` is a comma-separated string (``"2.02,9.01"``); 6-K rows have no items.
* ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``:
  ``facts.us-gaap.EarningsPerShareDiluted.units["USD/shares"]`` rows carry ``start, end, val,
  accn, fy, fp, form, filed`` (+ ``frame`` on the canonical row). The same ``fp`` appears twice
  in a 10-Q: the quarter and the fiscal year-to-date, distinguishable only by ``start``.
* The filing index ``https://www.sec.gov/Archives/edgar/data/<cik>/<accession-no-dashes>/
  <accession>-index.htm`` has a "Document Format Files" table whose *Type* column names each
  exhibit (``EX-99.1``); the exhibit itself is served with an SGML ``<DOCUMENT>...<TEXT>``
  wrapper around the HTML.

Timestamps: ``accepted`` is the exact UTC instant. Calendar dates (``filing_date``,
``period_end``, ``period_start``, ``filed``) are EDGAR dates represented as tz-aware UTC
*midnight* Timestamps so that every datetime column leaving this client is UTC; read them with
``edgar_date`` (``.dt.date``), never by converting to another zone, which shifts the day.

Errors: EDGAR answers 404 for a document that does not exist (``press_release_text`` returns
None and remembers the miss for ``TTL_MISSING``) and 403 for policy blocks (rejected User-Agent,
request-rate threshold). A 403 raises ``ProviderUnavailable`` so the run aborts instead of
reporting "no press release" for every filing.
"""

from __future__ import annotations

import re
import threading
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from ..config import Settings
from ..data.base import (
    DiskCache,
    HttpClient,
    ProviderUnavailable,
    TokenBucket,
    _is_retryable,
    cache_key,
)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/"

DAY = 24 * 3600
TTL_TICKERS = 7 * DAY
TTL_SUBMISSIONS = 1 * DAY
TTL_FACTS = 7 * DAY
TTL_FOREVER = 100 * 365 * DAY  # filed documents never change; HttpClient treats None as "no cache"
TTL_MISSING = 1 * DAY  # a 404 is re-checked daily: a just-accepted filing can lag on the Archives

EARNINGS_ITEM = "2.02"  # 8-K item: Results of Operations and Financial Condition
PRESS_RELEASE_EXHIBIT = "EX-99.1"

TICKER_COLUMNS = ["ticker", "cik", "title"]
SUBMISSION_COLUMNS = [
    "accession", "form", "filing_date", "accepted", "items", "primary_doc", "description",
]
EPS_COLUMNS = ["period_end", "value", "fp", "form", "filed"]
EPS_EXTRA_COLUMNS = ["period_start", "fy", "accession", "frame"]

_ACCESSION_RE = re.compile(r"^(\d{10})-?(\d{2})-?(\d{6})$")


# ---- plumbing ----------------------------------------------------------------------------------
class _SecLimiter(TokenBucket):
    """Token bucket for the sustained rate plus a minimum spacing between requests, so that a
    burst of cache-miss requests never exceeds SEC's per-second fair-access limit."""

    def __init__(self, requests_per_second: int):
        rps = max(int(requests_per_second), 1)
        super().__init__(rps * 60)
        self.min_interval = 1.0 / rps
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self, weight: float = 1.0) -> None:
        super().acquire(weight)
        with self._lock:
            now = time.monotonic()
            wait = self._last + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last = now


def _policy_block(url: str, exc: httpx.HTTPStatusError) -> ProviderUnavailable | None:
    """EDGAR answers 403 for policy blocks ("Undeclared Automated Tool" when the User-Agent is
    rejected, "Request Rate Threshold Exceeded" when throttled) and never for a missing document
    (that is a 404). Treating a 403 as "no document" would silently empty the whole run."""
    if exc.response.status_code != 403:
        return None
    reason = " ".join(html_to_text(exc.response.text).split())[:200] or "(empty body)"
    return ProviderUnavailable(
        f"sec: EDGAR refused {url} (HTTP 403: {reason}). SEC's fair-access policy requires a "
        "User-Agent that identifies you: set FREEDOM_SEC_USER_AGENT to "
        "'Company Name contact@example.com'; if the message mentions the request rate, lower "
        "FREEDOM_SEC_REQUESTS_PER_SECOND or wait before rerunning."
    )


_MISSING_KEY = "missing_until"  # cached payload of a 404: {"missing_until": <epoch seconds>}


class _SecHttp(HttpClient):
    """HttpClient plus a text GET (filing indexes and exhibits are HTML, not JSON) that goes
    through the same cache, limiter and retry policy. Both GETs turn an EDGAR 403 into
    ProviderUnavailable (see _policy_block)."""

    def get_text(self, url: str, *, cache_ttl: int | None) -> str | None:
        """Body of a document, or None when EDGAR has no such document (HTTP 404).

        A 404 is remembered for TTL_MISSING so that repeated lookups of a filing without an
        index page or exhibit do not re-hit EDGAR. Any other error propagates."""
        key = cache_key(self.provider, f"GET-TEXT {url}", None)
        if cache_ttl is not None:
            hit = self.cache.get(self.provider, key, cache_ttl)
            if isinstance(hit, str):
                return hit
            if isinstance(hit, dict) and hit.get(_MISSING_KEY, 0) > time.time():
                return None
        if self.limiter is not None:
            self.limiter.acquire(1.0)
        if self.budget is not None:
            self.budget.consume(1)
        try:
            text = self._request_text(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                self.cache.set(self.provider, key, {_MISSING_KEY: time.time() + TTL_MISSING})
                return None
            blocked = _policy_block(url, exc)
            if blocked is not None:
                raise blocked from exc
            raise
        self.cache.set(self.provider, key, text)
        return text

    def _request(self, method: str, url: str, *, params=None, json_body=None, headers=None) -> Any:
        try:
            return super()._request(method, url, params=params, json_body=json_body, headers=headers)
        except httpx.HTTPStatusError as exc:
            blocked = _policy_block(url, exc)
            if blocked is not None:
                raise blocked from exc
            raise

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _request_text(self, url: str) -> str:
        r = self._client.get(url)
        r.raise_for_status()
        return r.text


def normalise_accession(accession: str) -> str:
    """Return the dashed form ``0001045810-26-000073`` from either dashed or 18-digit input."""
    m = _ACCESSION_RE.match(str(accession).strip())
    if not m:
        raise ValueError(f"not an EDGAR accession number: {accession!r}")
    return "-".join(m.groups())


def split_items(items: str | None) -> list[str]:
    return [x.strip() for x in str(items or "").split(",") if x.strip()]


def edgar_date(values: pd.Series) -> pd.Series:
    """Calendar dates (``datetime.date``; NaT where missing) of an EDGAR date column:
    ``filing_date``, ``filed``, ``period_start`` or ``period_end``.

    These columns hold EDGAR dates as tz-aware UTC *midnight* Timestamps. This is the one place
    that turns them back into dates: ``tz_convert("America/New_York")`` would move midnight UTC
    to 20:00 the previous evening and shift the day. Derive ``E.report_date_ny`` from
    ``filing_date`` through this helper.
    """
    return values.dt.date


def _dates_utc(values: list[Any]) -> pd.DatetimeIndex:
    """ISO dates / instants -> tz-aware UTC (NaT for blanks or garbage)."""
    cleaned = [v if isinstance(v, str) and v.strip() else None for v in values]
    parsed = pd.to_datetime(cleaned, utc=True, format="ISO8601", errors="coerce")
    return parsed.as_unit("ns")  # pandas>=3 infers [s]/[us]; keep one resolution everywhere


def _submissions_page_to_frame(page: dict[str, Any] | None) -> pd.DataFrame:
    page = page or {}
    accessions = list(page.get("accessionNumber") or [])
    n = len(accessions)

    def col(name: str) -> list[Any]:
        v = list(page.get(name) or [])
        return (v + [""] * (n - len(v)))[:n]

    return pd.DataFrame(
        {
            "accession": accessions,
            "form": [str(x or "") for x in col("form")],
            "filing_date": _dates_utc(col("filingDate")),
            "accepted": _dates_utc(col("acceptanceDateTime")),
            "items": [str(x or "") for x in col("items")],
            "primary_doc": [str(x or "") for x in col("primaryDocument")],
            "description": [str(x or "") for x in col("primaryDocDescription")],
        },
        columns=SUBMISSION_COLUMNS,
    )


# ---- HTML helpers (stdlib only) ----------------------------------------------------------------
class _IndexTableParser(HTMLParser):
    """Rows of the filing-index tables as (cell texts, first href in the row)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[list[str], str | None]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row, self._href = [], None
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "a" and self._cell is not None and self._href is None:
            self._href = dict(attrs).get("href") or None

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append((self._row, self._href))
            self._row, self._href = None, None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def find_exhibit_path(index_html: str, exhibit_type: str = PRESS_RELEASE_EXHIBIT) -> str | None:
    """Path (``/Archives/...``) of the document typed `exhibit_type` in a filing index page.

    Exact type match first (``EX-99.1``); otherwise the first ``EX-99*`` document, because some
    filers label the press release ``EX-99`` or ``EX-99.01``. None when there is no such row.
    """
    parser = _IndexTableParser()
    parser.feed(index_html)
    parser.close()
    want = exhibit_type.upper()
    family = want.split(".")[0]
    fallback: str | None = None
    for cells, href in parser.rows:
        if not href or len(cells) < 4:
            continue
        typ = cells[3].upper()  # Seq | Description | Document | Type | Size
        path = href.split("/ix?doc=", 1)[1] if "/ix?doc=" in href else href
        if typ == want:
            return path
        if fallback is None and typ.startswith(family):
            fallback = path
    return fallback


# paragraph-level containers: a line break on open and on close (so a blank line separates them)
_PARAGRAPH_TAGS = {
    "p", "div", "table", "hr", "ul", "ol", "blockquote", "pre", "dl", "section", "article",
    "header", "footer", "center", "h1", "h2", "h3", "h4", "h5", "h6",
}
# line-level elements: a line break on open only, so consecutive rows/items stay adjacent
_LINE_TAGS = {"br", "tr", "li", "dt", "dd"}
_SKIP_TAGS = {"script", "style", "head", "title"}
_CELL_TAGS = {"td", "th"}
_BULLET_RE = re.compile(r"^([•·▪●■–—])(?=\S)")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _PARAGRAPH_TAGS or tag in _LINE_TAGS:
            self.parts.append("\n")
        elif tag in _CELL_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _PARAGRAPH_TAGS:
            self.parts.append("\n")
        elif tag in _CELL_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    """Plain text of an EDGAR HTML document: SGML wrapper removed, scripts/styles dropped,
    entities decoded, block elements on their own lines, whitespace collapsed."""
    m = re.search(r"<TEXT>", raw, flags=re.IGNORECASE)
    if m:
        raw = raw[m.end():]
        end = raw.lower().rfind("</text>")
        if end >= 0:
            raw = raw[:end]
    parser = _TextExtractor()
    parser.feed(raw)
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    # bullets and their text usually sit in adjacent inline <font> runs: "•Revenue" -> "• Revenue"
    lines = [_BULLET_RE.sub(r"\1 ", ln) for ln in lines]
    out: list[str] = []
    blank = False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


# ---- client ------------------------------------------------------------------------------------
class SECClient:
    def __init__(self, settings: Settings, cache: DiskCache | None = None):
        self.settings = settings
        self.http = _SecHttp(
            provider="sec",
            cache=cache or DiskCache(settings.cache_dir),
            limiter=_SecLimiter(settings.sec_requests_per_second),
            timeout=settings.http_timeout_seconds,
            default_headers={"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
        )

    # -- tickers -------------------------------------------------------------------------------
    def ticker_map(self) -> pd.DataFrame:
        """company_tickers.json as columns ticker, cik (int), title."""
        raw = self.http.get_json(TICKERS_URL, cache_ttl=TTL_TICKERS)
        rows = list(raw.values()) if isinstance(raw, dict) else list(raw or [])
        df = pd.DataFrame(
            {
                "ticker": [str(r.get("ticker") or "").strip().upper() for r in rows],
                "cik": [int(r.get("cik_str")) for r in rows],
                "title": [str(r.get("title") or "") for r in rows],
            },
            columns=TICKER_COLUMNS,
        )
        df = df[df["ticker"] != ""].drop_duplicates("ticker", keep="first")
        df["cik"] = df["cik"].astype("int64")
        return df.reset_index(drop=True)

    # -- filings -------------------------------------------------------------------------------
    def submissions(self, cik: int, *, include_older: bool = True) -> pd.DataFrame:
        """All filings for a CIK (recent + paginated older files): columns accession, form,
        filing_date, accepted (UTC), items (str), primary_doc, description.

        Sorted by accepted ascending; duplicate accessions (recent wins) dropped."""
        cik = int(cik)
        root = self.http.get_json(f"{SUBMISSIONS_URL}CIK{cik:010d}.json", cache_ttl=TTL_SUBMISSIONS)
        filings = (root or {}).get("filings") or {}
        pages: list[dict[str, Any] | None] = [filings.get("recent")]
        if include_older:
            for entry in filings.get("files") or []:
                name = (entry or {}).get("name")
                if name:
                    pages.append(self.http.get_json(f"{SUBMISSIONS_URL}{name}", cache_ttl=TTL_SUBMISSIONS))
        df = pd.concat([_submissions_page_to_frame(p) for p in pages], ignore_index=True)
        df = df.drop_duplicates("accession", keep="first")
        df = df.sort_values(["accepted", "accession"], kind="mergesort", na_position="last")
        return df.reset_index(drop=True)

    def earnings_filings(self, cik: int) -> pd.DataFrame:
        """8-K rows whose items include 2.02, plus 6-K rows, sorted by accepted.

        Amendments (8-K/A, 6-K/A) are excluded: their acceptance time is not a release time."""
        df = self.submissions(cik)
        has_item = df["items"].map(lambda s: EARNINGS_ITEM in split_items(s))
        keep = ((df["form"] == "8-K") & has_item) | (df["form"] == "6-K")
        out = df[keep].sort_values(["accepted", "accession"], kind="mergesort", na_position="last")
        return out.reset_index(drop=True)

    # -- XBRL facts ----------------------------------------------------------------------------
    def company_facts_eps(self, cik: int) -> pd.DataFrame:
        """Diluted EPS facts: columns period_end, value, fp, form, filed.

        Extra columns period_start, fy, accession, frame follow: a 10-Q reports both the quarter
        and the fiscal year-to-date under the same fp, so period_start is needed to tell them
        apart. Sorted by period_end, filed, period_start."""
        cik = int(cik)
        raw = self.http.get_json(f"{COMPANYFACTS_URL}CIK{cik:010d}.json", cache_ttl=TTL_FACTS)
        gaap = ((raw or {}).get("facts") or {}).get("us-gaap") or {}
        units = (gaap.get("EarningsPerShareDiluted") or {}).get("units") or {}
        rows = list(units.get("USD/shares") or [])
        df = pd.DataFrame(
            {
                "period_end": _dates_utc([r.get("end") for r in rows]),
                "value": pd.to_numeric([r.get("val") for r in rows], errors="coerce"),
                "fp": [r.get("fp") for r in rows],
                "form": [r.get("form") for r in rows],
                "filed": _dates_utc([r.get("filed") for r in rows]),
                "period_start": _dates_utc([r.get("start") for r in rows]),
                "fy": pd.array([r.get("fy") for r in rows], dtype="Int64"),
                "accession": [r.get("accn") for r in rows],
                "frame": [r.get("frame") for r in rows],
            },
            columns=EPS_COLUMNS + EPS_EXTRA_COLUMNS,
        )
        df["value"] = df["value"].astype("float64")
        df = df.drop_duplicates(["period_start", "period_end", "value", "accession"])
        df = df.sort_values(["period_end", "filed", "period_start"], kind="mergesort", na_position="last")
        return df.reset_index(drop=True)

    # -- documents -----------------------------------------------------------------------------
    def filing_folder_url(self, cik: int, accession: str) -> str:
        acc = normalise_accession(accession)
        return f"{ARCHIVES_URL}{int(cik)}/{acc.replace('-', '')}/"

    def press_release_text(self, cik: int, accession: str) -> str | None:
        """Plain text of the EX-99.1 exhibit for a filing, or None."""
        acc = normalise_accession(accession)
        index_url = f"{self.filing_folder_url(cik, acc)}{acc}-index.htm"
        index_html = self.http.get_text(index_url, cache_ttl=TTL_FOREVER)
        if index_html is None:
            return None
        path = find_exhibit_path(index_html, PRESS_RELEASE_EXHIBIT)
        if path is None:
            return None
        url = urljoin(index_url, path)  # absolute /Archives/... paths and bare file names alike
        if not url.lower().endswith((".htm", ".html", ".txt")):
            return None  # e.g. a PDF-only exhibit: nothing we can strip to text
        raw = self.http.get_text(url, cache_ttl=TTL_FOREVER)
        if raw is None:
            return None
        text = html_to_text(raw)
        return text or None
