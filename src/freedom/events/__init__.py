"""Earnings event table and release-time resolver. Output uses schemas.E columns.

Decisions from review (docs/design.md §2, §5):
* detection may only move an 8-K time EARLIER (within 15 min); a later detection never changes
  t0 and is recorded as `t0_lag_s`; 8-K confidence does not depend on detection;
* a detection on the very first bar of the extended session is flagged `detection_first_bar`
  and downgraded to calendar-flag confidence;
* calendar-flag defaults: AMC -> 16:05, BMO -> 07:00 America/New_York, confidence 0.5;
* event_id = f"{underlying}:{fiscal_period}" with fiscal_period = fiscal quarter-end month;
  when the period's own facts are not on file yet (a fresh or upcoming event) the month is
  projected from the issuer's latest SEC period end in whole quarters, so the id is the same
  before and after the 10-Q lands;
* vendor estimates for past events are `estimate_source = fmp_final` (not point-in-time),
  upcoming rows carry the vendor's current, not-yet-final value as `fmp_calendar`;
  archived consensus snapshots (data/archive/consensus/*.parquet) win when they exist and
  give `estimate_source = consensus_snapshot` with `estimate_snapshot_time`;
* budget exhaustion is a checkpoint: rows already resolved are written, the rest are marked
  `pending = True`, the command exits non-zero, and a rerun resumes from the cache. A row
  that an earlier run completed is never replaced by a pending one.

Implementation notes (what the code below relies on):

* Sources per event: FMP ``earnings_history`` is the list of events (past rows have
  ``eps_actual``; future rows are kept with ``flags += upcoming``). SEC submissions give the
  8-K item 2.02 acceptance instant (US filers) and companyfacts the fiscal period end. Alpha
  Vantage (key and budget permitting) gives ``report_time`` and ``fiscalDateEnding`` for
  foreign private issuers. The Nasdaq calendar for each FMP report date is a consensus /
  surprise cross-check and supplies ``n_estimates``.
* ``report_date_ny`` is the New York date of the 8-K acceptance when one is found (FMP's date
  is kept otherwise). The intraday window for detection is exactly the window
  ``targets.loaders`` derives from ``t0`` -- ``[report_date_ny - 1 day, report_date_ny + 2 days]``
  of 1-minute extended-hours bars -- so both modules share one cached FMP request per event.
* The volume baseline for detection is what that window can supply: the same clock minute
  (+/- 15 min, same session segment) on the prior sessions in the window plus the preceding
  bars of the same segment on the report day. Ten same-clock sessions are used when a caller
  passes a wider frame; the loaders window holds at most one.
* FMP 1-minute depth was measured to 2025-02 (docs/data-sources.md): events before
  ``INTRADAY_1MIN_FLOOR`` skip detection (``flags += no_intraday``) rather than spend a
  budgeted request on an empty answer every week.
* Nasdaq / Alpha Vantage rows are matched by nearest report date within +/-10 days; a matched
  pair whose dates differ by more than one day gets ``flags += date_conflict`` and confidence 0.
* Events with no timing source at all (no 8-K, no detection, no flag) get the AMC default with
  ``CONFIDENCE_UNKNOWN`` and ``flags += timing_unknown``; they stay in the table and are
  excluded from training by ``min_t0_confidence`` like every other calendar-flag row.
* An FMP error on one event's intraday request (a bad symbol, a transport failure) is not a
  checkpoint: the event is resolved without bars (``flags += intraday_error;no_intraday``)
  and the build goes on. Only ProviderUnavailable / BudgetExhausted stop it.
* Corporate actions (docs/design.md §2): the FMP splits calendar is fetched once per
  underlying (cached for a week); an ex-date inside ``[t0 - 60 d, t0 + horizon]`` sets
  ``flags += corporate_action`` and ``corporate_action_ex_date`` (the ex-date nearest t0, as
  00:00 New York in UTC). The targets NaN the headline label when it lies in
  ``[p0_time, t0 + horizon]``. A failed splits request is ``flags += splits_error``, not a
  checkpoint; budget exhaustion on it is one, like the intraday request.
* Optional manual overrides live in ``configs/t0_overrides.yaml`` as
  ``{"NVDA:2026-07": "2026-08-26T20:20:00Z"}`` (keys may also be ``"NVDA:2026-08-26"``).
* ``upcoming_events`` takes ``expected_t0`` from, in order: a manual override recorded on the
  matching upcoming row of events.parquet, the issuer's median sec_8k acceptance clock, the
  calendar-flag time of that upcoming row, the Nasdaq calendar's time flag, the AMC default.
"""

from __future__ import annotations

import calendar
import logging
import math
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yaml

from ..config import Settings
from ..data.base import BudgetExhausted, ProviderUnavailable, utcnow
from ..data.fmp import FMPError
from ..data.sec import split_items
from ..schemas import EVENT_KINDS, NY, SCHEMA_VERSION, UTC, C, E, Kind, T0Source, U
from ..timeutil import classify_timing, to_utc

log = logging.getLogger(__name__)

CONFIDENCE = {"manual": 1.0, "sec_8k": 0.95, "detected": 0.8, "calendar_flag": 0.5}
CONFIDENCE_UNKNOWN = 0.25  # no timing source at all: AMC default, flagged timing_unknown
DETECTION_WINDOW = pd.Timedelta(minutes=15)
DATE_MATCH_WINDOW_DAYS = 10  # Nasdaq / Alpha Vantage rows match the FMP row within this
DATE_CONFLICT_DAYS = 1  # a matched pair further apart than this is a date conflict
FISCAL_PERIOD_MAX_LAG_DAYS = 120  # SEC period end must be at most this far before the report
CALENDAR_DEFAULT_CLOCK = {"AMC": time(16, 5), "BMO": time(7, 0)}  # America/New_York
INTRADAY_1MIN_FLOOR = date(2025, 1, 1)  # FMP 1-minute bars measured back to 2025-02
CLOCK_WINDOW_MINUTES = 15  # same-clock baseline: +/- this many minutes on prior sessions
SAME_DAY_BASELINE_BARS = 60  # preceding same-segment bars of the report day used as baseline
MIN_BASELINE_BARS = 10  # a candidate bar without this many baseline bars cannot be judged
FLAG_DETECTION_FIRST_BAR = "detection_first_bar"
CONSENSUS_ITEM = "consensus"  # market/interval label of the archiver's consensus summary row

SNAPSHOT_COLUMNS: list[str] = [
    "snapshot_time", "symbol", "report_date_ny", "eps_estimate", "rev_estimate", "n_estimates",
    "vendor", "vendor_last_updated",
]
EXTRA_COLUMNS: list[str] = ["t0_acceptance", "t0_lag_s", "t0_detail"]
EVENT_COLUMNS: list[str] = [
    E.event_id, E.underlying, E.market, E.cik, E.kind, E.fiscal_period, E.fiscal_period_source,
    E.report_date_ny, E.t0, E.t0_confidence, E.t0_source, E.timing, E.eps_actual, E.eps_estimate,
    E.eps_surprise_pct, E.rev_actual, E.rev_estimate, E.rev_surprise_pct, E.n_estimates,
    E.estimate_source, E.estimate_snapshot_time, E.sources_used, E.has_perp_at_t0,
    E.listing_start, E.pending, E.flags, E.ca_ex_date, *EXTRA_COLUMNS,
]
UPCOMING_COLUMNS: list[str] = [
    E.underlying, E.market, E.kind, E.report_date_ny, "expected_t0", "expected_t0_source",
    E.eps_estimate, E.rev_estimate, E.n_estimates, E.estimate_source, E.estimate_snapshot_time,
]
DATETIME_COLUMNS = (E.t0, E.estimate_snapshot_time, E.listing_start, E.ca_ex_date, "t0_acceptance")
CORPORATE_ACTION_LOOKBACK = pd.Timedelta(days=60)  # design §2: ex-date in [t0 - 60 d, t0 + horizon]
FLAG_CORPORATE_ACTION = "corporate_action"

_AMC_FLAGS = frozenset({"amc", "post-market", "postmarket", "after-hours", "afterhours",
                        "after market close", "after-market", "post market"})
_BMO_FLAGS = frozenset({"bmo", "pre-market", "premarket", "before market open", "pre market",
                        "before-market"})
_SEG_PRE, _SEG_RTH, _SEG_AH = 0, 1, 2
_RTH_OPEN_MIN, _RTH_CLOSE_MIN = 9 * 60 + 30, 16 * 60


@dataclass(frozen=True)
class ResolvedT0:
    t0: pd.Timestamp  # UTC
    confidence: float  # 0..1
    source: str  # schemas.T0Source value
    detail: str = ""
    t0_lag_s: float | None = None  # acceptance - detected start, seconds (8-K events only)
    flags: tuple[str, ...] = ()


# ---- small helpers -----------------------------------------------------------------------------
def _as_date(value: Any) -> date:
    """Calendar date: tz-aware instants are read in New York, naive ones as given."""
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    t = pd.Timestamp(value)
    if t.tzinfo is not None:
        t = t.tz_convert(NY)
    return t.date()


def _isna(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _today_ny() -> date:
    return pd.Timestamp.now(tz=NY).date()


def _fmt_ts(ts: pd.Timestamp | None) -> str:
    return "?" if ts is None or _isna(ts) else to_utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def timing_from_flag(flag: Any) -> str | None:
    """'AMC' / 'BMO' from a vendor time-of-day flag ('post-market', 'pre-market', 'after-hours',
    Nasdaq's 'time-after-hours' / 'time-pre-market', ...); None for missing or uninformative
    flags such as 'time-not-supplied'."""
    if flag is None or _isna(flag):
        return None
    s = str(flag).strip().lower().removeprefix("time-")
    if s in _AMC_FLAGS:
        return "AMC"
    if s in _BMO_FLAGS:
        return "BMO"
    return None


def calendar_default_t0(report_date_ny: Any, timing: str) -> pd.Timestamp:
    """The default clock time for a calendar flag on the report date, as a UTC instant."""
    clock = CALENDAR_DEFAULT_CLOCK[timing]
    d = _as_date(report_date_ny)
    return to_utc(pd.Timestamp.combine(d, clock), assume_tz=NY)


def _surprise_pct(actual: Any, estimate: Any) -> float:
    if _isna(actual) or _isna(estimate) or float(estimate) == 0.0:
        return math.nan
    return (float(actual) - float(estimate)) / abs(float(estimate)) * 100.0


def _nearest_date(candidates: list[date], target: date,
                  window_days: int = DATE_MATCH_WINDOW_DAYS) -> tuple[date, int] | None:
    """(nearest candidate, signed day difference candidate - target) within the window."""
    best: tuple[date, int] | None = None
    for d in candidates:
        if d is None or _isna(d):
            continue
        diff = (d - target).days
        if abs(diff) > window_days:
            continue
        if best is None or abs(diff) < abs(best[1]):
            best = (d, diff)
    return best


def _to_utc_ns(values: Any) -> pd.Series:
    s = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    return s.dt.as_unit("ns")


# ---- release-time resolver ---------------------------------------------------------------------
def find_8k_acceptance(sec_filings: pd.DataFrame | None, report_date_ny: Any) -> tuple[pd.Timestamp, str] | None:
    """Earliest 8-K item 2.02 acceptance (UTC, accession) attributable to the report date:
    accepted on that New York date, on the next calendar day before 04:00 NY (late
    acceptance), or on the previous day after 16:00 NY (the vendor dated an evening release
    with the following day). 6-K rows and amendments are never a time source."""
    if sec_filings is None or len(sec_filings) == 0 or "accepted" not in sec_filings.columns:
        return None
    d = _as_date(report_date_ny)
    f = sec_filings
    form = f["form"].astype(str).str.strip() if "form" in f.columns else pd.Series("8-K", index=f.index)
    items = f["items"] if "items" in f.columns else pd.Series("2.02", index=f.index)
    is_8k = (form == "8-K") & items.map(lambda s: "2.02" in split_items(s))
    f = f[is_8k & f["accepted"].notna()]
    if len(f) == 0:
        return None
    accepted = pd.to_datetime(f["accepted"], utc=True)
    ny = accepted.dt.tz_convert(NY)
    days = ny.dt.date
    clock = ny.dt.time
    ok = (days == d) | ((days == d + timedelta(days=1)) & (clock < time(4, 0))) | (
        (days == d - timedelta(days=1)) & (clock >= time(16, 0))
    )
    if not ok.any():
        return None
    idx = accepted[ok].idxmin()
    accession = str(f.loc[idx, "accession"]) if "accession" in f.columns else ""
    return pd.Timestamp(accepted.loc[idx]).tz_convert(UTC), accession


def _crosses_session_boundary(acceptance: pd.Timestamp, detected_start: pd.Timestamp) -> bool:
    """True when the acceptance is outside regular hours but the detected bar starts inside
    them (or vice versa): a detection may not move t0 across the open/close boundary."""
    from ..timeutil import is_rth

    return bool(is_rth(detected_start)) != bool(is_rth(acceptance))


def resolve_release_time(
    *,
    report_date_ny: pd.Timestamp,
    sec_filings: pd.DataFrame | None,
    intraday: pd.DataFrame | None,
    calendar_flag: str | None,
    manual: pd.Timestamp | None = None,
) -> ResolvedT0:
    """Priority: manual > SEC 8-K item 2.02 acceptance on the report date (or the next calendar
    day before 04:00 NY for late acceptances) > detection from 1-minute extended-hours bars >
    calendar flag. When both an 8-K time and a detection exist: t0 = min(acceptance, detected
    bar start) if detected is within DETECTION_WINDOW before acceptance, else t0 = acceptance;
    t0_lag_s = acceptance - detected_start is recorded either way. 6-K rows are never a time
    source. Returns a ResolvedT0 with the CONFIDENCE of its source; `detection_first_bar`
    lowers a detected source to calendar-flag confidence."""
    d = _as_date(report_date_ny)
    if manual is not None and not _isna(manual):
        t0 = to_utc(manual, assume_tz=UTC)
        return ResolvedT0(t0, CONFIDENCE["manual"], T0Source.manual.value,
                          detail=f"manual override {_fmt_ts(t0)}")

    acceptance = find_8k_acceptance(sec_filings, d)
    if acceptance is not None:
        acc, accession = acceptance
        detail = f"8-K {accession} accepted {_fmt_ts(acc)}".rstrip()
        flags: tuple[str, ...] = ()
        lag: float | None = None
        t0 = acc
        # only bars from the 15 minutes before acceptance onwards can move t0 or measure the
        # reaction lag; an unrelated spike earlier in the day is not the release
        det = detect_release_from_bars(intraday, d, not_before=acc - DETECTION_WINDOW) \
            if intraday is not None and len(intraday) else None
        if det is not None:
            start, first_bar = det
            lag = float((acc - start).total_seconds())
            if 0 <= lag <= DETECTION_WINDOW.total_seconds() and _crosses_session_boundary(acc, start):
                # a filing accepted after the close cannot have been released during the
                # session: the spike is the closing auction (measured: MU 2026-06-24, 15:59 ET)
                detail += (f"; detection {_fmt_ts(start)} inside the regular session ignored "
                           f"(closing auction), t0 stays at acceptance")
                lag = None
            elif 0 <= lag <= DETECTION_WINDOW.total_seconds():
                t0 = start
                detail += f"; detection {_fmt_ts(start)} moved t0 earlier by {lag:.0f}s"
            else:
                detail += f"; detection {_fmt_ts(start)} (reaction lag {-lag:.0f}s) left t0 unchanged"
            if first_bar and t0 == start:
                flags += (FLAG_DETECTION_FIRST_BAR,)
        return ResolvedT0(t0, CONFIDENCE["sec_8k"], T0Source.sec_8k.value, detail=detail,
                          t0_lag_s=lag, flags=flags)

    det = detect_release_from_bars(intraday, d) if intraday is not None and len(intraday) else None
    if det is not None:
        start, first_bar = det
        if first_bar:
            return ResolvedT0(start, CONFIDENCE["calendar_flag"], T0Source.detected.value,
                              detail=f"detected on the first bar of the session {_fmt_ts(start)}",
                              flags=(FLAG_DETECTION_FIRST_BAR,))
        return ResolvedT0(start, CONFIDENCE["detected"], T0Source.detected.value,
                          detail=f"detected {_fmt_ts(start)}")

    timing = timing_from_flag(calendar_flag)
    if timing is None:
        return ResolvedT0(calendar_default_t0(d, "AMC"), CONFIDENCE_UNKNOWN,
                          T0Source.calendar_flag.value,
                          detail="no timing source; AMC default", flags=("timing_unknown",))
    return ResolvedT0(calendar_default_t0(d, timing), CONFIDENCE["calendar_flag"],
                      T0Source.calendar_flag.value,
                      detail=f"calendar flag {calendar_flag!r} -> {timing} default")


def detect_release_from_bars(bars: pd.DataFrame, report_date_ny: pd.Timestamp,
                             *, vol_z_threshold: float = 6.0, abs_ret_threshold: float = 0.01,
                             baseline_days: int = 10,
                             not_before: pd.Timestamp | None = None) -> tuple[pd.Timestamp, bool] | None:
    """First 1-minute bar on the report date (New York) whose volume z-score against the same
    clock-minute over the previous `baseline_days` sessions exceeds `vol_z_threshold` and whose
    |return| over the bar exceeds `abs_ret_threshold`. Returns (bar START time UTC,
    is_first_bar_of_session) or None. Bars must be schemas.C rows with tz-aware UTC t/t_end.

    Baseline = bars within +/-CLOCK_WINDOW_MINUTES of the candidate's clock minute, in the same
    session segment (pre-market before 09:30, regular hours through the 16:00 minute that
    carries the closing auction, after-hours from 16:01), on up to `baseline_days` prior
    sessions present in `bars`, plus the preceding same-segment bars of the report day
    (SAME_DAY_BASELINE_BARS at most). The z-score is robust so that one heavy print (the
    closing cross, an earlier release bar) cannot hide a spike: z = (v - median) /
    max(1.4826 MAD, 0.25 median, 1). A candidate with fewer than MIN_BASELINE_BARS baseline
    bars cannot be judged and is skipped. The bar return is max(|close/open - 1|,
    |open/previous close - 1|) so a gap into the bar counts too. `not_before` restricts the
    candidates (not the baseline) to bars starting at or after it."""
    if bars is None or len(bars) == 0:
        return None
    b = bars.sort_values(C.t, kind="mergesort").reset_index(drop=True)
    t = pd.to_datetime(b[C.t], utc=True)
    ny = t.dt.tz_convert(NY)
    day_key = (ny.dt.year * 10000 + ny.dt.month * 100 + ny.dt.day).to_numpy(dtype="int64")
    clock = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(dtype="int64")
    seg = np.where(clock < _RTH_OPEN_MIN, _SEG_PRE, np.where(clock <= _RTH_CLOSE_MIN, _SEG_RTH, _SEG_AH))
    vol = pd.to_numeric(b[C.volume], errors="coerce").to_numpy(dtype="float64")
    opn = pd.to_numeric(b[C.open], errors="coerce").to_numpy(dtype="float64")
    close = pd.to_numeric(b[C.close], errors="coerce").to_numpy(dtype="float64")
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        intra = np.abs(close / opn - 1.0)
        gap = np.abs(opn / prev_close - 1.0)
    ret = np.fmax(np.nan_to_num(intra, nan=0.0), np.nan_to_num(gap, nan=0.0))

    d = _as_date(report_date_ny)
    rd_key = d.year * 10000 + d.month * 100 + d.day
    is_day = day_key == rd_key
    if not is_day.any():
        return None
    prior_keys = sorted({int(k) for k in day_key[day_key < rd_key]})[-max(int(baseline_days), 0):]
    prior = np.isin(day_key, prior_keys) if prior_keys else np.zeros(len(b), dtype=bool)
    idx_day = np.flatnonzero(is_day)
    first_idx = int(idx_day[0])
    lo = to_utc(not_before, assume_tz=UTC) if not_before is not None else None
    positions = np.arange(len(b))
    for i in idx_day:
        i = int(i)
        if lo is not None and t.iloc[i] < lo:
            continue
        if not (vol[i] > 0) or not (ret[i] >= abs_ret_threshold):
            continue
        near = np.abs(clock - clock[i]) <= CLOCK_WINDOW_MINUTES
        cross = prior & (seg == seg[i]) & near
        same = is_day & (seg == seg[i]) & (positions < i)
        sample = np.concatenate([vol[cross], vol[same][-SAME_DAY_BASELINE_BARS:]])
        sample = sample[np.isfinite(sample)]
        if len(sample) < MIN_BASELINE_BARS:
            continue
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        scale = max(1.4826 * mad, 0.25 * median, 1.0)
        if (vol[i] - median) / scale >= vol_z_threshold:
            return pd.Timestamp(t.iloc[i]).tz_convert(UTC), i == first_idx
    return None


# ---- fiscal period -------------------------------------------------------------------------------
def _quarter_end_before(d: date) -> date:
    """Last day of the calendar quarter that ended strictly before `d`."""
    month = ((d.month - 1) // 3) * 3  # last month of the previous quarter: 0 (=Dec), 3, 6, 9
    year = d.year
    if month == 0:
        year, month = year - 1, 12
    first_of_next = date(year + month // 12, month % 12 + 1, 1)
    return first_of_next - timedelta(days=1)


def _add_months(d: date, n: int) -> date:
    """`d` shifted by `n` calendar months, the day clipped to the target month's length."""
    year, month0 = divmod(d.year * 12 + d.month - 1 + n, 12)
    month = month0 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def project_quarter_end(anchor: date, d: date) -> date:
    """The issuer's fiscal quarter end strictly before `d`, projected from one known quarter
    end `anchor` in whole quarters (3 months) forwards or backwards. Deterministic in the
    issuer's fiscal calendar, so the value does not change when later periods are filed."""
    k = ((d.year - anchor.year) * 12 + d.month - anchor.month) // 3
    while _add_months(anchor, 3 * k) >= d:
        k -= 1
    while _add_months(anchor, 3 * (k + 1)) < d:
        k += 1
    return _add_months(anchor, 3 * k)


def fiscal_period_for(report_date_ny: pd.Timestamp, *, sec_eps_facts: pd.DataFrame | None,
                      av_rows: pd.DataFrame | None) -> tuple[str, str, bool]:
    """(fiscal_period 'YYYY-MM', source, derived_flag). SEC companyfacts period_end nearest
    before the report date (within 120 days, first filed on or after it) for US filers
    ('sec_facts'), Alpha Vantage fiscalDateEnding of the row matched by report date for FPIs
    ('alphavantage'). When the period's own facts are not on file yet -- a fresh event before
    its 10-Q, an upcoming one -- the quarter end is projected from the issuer's latest known
    period end in whole quarters ('sec_facts_projected' / 'alphavantage_projected'), which
    gives the same month the eventual filing will. Only a name with no period end from either
    source falls back to the calendar quarter end preceding the report date ('derived',
    derived_flag=True)."""
    d = _as_date(report_date_ny)
    known_ends: list[date] = []
    if sec_eps_facts is not None and len(sec_eps_facts) and "period_end" in sec_eps_facts.columns:
        ends = pd.to_datetime(sec_eps_facts["period_end"], utc=True, errors="coerce").dt.date
        # the results of a period are public from the day they are first filed: a period whose
        # facts were already on file before the report date is an earlier event's quarter
        first_filed: dict[date, date] = {}
        if "filed" in sec_eps_facts.columns:
            filed = pd.to_datetime(sec_eps_facts["filed"], utc=True, errors="coerce").dt.date
            for e, f in zip(ends, filed, strict=True):
                if e is None or _isna(e) or f is None or _isna(f):
                    continue
                if e not in first_filed or f < first_filed[e]:
                    first_filed[e] = f
        best: date | None = None
        for e in ends.dropna().unique():
            known_ends.append(e)
            lag = (d - e).days
            if not 0 < lag <= FISCAL_PERIOD_MAX_LAG_DAYS:
                continue
            if e in first_filed and first_filed[e] < d:
                continue
            if best is None or e > best:
                best = e
        if best is not None:
            return f"{best.year:04d}-{best.month:02d}", "sec_facts", False
    av_ends: list[date] = []
    if av_rows is not None and len(av_rows) and "report_date_ny" in av_rows.columns:
        dates = [x for x in av_rows["report_date_ny"].tolist() if x is not None and not _isna(x)]
        hit = _nearest_date(dates, d)
        if hit is not None:
            row = av_rows[av_rows["report_date_ny"] == hit[0]].iloc[0]
            fe = row.get("fiscal_period_end")
            if fe is not None and not _isna(fe):
                fe = _as_date(fe)
                return f"{fe.year:04d}-{fe.month:02d}", "alphavantage", False
        if "fiscal_period_end" in av_rows.columns:
            av_ends = [_as_date(x) for x in av_rows["fiscal_period_end"].tolist()
                       if x is not None and not _isna(x)]
    for anchors, source in ((known_ends, "sec_facts_projected"), (av_ends, "alphavantage_projected")):
        if anchors:
            q = project_quarter_end(max(anchors), d)
            return f"{q.year:04d}-{q.month:02d}", source, False
    q = _quarter_end_before(d)
    return f"{q.year:04d}-{q.month:02d}", "derived", True


# ---- consensus snapshots -----------------------------------------------------------------------
def _empty_snapshots() -> pd.DataFrame:
    return pd.DataFrame({
        "snapshot_time": pd.Series(dtype="datetime64[ns, UTC]"),
        "symbol": pd.Series(dtype=str),
        "report_date_ny": pd.Series(dtype=object),
        "eps_estimate": pd.Series(dtype="float64"),
        "rev_estimate": pd.Series(dtype="float64"),
        "n_estimates": pd.Series(dtype="Int64"),
        "vendor": pd.Series(dtype=str),
        "vendor_last_updated": pd.Series(dtype="datetime64[ns, UTC]"),
    })[SNAPSHOT_COLUMNS]


def load_consensus_snapshots(settings: Settings) -> pd.DataFrame:
    """Every archived consensus snapshot row (SNAPSHOT_COLUMNS), oldest first; empty frame when
    the archive has none."""
    root = settings.consensus_archive_dir
    files = sorted(root.glob("*.parquet")) if root.exists() else []
    frames = []
    for p in files:
        try:
            frames.append(pd.read_parquet(p))
        except Exception as exc:  # a corrupt file must not hide the others
            log.warning("consensus snapshot %s unreadable: %s", p, exc)
    frames = [f for f in frames if len(f)]
    if not frames:
        return _empty_snapshots()
    df = pd.concat(frames, ignore_index=True).reindex(columns=SNAPSHOT_COLUMNS)
    df["snapshot_time"] = _to_utc_ns(df["snapshot_time"])
    df["vendor_last_updated"] = _to_utc_ns(df["vendor_last_updated"])
    df["report_date_ny"] = [None if _isna(x) else _as_date(x) for x in df["report_date_ny"]]
    df["n_estimates"] = pd.array(pd.to_numeric(df["n_estimates"], errors="coerce").round(), dtype="Int64")
    return df.sort_values(["snapshot_time", "symbol"], kind="mergesort").reset_index(drop=True)


def consensus_before(snapshots: pd.DataFrame, symbol: str, report_date_ny: Any,
                     before: pd.Timestamp | None) -> dict | None:
    """The latest archived consensus for (symbol, report date +/- 1 day) captured strictly
    before `before` (any time when `before` is None): {eps_estimate, rev_estimate,
    n_estimates, snapshot_time, vendor}. FMP rows are preferred for the estimates at the
    winning snapshot time; n_estimates comes from whichever vendor reports it."""
    if snapshots is None or len(snapshots) == 0:
        return None
    d = _as_date(report_date_ny)
    s = snapshots[snapshots["symbol"].astype(str).str.upper() == symbol.upper()]
    if len(s) == 0:
        return None
    dd = pd.to_numeric(s["report_date_ny"].map(
        lambda x: math.nan if x is None or _isna(x) else abs((_as_date(x) - d).days)
    ), errors="coerce")
    s = s[(dd <= 1).fillna(False).to_numpy()]
    if before is not None:
        s = s[s["snapshot_time"] < to_utc(before, assume_tz=UTC)]
    s = s[s["snapshot_time"].notna()]
    if len(s) == 0:
        return None
    latest = s["snapshot_time"].max()
    at = s[s["snapshot_time"] == latest]
    fmp_rows = at[at["vendor"].astype(str) == "fmp"]
    pick = fmp_rows.iloc[0] if len(fmp_rows) else at.iloc[0]
    n_est = pd.NA
    for _, r in at.iterrows():
        if not _isna(r["n_estimates"]):
            n_est = int(r["n_estimates"])
            break
    return {
        "eps_estimate": float(pick["eps_estimate"]) if not _isna(pick["eps_estimate"]) else math.nan,
        "rev_estimate": float(pick["rev_estimate"]) if not _isna(pick["rev_estimate"]) else math.nan,
        "n_estimates": n_est,
        "snapshot_time": pd.Timestamp(latest).tz_convert(UTC),
        "vendor": str(pick["vendor"]),
    }


# ---- universe view -------------------------------------------------------------------------------
@dataclass
class _Name:
    underlying: str
    kind: str
    cik: int | None
    primary: str | None
    markets: list[tuple[str, pd.Timestamp | None]] = field(default_factory=list)  # (market, listing)
    flags: tuple[str, ...] = ()

    @property
    def earliest_listing(self) -> pd.Timestamp | None:
        starts = [s for _, s in self.markets if s is not None and not _isna(s)]
        return min(starts) if starts else None

    def market_at(self, t0: pd.Timestamp | None) -> tuple[str | None, bool]:
        """(market to price the event with, has_perp_at_t0): the primary when it was listed at
        t0, else the earliest-listed alternate that was, else the primary with False."""
        if t0 is None or _isna(t0):
            return self.primary, False
        t0 = to_utc(t0, assume_tz=UTC)
        listed = [(m, s) for m, s in self.markets if s is not None and not _isna(s) and s <= t0]
        if not listed:
            return self.primary, False
        if any(m == self.primary for m, _ in listed):
            return self.primary, True
        listed.sort(key=lambda ms: ms[1])
        return listed[0][0], True


def _names_from_universe(universe: pd.DataFrame, underlyings: list[str] | None) -> list[_Name]:
    from ..universe import event_universe

    ev = event_universe(universe)
    want = {u.strip().upper() for u in underlyings} if underlyings else None
    kinds = {k.value for k in EVENT_KINDS}
    names: list[_Name] = []
    for _, row in ev.iterrows():
        u = row[U.underlying]
        if u is None or _isna(u):
            continue
        u = str(u).upper()
        if want is not None and u not in want:
            continue
        same = universe[U.underlying].fillna("").astype(str).str.upper() == u
        alts = universe[same & universe[U.kind].isin(kinds)]
        markets = []
        for _, m in alts.iterrows():
            start = m.get(U.listing_start)
            start = None if start is None or _isna(start) else to_utc(start, assume_tz=UTC)
            markets.append((str(m[U.market]), start))
        cik = row.get(U.cik)
        names.append(_Name(u, str(row[U.kind]), None if _isna(cik) else int(cik), str(row[U.market]), markets))
    if want is not None:
        missing = want - {n.underlying for n in names}
        if missing:
            log.warning("not in the event universe, skipped: %s", ", ".join(sorted(missing)))
    return names


def _names_without_universe(settings: Settings, underlyings: list[str]) -> list[_Name]:
    """Degraded mode for an explicit list of tickers when data/universe.parquet is missing: CIK
    from the SEC ticker map, kind guessed from the earnings filings (no 8-K item 2.02 and at
    least one 6-K -> equity_fpi, else equity_us), no market. The submissions request is cached,
    so the guess costs nothing extra: the resolver reads the same filings."""
    from ..data.sec import SECClient

    sec = SECClient(settings)
    try:
        tickers = sec.ticker_map().set_index("ticker")
    except Exception as exc:  # noqa: BLE001 - a missing ticker map only costs the CIKs
        log.warning("SEC ticker map unavailable (%s)", exc)
        tickers = pd.DataFrame(columns=["cik", "title"])
    names = []
    for u in underlyings:
        u = u.strip().upper()
        cik = int(tickers.loc[u, "cik"]) if u in tickers.index else None
        kind = Kind.equity_us.value
        if cik is not None:
            try:
                forms = sec.earnings_filings(cik)["form"].astype(str)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.warning("SEC filings unavailable for %s (CIK %s): %s; treating it as equity_us", u, cik, exc)
            else:
                if not (forms == "8-K").any() and (forms == "6-K").any():
                    kind = Kind.equity_fpi.value
        names.append(_Name(u, kind, cik, None, [], flags=("no_universe",)))
    return names


# ---- per-run data access -------------------------------------------------------------------------
@dataclass
class _SecData:
    filings: pd.DataFrame | None = None
    facts: pd.DataFrame | None = None


class _Providers:
    """Lazy, memoised access to every provider used by build_events."""

    def __init__(self, settings: Settings):
        from ..data.fmp import FMPClient
        from ..data.nasdaq import NasdaqClient
        from ..data.sec import SECClient

        self.settings = settings
        self.fmp = FMPClient(settings)
        self.sec = SECClient(settings)
        self.nasdaq = NasdaqClient(settings)
        self.av = None
        if settings.alphavantage_api_key:
            from ..data.alphavantage import AlphaVantageClient

            self.av = AlphaVantageClient(settings)
        self._sec: dict[int, _SecData] = {}
        self._av: dict[str, pd.DataFrame | None] = {}
        self._nasdaq: dict[date, pd.DataFrame | None] = {}
        self._nasdaq_by_symbol: dict[str, list[dict]] = {}
        self._splits: dict[str, pd.DataFrame | None] = {}
        self.nasdaq_failed = 0

    def splits(self, symbol: str) -> pd.DataFrame | None:
        """The FMP splits calendar of `symbol` (memoised: one request per underlying per run,
        one per week through the cache); None when FMP answered with an error. Budget
        exhaustion propagates: it is a checkpoint like the intraday request."""
        symbol = symbol.upper()
        if symbol not in self._splits:
            try:
                self._splits[symbol] = self.fmp.splits(symbol)
            except (FMPError, httpx.HTTPError, ValueError) as exc:
                log.warning("FMP splits unavailable for %s: %s", symbol, exc)
                self._splits[symbol] = None
        return self._splits[symbol]

    def sec_data(self, cik: int | None) -> _SecData:
        if cik is None:
            return _SecData()
        if cik not in self._sec:
            data = _SecData()
            try:
                data.filings = self.sec.earnings_filings(cik)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.warning("SEC filings unavailable for CIK %s: %s", cik, exc)
            try:
                data.facts = self.sec.company_facts_eps(cik)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.warning("SEC companyfacts unavailable for CIK %s: %s", cik, exc)
            self._sec[cik] = data
        return self._sec[cik]

    def av_rows(self, symbol: str) -> pd.DataFrame | None:
        """Alpha Vantage quarterly earnings, or None when the provider is unavailable. Budget
        exhaustion here is not a checkpoint: the source is optional."""
        if self.av is None:
            return None
        if symbol not in self._av:
            try:
                self._av[symbol] = self.av.earnings(symbol)
            except ProviderUnavailable as exc:
                log.warning("Alpha Vantage unavailable (%s); skipping it for the rest of the run", exc)
                self.av = None
                return None
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("Alpha Vantage failed for %s: %s", symbol, exc)
                self._av[symbol] = None
        return self._av[symbol]

    def nasdaq_calendar(self, day: date) -> pd.DataFrame | None:
        """The Nasdaq calendar for one New York date (memoised); rows are also indexed by
        symbol so that later events can match against every date fetched so far."""
        if day not in self._nasdaq:
            cal = None
            try:
                cal = self.nasdaq.earnings_calendar(pd.Timestamp(day))
            except (httpx.HTTPError, ValueError) as exc:
                self.nasdaq_failed += 1
                if self.nasdaq_failed <= 3:
                    log.warning("Nasdaq calendar unavailable for %s: %s", day, exc)
            self._nasdaq[day] = cal
            if cal is not None and len(cal):
                for rec in cal.to_dict("records"):
                    self._nasdaq_by_symbol.setdefault(str(rec["symbol"]).upper(), []).append(rec)
        return self._nasdaq[day]

    def nasdaq_rows(self, symbol: str, days: list[date]) -> list[dict]:
        """Nasdaq calendar rows for `symbol` on every date fetched so far, after fetching
        `days`. Events run newest-first, so other dates of the same season are usually already
        indexed and a shifted date shows up as a date conflict."""
        for day in days:
            self.nasdaq_calendar(day)
        return self._nasdaq_by_symbol.get(symbol.upper(), [])


def _load_manual_overrides(settings: Settings) -> dict[str, pd.Timestamp]:
    path = settings.configs_dir / "t0_overrides.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out: dict[str, pd.Timestamp] = {}
    for k, v in (raw.items() if isinstance(raw, dict) else []):
        try:
            out[str(k).strip().upper()] = to_utc(pd.Timestamp(str(v)), assume_tz=UTC)
        except (ValueError, TypeError) as exc:
            log.warning("ignoring t0 override %r: %s", k, exc)
    return out


# ---- row assembly --------------------------------------------------------------------------------
@dataclass
class _Event:
    name: _Name
    report_date_ny: date
    eps_actual: float
    eps_estimate: float
    rev_actual: float
    rev_estimate: float


def _base_row(ev: _Event) -> dict[str, Any]:
    n = ev.name
    return {
        E.event_id: None, E.underlying: n.underlying, E.market: n.primary, E.cik: n.cik,
        E.kind: n.kind, E.fiscal_period: None, E.fiscal_period_source: None,
        E.report_date_ny: ev.report_date_ny, E.t0: pd.NaT, E.t0_confidence: math.nan,
        E.t0_source: None, E.timing: None, E.eps_actual: ev.eps_actual,
        E.eps_estimate: ev.eps_estimate, E.eps_surprise_pct: _surprise_pct(ev.eps_actual, ev.eps_estimate),
        E.rev_actual: ev.rev_actual, E.rev_estimate: ev.rev_estimate,
        E.rev_surprise_pct: _surprise_pct(ev.rev_actual, ev.rev_estimate), E.n_estimates: pd.NA,
        E.estimate_source: "fmp_final", E.estimate_snapshot_time: pd.NaT, E.sources_used: "fmp",
        E.has_perp_at_t0: False, E.listing_start: n.earliest_listing, E.pending: False,
        E.flags: "", E.ca_ex_date: pd.NaT, "t0_acceptance": pd.NaT, "t0_lag_s": math.nan,
        "t0_detail": "",
    }


def corporate_action_near(splits: pd.DataFrame | None, t0: pd.Timestamp, horizon_hours: int) -> pd.Timestamp | None:
    """UTC instant (00:00 America/New_York) of the split ex-date nearest t0 inside
    [t0 - CORPORATE_ACTION_LOOKBACK, t0 + horizon_hours]; None when there is none."""
    if splits is None or len(splits) == 0 or "ex_date" not in splits.columns:
        return None
    t0 = to_utc(t0, assume_tz=UTC)
    lo, hi = t0 - CORPORATE_ACTION_LOOKBACK, t0 + pd.Timedelta(hours=horizon_hours)
    best: pd.Timestamp | None = None
    for d in splits["ex_date"]:
        if d is None or _isna(d):
            continue
        ex = to_utc(pd.Timestamp(_as_date(d)), assume_tz=NY)
        if lo <= ex <= hi and (best is None or abs(ex - t0) < abs(best - t0)):
            best = ex
    return best


def _pending_row(ev: _Event, providers: _Providers | None, *, today_ny: date | None = None) -> dict[str, Any]:
    row = _base_row(ev)
    facts = providers.sec_data(ev.name.cik).facts if providers is not None else None
    fp, src, derived = fiscal_period_for(ev.report_date_ny, sec_eps_facts=facts, av_rows=None)
    row[E.event_id] = f"{ev.name.underlying}:{fp}"
    row[E.fiscal_period], row[E.fiscal_period_source] = fp, src
    row[E.pending] = True
    if ev.report_date_ny > (today_ny if today_ny is not None else _today_ny()):
        row[E.estimate_source] = "fmp_calendar"
    row[E.flags] = ";".join(["pending"] + (["fiscal_period_derived"] if derived else []) + list(ev.name.flags))
    return row


def _resolve_event(ev: _Event, providers: _Providers, *, snapshots: pd.DataFrame,
                   manual: dict[str, pd.Timestamp], today_ny: date) -> dict[str, Any]:
    """One completed event row. Raises BudgetExhausted from the FMP intraday request."""
    name, d_fmp = ev.name, ev.report_date_ny
    row = _base_row(ev)
    flags: list[str] = list(name.flags)
    sources: list[str] = ["fmp"]

    sec = providers.sec_data(name.cik)
    if sec.filings is not None and len(sec.filings):
        sources.append("sec")
    acc = find_8k_acceptance(sec.filings, d_fmp)
    d_eff = d_fmp
    if acc is not None:
        d_eff = acc[0].tz_convert(NY).date()
        if d_eff != d_fmp:
            flags.append("report_date_from_8k")
    row[E.report_date_ny] = d_eff

    av_rows = providers.av_rows(name.underlying) if name.kind == Kind.equity_fpi.value else None
    fp, fp_src, derived = fiscal_period_for(d_eff, sec_eps_facts=sec.facts, av_rows=av_rows)
    if derived:
        flags.append("fiscal_period_derived")
    event_id = f"{name.underlying}:{fp}"
    row[E.event_id], row[E.fiscal_period], row[E.fiscal_period_source] = event_id, fp, fp_src

    # cross-checks: Alpha Vantage (FPIs) and the Nasdaq calendar, matched by nearest date to
    # the event's (8-K-corrected) report date; the Nasdaq calendar is fetched for that date,
    # its neighbours and FMP's date, and matched against every date fetched so far
    date_conflict = False
    calendar_flag: str | None = None
    if av_rows is not None and len(av_rows):
        hit = _nearest_date([x for x in av_rows["report_date_ny"].tolist()], d_eff)
        if hit is not None:
            sources.append("alphavantage")
            if abs(hit[1]) > DATE_CONFLICT_DAYS:
                date_conflict = True
            av_row = av_rows[av_rows["report_date_ny"] == hit[0]].iloc[0]
            calendar_flag = av_row.get("report_time") if not _isna(av_row.get("report_time")) else None
    nq_days = sorted({d_eff - timedelta(days=1), d_eff, d_eff + timedelta(days=1), d_fmp})
    nq = providers.nasdaq_rows(name.underlying, [d for d in nq_days if d <= today_ny])
    if nq:
        hit = _nearest_date([r.get("report_date_ny") for r in nq], d_eff)
        if hit is not None:
            sources.append("nasdaq")
            if abs(hit[1]) > DATE_CONFLICT_DAYS:
                date_conflict = True
            nq_row = next(r for r in nq if r.get("report_date_ny") == hit[0])
            if calendar_flag is None and timing_from_flag(nq_row.get("time_flag")) is not None:
                calendar_flag = str(nq_row.get("time_flag"))
            if not _isna(nq_row.get("n_estimates")):
                row[E.n_estimates] = int(nq_row["n_estimates"])
            if _isna(row[E.eps_estimate]) and not _isna(nq_row.get("eps_estimate")):
                row[E.eps_estimate] = float(nq_row["eps_estimate"])
                row[E.estimate_source] = "nasdaq_final"
            a_fmp, a_nq = row[E.eps_actual], nq_row.get("eps_actual")
            if not _isna(a_fmp) and not _isna(a_nq) and abs(float(a_fmp) - float(a_nq)) > max(0.02, 0.05 * abs(float(a_nq))):
                flags.append("eps_actual_conflict")
    if date_conflict:
        flags.append("date_conflict")

    # intraday bars: the very window targets.loaders requests, so the cache entry is shared
    bars = None
    if d_eff > today_ny:
        flags.append("upcoming")
        row[E.estimate_source] = "fmp_calendar"  # the vendor's current, not-yet-final value
    elif d_eff < INTRADAY_1MIN_FLOOR:
        flags.append("no_intraday")
    else:
        start = pd.Timestamp(d_eff) - pd.Timedelta(days=1)
        end = pd.Timestamp(d_eff) + pd.Timedelta(days=1)  # same window as targets.loaders (3 days, 1 request)
        try:
            bars = providers.fmp.intraday(name.underlying, "1min", start, end, extended=True)
        except (FMPError, httpx.HTTPError) as exc:
            # one symbol's bad answer is not a checkpoint: resolve without bars and go on
            log.warning("FMP 1-minute bars unavailable for %s %s: %s", name.underlying, d_eff, exc)
            bars = None
            flags.append("intraday_error")
        if bars is not None and len(bars):
            sources.append("fmp_intraday")
        else:
            bars = None
            flags.append("no_intraday")

    if calendar_flag is None and d_eff > today_ny:
        hist = _typical_timing_from_filings(sec.filings)
        if hist is not None:
            calendar_flag, _ = hist
            flags.append("timing_from_history")

    man = manual.get(event_id) or manual.get(f"{name.underlying}:{d_fmp.isoformat()}")
    res = resolve_release_time(report_date_ny=d_eff, sec_filings=sec.filings, intraday=bars,
                               calendar_flag=calendar_flag, manual=man)
    flags.extend(res.flags)
    t0 = res.t0
    row[E.t0], row[E.t0_source], row[E.t0_confidence] = t0, res.source, res.confidence
    row["t0_acceptance"] = acc[0] if acc is not None else pd.NaT
    row["t0_lag_s"] = math.nan if res.t0_lag_s is None else float(res.t0_lag_s)
    row["t0_detail"] = res.detail
    if res.source == T0Source.sec_8k.value:
        sources.append("sec_8k")
    if date_conflict:
        row[E.t0_confidence] = 0.0
    row[E.timing] = _timing_of(t0)

    # corporate actions: a split ex-date near the release (design §2); the targets decide
    # from E.ca_ex_date whether the headline label survives
    splits = providers.splits(name.underlying)
    if splits is None:
        flags.append("splits_error")
    else:
        ex = corporate_action_near(splits, t0, providers.settings.horizon_hours)
        if ex is not None:
            flags.append(FLAG_CORPORATE_ACTION)
            row[E.ca_ex_date] = ex

    # consensus provenance: an archived snapshot captured before t0 beats the vendor's final value
    snap = consensus_before(snapshots, name.underlying, d_eff, t0)
    if snap is not None:
        row[E.eps_estimate], row[E.rev_estimate] = snap["eps_estimate"], snap["rev_estimate"]
        if not _isna(snap["n_estimates"]):
            row[E.n_estimates] = int(snap["n_estimates"])
        row[E.estimate_source] = "consensus_snapshot"
        row[E.estimate_snapshot_time] = snap["snapshot_time"]
        sources.append("consensus_snapshot")
    row[E.eps_surprise_pct] = _surprise_pct(row[E.eps_actual], row[E.eps_estimate])
    row[E.rev_surprise_pct] = _surprise_pct(row[E.rev_actual], row[E.rev_estimate])

    market, has_perp = name.market_at(t0)
    row[E.market], row[E.has_perp_at_t0] = market, has_perp
    row[E.sources_used] = ";".join(dict.fromkeys(sources))
    row[E.flags] = ";".join(dict.fromkeys(f for f in flags if f))
    return row


def _timing_of(t0: pd.Timestamp) -> str | None:
    """schemas.Timing of the release instant; None when the exchange calendar cannot place it
    (far-future calendar dates)."""
    try:
        return str(classify_timing(t0))
    except Exception as exc:  # noqa: BLE001 - exchange_calendars raises its own error types
        log.warning("timing of %s not classified: %s", t0, exc)
        return None


def _typical_timing_from_filings(filings: pd.DataFrame | None) -> tuple[str, int] | None:
    """('AMC' | 'BMO', n) from the issuer's past 8-K item 2.02 acceptance clocks (majority)."""
    if filings is None or len(filings) == 0:
        return None
    f = filings[(filings["form"].astype(str) == "8-K") & filings["items"].map(lambda s: "2.02" in split_items(s))]
    acc = pd.to_datetime(f["accepted"], utc=True).dropna()
    if len(acc) == 0:
        return None
    clocks = acc.dt.tz_convert(NY).dt.time
    n_bmo = int((clocks < time(9, 30)).sum())
    n_amc = int(len(clocks) - n_bmo)
    return ("BMO", n_bmo) if n_bmo > n_amc else ("AMC", n_amc)


def _events_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    for col in DATETIME_COLUMNS:
        df[col] = _to_utc_ns(df[col]) if len(df) else pd.Series(dtype="datetime64[ns, UTC]")
    df[E.cik] = pd.array(pd.to_numeric(df[E.cik], errors="coerce"), dtype="Int64")
    df[E.n_estimates] = pd.array(pd.to_numeric(df[E.n_estimates], errors="coerce"), dtype="Int64")
    for col in (E.t0_confidence, E.eps_actual, E.eps_estimate, E.eps_surprise_pct, E.rev_actual,
                E.rev_estimate, E.rev_surprise_pct, "t0_lag_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df[E.has_perp_at_t0] = df[E.has_perp_at_t0].fillna(False).astype(bool)
    df[E.pending] = df[E.pending].fillna(False).astype(bool)
    for col in (E.flags, E.sources_used, "t0_detail"):
        df[col] = df[col].fillna("").astype(str)
    df[E.report_date_ny] = [None if _isna(x) else _as_date(x) for x in df[E.report_date_ny]]
    if len(df):
        df = df.sort_values([E.report_date_ny, E.underlying], ascending=[False, True], kind="mergesort")
        dup = df[E.event_id].duplicated(keep="first")
        if dup.any():
            log.warning("dropping %d rows whose fiscal period repeats an existing event_id: %s",
                        int(dup.sum()), ", ".join(df.loc[dup, E.event_id].astype(str).head(5)))
        df = df[~dup]
    df = df.reset_index(drop=True)
    df.attrs["schema_version"] = SCHEMA_VERSION
    return df


def _write_events(settings: Settings, df: pd.DataFrame) -> None:
    from ..data.archive import write_parquet_atomic

    df = df.copy()
    df.attrs["schema_version"] = SCHEMA_VERSION
    settings.events_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(df, settings.events_path)


def _days_apart(a: Any, b: Any) -> float:
    """|a - b| in days for two calendar dates; NaN when either is missing."""
    if a is None or b is None or _isna(a) or _isna(b):
        return math.nan
    return float(abs((_as_date(a) - _as_date(b)).days))


def _merge_existing(settings: Settings, new: pd.DataFrame, processed: set[str],
                    since: date | None) -> pd.DataFrame:
    """Keep rows of an existing events.parquet that this run did not rebuild: other
    underlyings, rows of processed underlyings older than `since`, every row of a name whose
    earnings history could not be fetched (the new frame carries only its `<U>:pending`
    placeholder), and completed rows that this run could only mark pending -- same event_id,
    or same name and report date within a day. A resolved row is never replaced by a pending
    one; the pending row is dropped instead, so a budget-hit rerun keeps what earlier runs
    finished."""
    if not settings.events_path.exists():
        return new
    try:
        old = load_events(settings)
    except Exception as exc:  # noqa: BLE001 - an unreadable old table is replaced
        log.warning("existing %s not merged: %s", settings.events_path, exc)
        return new
    if old.attrs.get("schema_version") != SCHEMA_VERSION or list(old.columns) != list(new.columns):
        log.warning("existing %s has another schema; replacing it", settings.events_path)
        return new
    old_u = old[E.underlying].astype(str).str.upper()
    new_u = new[E.underlying].astype(str).str.upper()
    new_pending = new[E.pending].astype(bool)
    placeholder = new_pending & new[E.flags].astype(str).map(lambda f: "earnings_history_pending" in f.split(";"))
    rebuilt = {u.upper() for u in processed} - set(new_u[placeholder])
    keep = ~old_u.isin(rebuilt)
    if since is not None:
        older = old[E.report_date_ny].map(lambda x: x is not None and not _isna(x) and _as_date(x) < since)
        keep |= older
    old_done = ~old[E.pending].astype(bool)
    superseded = pd.Series(False, index=new.index)
    for i in new.index[new_pending & ~placeholder]:
        same_name = old_done & (old_u == new_u[i])
        if not same_name.any():
            continue
        near = old[E.report_date_ny].map(lambda x, d=new.at[i, E.report_date_ny]: _days_apart(x, d))
        hit = same_name & ((old[E.event_id] == new.at[i, E.event_id]) | (near <= DATE_CONFLICT_DAYS).fillna(False))
        if hit.any():
            keep |= hit
            superseded[i] = True
    if superseded.any():
        log.info("keeping %d completed rows that this run could only mark pending", int(superseded.sum()))
        new = new[~superseded]
    keep &= ~old[E.event_id].isin(new[E.event_id])
    if not keep.any():
        return new
    return _events_frame(pd.concat([new, old[keep]], ignore_index=True).to_dict("records"))


def build_events(settings: Settings, *, underlyings: list[str] | None = None,
                 since: pd.Timestamp | None = None, write: bool = True) -> pd.DataFrame:
    """Assemble events from FMP earnings history (primary), SEC filings (timing and fiscal
    period), archived consensus snapshots, and optional Nasdaq / Alpha Vantage cross-checks.
    Nasdaq/AV rows match the FMP row by nearest report date within +/-10 days; a matched pair
    whose dates differ by more than one day gets flags += 'date_conflict' and confidence 0.
    Events are processed newest-first; on BudgetExhausted the completed rows are written, the
    remaining ones are written with pending=True, and the function raises after writing."""
    from ..universe import load_universe

    settings.ensure_dirs()
    since_d = _as_date(since) if since is not None else None
    try:
        universe = load_universe(settings)
        names = _names_from_universe(universe, underlyings)
    except FileNotFoundError:
        if not underlyings:
            raise
        log.warning("%s missing; building %s without market information", settings.universe_path,
                    ", ".join(underlyings))
        names = _names_without_universe(settings, underlyings)
    providers = _Providers(settings)
    snapshots = load_consensus_snapshots(settings)
    manual = _load_manual_overrides(settings)
    today_ny = _today_ny()

    budget_error: BudgetExhausted | None = None
    events: list[_Event] = []
    unfetched: list[_Name] = []
    for name in sorted(names, key=lambda n: n.underlying):
        try:
            hist = providers.fmp.earnings_history(name.underlying)
        except BudgetExhausted as exc:
            budget_error = budget_error or exc
            unfetched.append(name)
            continue
        for _, r in hist.iterrows():
            d = r[E.report_date_ny]
            if d is None or _isna(d):
                continue
            d = _as_date(d)
            if since_d is not None and d < since_d:
                continue
            events.append(_Event(name, d, float(r[E.eps_actual]), float(r[E.eps_estimate]),
                                 float(r[E.rev_actual]), float(r[E.rev_estimate])))
    events.sort(key=lambda e: (e.report_date_ny, e.name.underlying), reverse=True)

    rows: list[dict[str, Any]] = []
    n_done = 0
    for ev in events:
        if budget_error is not None:
            rows.append(_pending_row(ev, providers, today_ny=today_ny))
            continue
        try:
            rows.append(_resolve_event(ev, providers, snapshots=snapshots, manual=manual,
                                       today_ny=today_ny))
            n_done += 1
        except BudgetExhausted as exc:
            budget_error = exc
            rows.append(_pending_row(ev, providers, today_ny=today_ny))
    for name in unfetched:
        row = _base_row(_Event(name, None, math.nan, math.nan, math.nan, math.nan))  # type: ignore[arg-type]
        row[E.event_id] = f"{name.underlying}:pending"
        row[E.pending] = True
        row[E.flags] = ";".join(["pending", "earnings_history_pending", *name.flags])
        rows.append(row)

    out = _events_frame(rows)
    if write:
        merged = _merge_existing(settings, out, {n.underlying for n in names}, since_d)
        _write_events(settings, merged)
    n_pending = int(out[E.pending].sum())
    log.info("events: %d resolved, %d pending, %d underlyings", n_done, n_pending, len(names))
    if budget_error is not None:
        if write:
            outcome = (f"{n_done} events resolved and {n_pending} written as pending to "
                       f"{settings.events_path}; rerun to resume from the cache.")
        else:
            outcome = (f"{n_done} events resolved and {n_pending} pending; nothing written "
                       f"(write=False), rerun to resume from the cache.")
        raise BudgetExhausted(f"{budget_error} -- {outcome}") from budget_error
    return out


def load_events(settings: Settings) -> pd.DataFrame:
    if not settings.events_path.exists():
        raise FileNotFoundError(f"{settings.events_path} missing; run `freedom events` first")
    df = pd.read_parquet(settings.events_path)
    for col in DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = _to_utc_ns(df[col])
    if E.report_date_ny in df.columns:
        df[E.report_date_ny] = [None if _isna(x) else _as_date(x) for x in df[E.report_date_ny]]
    df.attrs.setdefault("schema_version", None)
    return df


def expected_release_clock(events: pd.DataFrame, underlying: str) -> tuple[str, str] | None:
    """(HH:MM America/New_York, source) = the issuer's median acceptance clock over its past
    sec_8k events, or None when it has none (caller falls back to the calendar-flag default).
    Used by `freedom predict` for pre-release decision times (docs/design.md §10)."""
    if events is None or len(events) == 0:
        return None
    e = events[(events[E.underlying].astype(str).str.upper() == underlying.upper())
               & (events[E.t0_source].astype(str) == T0Source.sec_8k.value)]
    if E.pending in e.columns:
        e = e[~e[E.pending].astype(bool)]
    if len(e) == 0:
        return None
    src = e["t0_acceptance"] if "t0_acceptance" in e.columns else e[E.t0]
    src = pd.to_datetime(src, utc=True, errors="coerce")
    fallback = pd.to_datetime(e[E.t0], utc=True, errors="coerce")
    inst = src.where(src.notna(), fallback).dropna()
    if len(inst) == 0:
        return None
    ny = inst.dt.tz_convert(NY)
    seconds = (ny.dt.hour * 3600 + ny.dt.minute * 60 + ny.dt.second).to_numpy()
    med = int(round(float(np.median(seconds))))
    return f"{med // 3600:02d}:{(med % 3600) // 60:02d}", f"median of {len(inst)} sec_8k acceptances"


def detect_release_live(bars: pd.DataFrame, expected_date_ny: pd.Timestamp, **kw) -> pd.Timestamp | None:
    """Live wrapper around detect_release_from_bars for bars ending at or before now; returns the
    detected bar start or None. Same thresholds as the historical detector."""
    now = kw.pop("now", None)
    now = to_utc(now, assume_tz=UTC) if now is not None else pd.Timestamp(utcnow())
    if bars is None or len(bars) == 0:
        return None
    closed = bars[pd.to_datetime(bars[C.t_end], utc=True) <= now]
    hit = detect_release_from_bars(closed, expected_date_ny, **kw)
    return None if hit is None else hit[0]


def _events_table_clock(events: pd.DataFrame | None, underlying: str, d: date) -> tuple[time, str, str] | None:
    """(clock America/New_York, t0_source, label) of the completed events.parquet row for
    (underlying, report date within a day) whose release time has a real source -- a manual
    override or a calendar flag -- or None when there is no such row or it only carries the
    timing_unknown default."""
    if events is None or len(events) == 0:
        return None
    e = events[(events[E.underlying].astype(str).str.upper() == underlying.upper())
               & ~events[E.pending].astype(bool) & events[E.t0].notna()]
    if len(e) == 0:
        return None
    apart = e[E.report_date_ny].map(lambda x: _days_apart(x, d))
    e = e.assign(_apart=apart)[(apart <= DATE_CONFLICT_DAYS).fillna(False).to_numpy()]
    if len(e) == 0:
        return None
    row = e.sort_values("_apart", kind="mergesort").iloc[0]
    flags = set(str(row[E.flags]).split(";"))
    if "timing_unknown" in flags:
        return None
    src = str(row[E.t0_source])
    label = f"events table: {src}" + (" (timing_from_history)" if "timing_from_history" in flags else "")
    return pd.Timestamp(row[E.t0]).tz_convert(NY).time(), src, label


def expected_t0_for(events: pd.DataFrame | None, underlying: str, report_date_ny: Any,
                    nasdaq_flag: Any = None) -> tuple[pd.Timestamp, str]:
    """(expected_t0 UTC, source) for an upcoming event, in order: a manual override recorded on
    the matching events.parquet row, the issuer's median sec_8k acceptance clock, the
    calendar-flag time of the matching events.parquet row, the Nasdaq calendar's time flag
    (BMO 07:00 / AMC 16:05), else the AMC default."""
    d = _as_date(report_date_ny)
    table = _events_table_clock(events, underlying, d)
    if table is not None and table[1] == T0Source.manual.value:
        return to_utc(pd.Timestamp.combine(d, table[0]), assume_tz=NY), table[2]
    clock = expected_release_clock(events, underlying) if events is not None else None
    if clock is not None:
        hh, mm = (int(x) for x in clock[0].split(":"))
        return to_utc(pd.Timestamp.combine(d, time(hh, mm)), assume_tz=NY), clock[1]
    if table is not None:
        return to_utc(pd.Timestamp.combine(d, table[0]), assume_tz=NY), table[2]
    timing = timing_from_flag(nasdaq_flag)
    if timing is not None:
        return calendar_default_t0(d, timing), f"nasdaq flag {str(nasdaq_flag)!r} ({timing} default)"
    return calendar_default_t0(d, "AMC"), "calendar default (AMC)"


def upcoming_events(settings: Settings, days: int = 14) -> pd.DataFrame:
    """Future events for the event universe from the FMP calendar, with the consensus taken from
    the newest archived snapshot when present and `expected_t0` from `expected_t0_for`."""
    from ..data.fmp import FMPClient
    from ..data.nasdaq import NasdaqClient
    from ..universe import event_universe, load_universe

    universe = load_universe(settings)
    ev = event_universe(universe)
    by_underlying = {str(r[U.underlying]).upper(): r for _, r in ev.iterrows() if not _isna(r[U.underlying])}
    today = _today_ny()
    cal = FMPClient(settings).earnings_calendar(pd.Timestamp(today), pd.Timestamp(today) + pd.Timedelta(days=int(days)))
    snapshots = load_consensus_snapshots(settings)
    events = load_events(settings) if settings.events_path.exists() else None
    nasdaq = NasdaqClient(settings)
    nasdaq_flags: dict[date, dict[str, Any]] = {}

    def nasdaq_flag(sym: str, day: date) -> Any:
        if day not in nasdaq_flags:
            flags: dict[str, Any] = {}
            try:
                rows = nasdaq.earnings_calendar(pd.Timestamp(day))
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("Nasdaq calendar unavailable for %s: %s", day, exc)
            else:
                for rec in rows.to_dict("records"):
                    flags.setdefault(str(rec["symbol"]).upper(), rec.get("time_flag"))
            nasdaq_flags[day] = flags
        return nasdaq_flags[day].get(sym)

    rows = []
    for _, r in cal.iterrows():
        sym = str(r[U.symbol]).upper()
        u = by_underlying.get(sym)
        if u is None:
            continue
        d = _as_date(r[E.report_date_ny])
        expected_t0, expected_src = expected_t0_for(events, sym, d, nasdaq_flag(sym, d))
        snap = consensus_before(snapshots, sym, d, None)
        rows.append({
            E.underlying: sym, E.market: u[U.market], E.kind: u[U.kind], E.report_date_ny: d,
            "expected_t0": expected_t0, "expected_t0_source": expected_src,
            E.eps_estimate: snap["eps_estimate"] if snap else r[E.eps_estimate],
            E.rev_estimate: snap["rev_estimate"] if snap else r[E.rev_estimate],
            E.n_estimates: snap["n_estimates"] if snap else pd.NA,
            E.estimate_source: "consensus_snapshot" if snap else "fmp_calendar",
            E.estimate_snapshot_time: snap["snapshot_time"] if snap else pd.NaT,
        })
    out = pd.DataFrame(rows, columns=UPCOMING_COLUMNS)
    out["expected_t0"] = _to_utc_ns(out["expected_t0"]) if len(out) else pd.Series(dtype="datetime64[ns, UTC]")
    out[E.estimate_snapshot_time] = _to_utc_ns(out[E.estimate_snapshot_time]) if len(out) else pd.Series(dtype="datetime64[ns, UTC]")
    out[E.n_estimates] = pd.array(pd.to_numeric(out[E.n_estimates], errors="coerce"), dtype="Int64")
    return out.sort_values([E.report_date_ny, E.underlying], kind="mergesort").reset_index(drop=True)


def snapshot_consensus(settings: Settings, *, days: int = 14,
                       now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Archiver hook: fetch the FMP earnings calendar (one request) and the Nasdaq calendar for
    the next `days` days and append to data/archive/consensus/<UTC date>.parquet with columns
    snapshot_time, symbol, report_date_ny, eps_estimate, rev_estimate, n_estimates, vendor,
    vendor_last_updated. Returns the rows written."""
    from ..data.archive import append_dedup
    from ..data.fmp import FMPClient
    from ..data.nasdaq import NasdaqClient

    now_ts = to_utc(now, assume_tz=UTC) if now is not None else pd.Timestamp(utcnow())
    now_ts = now_ts.as_unit("ns")
    today = now_ts.tz_convert(NY).date()
    last = today + timedelta(days=int(days))
    frames: list[pd.DataFrame] = []
    cal = FMPClient(settings).earnings_calendar(pd.Timestamp(today), pd.Timestamp(last))
    if len(cal):
        frames.append(pd.DataFrame({
            "snapshot_time": now_ts, "symbol": cal[U.symbol].astype(str).str.upper().to_numpy(),
            "report_date_ny": cal[E.report_date_ny].to_numpy(),
            "eps_estimate": cal[E.eps_estimate].astype("float64").to_numpy(),
            "rev_estimate": cal[E.rev_estimate].astype("float64").to_numpy(),
            "n_estimates": pd.array([pd.NA] * len(cal), dtype="Int64"),
            "vendor": "fmp",
            "vendor_last_updated": _to_utc_ns(cal["last_updated"]).to_numpy(),
        }))
    nasdaq = NasdaqClient(settings)
    for i in range(int(days) + 1):
        day = today + timedelta(days=i)
        try:
            rows = nasdaq.earnings_calendar(pd.Timestamp(day))
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("Nasdaq calendar for %s not snapshotted: %s", day, exc)
            continue
        if len(rows) == 0:
            continue
        frames.append(pd.DataFrame({
            "snapshot_time": now_ts, "symbol": rows["symbol"].astype(str).str.upper().to_numpy(),
            "report_date_ny": rows["report_date_ny"].to_numpy(),
            "eps_estimate": rows["eps_estimate"].astype("float64").to_numpy(),
            "rev_estimate": math.nan,
            "n_estimates": pd.array(rows["n_estimates"].tolist(), dtype="Int64"),
            "vendor": "nasdaq",
            "vendor_last_updated": pd.NaT,
        }))
    if not frames:
        return _empty_snapshots()
    out = pd.concat(frames, ignore_index=True).reindex(columns=SNAPSHOT_COLUMNS)
    out["snapshot_time"] = _to_utc_ns(out["snapshot_time"])
    out["vendor_last_updated"] = _to_utc_ns(out["vendor_last_updated"])
    out["report_date_ny"] = [None if _isna(x) else _as_date(x) for x in out["report_date_ny"]]
    out = out.sort_values(["symbol", "report_date_ny", "vendor"], kind="mergesort").reset_index(drop=True)
    path = consensus_path(settings, now_ts.date())
    append_dedup(path, out, key=["snapshot_time", "symbol", "report_date_ny", "vendor"],
                 sort=["snapshot_time", "symbol", "report_date_ny", "vendor"])
    return out


def consensus_path(settings: Settings, day: date) -> Path:
    return settings.consensus_archive_dir / f"{day.isoformat()}.parquet"
