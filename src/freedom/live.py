"""Live prediction for one event at one decision time (docs/design.md §10).

`predict_event` fixes the decision instant `as_of`, builds features as of that instant through
the same feature groups the dataset uses, scores the trained model for the decision time and
appends one row to data/live_predictions.parquet:

* pre-release decisions: `as_of = expected_t0 + offset` where `expected_t0` follows the
  design's precedence (events.expected_t0_for): a manual override on the matching
  events.parquet row (`t0_source_live = expected_manual`), the issuer's median 8-K acceptance
  clock over acceptances at or before `now` (`expected_sec_8k`; the gate keeps a replay from
  learning the clock from the event it predicts), then that row's calendar-flag time, the
  Nasdaq flag or the AMC/BMO default (16:05 / 07:00 America/New_York, all
  `expected_calendar_flag`). A row from the upcoming calendar already carries the result. The
  provenance text goes into `schedule_note`. The row is `off_schedule` when `now` is not
  inside [as_of - PRE_WINDOW, expected_t0].
* post-release decisions: `t0_live` comes from events.detect_release_live on 1-minute bars —
  archived plus live Hyperliquid perp candles when the market exists, else FMP extended-hours
  bars — and `as_of = t0_live + k`. The row is `off_schedule` (and must not be traded) when
  `now - t0_live` is outside [k - 1 min, k + max_fill_lag_minutes]. When the 8-K is already
  on EDGAR its acceptance is recorded as `t0_actual` with `t0_lag_s`; otherwise
  `freedom evaluate --live` back-fills it later.
* only bars closed at `now` (t_end <= now) are used anywhere: the forming 1-minute candle the
  providers return is dropped before the detector, the features and the input lags see it.
* the features see the same loader inputs the dataset was built from (features.loaders
  .ContextLoader: funding, perp 1d candles, leverage cap, listing fallback, sector ETF bars,
  VIX bars, same-day event count) so a trained model never scores a block of features it
  always had as missing.
* every row records model_id, the sources used, `input_lag_s_<source>` (now minus the newest
  closed bar / filing each source served) and the feature values, so the live record can be
  scored and compared with the backtest's decision instants. `run_at` is the wall clock; a
  `--now` override is recorded as `now_override` with `replay = True`, and a replay is never a
  live prediction (`freedom evaluate --live` reports it as its own stratum).
* an event is found in data/events.parquet by id, else in the upcoming calendar under the id
  `freedom upcoming` prints — the table's own id when the table has the row, otherwise
  `upcoming_event_id` (<underlying>:<calendar quarter before the report date>) — or under its
  bare underlying when that has a single upcoming event; a calendar hit whose id the table
  knows is predicted from the table row (cik, timing, flags). The row also records
  `report_date_ny` so a minted id can be matched once `freedom events` assigns the fiscal period.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import events as events_mod
from . import features as features_mod
from . import models as models_mod
from .card import build_card
from .config import Settings
from .data.archive import CTX_SUBDIR, load_archive, read_parquet_or_none, write_parquet_atomic
from .data.base import ProviderUnavailable
from .errors import EventNotFound
from .schemas import DECISION_TIMES, NY, UTC, C, D, E, Timing
from .timeutil import to_ny, to_utc

log = logging.getLogger(__name__)

PRE_WINDOW = pd.Timedelta(minutes=60)  # a pre-release prediction is on schedule this long before as_of
BAR_LOOKBACK_DAYS = 15  # 1-minute bars before the report day: the detector's volume baseline
DAILY_LOOKBACK_DAYS = 400  # daily bars for 52-week features
CTX_LOOKBACK_DAYS = 35
UPCOMING_LOOKAHEAD_DAYS = 30
BENCHMARK_MARKET = "xyz:SP500"
BENCHMARK_EQUITY = "SPY"
LIVE_PREDICTIONS_FILE = "live_predictions.parquet"
TOP_CONTRIBUTIONS = 5


class ReleaseNotDetected(RuntimeError):
    """No release could be detected yet on the live bars (nothing to predict)."""


class ModelNotFound(FileNotFoundError):
    """No trained model for the decision time; the message names the train command."""


def live_predictions_path(settings: Settings) -> Path:
    return settings.data_dir / LIVE_PREDICTIONS_FILE


# ---- clients (module-level factories so tests can swap them) ---------------------------------------
def hl_client(settings: Settings):
    from .data.hyperliquid import HyperliquidClient

    return HyperliquidClient(settings)


def fmp_client(settings: Settings):
    from .data.fmp import FMPClient

    try:
        return FMPClient(settings)
    except ProviderUnavailable as exc:
        log.warning("FMP unavailable for live prediction: %s", exc)
        return None


def sec_client(settings: Settings):
    from .data.sec import SECClient

    return SECClient(settings)


# ---- event and model lookup ------------------------------------------------------------------------
def derived_fiscal_period(report_date_ny) -> str:
    """'YYYY-MM' of the calendar quarter end preceding the report date — the fiscal period
    events.fiscal_period_for derives when no filing has fixed it yet."""
    d = pd.Timestamp(report_date_ny)
    quarter_start = pd.Timestamp(year=d.year, month=(d.month - 1) // 3 * 3 + 1, day=1)
    q_end = quarter_start - pd.Timedelta(days=1)
    return f"{q_end.year:04d}-{q_end.month:02d}"


def upcoming_event_id(underlying: str, report_date_ny) -> str:
    """The id `freedom predict --event` takes for a calendar event that has no events.parquet
    row yet: <underlying>:<derived fiscal period>. Once `freedom events` resolves the period
    from filings the table's id can differ (off-calendar fiscal years) — from then on the
    upcoming frame carries the table's id and this one is not used — which is why the live row
    also records report_date_ny."""
    return f"{str(underlying).upper()}:{derived_fiscal_period(report_date_ny)}"


def with_event_ids(upcoming: pd.DataFrame) -> pd.DataFrame:
    """The upcoming-events frame with an event_id column first: an id the events table
    supplied is kept; rows without one (no events.parquet row yet) get
    upcoming_event_id(underlying, report_date_ny)."""
    df = upcoming.copy()
    ids = df[E.event_id].astype(object) if E.event_id in df.columns else pd.Series(None, index=df.index, dtype=object)
    days = df[E.report_date_ny] if E.report_date_ny in df.columns else [None] * len(df)
    minted = [upcoming_event_id(u, d) if not pd.isna(d) and isinstance(u, str) and u else None
              for u, d in zip(df[E.underlying], days, strict=True)]
    ids = ids.where(ids.notna(), pd.Series(minted, index=df.index, dtype=object))
    if E.event_id in df.columns:
        df = df.drop(columns=[E.event_id])
    df.insert(0, E.event_id, ids)
    return df


def find_event(settings: Settings, event_id: str, *, days: int = UPCOMING_LOOKAHEAD_DAYS) -> tuple[pd.Series, pd.DataFrame]:
    """(event row, full events table). The row comes from data/events.parquet, else from the
    upcoming calendar (events.upcoming_events) under the id `freedom upcoming` prints
    (with_event_ids) — or under its bare underlying when that has exactly one upcoming event.
    A calendar hit whose id the table knows returns the table row (cik, timing, flags, ...)."""
    events = events_mod.load_events(settings)
    hit = events[events[E.event_id] == event_id]
    if len(hit):
        return hit.iloc[0].copy(), events
    upcoming = events_mod.upcoming_events(settings, days=days)
    if upcoming is not None and len(upcoming):
        upcoming = with_event_ids(upcoming)
        hit = upcoming[upcoming[E.event_id] == event_id]
        if len(hit) == 0 and ":" not in event_id:
            hit = upcoming[upcoming[E.underlying].astype(str).str.upper() == event_id.upper()]
        if len(hit) == 1:
            row = hit.iloc[0]
            table = events[events[E.event_id] == row[E.event_id]]
            return (table.iloc[0].copy() if len(table) else row.copy()), events
        if len(hit) > 1:
            raise EventNotFound(f"{event_id!r} matches {len(hit)} upcoming events "
                                f"({', '.join(hit[E.event_id].astype(str))}): pass one of these ids")
    raise EventNotFound(f"event {event_id!r} is neither in {settings.events_path} nor in the next {days} days "
                        "of the calendar: run `freedom events` (past events) or `freedom upcoming` to list event ids")


def load_model(settings: Settings, decision: str, model_name: str | None = None) -> tuple[models_mod.BaseModel, dict, Path]:
    """(model, model.json metadata, path) for the trained model of a decision time."""
    root = settings.models_dir / decision
    if model_name is None:
        candidates = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
        if len(candidates) != 1:
            raise ModelNotFound(
                f"{root} holds {len(candidates)} trained models ({', '.join(candidates) or 'none'}); "
                f"run `freedom train --decision-time {decision} --model <name>` first or pass --model")
        model_name = candidates[0]
    path = root / model_name
    meta_path = path / "model.json"
    if not meta_path.exists():
        raise ModelNotFound(f"{path} has no trained model: run `freedom train --model {model_name} "
                            f"--decision-time {decision}` first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    family = str(meta.get("model_name") or meta.get("model") or model_name)
    cls = models_mod.REGISTRY.get(family) or models_mod.REGISTRY.get(model_name)
    if cls is None:
        raise ModelNotFound(f"model family {family!r} is not registered; available: {models_mod.available_models()}")
    model = cls.load(path)
    meta.setdefault("model_name", family)
    return model, meta, path


# ---- schedule -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Schedule:
    decision: str
    offset_min: int
    as_of: pd.Timestamp
    t0: pd.Timestamp  # expected (pre) or detected (post) release instant, UTC
    t0_source: str  # expected_manual | expected_sec_8k | expected_calendar_flag | detected (the live stratum key)
    off_schedule: bool
    note: str  # free text: schedule state and where expected_t0 came from


# events.expected_t0_for's provenance text -> the fixed live stratum key (anything else is a
# calendar-level default: the table's calendar flag, the Nasdaq flag or the AMC/BMO clock)
EXPECTED_T0_SOURCES: tuple[tuple[str, str], ...] = (("events table: manual", "expected_manual"),
                                                    ("median of", "expected_sec_8k"))


def expected_t0_source_key(detail: str) -> str:
    for prefix, key in EXPECTED_T0_SOURCES:
        if detail.startswith(prefix):
            return key
    return "expected_calendar_flag"


def report_day(event: pd.Series) -> pd.Timestamp:
    """The report date as a naive New York calendar day (from report_date_ny, else t0)."""
    d = event.get(E.report_date_ny)
    if d is not None and not pd.isna(d):
        return pd.Timestamp(d).tz_localize(None).normalize()
    t0 = event.get(E.t0)
    if t0 is None or pd.isna(t0):
        raise ValueError(f"event {event.get(E.event_id)} has neither report_date_ny nor t0")
    return to_ny(pd.Timestamp(t0)).tz_localize(None).normalize()


def ny_day_start_utc(day: pd.Timestamp) -> pd.Timestamp:
    return to_utc(pd.Timestamp(day).normalize(), assume_tz=NY)


def pre_schedule(settings: Settings, event: pd.Series, events: pd.DataFrame, decision: str,
                 now: pd.Timestamp) -> Schedule:
    """The pre-release schedule: expected_t0 through events.expected_t0_for (manual override >
    median 8-K clock over acceptances <= now > table calendar flag > the row's AMC/BMO class),
    unless the row came from the upcoming calendar, whose expected_t0 already went through the
    same chain with the Nasdaq flag."""
    offset = DECISION_TIMES[decision]
    day = report_day(event)
    row_t0 = event.get("expected_t0")
    if row_t0 is not None and not pd.isna(row_t0):
        expected_t0 = to_utc(pd.Timestamp(row_t0), assume_tz=UTC)
        detail = str(event.get("expected_t0_source") or "upcoming calendar row")
    else:
        timing = event.get(E.timing)
        timing = str(timing) if isinstance(timing, str) and timing else Timing.amc.value
        expected_t0, detail = events_mod.expected_t0_for(events, str(event[E.underlying]), day.date(), None,
                                                         before=now, timing=timing)
        expected_t0 = to_utc(expected_t0, assume_tz=UTC)
    source = expected_t0_source_key(detail)
    hhmm = to_ny(expected_t0).strftime("%H:%M")
    as_of = expected_t0 + pd.Timedelta(minutes=offset)
    if now > expected_t0:
        off, state = True, f"now is {(now - expected_t0)} after the expected release"
    elif now < as_of - PRE_WINDOW:
        off, state = True, f"now is {(as_of - now)} before the decision instant"
    else:
        off, state = False, "on schedule"
    note = f"{state}; expected_t0 {hhmm} New York from {detail}"
    return Schedule(decision, offset, as_of, expected_t0, source, off, note)


def post_schedule(settings: Settings, event: pd.Series, decision: str, now: pd.Timestamp,
                  bars: pd.DataFrame | None) -> Schedule:
    k = DECISION_TIMES[decision]
    day = report_day(event)
    if bars is None or len(bars) == 0:
        raise ReleaseNotDetected(f"no 1-minute bars for {event.get(E.market) or event[E.underlying]} on "
                                 f"{day.date()}; cannot detect the release")
    t0_live = events_mod.detect_release_live(bars, day, now=now)
    if t0_live is None:
        raise ReleaseNotDetected(f"no release detected yet for {event[E.event_id]} on {day.date()} "
                                 f"(bars up to {bars[C.t_end].max()})")
    t0_live = to_utc(t0_live)
    as_of = t0_live + pd.Timedelta(minutes=k)
    elapsed = now - t0_live
    lo = pd.Timedelta(minutes=k - 1)
    hi = pd.Timedelta(minutes=k + settings.max_fill_lag_minutes)
    if lo <= elapsed <= hi:
        off, note = False, "on schedule"
    else:
        off, note = True, (f"now - t0_live = {elapsed} is outside [{lo}, {hi}]; not tradable at {decision}")
    return Schedule(decision, k, as_of, t0_live, "detected", off, note)


# ---- inputs -------------------------------------------------------------------------------------------
def _concat_bars(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(subset=C.t, keep="last").sort_values(C.t).reset_index(drop=True)


def perp_bars(settings: Settings, hl, market: str, start: pd.Timestamp, end: pd.Timestamp,
              interval: str = "1m") -> pd.DataFrame | None:
    """Archived plus live candles for [start, end); None when neither has any."""
    parts = []
    try:
        archived = load_archive(settings, market, interval, start, end)
        if len(archived):
            parts.append(archived)
    except FileNotFoundError:
        pass
    live = hl.candles(market, interval, start, end) if hl is not None else None
    if live is not None and len(live):
        parts.append(live)
    return _concat_bars(parts) if parts else None


def equity_bars(fmp, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    if fmp is None:
        return None
    b = fmp.intraday(symbol, "1min", to_ny(start).normalize(), to_ny(end).normalize(), extended=True)
    return b if b is not None and len(b) else None


def closed_bars(bars: pd.DataFrame | None, now: pd.Timestamp) -> pd.DataFrame | None:
    """Bars closed at `now` (t_end <= now): providers include the forming candle when asked up
    to now, and a partial bar must reach neither the detector, the features nor the lags.
    None when nothing is closed."""
    if bars is None or len(bars) == 0:
        return None
    keep = bars[pd.to_datetime(bars[C.t_end], utc=True) <= now]
    return keep.reset_index(drop=True) if len(keep) else None


def live_bars(settings: Settings, event: pd.Series, *, hl, fmp, start: pd.Timestamp,
              end: pd.Timestamp) -> tuple[pd.DataFrame | None, str | None]:
    """(1-minute bars for the event's instrument closed at `end`, source) — the perp when the
    market has candles, else the underlying's FMP extended-hours bars; (None, None) when
    neither has a closed bar."""
    market = event.get(E.market)
    if isinstance(market, str) and market:
        try:
            b = closed_bars(perp_bars(settings, hl, market, start, end), end)
        except Exception as exc:  # the FMP proxy is the fallback, not a crash
            log.warning("Hyperliquid bars unavailable for %s: %s", market, exc)
            b = None
        if b is not None:
            return b, "hyperliquid"
    try:
        b = closed_bars(equity_bars(fmp, str(event[E.underlying]), start, end), end)
    except Exception as exc:
        log.warning("FMP bars unavailable for %s: %s", event[E.underlying], exc)
        b = None
    return (b, "fmp") if b is not None else (None, None)


def load_perp_ctx(settings: Settings, market: str | None, as_of: pd.Timestamp,
                  days: int = CTX_LOOKBACK_DAYS) -> pd.DataFrame | None:
    """Archived asset-context snapshots of the market with t <= as_of."""
    if not isinstance(market, str) or ":" not in market:
        return None
    root = settings.archive_dir / CTX_SUBDIR / market.split(":", 1)[0]
    if not root.exists():
        return None
    first = (as_of - pd.Timedelta(days=days)).date().isoformat()
    frames = []
    for p in sorted(root.glob("*.parquet")):
        if p.stem < first or p.stem > as_of.date().isoformat():
            continue
        df = read_parquet_or_none(p)
        if df is not None and len(df):
            frames.append(df[df["market"] == market])
    if not frames:
        return None
    ctx = pd.concat(frames, ignore_index=True)
    ctx["t"] = pd.to_datetime(ctx["t"], utc=True)
    return ctx[ctx["t"] <= as_of].sort_values("t").reset_index(drop=True)


def load_targets(settings: Settings) -> pd.DataFrame | None:
    df = read_parquet_or_none(settings.targets_path)
    if df is None:
        log.warning("%s missing: history features will be empty (run `freedom dataset`)", settings.targets_path)
    return df


def lag_seconds(now: pd.Timestamp, bars: pd.DataFrame | None, col: str = C.t_end) -> float:
    if bars is None or len(bars) == 0 or col not in bars.columns:
        return float("nan")
    latest = pd.to_datetime(bars[col], utc=True).max()
    return float((now - latest).total_seconds())


def sec_backfill(sec, event: pd.Series, day: pd.Timestamp, now: pd.Timestamp) -> tuple[pd.Timestamp | None, float]:
    """(acceptance of the 8-K filed on the report day if already on EDGAR, seconds between now
    and the newest earnings filing served). Never raises."""
    cik = event.get(E.cik)
    if sec is None or cik is None or pd.isna(cik):
        return None, float("nan")
    try:
        filings = sec.earnings_filings(int(cik))
    except Exception as exc:
        log.warning("SEC submissions unavailable: %s", exc)
        return None, float("nan")
    if filings is None or len(filings) == 0:
        return None, float("nan")
    accepted = pd.to_datetime(filings["accepted"], utc=True)
    lo, hi = ny_day_start_utc(day), ny_day_start_utc(day + pd.Timedelta(days=1)) + pd.Timedelta(hours=4)
    same_day = accepted[(accepted >= lo) & (accepted <= hi) & (filings["form"] == "8-K")]
    t0_actual = pd.Timestamp(same_day.iloc[0]) if len(same_day) else None
    return t0_actual, float((now - accepted.max()).total_seconds())


# ---- prediction ---------------------------------------------------------------------------------------
def n_events_same_day(events: pd.DataFrame, event_id: str, day: pd.Timestamp,
                      known: dict | None = None) -> float:
    """Universe events released on the report day (New York date), this one included: the
    dataset loader's count when it has one for this id (a table row with a t0), else the other
    table rows on that date plus this event (a calendar row, or a table row without a t0)."""
    if known is not None and known.get(event_id) is not None:
        return float(known[event_id])
    if events is None or len(events) == 0 or E.t0 not in events.columns:
        return 1.0
    t0 = pd.to_datetime(events[E.t0], utc=True, errors="coerce")
    same = t0.dt.tz_convert(NY).dt.date == day.date()
    others = same & (events[E.event_id].astype(str) != event_id)
    return float(int(others.sum()) + 1)


def loader_extras(settings: Settings, ev: pd.Series, *, hl, fmp, events: pd.DataFrame, as_of: pd.Timestamp,
                  bars: pd.DataFrame | None, daily: pd.DataFrame | None) -> dict:
    """ctx.extra as features.loaders.ContextLoader fills it for the dataset (funding, perp 1d
    candles, leverage cap, listing fallback, sector ETF bars, VIX bars, same-day count), so the
    live features are built from the inputs the model was trained on. The loader is created
    with now=as_of; the underlying's and the sector ETF's daily bars are fetched here through
    as_of (the loader's span stops at yesterday, which would drop the release-day session of an
    AMC event)."""
    from .features.groups import (
        X_FUNDING,
        X_LISTING_START,
        X_MAX_LEVERAGE,
        X_N_EVENTS_SAME_DAY,
        X_PERP_DAILY,
        X_SECTOR_DAILY,
        X_VIX_DAILY,
    )
    from .features.loaders import ContextLoader, UnderlyingInputs

    underlying = str(ev[E.underlying])
    loader = ContextLoader(settings, events, hl=hl, fmp=fmp, now=as_of)
    etf = loader.sector_etf(underlying) if fmp is not None else None
    sector_daily = None
    if etf:
        try:
            sector_daily = fmp.daily(etf, as_of - pd.Timedelta(days=DAILY_LOOKBACK_DAYS), as_of)
        except Exception as exc:
            log.warning("sector ETF %s daily bars unavailable: %s", etf, exc)
    uinputs = UnderlyingInputs(underlying, daily=daily, sector_etf=etf, sector_daily=sector_daily)
    einputs = loader.event_inputs(ev, uinputs, bars)
    day = report_day(ev)
    return {X_N_EVENTS_SAME_DAY: n_events_same_day(events, str(ev[E.event_id]), day, loader.n_same_day),
            X_VIX_DAILY: einputs.vix_daily, X_SECTOR_DAILY: sector_daily, X_PERP_DAILY: einputs.perp_daily,
            X_FUNDING: einputs.funding, X_MAX_LEVERAGE: einputs.max_leverage,
            X_LISTING_START: einputs.listing_start, "sector_etf": etf, **einputs.extra}


def feature_context(settings: Settings, event: pd.Series, schedule: Schedule, *, bars, bar_source,
                    hl, fmp, events: pd.DataFrame, targets: pd.DataFrame | None,
                    sources: dict[str, bool]) -> features_mod.FeatureContext:
    as_of = schedule.as_of
    ev = event.copy()
    ev[E.t0] = schedule.t0
    ev[E.t0_source] = schedule.t0_source
    underlying = str(event[E.underlying])
    daily = market_daily = market_bars = None
    if fmp is not None:
        try:
            daily = fmp.daily(underlying, as_of - pd.Timedelta(days=DAILY_LOOKBACK_DAYS), as_of)
            market_daily = fmp.daily(BENCHMARK_EQUITY, as_of - pd.Timedelta(days=DAILY_LOOKBACK_DAYS), as_of)
            sources["fmp"] = True
        except Exception as exc:
            log.warning("FMP daily bars unavailable: %s", exc)
    start = ny_day_start_utc(report_day(event) - pd.Timedelta(days=BAR_LOOKBACK_DAYS))
    try:
        if bar_source == "hyperliquid":
            market_bars = perp_bars(settings, hl, BENCHMARK_MARKET, start, as_of)
        elif fmp is not None:
            market_bars = equity_bars(fmp, BENCHMARK_EQUITY, start, as_of)
    except Exception as exc:
        log.warning("benchmark bars unavailable: %s", exc)
    history = None
    if targets is not None:
        history = features_mod.history_view(events, targets, underlying, as_of, settings.horizon_hours)
    extra = loader_extras(settings, ev, hl=hl, fmp=fmp, events=events, as_of=as_of, bars=bars, daily=daily)
    return features_mod.FeatureContext(
        event=ev, as_of=as_of, decision_time=schedule.decision, bars=bars, daily=daily,
        market_bars=market_bars, market_daily=market_daily, history=history,
        perp_ctx=load_perp_ctx(settings, event.get(E.market), as_of), horizon_hours=settings.horizon_hours,
        p0_buffer_minutes_sec_8k=float(settings.p0_buffer_minutes_sec_8k),
        extra=extra)


def top_contributions(model: models_mod.BaseModel, X: pd.DataFrame, n: int = TOP_CONTRIBUTIONS) -> list[dict]:
    """Highest-importance features with their current values (importance-ranked, not SHAP)."""
    imp = model.feature_importance()
    if imp is None or len(imp) == 0:
        return []
    imp = pd.Series(imp).astype(float).abs().sort_values(ascending=False)
    out = []
    for name, weight in imp.head(n).items():
        value = X[name].iloc[0] if name in X.columns else float("nan")
        out.append({"feature": str(name), "importance": float(weight),
                    "value": float(value) if value is not None and not pd.isna(value) else float("nan")})
    return out


def _num(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def predict_event(settings: Settings, *, event_id: str, decision: str, model_name: str | None = None,
                  now: pd.Timestamp | str | None = None, hl=None, fmp=None, sec=None,
                  append: bool = True) -> dict:
    """Build features as of the decision instant, score the trained model and append the live
    row. Returns {row, features, contributions, card, schedule, model_meta, consensus}; `card` is
    card.build_card's LONG/SHORT/NO TRADE summary with its reasons. `now` is the
    replay override (`predict --now`): the row then carries replay=True and now_override, and
    run_at stays the wall clock, so a replay can never pass for a live prediction."""
    if decision not in DECISION_TIMES:
        raise ValueError(f"unknown decision time {decision!r}; choose from {sorted(DECISION_TIMES)}")
    wall = pd.Timestamp.now(tz=UTC)
    replay = now is not None
    now_ts = to_utc(now, assume_tz=UTC) if replay else wall
    event, events = find_event(settings, event_id)
    event_id = str(event.get(E.event_id) or event_id)  # the resolved id (a bare underlying resolves to one)
    model, meta, model_path = load_model(settings, decision, model_name)
    hl = hl if hl is not None else hl_client(settings)
    fmp = fmp if fmp is not None else fmp_client(settings)
    day = report_day(event)
    start = ny_day_start_utc(day - pd.Timedelta(days=BAR_LOOKBACK_DAYS))
    bars, bar_source = live_bars(settings, event, hl=hl, fmp=fmp, start=start, end=now_ts)
    sources = {"hyperliquid": bar_source == "hyperliquid", "fmp": bar_source == "fmp", "sec": False}
    if DECISION_TIMES[decision] < 0:
        schedule = pre_schedule(settings, event, events, decision, now_ts)
        t0_actual, sec_lag = None, float("nan")
    else:
        schedule = post_schedule(settings, event, decision, now_ts, bars)
        sec = sec if sec is not None else _sec_or_none(settings)
        t0_actual, sec_lag = sec_backfill(sec, event, day, now_ts)
        sources["sec"] = not math.isnan(sec_lag)
    ctx = feature_context(settings, event, schedule, bars=bars, bar_source=bar_source, hl=hl, fmp=fmp,
                          events=events, targets=load_targets(settings), sources=sources)
    feats = features_mod.build_features(ctx)
    X = pd.DataFrame([feats])
    names = list(model.feature_names_) or list(X.columns)
    X = X.reindex(columns=names)
    p_up = float(np.asarray(model.predict_proba_up(X), dtype=float).reshape(-1)[0])
    r_hat = float(np.asarray(model.predict_return(X), dtype=float).reshape(-1)[0])
    magnitude = float(np.asarray(model.predict_magnitude(X), dtype=float).reshape(-1)[0])
    q = model.residual_q_
    r_lo, r_hi = (r_hat + float(q[0]), r_hat + float(q[1])) if q is not None else (float("nan"), float("nan"))
    model_id = str(meta.get("model_id") or f"{decision}/{meta['model_name']}@{str(meta.get('dataset_sha256', ''))[:8]}")
    missing = [k for k in names if k.endswith(D.missing_suffix) and _num(feats.get(k)) == 1.0]
    row: dict = {
        E.event_id: event_id, E.underlying: str(event[E.underlying]), E.market: event.get(E.market),
        E.report_date_ny: day.date().isoformat(),  # matches the events table once its fiscal period is known
        D.decision_time: decision, D.as_of: schedule.as_of, "run_at": wall,
        "now_override": now_ts if replay else pd.NaT, "replay": replay,
        "t0_used": schedule.t0, "t0_source_live": schedule.t0_source,
        "expected_t0": schedule.t0 if schedule.offset_min < 0 else pd.NaT,
        "t0_live": schedule.t0 if schedule.offset_min >= 0 else pd.NaT,
        "t0_actual": t0_actual if t0_actual is not None else pd.NaT,
        "t0_lag_s": float((t0_actual - schedule.t0).total_seconds()) if t0_actual is not None else float("nan"),
        "off_schedule": bool(schedule.off_schedule), "schedule_note": schedule.note,
        "model_id": model_id, "model_name": meta["model_name"], "model_path": str(model_path),
        "dataset_sha256": meta.get("dataset_sha256"),
        "p_up": p_up, "r_hat": r_hat, "r_lo": r_lo, "r_hi": r_hi, "magnitude_hat": magnitude,
        "bar_source": bar_source, "sources_used": ";".join(k for k, v in sources.items() if v),
        "input_lag_s_hyperliquid": lag_seconds(now_ts, bars) if bar_source == "hyperliquid" else float("nan"),
        "input_lag_s_fmp": lag_seconds(now_ts, bars) if bar_source == "fmp" else float("nan"),
        "input_lag_s_sec": sec_lag,
        E.estimate_source: event.get(E.estimate_source),
        E.estimate_snapshot_time: event.get(E.estimate_snapshot_time, pd.NaT),
        E.eps_estimate: _num(event.get(E.eps_estimate)), E.rev_estimate: _num(event.get(E.rev_estimate)),
        E.n_estimates: _num(event.get(E.n_estimates)),
        "n_features": len(names), "n_features_missing": len(missing),
    }
    row.update({k: feats.get(k, float("nan")) for k in names})
    contribs = top_contributions(model, X)
    card = build_card(row, model=model, X=X, band=float(settings.no_trade_band), fallback=contribs)
    row["call"], row["no_trade_band"] = card["call"], card["band"]  # the call as recorded
    if append:
        append_live_prediction(settings, row)
    return {"row": row, "features": feats, "contributions": contribs, "card": card,
            "schedule": asdict(schedule), "model_meta": meta,
            "consensus": {k: row[k] for k in (E.estimate_source, E.estimate_snapshot_time, E.eps_estimate,
                                                 E.rev_estimate, E.n_estimates)}}


def _sec_or_none(settings: Settings):
    try:
        return sec_client(settings)
    except Exception as exc:  # SEC is a bonus at predict time
        log.warning("SEC client unavailable: %s", exc)
        return None


def append_live_prediction(settings: Settings, row: dict) -> Path:
    path = live_predictions_path(settings)
    new = pd.DataFrame([row])
    for col in (D.as_of, "run_at", "now_override", "t0_used", "expected_t0", "t0_live", "t0_actual",
                E.estimate_snapshot_time):
        if col in new.columns:
            new[col] = pd.to_datetime(new[col], utc=True)
    old = read_parquet_or_none(path)
    merged = new if old is None or old.empty else pd.concat([old, new], ignore_index=True)
    write_parquet_atomic(merged, path)
    return path


def load_live_predictions(settings: Settings) -> pd.DataFrame:
    df = read_parquet_or_none(live_predictions_path(settings))
    if df is None:
        raise FileNotFoundError(f"{live_predictions_path(settings)} not found; run `freedom predict` first")
    return df


__all__ = ["EventNotFound", "ModelNotFound", "ReleaseNotDetected", "Schedule", "append_live_prediction",
           "closed_bars", "derived_fiscal_period", "find_event", "live_predictions_path",
           "load_live_predictions", "load_model", "predict_event", "upcoming_event_id", "with_event_ids"]
