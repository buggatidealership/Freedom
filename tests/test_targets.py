import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freedom.schemas import CHECKPOINTS, CONTINUATION_DEAD_BAND, C, E, T
from freedom.targets import (
    build_price_path,
    build_targets,
    checkpoint_times,
    compute_targets,
    corporate_action_ex,
    p0_buffer_for,
    price_at,
)
from freedom.timeutil import to_utc

FIX = Path(__file__).parent / "fixtures" / "hyperliquid"


def _hl_bars(name: str, interval: str) -> pd.DataFrame:
    raw = json.load(open(FIX / name))
    td = {"5m": 5, "1h": 60}[interval]
    df = pd.DataFrame({
        C.market: "xyz:NVDA", C.interval: interval,
        C.t: pd.to_datetime([r["t"] for r in raw], unit="ms", utc=True),
        C.open: [float(r["o"]) for r in raw], C.high: [float(r["h"]) for r in raw],
        C.low: [float(r["l"]) for r in raw], C.close: [float(r["c"]) for r in raw],
        C.volume: [float(r["v"]) for r in raw], C.n_trades: [int(r["n"]) for r in raw],
        C.source: "hl_live",
    })
    df[C.t_end] = df[C.t] + pd.Timedelta(minutes=td)
    return df.sort_values(C.t).reset_index(drop=True)


T0 = to_utc("2026-08-26 20:21:19", assume_tz="UTC")  # NVDA 8-K acceptance


def _event(source: str = "detected"):
    return pd.Series({E.event_id: "NVDA:2026-07", E.underlying: "NVDA", E.market: "xyz:NVDA", E.t0: T0,
                      E.t0_source: source})


def test_checkpoints_use_exchange_calendar():
    cps = checkpoint_times(T0)
    assert cps["5m"] == T0 + pd.Timedelta(minutes=5)
    assert cps["next_open"] == to_utc("2026-08-27 13:30", assume_tz="UTC")
    assert cps["next_open_30m"] == to_utc("2026-08-27 14:00", assume_tz="UTC")
    assert cps["next_close"] == to_utc("2026-08-27 20:00", assume_tz="UTC")
    assert cps["24h"] == T0 + pd.Timedelta(hours=24)
    # a Friday-evening release: next open is Monday
    fri = to_utc("2026-08-28 20:30", assume_tz="UTC")
    assert checkpoint_times(fri)["next_open"] == to_utc("2026-08-31 13:30", assume_tz="UTC")


def test_price_at_never_uses_a_bar_containing_the_instant():
    bars = _hl_bars("candles_xyzNVDA_5m_20260826.json", "5m")
    # the 20:20-20:25 bar contains the release and crashed to 207.94; it must not be used
    p, t_end = price_at(bars, T0)
    assert t_end == to_utc("2026-08-26 20:20", assume_tz="UTC")
    assert p == pytest.approx(211.07)
    # strictly_before at an exact boundary steps back one more bar
    p2, t2 = price_at(bars, to_utc("2026-08-26 20:20", assume_tz="UTC"), strictly_before=True)
    assert t2 == to_utc("2026-08-26 20:15", assume_tz="UTC")
    assert price_at(bars, bars[C.t].min() - pd.Timedelta(minutes=1)) is None
    assert price_at(pd.DataFrame(), T0) is None


def test_compute_targets_on_real_reaction():
    bars = _hl_bars("candles_xyzNVDA_5m_20260826.json", "5m")
    ev = _event()
    tg = compute_targets(ev, bars, None)
    assert tg[T.p0] == pytest.approx(211.07)
    assert tg[T.p0_time] == to_utc("2026-08-26 20:20", assume_tz="UTC")
    # +5m: last bar ending <= 20:26:19 is 20:20-20:25 (close 207.94): the initial dump
    assert tg[T.p("5m")] == pytest.approx(207.94)
    assert tg[T.r("5m")] == pytest.approx(math.log(207.94 / 211.07))
    assert tg[T.r("5m")] < 0
    # independent recomputation of the 24h endpoint from the raw bars
    end = T0 + pd.Timedelta(hours=24)
    last = bars[bars[C.t_end] <= end].iloc[-1]
    assert tg[T.p("24h")] == pytest.approx(float(last[C.close]))
    assert tg[T.r("24h")] == pytest.approx(math.log(float(last[C.close]) / 211.07))
    assert tg[T.r("24h")] > 0.05
    assert tg[T.direction] == 1.0 and tg[T.magnitude] == pytest.approx(abs(tg[T.r("24h")]))
    # the first 30 minutes were negative (-0.62%) and the day ended +7%: a REVERSAL, not a continuation
    assert tg[T.r("30m")] < -CONTINUATION_DEAD_BAND
    assert tg[T.continuation_30m] == -1.0
    assert tg[T.continuation_15m] == -1.0
    assert tg[T.price_interval] == "5m" and tg[T.price_source] == "hl_live"
    assert tg[T.price_market] == "xyz:NVDA"
    assert tg[T.s("5m")] == pytest.approx(79 / 60, abs=0.01)  # 20:26:19 minus bar end 20:25:00
    assert 23 < tg[T.horizon_actual_h] < 24.2
    assert bool(tg[T.h24_in_closure]) is True  # 16:21 ET next day is after the regular close
    assert np.isnan(tg[T.ar("24h")])  # no market path given


def test_sec_8k_source_backs_off_three_minutes():
    bars = _hl_bars("candles_xyzNVDA_5m_20260826.json", "5m")
    tg = compute_targets(_event("sec_8k"), bars, None)
    # t0 - 3 min = 20:18:19 -> last bar ending at or before that is [20:10, 20:15)
    assert tg[T.p0_time] == to_utc("2026-08-26 20:15", assume_tz="UTC")
    assert tg[T.p0] == pytest.approx(210.63)
    assert tg[T.p0_staleness_min] == pytest.approx(3 + 19 / 60, abs=0.01)


def test_coarse_bars_never_resolve_targets():
    coarse = _hl_bars("candles_xyzNVDA_1h_20260824_29.json", "1h")
    tg = compute_targets(_event(), coarse, None)
    assert np.isnan(tg[T.p0]) and tg[T.price_interval] is None
    assert tg[T.label_reason] == "coarse_bars"


def _flat_1m_bars(start: str, end: str, price: float, source: str = "fmp_intraday") -> pd.DataFrame:
    t = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    return pd.DataFrame({C.market: "ASML", C.interval: "1m", C.t: t, C.t_end: t + pd.Timedelta(minutes=1),
                         C.open: price, C.high: price, C.low: price, C.close: price, C.volume: 1000.0,
                         C.n_trades: pd.array([pd.NA] * len(t), dtype="Int64"), C.source: source})


def test_stale_p0_keeps_the_anchor_but_yields_no_labels():
    """An overnight release on the FMP proxy (no bars 20:00-04:00 ET): the last pre-release bar
    is hours old, so p0 is recorded and every checkpoint and label stays NaN."""
    t0 = to_utc("2026-10-15 05:00", assume_tz="UTC")  # ASML: 07:00 CEST, no P0 buffer for this source
    ev = pd.Series({E.event_id: "ASML:2026-09", E.underlying: "ASML", E.market: "xyz:ASML", E.t0: t0,
                    E.t0_source: "issuer_clock"})
    evening = _flat_1m_bars("2026-10-14 20:00", "2026-10-15 00:00", 100.0)  # last bar ends 00:00 UTC
    after = _flat_1m_bars("2026-10-15 05:00", "2026-10-16 06:00", 110.0)  # the reaction
    path = pd.concat([evening, after], ignore_index=True)
    flat = {C.open: 100.0, C.high: 100.0, C.low: 100.0, C.close: 100.0}
    tg = compute_targets(ev, path, path.assign(**flat))
    assert tg[T.p0] == pytest.approx(100.0) and tg[T.p0_time] == to_utc("2026-10-15 00:00", assume_tz="UTC")
    assert tg[T.p0_staleness_min] == pytest.approx(300.0)
    assert tg[T.label_reason] == "p0_stale"
    assert tg[T.price_source] == "fmp_intraday" and tg[T.price_interval] == "1m"
    for cp in CHECKPOINTS:
        assert np.isnan(tg[T.r(cp)]) and np.isnan(tg[T.ar(cp)]) and np.isnan(tg[T.p(cp)]), cp
        assert pd.isna(tg[T.t(cp)]) and np.isnan(tg[T.s(cp)]), cp
    for col in (T.direction, T.magnitude, T.continuation_15m, T.continuation_30m, T.horizon_actual_h):
        assert np.isnan(tg[col]), col
    # a bar ending two minutes before the release is inside the limit: the same event has labels
    fresh = pd.concat([evening, _flat_1m_bars("2026-10-15 00:00", "2026-10-15 04:58", 100.0), after],
                      ignore_index=True)
    tg2 = compute_targets(ev, fresh, fresh.assign(**flat))
    assert tg2[T.p0_time] == to_utc("2026-10-15 04:58", assume_tz="UTC")
    assert tg2[T.p0_staleness_min] == pytest.approx(2.0) and tg2[T.label_reason] is None
    assert tg2[T.r("5m")] == pytest.approx(math.log(1.1)) and tg2[T.r("24h")] == pytest.approx(math.log(1.1))
    assert tg2[T.ar("24h")] == pytest.approx(math.log(1.1)) and tg2[T.direction] == 1.0
    assert not np.isnan(tg2[T.r("next_open")]) and not np.isnan(tg2[T.r("next_close")])
    # the 8-K buffer counts against the limit: three minutes more and the 04:58 bar is stale again
    ev8k = ev.copy()
    ev8k[E.t0_source] = "sec_8k"
    tg3 = compute_targets(ev8k, fresh, None)
    assert tg3[T.p0_time] == to_utc("2026-10-15 04:57", assume_tz="UTC") and tg3[T.label_reason] is None
    tg4 = compute_targets(ev8k, fresh, None, p0_buffer=pd.Timedelta(minutes=8))
    assert tg4[T.p0_time] == to_utc("2026-10-15 04:52", assume_tz="UTC")
    assert tg4[T.p0_staleness_min] == pytest.approx(0.0)
    assert tg4[T.label_reason] is None and not np.isnan(tg4[T.r("24h")])
    # a stale benchmark anchor gives no abnormal return, the plain returns stand
    stale_market = pd.concat([evening, after], ignore_index=True).assign(**flat)
    tg5 = compute_targets(ev, fresh, stale_market)
    assert tg5[T.r("24h")] == pytest.approx(math.log(1.1)) and np.isnan(tg5[T.ar("24h")])
    # the other empty rows say why
    assert compute_targets(ev, pd.DataFrame(), None)[T.label_reason] == "no_path"
    assert compute_targets(ev, after, None)[T.label_reason] == "no_p0"
    short = fresh[fresh[C.t_end] <= t0 + pd.Timedelta(hours=3)]
    assert compute_targets(ev, short, None)[T.label_reason] == "no_24h_bar"


def test_continuation_label_follows_the_early_reaction_sign():
    bars = _hl_bars("candles_xyzNVDA_5m_20260826.json", "5m")
    # synthetic: a negative early reaction that keeps falling must be a continuation (+1)
    down = bars.copy()
    post = down[C.t_end] > T0
    minutes = ((down[C.t_end] - T0) / pd.Timedelta(minutes=1)).clip(lower=0)
    down.loc[post, C.close] = 211.07 * (1 - 0.01 - 0.0005 * minutes[post])
    tg = compute_targets(_event(), down, None)
    assert tg[T.r("30m")] < 0 and tg[T.r("24h")] < tg[T.r("30m")]
    assert tg[T.continuation_30m] == 1.0 and tg[T.continuation_15m] == 1.0
    # dead band: an early move inside 25 bp has no sign to extend
    flat = bars.copy()
    flat.loc[post, C.close] = 211.07 * (1 + 0.001)
    flat.loc[down[C.t_end] > T0 + pd.Timedelta(hours=2), C.close] = 211.07 * 1.05
    tg2 = compute_targets(_event(), flat, None)
    assert np.isnan(tg2[T.continuation_30m]) and tg2[T.direction] == 1.0


def test_compute_targets_marks_missing_checkpoints():
    bars = _hl_bars("candles_xyzNVDA_5m_20260826.json", "5m")
    truncated = bars[bars[C.t_end] <= T0 + pd.Timedelta(hours=3)]
    tg = compute_targets(_event(), truncated, None)
    assert not np.isnan(tg[T.r("2h")])
    assert np.isnan(tg[T.r("24h")]) and np.isnan(tg[T.direction])
    # no bars at all after the release -> everything NaN but p0 still resolvable
    pre = bars[bars[C.t_end] <= T0]
    tg2 = compute_targets(_event(), pre, None)
    assert tg2[T.p0] == pytest.approx(211.07) and np.isnan(tg2[T.r("5m")])
    assert np.isnan(compute_targets(_event(), pd.DataFrame(), None)[T.p0])


def test_abnormal_return_subtracts_market():
    bars = _hl_bars("candles_xyzNVDA_5m_20260826.json", "5m")
    mkt = bars.copy()
    mkt[C.close] = 100.0  # flat benchmark
    tg = compute_targets(_event(), bars, mkt)
    assert tg[T.ar("24h")] == pytest.approx(tg[T.r("24h")])
    mkt2 = bars.copy()
    mkt2[C.close] = np.where(mkt2[C.t_end] > T0, 102.0, 100.0)  # benchmark +2% after t0
    tg2 = compute_targets(_event(), bars, mkt2)
    assert tg2[T.ar("24h")] == pytest.approx(tg2[T.r("24h")] - math.log(1.02))


def test_build_price_path_prefers_fine_perp_then_equity(settings):
    fine = _hl_bars("candles_xyzNVDA_5m_20260826.json", "5m")
    coarse = _hl_bars("candles_xyzNVDA_1h_20260824_29.json", "1h")
    equity = fine.copy()
    equity[C.source] = "fmp_intraday"
    ev = _event()
    assert build_price_path(settings, ev, market_bars=fine, equity_bars=equity)[C.source].iloc[0] == "hl_live"
    assert build_price_path(settings, ev, market_bars=coarse, equity_bars=equity)[C.source].iloc[0] == "fmp_intraday"
    # coarse perp bars are never a fallback
    assert len(build_price_path(settings, ev, market_bars=coarse, equity_bars=None)) == 0
    short = fine[fine[C.t_end] <= T0 + pd.Timedelta(hours=2)]
    assert len(build_price_path(settings, ev, market_bars=short, equity_bars=None)) == 0



def test_upcoming_events_never_hit_providers(settings):
    """A release in the future has no bars; the loader must not spend a request on it."""
    from freedom.targets.loaders import load_event_bars

    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"provider called: {name}")

    ev = pd.Series({E.event_id: "NVDA:2026-10", E.underlying: "NVDA", E.market: "xyz:NVDA",
                    E.t0: to_utc("2026-11-18 21:05", assume_tz="UTC"), E.t0_source: "calendar_flag"})
    path, mkt = load_event_bars(settings, ev, hl=Boom(), fmp=Boom(), benchmark_market="xyz:SP500",
                                benchmark_equity="SPY", now=to_utc("2026-09-02 12:00", assume_tz="UTC"))
    assert len(path) == 0 and mkt is None


class Recorder:
    """Provider stand-in that records the method asked for and refuses to answer."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name):
        self.calls.append(name)
        raise AssertionError(f"provider called: {name}")


def test_context_loader_never_asks_for_a_window_entirely_in_the_future(settings):
    """The feature loader's fallback (raw bars around the release when no source covers the
    target window) must not spend a request on an upcoming event either; a release later
    today still gets its pre-release bars."""
    from freedom.features.loaders import ContextLoader, EventInputs, UnderlyingInputs

    ev = pd.Series({E.event_id: "NVDA:2026-10", E.underlying: "NVDA", E.market: "xyz:NVDA",
                    E.t0: to_utc("2026-11-18 21:05", assume_tz="UTC"), E.t0_source: "calendar_flag",
                    E.has_perp_at_t0: True})
    events = pd.DataFrame([ev])
    hl, fmp = Recorder(), Recorder()
    loader = ContextLoader(settings, events, hl=hl, fmp=fmp, now=to_utc("2026-09-02 12:00", assume_tz="UTC"))
    assert loader.event_bars(ev) == (None, None)
    assert hl.calls == [] and fmp.calls == []
    # the context carries the P0 buffer setting to the feature groups
    settings.p0_buffer_minutes_sec_8k = 1.5
    ctx = loader.context(ev, "post_30m", ev[E.t0] + pd.Timedelta(minutes=30), history=None, bars=None,
                         market_bars=None, uinputs=UnderlyingInputs(underlying="NVDA"), einputs=EventInputs())
    assert ctx.p0_buffer_minutes_sec_8k == 1.5 and ctx.horizon_hours == settings.horizon_hours
    # the release is later today: [t0 - 1d, ...] has started, the pre-release bars are wanted
    later_today = ContextLoader(settings, events, hl=Recorder(), fmp=fmp,
                                now=to_utc("2026-11-18 12:00", assume_tz="UTC"))
    assert later_today.event_bars(ev) == (None, None)  # the fakes refuse, the loader degrades
    assert "intraday" in fmp.calls


def _resolved_events(*rows: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_p0_buffer_setting_reaches_the_targets(settings, monkeypatch):
    """Settings.p0_buffer_minutes_sec_8k is part of the config hash, so it must change P0."""
    from freedom.targets import loaders

    bars = _hl_bars("candles_xyzNVDA_5m_20260826.json", "5m")
    monkeypatch.setattr(loaders, "load_event_bars", lambda *_a, **_k: (bars, None))
    detected = _event("detected")
    detected[E.event_id] = "NVDA:detected"
    events = _resolved_events(_event("sec_8k"), detected)
    default = build_targets(settings, events, write=False).set_index(T.event_id)
    assert default.loc["NVDA:2026-07", T.p0_time] == to_utc("2026-08-26 20:15", assume_tz="UTC")
    assert default.loc["NVDA:2026-07", T.p0] == pytest.approx(210.63)
    settings.p0_buffer_minutes_sec_8k = 0.0
    zero = build_targets(settings, events, write=False).set_index(T.event_id)
    assert zero.loc["NVDA:2026-07", T.p0_time] == to_utc("2026-08-26 20:20", assume_tz="UTC")  # last bar ending <= t0
    assert zero.loc["NVDA:2026-07", T.p0] == pytest.approx(211.07)
    assert zero.loc["NVDA:2026-07", T.p0_staleness_min] == pytest.approx(79 / 60, abs=0.01)
    # other sources never back off, whatever the setting
    for tg in (default, zero):
        assert tg.loc["NVDA:detected", T.p0_time] == to_utc("2026-08-26 20:20", assume_tz="UTC")
    assert p0_buffer_for(_event("sec_8k"), 1.5) == pd.Timedelta(minutes=1.5)
    assert p0_buffer_for(_event("sec_8k")) == pd.Timedelta(minutes=3)
    assert p0_buffer_for(_event("manual"), 1.5) == pd.Timedelta(0)


def test_pending_rows_get_nan_targets_without_a_provider_request(settings, monkeypatch):
    """`freedom events` writes pending rows (t0 NaT) at budget exhaustion; the targets pass
    must keep one row per event_id and not die on them."""
    from freedom.targets import loaders

    bars = _hl_bars("candles_xyzNVDA_5m_20260826.json", "5m")
    asked: list[str] = []

    def fake_load(_settings, ev, **_kw):
        asked.append(ev[E.event_id])
        return bars, None

    monkeypatch.setattr(loaders, "load_event_bars", fake_load)
    placeholder = pd.Series({E.event_id: "AAPL:pending", E.underlying: "AAPL", E.market: "xyz:AAPL",
                             E.t0: pd.NaT, E.t0_source: None, E.pending: True,
                             E.flags: "pending;earnings_history_pending"})
    pending = pd.Series({E.event_id: "NVDA:2026-04", E.underlying: "NVDA", E.market: "xyz:NVDA",
                         E.t0: pd.NaT, E.t0_source: None, E.pending: True, E.flags: "pending"})
    resolved = _event("sec_8k")
    resolved[E.pending] = False
    events = _resolved_events(placeholder, pending, resolved)  # newest first, like the table
    settings.fmp_api_key = None
    tg = build_targets(settings, events, write=True)
    assert list(tg[T.event_id]) == ["AAPL:pending", "NVDA:2026-04", "NVDA:2026-07"]
    assert tg[T.r("24h")].isna().tolist() == [True, True, False]
    assert tg[T.h24_in_closure].isna().tolist() == [True, True, False]
    assert asked == ["NVDA:2026-07"]
    back = pd.read_parquet(settings.targets_path)
    assert len(back) == 3 and back[T.p0].isna().tolist() == [True, True, False]
    # the empty row can be built without a t0 (this is what the failure handler falls back to)
    row = compute_targets(pending, pd.DataFrame(), None)
    assert row[T.event_id] == "NVDA:2026-04" and np.isnan(row[T.p0]) and np.isnan(row[T.r("24h")])
    assert pd.isna(row[T.h24_in_closure]) and pd.isna(row[T.p0_time]) and row[T.label_reason] == "no_t0"
    # ... and the resolved row with a NaN pending column is still resolved
    resolved[E.pending] = np.nan
    assert not np.isnan(build_targets(settings, _resolved_events(resolved), write=False)[T.r("24h")].iloc[0])


def _split_path(bars: pd.DataFrame, ex: pd.Timestamp, ratio: float = 10.0) -> pd.DataFrame:
    """An unadjusted perp path across a `ratio`:1 forward split at `ex`."""
    out = bars.copy()
    after = (out[C.t] >= ex).to_numpy()
    for col in (C.open, C.high, C.low, C.close):
        out.loc[after, col] = out.loc[after, col] / ratio
    return out


def test_split_ex_date_inside_the_window_voids_the_headline_label():
    bars = _hl_bars("candles_xyzNVDA_5m_20260826.json", "5m")
    ex = to_utc("2026-08-27 00:00", assume_tz="America/New_York")  # 04:00 UTC, between t0 and t0 + 24h
    clean = compute_targets(_event(), bars, None)
    # the hazard: an unadjusted perp path across a 10:1 split reads as a -230% day
    naive = compute_targets(_event(), _split_path(bars, ex), None)
    assert naive[T.r("24h")] < -2 and naive[T.direction] == -1.0
    for ex_value in (ex, ex.tz_convert("America/New_York").date(), "2026-08-27"):
        ev = _event()
        ev[E.ca_ex_date] = ex_value
        assert corporate_action_ex(ev) == ex
        tg = compute_targets(ev, _split_path(bars, ex), None)
        for cp in ("5m", "15m", "30m", "60m", "2h"):  # bars before the ex-date: same as the clean path
            assert tg[T.r(cp)] == pytest.approx(clean[T.r(cp)]), cp
        for cp in ("next_open", "next_open_30m", "next_close", "24h"):  # perp bars from the ex-date on
            assert np.isnan(tg[T.r(cp)]) and pd.isna(tg[T.t(cp)]) and np.isnan(tg[T.s(cp)]), cp
        assert np.isnan(tg[T.direction]) and np.isnan(tg[T.magnitude])
        assert np.isnan(tg[T.continuation_15m]) and np.isnan(tg[T.continuation_30m])
        assert np.isnan(tg[T.horizon_actual_h]) and tg[T.p0] == pytest.approx(clean[T.p0])
        assert tg[T.label_reason] == "corporate_action" and clean[T.label_reason] is None
    # the FMP proxy is split-adjusted: only the headline label goes, the intermediate checkpoints stay
    proxy = bars.copy()
    proxy[C.source] = "fmp_intraday"
    ev = _event()
    ev[E.ca_ex_date] = ex
    tg = compute_targets(ev, proxy, None)
    assert np.isnan(tg[T.r("24h")]) and np.isnan(tg[T.direction])
    for cp in ("5m", "2h", "next_open", "next_close"):
        assert tg[T.r(cp)] == pytest.approx(clean[T.r(cp)]), cp
    # an ex-date before the P0 bar (whole path on the new basis) or after the horizon changes nothing
    for outside in ("2026-08-26", "2026-08-28"):
        ev = _event()
        ev[E.ca_ex_date] = outside
        pd.testing.assert_series_equal(compute_targets(ev, bars, None), clean)
    ev = _event()
    ev[E.ca_ex_date] = pd.NaT
    pd.testing.assert_series_equal(compute_targets(ev, bars, None), clean)
    assert corporate_action_ex(ev) is None and corporate_action_ex(_event()) is None


def _minute_bars(market: str, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    t = pd.date_range(lo.floor("min"), hi.ceil("min"), freq="1min", tz="UTC")
    return pd.DataFrame({C.market: market, C.interval: "1m", C.t: t, C.t_end: t + pd.Timedelta(minutes=1),
                         C.open: 100.0, C.high: 101.0, C.low: 99.0, C.close: 100.5, C.volume: 10.0, C.n_trades: 5,
                         C.source: None})


class _BudgetFMP:
    """FMP stand-in: 1-minute bars for any symbol except those whose budget is 'exhausted'."""

    def __init__(self, exhausted: tuple[str, ...] = ()) -> None:
        self.exhausted, self.calls = set(exhausted), []

    def intraday(self, symbol, interval, start_day, end_day, *, extended=True):
        from freedom.data.base import BudgetExhausted

        self.calls.append(symbol)
        if symbol in self.exhausted:
            raise BudgetExhausted(f"fmp: daily budget exhausted ({symbol})")
        def ny(day):  # the loader passes tz-aware New York midnights, the resolver naive dates
            ts = pd.Timestamp(day)
            return (ts.tz_localize("America/New_York") if ts.tzinfo is None else ts.tz_convert("America/New_York")).tz_convert("UTC")

        lo, hi = ny(start_day), ny(end_day) + pd.Timedelta(days=1)
        return _minute_bars(symbol, lo, hi)


class _LiveHL:
    def candles(self, market, interval, lo, hi):
        return _minute_bars(market, lo - pd.Timedelta(hours=2), hi + pd.Timedelta(hours=2)) if interval == "1m" else None


def test_an_exhausted_benchmark_request_does_not_void_the_event(settings):
    """The SPY bars are one request per event window that the release resolver never made; when
    the budget is gone the event keeps its own path and labels (abnormal returns stay NaN)."""
    from freedom.data.base import BudgetExhausted
    from freedom.targets.loaders import load_event_bars

    t0 = to_utc("2026-07-29 20:05", assume_tz="UTC")
    ev = pd.Series({E.event_id: "HOOD:2026-06", E.underlying: "HOOD", E.market: None, E.t0: t0, E.t0_source: "sec_8k"})
    fmp = _BudgetFMP(exhausted=("SPY",))
    path, mkt = load_event_bars(settings, ev, hl=None, fmp=fmp, benchmark_market="xyz:SP500", benchmark_equity="SPY",
                                now=to_utc("2026-09-04 06:00", assume_tz="UTC"))
    assert len(path) > 0 and path[C.source].iloc[0] == "fmp_intraday" and mkt is None
    assert fmp.calls == ["HOOD", "SPY"]
    # the event's own bars gone with no perp path: the budget checkpoint still reaches the caller
    with pytest.raises(BudgetExhausted):
        load_event_bars(settings, ev, hl=None, fmp=_BudgetFMP(exhausted=("HOOD",)), benchmark_market="xyz:SP500",
                        benchmark_equity="SPY", now=to_utc("2026-09-04 06:00", assume_tz="UTC"))


def test_the_perp_path_survives_an_exhausted_underlying_request(settings):
    from freedom.targets.loaders import load_event_bars

    t0 = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(days=2)  # inside the live 1m window
    ev = pd.Series({E.event_id: "NVDA:2026-07", E.underlying: "NVDA", E.market: "xyz:NVDA", E.t0: t0,
                    E.t0_source: "sec_8k"})
    path, mkt = load_event_bars(settings, ev, hl=_LiveHL(), fmp=_BudgetFMP(exhausted=("NVDA", "SPY")),
                                benchmark_market="xyz:SP500", benchmark_equity="SPY")
    assert len(path) > 0 and path[C.source].iloc[0] == "hl_live"
    assert mkt is not None and mkt[C.source].iloc[0] == "hl_live"  # the perp benchmark needs no FMP request
