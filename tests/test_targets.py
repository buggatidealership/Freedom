import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freedom.schemas import CONTINUATION_DEAD_BAND, C, E, T
from freedom.targets import build_price_path, checkpoint_times, compute_targets, price_at
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
