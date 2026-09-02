"""Offline tests for the feature groups (fixtures only, no network).

The two look-ahead traps from docs/design.md section 6 are here: a bar that starts before the
decision instant but ends after it must not affect any feature, and at post_60m the event's own
r_24h must not either. Plus: history gating, pre/post admissibility, reaction features equal to
the targets' returns, and the __missing companions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freedom.features import (
    REGISTRY,
    FeatureContext,
    admissible_groups,
    build_features,
    decision_as_of,
    history_view,
    phase_of,
)
from freedom.features.groups import (
    GROUP_KEYS,
    X_FUNDING,
    X_MAX_LEVERAGE,
    X_N_EVENTS_SAME_DAY,
    X_PERP_DAILY,
    X_SECTOR_DAILY,
    X_VIX_DAILY,
    cut,
    cut_daily,
    pre_cut,
    session_ends,
)
from freedom.schemas import DECISION_TIMES, C, D, E, T
from freedom.targets import compute_targets
from freedom.timeutil import to_utc

FIX = Path(__file__).parent / "fixtures"
T0 = to_utc("2026-08-26 20:21:19", assume_tz="UTC")  # NVDA 8-K acceptance
T0_RTH = to_utc("2026-08-26 19:16:19", assume_tz="UTC")  # 15:16 ET: in session, the day's bar still open
PRE_GROUPS = {"calendar", "pre_price", "history", "market", "perp_state"}
POST_GROUPS = {"surprise", "reaction"}
POST_TIMES = [d for d in DECISION_TIMES if DECISION_TIMES[d] > 0]


def utc(s: str) -> pd.Timestamp:
    return to_utc(s, assume_tz="UTC")


def hl_bars(name: str = "candles_xyzNVDA_5m_20260826.json", interval: str = "5m") -> pd.DataFrame:
    raw = json.load(open(FIX / "hyperliquid" / name))
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


def fmp_daily(symbol: str = "NVDA") -> pd.DataFrame:
    raw = json.load(open(FIX / "fmp" / "historical-price-eod_NVDA_20260601_0901.json"))
    df = pd.DataFrame(raw)
    days = pd.DatetimeIndex(pd.to_datetime(df["date"])).tz_localize("America/New_York").tz_convert("UTC")
    out = pd.DataFrame({
        C.market: symbol, C.interval: "1d", C.t: days, C.t_end: days + pd.Timedelta(days=1),
        C.open: df["open"].astype(float), C.high: df["high"].astype(float),
        C.low: df["low"].astype(float), C.close: df["close"].astype(float),
        C.volume: df["volume"].astype(float), C.n_trades: np.nan, C.source: "fmp_daily",
    })
    return out.sort_values(C.t).reset_index(drop=True)


def daily_close(day: str) -> float:
    raw = json.load(open(FIX / "fmp" / "historical-price-eod_NVDA_20260601_0901.json"))
    return float(next(r["close"] for r in raw if r["date"] == day))


def funding_frame() -> pd.DataFrame:
    raw = json.load(open(FIX / "hyperliquid" / "funding_xyzNVDA_20260826_28.json"))
    return pd.DataFrame({
        "market": "xyz:NVDA",
        "t": pd.to_datetime([r["time"] for r in raw], unit="ms", utc=True).floor("h"),
        "funding_rate": [float(r["fundingRate"]) for r in raw],
        "premium": [float(r["premium"]) for r in raw],
    })


def event(source: str = "sec_8k", **over) -> pd.Series:
    row = {E.event_id: "NVDA:2026-07", E.underlying: "NVDA", E.market: "xyz:NVDA", E.t0: T0,
           E.t0_source: source, E.timing: "AMC", E.eps_actual: 2.22, E.eps_estimate: 2.09,
           E.rev_actual: 96221000000.0, E.rev_estimate: 92270940000.0, E.n_estimates: 30,
           E.has_perp_at_t0: True, E.listing_start: utc("2025-11-12")}
    row.update(over)
    return pd.Series(row)


def empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=[E.event_id, E.underlying, E.t0, T.r("24h")])


def ctx_for(decision_time: str, ev: pd.Series | None = None, **kw) -> FeatureContext:
    ev = event() if ev is None else ev
    as_of = kw.pop("as_of", None) or decision_as_of(ev[E.t0], decision_time)
    defaults = dict(bars=hl_bars(), daily=fmp_daily(), history=empty_history())
    defaults.update(kw)
    return FeatureContext(event=ev, as_of=as_of, decision_time=decision_time, **defaults)


def values(feats: dict) -> dict:
    return {k: v for k, v in feats.items() if not k.endswith(D.missing_suffix)}


def present(feats: dict) -> dict:
    return {k: v for k, v in values(feats).items() if not math.isnan(v)}


def assert_same_features(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        va, vb = a[k], b[k]
        assert (math.isnan(va) and math.isnan(vb)) or va == vb, k


# ---- registry and admissibility ----------------------------------------------------------------
def test_registry_names_and_admissibility():
    assert set(REGISTRY) == PRE_GROUPS | POST_GROUPS
    for name in PRE_GROUPS:
        assert REGISTRY[name][1] == ("pre", "post"), name
    for name in POST_GROUPS:
        assert REGISTRY[name][1] == ("post",), name
    assert phase_of("pre_5m") == "pre" and phase_of("post_1m") == "post"
    assert set(admissible_groups("pre_5m")) == PRE_GROUPS
    assert set(admissible_groups("post_30m")) == PRE_GROUPS | POST_GROUPS
    assert decision_as_of(T0, "pre_5m") == T0 - pd.Timedelta(minutes=5)
    assert decision_as_of(T0, "post_60m") == T0 + pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        decision_as_of(T0, "post_7m")


def test_pre_only_vs_post_only_admissibility():
    pre = build_features(ctx_for("pre_5m"))
    post = build_features(ctx_for("post_30m"))
    pre_keys = {D.feature_prefix + k for g in PRE_GROUPS for k in GROUP_KEYS[g]}
    post_keys = {D.feature_prefix + k for g in POST_GROUPS for k in GROUP_KEYS[g]}
    assert set(values(pre)) == pre_keys
    assert set(values(post)) == pre_keys | post_keys
    assert not any(k.startswith("f_r_") or k.startswith("f_eps") for k in pre)
    # a post-only group requested at a pre decision time yields nothing rather than leaking
    assert build_features(ctx_for("pre_5m"), groups=["reaction", "surprise"]) == {}
    only_cal = build_features(ctx_for("post_30m"), groups=["calendar"])
    assert set(values(only_cal)) == {D.feature_prefix + k for k in GROUP_KEYS["calendar"]}
    with pytest.raises(ValueError):
        build_features(ctx_for("post_30m"), groups=["calendar", "text"])
    # pre groups are anchored before the release: identical at every post decision time
    posts = {d: build_features(ctx_for(d), groups=sorted(PRE_GROUPS)) for d in POST_TIMES}
    for d in POST_TIMES[1:]:
        assert_same_features(posts[POST_TIMES[0]], posts[d])
    # and the reaction differs between post_30m and post_60m (it really reads later bars)
    assert post["f_r_now"] != build_features(ctx_for("post_60m"))["f_r_now"]


# ---- trap 1: a bar spanning the decision instant -----------------------------------------------
def _corrupt_after(bars: pd.DataFrame, as_of: pd.Timestamp, ends: pd.Series | None = None) -> tuple[pd.DataFrame, int]:
    """Absurd values in every bar that ends after as_of (including the one that started
    before it). `ends` overrides the bar end times (the session closes of daily bars).
    Returns the frame and how many bars span as_of."""
    out = bars.copy()
    after = ((out[C.t_end] if ends is None else ends) > as_of).to_numpy()
    spanning = int((after & (out[C.t] < as_of).to_numpy()).sum())
    for col in (C.open, C.high, C.low, C.close):
        out.loc[after, col] = 9999.0
    out.loc[after, C.volume] = 1e12
    return out, spanning


@pytest.mark.parametrize("decision_time,t0", [("pre_5m", T0), ("post_1m", T0), ("post_30m", T0),
                                              ("post_30m", T0_RTH)])
def test_lookahead_trap_bar_ending_after_as_of_is_ignored(decision_time, t0):
    ev = event() if t0 == T0 else event(**{E.t0: t0, E.timing: "RTH"})
    as_of = decision_as_of(t0, decision_time)
    fine = hl_bars()
    daily = fmp_daily()
    fund = funding_frame()
    snaps = pd.DataFrame({
        "t": [as_of - pd.Timedelta(hours=30), as_of - pd.Timedelta(hours=1), as_of + pd.Timedelta(seconds=1)],
        "market": "xyz:NVDA", "funding": [1e-5, 2e-5, 9.0], "open_interest": [1000.0, 1100.0, 1e9],
        "premium": [1e-4, 2e-4, 9.0], "mark_px": [210.0, 211.0, 9999.0], "oracle_px": [210.0, 211.0, 9999.0],
        "day_ntl_vlm": [5e7, 6e7, 1e15],
    })
    clean = FeatureContext(event=ev, as_of=as_of, decision_time=decision_time, bars=fine,
                           daily=daily, market_bars=fine, market_daily=daily, history=empty_history(),
                           perp_ctx=snaps, extra={X_FUNDING: fund, X_VIX_DAILY: daily, X_SECTOR_DAILY: daily,
                                                  X_PERP_DAILY: daily})
    ref = build_features(clean)

    bad_fine, n_span = _corrupt_after(fine, as_of)
    assert n_span == 1, "the 5-minute bar containing as_of must exist in the fixture"
    # a daily bar is known from its session close (not from its next-midnight t_end): the
    # release-day bar is still open, and so a spanning bar, only for the in-session release
    bad_daily, n_span_daily = _corrupt_after(daily, as_of, session_ends(daily))
    assert n_span_daily == (1 if t0 == T0_RTH else 0)
    bad_fund = fund.copy()
    bad_fund.loc[bad_fund["t"] >= as_of - pd.Timedelta(minutes=1), ["funding_rate", "premium"]] = 9.0
    trap = FeatureContext(event=ev, as_of=as_of, decision_time=decision_time, bars=bad_fine,
                          daily=bad_daily, market_bars=bad_fine, market_daily=bad_daily,
                          history=empty_history(), perp_ctx=snaps,
                          extra={X_FUNDING: bad_fund, X_VIX_DAILY: bad_daily, X_SECTOR_DAILY: bad_daily,
                                 X_PERP_DAILY: bad_daily})
    got = build_features(trap)
    assert_same_features(ref, got)
    # the trap is only meaningful if the features actually read the bars
    have = present(ref)
    assert "f_drift_60m" in have and "f_ret_1d" in have and "f_funding_rate" in have
    assert have["f_premium"] == 2e-4 and have["f_oi_chg_24h"] == pytest.approx(math.log(1.1))
    # the after-close release sees the release-day session, the in-session one the day before
    last, prev = ("2026-08-26", "2026-08-25") if t0 == T0 else ("2026-08-25", "2026-08-24")
    assert have["f_ret_1d"] == pytest.approx(math.log(daily_close(last) / daily_close(prev)))
    if decision_time == "post_30m":
        assert "f_r_30m" in have and "f_path_max" in have and "f_vol_z" in have


def test_cut_uses_bar_end_times():
    bars = hl_bars()
    as_of = utc("2026-08-26 20:16:19")
    c = cut(bars, as_of)
    assert c[C.t_end].max() == utc("2026-08-26 20:15")  # [20:15, 20:20) contains as_of: excluded
    assert cut(bars, bars[C.t_end].min() - pd.Timedelta(seconds=1)) is None
    assert cut(None, as_of) is None and cut(bars.iloc[0:0], as_of) is None
    # pre groups are anchored at min(as_of, t0 - buffer)
    assert pre_cut(ctx_for("pre_5m")) == T0 - pd.Timedelta(minutes=5)
    assert pre_cut(ctx_for("post_30m")) == T0 - pd.Timedelta(minutes=3)
    assert pre_cut(ctx_for("post_30m", event("detected"))) == T0
    # daily bars count from their session close, not from the next-midnight t_end the loaders
    # stamp; every fixture day is a full 16:00 ET session, 16 hours after its NY midnight
    daily = fmp_daily()
    assert (session_ends(daily) == daily[C.t] + pd.Timedelta(hours=16)).all()
    assert cut_daily(daily, utc("2026-08-26 20:00"))[C.t].max() == utc("2026-08-26 04:00")
    assert cut_daily(daily, utc("2026-08-26 19:59:59"))[C.t].max() == utc("2026-08-25 04:00")
    assert cut(daily, utc("2026-08-26 20:00"))[C.t].max() == utc("2026-08-25 04:00")
    assert cut_daily(None, as_of) is None and cut_daily(daily.iloc[0:0], as_of) is None
    # perp 1d candles (UTC-midnight bars) are complete only at their t_end and keep it
    perp = daily.assign(**{C.t: daily[C.t].dt.normalize(), C.t_end: daily[C.t].dt.normalize() + pd.Timedelta(days=1)})
    assert (session_ends(perp) == perp[C.t_end]).all()


# ---- trap 2: the event's own targets ---------------------------------------------------------------
def _events_and_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = hl_bars()
    ev_now = event()
    ev_old = event(**{E.event_id: "NVDA:2026-04", E.t0: utc("2026-05-20 20:21:00"),
                      E.eps_surprise_pct: 6.25, E.rev_surprise_pct: 4.07})
    events = pd.DataFrame([ev_now, ev_old])
    tg_now = compute_targets(ev_now, bars, None)
    tg_old = tg_now.copy()
    tg_old[T.event_id] = "NVDA:2026-04"
    tg_old[T.r("24h")] = 0.05
    return events, pd.DataFrame([tg_now, tg_old])


def test_own_r24h_never_changes_features_at_post_60m():
    events, targets = _events_and_targets()
    as_of = decision_as_of(T0, "post_60m")
    ref = build_features(ctx_for("post_60m", history=history_view(events, targets, "NVDA", as_of)))
    poisoned = targets.copy()
    own = poisoned[T.event_id] == "NVDA:2026-07"
    poisoned.loc[own, T.r("24h")] = 5.0
    poisoned.loc[own, T.direction] = 1.0
    poisoned.loc[own, T.continuation_30m] = 1.0
    hist = history_view(events, poisoned, "NVDA", as_of)
    assert "NVDA:2026-07" not in hist[E.event_id].tolist()
    got = build_features(ctx_for("post_60m", history=hist))
    assert_same_features(ref, got)
    # the older event IS history and its r_24h does reach the history group
    assert got["f_hist_n"] == 1.0 and got["f_hist_last1_r24"] == pytest.approx(0.05)
    assert got["f_days_since_last_event"] == pytest.approx((T0 - utc("2026-05-20 20:21:00")) / pd.Timedelta(days=1))


# ---- history gating ------------------------------------------------------------------------------
def test_history_view_excludes_events_settling_after_as_of():
    as_of = utc("2026-08-26 21:21:19")
    rows = [
        event(**{E.event_id: "NVDA:a", E.t0: as_of - pd.Timedelta(hours=30)}),
        event(**{E.event_id: "NVDA:b", E.t0: as_of - pd.Timedelta(hours=24)}),  # exactly settled
        event(**{E.event_id: "NVDA:c", E.t0: as_of - pd.Timedelta(hours=23, minutes=59)}),
        event(**{E.event_id: "NVDA:d", E.t0: as_of + pd.Timedelta(days=30)}),
        event(**{E.event_id: "NVDA:e", E.t0: pd.NaT}),
        event(**{E.event_id: "AAPL:a", E.underlying: "AAPL", E.t0: as_of - pd.Timedelta(days=10)}),
    ]
    events = pd.DataFrame(rows)
    targets = pd.DataFrame({T.event_id: ["NVDA:a", "NVDA:b", "NVDA:c", "AAPL:a"],
                            T.r("24h"): [0.01, 0.02, 0.03, 0.04], T.continuation_30m: [1.0, -1.0, 1.0, 1.0]})
    h = history_view(events, targets, "NVDA", as_of)
    assert h[E.event_id].tolist() == ["NVDA:a", "NVDA:b"]  # sorted by t0, c/d/e/AAPL excluded
    assert h[T.r("24h")].tolist() == [0.01, 0.02]
    assert str(h[E.t0].dt.tz) == "UTC"
    # a longer horizon moves the boundary
    assert history_view(events, targets, "NVDA", as_of, horizon_hours=28)[E.event_id].tolist() == ["NVDA:a"]
    # no targets at all: the rows still come back, with no target columns joined
    only_ev = history_view(events, pd.DataFrame(), "NVDA", as_of)
    assert only_ev[E.event_id].tolist() == ["NVDA:a", "NVDA:b"] and T.r("24h") not in only_ev
    assert history_view(events, targets, "TSLA", as_of).empty
    assert history_view(events.iloc[0:0], targets, "NVDA", as_of).empty


def test_build_features_asserts_history_invariant():
    events, targets = _events_and_targets()
    as_of = decision_as_of(T0, "post_60m")
    leaked = events.merge(targets, on=E.event_id)  # includes the event itself (t0 + 24h > as_of)
    with pytest.raises(AssertionError, match="leaks"):
        build_features(ctx_for("post_60m", history=leaked, as_of=as_of))
    ok = history_view(events, targets, "NVDA", as_of)
    build_features(ctx_for("post_60m", history=ok, as_of=as_of))
    build_features(ctx_for("post_60m", history=None))


# ---- reaction == targets ---------------------------------------------------------------------------
@pytest.mark.parametrize("source", ["sec_8k", "detected"])
def test_reaction_matches_compute_targets(source):
    bars = hl_bars()
    ev = event(source)
    tg = compute_targets(ev, bars, None)
    f30 = build_features(ctx_for("post_30m", ev, bars=bars), groups=["reaction"])
    for cp in ("5m", "15m", "30m"):
        assert f30[f"f_r_{cp}"] == pytest.approx(tg[T.r(cp)]), cp
        assert f30[f"f_r_{cp}{D.missing_suffix}"] == 0.0
    assert f30["f_r_now"] == pytest.approx(tg[T.r("30m")])
    assert math.isnan(f30["f_r_60m"]) and f30["f_r_60m" + D.missing_suffix] == 1.0
    f60 = build_features(ctx_for("post_60m", ev, bars=bars), groups=["reaction"])
    assert f60["f_r_60m"] == pytest.approx(tg[T.r("60m")])
    assert f60["f_r_30m"] == pytest.approx(tg[T.r("30m")])
    # sec_8k backs P0 off by three minutes (210.63), detected uses the last bar before t0 (211.07)
    p0 = 210.63 if source == "sec_8k" else 211.07
    assert tg[T.p0] == pytest.approx(p0)
    close_30m = float(bars.loc[bars[C.t_end] <= T0 + pd.Timedelta(minutes=30), C.close].iloc[-1])
    assert f30["f_r_30m"] == pytest.approx(math.log(close_30m / p0))
    # r_1m on 5-minute bars: no bar ending after t0 has closed by t0 + 1m for either source. With
    # the 8-K buffer the bar between the P0 bar and t0 ends before the release: it must not
    # pass as the 1-minute reaction (nor as r_now / the path at post_1m)
    for src in ("detected", "sec_8k"):
        f1 = build_features(ctx_for("post_1m", event(src)), groups=["reaction"])
        assert math.isnan(f1["f_r_1m"]) and math.isnan(f1["f_r_now"]), src
        assert math.isnan(f1["f_path_max"]) and math.isnan(f1["f_vol_z"]), src
    # path range and volume z-score use only bars ending after t0 and up to as_of: the 8-K
    # buffer bar between the P0 bar and t0 is pre-release, neither path nor baseline
    post = bars[(bars[C.t_end] > T0) & (bars[C.t_end] <= T0 + pd.Timedelta(minutes=30))]
    assert f30["f_path_max"] == pytest.approx(math.log(post[C.high].max() / p0))
    assert f30["f_path_min"] == pytest.approx(math.log(post[C.low].min() / p0))
    assert f30["f_vol_z"] > 5 and f30["f_vol_ratio"] > 5
    if source == "sec_8k":
        buffered = bars[(bars[C.t_end] > tg[T.p0_time]) & (bars[C.t_end] <= T0 + pd.Timedelta(minutes=30))]
        assert len(buffered) == len(post) + 1 and buffered[C.high].max() > post[C.high].max()


def test_p0_buffer_setting_flows_through_the_context():
    """FeatureContext.p0_buffer_minutes_sec_8k (Settings.p0_buffer_minutes_sec_8k) moves P0 for
    8-K events in the reaction group and the anchor of the pre-release groups alike."""
    bars = hl_bars()
    ev = event("sec_8k")
    close_30m = float(bars.loc[bars[C.t_end] <= T0 + pd.Timedelta(minutes=30), C.close].iloc[-1])
    f_default = build_features(ctx_for("post_30m", ev, bars=bars), groups=["reaction"])
    f_zero = build_features(ctx_for("post_30m", ev, bars=bars, p0_buffer_minutes_sec_8k=0.0), groups=["reaction"])
    assert f_default["f_r_30m"] == pytest.approx(math.log(close_30m / 210.63))
    assert f_zero["f_r_30m"] == pytest.approx(math.log(close_30m / 211.07))
    assert pre_cut(ctx_for("post_30m", ev)) == T0 - pd.Timedelta(minutes=3)
    assert pre_cut(ctx_for("post_30m", ev, p0_buffer_minutes_sec_8k=0.0)) == T0
    assert pre_cut(ctx_for("post_30m", ev, p0_buffer_minutes_sec_8k=1.0)) == T0 - pd.Timedelta(minutes=1)
    # a detected time never backs off
    assert pre_cut(ctx_for("post_30m", event("detected"), p0_buffer_minutes_sec_8k=7.0)) == T0


def test_reaction_refuses_coarse_bars():
    """1h bars never resolve a reaction (targets.FINE_INTERVALS), although at post_60m a 1h bar
    ends inside the 2-hour staleness allowance that would otherwise apply."""
    hourly = hl_bars("candles_xyzNVDA_1h_20260824_29.json", "1h")
    f = build_features(ctx_for("post_60m", bars=hourly), groups=["reaction"])
    assert all(math.isnan(v) for v in values(f).values())
    assert math.isnan(compute_targets(event(), hourly, None)[T.r("60m")])


def test_reaction_abnormal_return_and_premium_after_release():
    bars = hl_bars()
    mkt = bars.copy()
    mkt[C.close] = np.where(mkt[C.t_end] > T0, 102.0, 100.0)  # benchmark +2% after the release
    f = build_features(ctx_for("post_30m", bars=bars, market_bars=mkt), groups=["reaction"])
    assert f["f_ar_now"] == pytest.approx(f["f_r_now"] - math.log(1.02))
    fund = funding_frame()
    f60 = build_features(ctx_for("post_60m", extra={X_FUNDING: fund}), groups=["reaction"])
    settled = fund[fund["t"] == utc("2026-08-26 21:00")]
    assert f60["f_premium_post"] == pytest.approx(float(settled["premium"].iloc[0]))
    f30 = build_features(ctx_for("post_30m", extra={X_FUNDING: fund}), groups=["reaction"])
    assert math.isnan(f30["f_premium_post"])  # the 21:00 settlement is after t0 + 30m


# ---- missing companions ------------------------------------------------------------------------------
def test_missing_companions_present_and_consistent():
    for d in ("pre_5m", "post_30m"):
        feats = build_features(ctx_for(d, market_daily=None, daily=None))
        vals = values(feats)
        assert vals, d
        for k, v in vals.items():
            assert k.startswith(D.feature_prefix) and isinstance(v, float)
            comp = feats[k + D.missing_suffix]
            assert comp == (1.0 if math.isnan(v) else 0.0), k
        assert len(feats) == 2 * len(vals)
        assert math.isnan(feats["f_ret_1d"]) and feats["f_ret_1d__missing"] == 1.0
        assert feats["f_drift_60m__missing"] == 0.0


def test_groups_never_raise_on_empty_inputs():
    bare = pd.Series({E.event_id: "X:2026-06", E.underlying: "X", E.t0: T0})
    for d in DECISION_TIMES:
        feats = build_features(FeatureContext(event=bare, as_of=decision_as_of(T0, d), decision_time=d))
        assert all(math.isnan(v) or k.startswith("f_hist_n") or k in {"f_amc", "f_bmo", "f_rth", "f_weekday", "f_friday", "f_hour_ny", "f_hours_to_next_open", "f_hours_to_next_close", "f_h24_closed", "f_holiday_adjacent"}
                   for k, v in values(feats).items()), d
    no_t0 = pd.Series({E.event_id: "X:2026-06", E.underlying: "X"})
    feats = build_features(FeatureContext(event=no_t0, as_of=T0, decision_time="post_30m", bars=hl_bars()))
    # nothing that needs the release instant; the pre-release drift is still anchored at as_of
    for k in ("f_r_now", "f_r_5m", "f_path_max", "f_amc", "f_weekday", "f_listing_age_d", "f_eps_surprise"):
        assert math.isnan(feats[k]), k
    assert not math.isnan(feats["f_drift_60m"])
    # a release far outside the exchange calendar range: calendar features degrade to None
    far = event(**{E.t0: utc("2099-01-05 21:05:00"), E.timing: None})
    feats = build_features(FeatureContext(event=far, as_of=utc("2099-01-05 21:00:00"), decision_time="pre_5m"))
    assert math.isnan(feats["f_hours_to_next_open"]) and feats["f_weekday"] == 0.0


# ---- individual groups ------------------------------------------------------------------------------
def test_calendar_group_values():
    hist = pd.DataFrame({E.event_id: ["NVDA:2026-04"], E.underlying: ["NVDA"], E.t0: [utc("2026-05-20 20:21")]})
    f = build_features(ctx_for("pre_5m", history=hist, extra={X_N_EVENTS_SAME_DAY: 3}), groups=["calendar"])
    assert f["f_amc"] == 1.0 and f["f_bmo"] == 0.0 and f["f_rth"] == 0.0
    assert f["f_weekday"] == 2.0 and f["f_friday"] == 0.0  # Wednesday
    assert f["f_hour_ny"] == pytest.approx(16 + 21 / 60 + 19 / 3600)
    assert f["f_hours_to_next_open"] == pytest.approx((utc("2026-08-27 13:30") - T0) / pd.Timedelta(hours=1))
    assert f["f_hours_to_next_close"] == pytest.approx((utc("2026-08-27 20:00") - T0) / pd.Timedelta(hours=1))
    assert f["f_h24_closed"] == 1.0  # 16:21 ET the next day is after the close
    assert f["f_holiday_adjacent"] == 0.0  # no weekday holiday around Wednesday Aug 26
    assert f["f_days_since_last_event"] == pytest.approx(98.0, abs=0.01)
    assert f["f_n_events_same_day"] == 3.0
    # timing falls back to the calendar when the event row has none; a Friday BMO release
    fri = event(**{E.t0: utc("2026-08-28 11:00"), E.timing: None})
    g = build_features(ctx_for("pre_5m", fri), groups=["calendar"])
    assert g["f_bmo"] == 1.0 and g["f_friday"] == 1.0 and g["f_h24_closed"] == 1.0
    assert g["f_holiday_adjacent"] == 0.0  # a plain weekend is not a holiday
    # explicit holiday adjacency: the Friday before Labor Day, the Tuesday after it, the day
    # before Thanksgiving, Thanksgiving itself and the half day after it; a Monday after an
    # ordinary weekend is not adjacent
    for day, want in (("2026-09-04", 1.0), ("2026-09-08", 1.0), ("2026-11-25", 1.0),
                      ("2026-11-26", 1.0), ("2026-11-27", 1.0), ("2026-08-31", 0.0)):
        h = build_features(ctx_for("pre_5m", event(**{E.t0: utc(f"{day} 20:21"), E.timing: None})), groups=["calendar"])
        assert h["f_holiday_adjacent"] == want, day


def test_pre_price_group_values():
    f = build_features(ctx_for("pre_5m"), groups=["pre_price"])
    # at 16:16 ET on the 26th the release-day session (closed 16:00 ET) is complete and known,
    # although the loaders stamp its t_end as the next NY midnight
    assert f["f_ret_1d"] == pytest.approx(math.log(daily_close("2026-08-26") / daily_close("2026-08-25")))
    assert f["f_ret_5d"] == pytest.approx(math.log(daily_close("2026-08-26") / daily_close("2026-08-19")))
    assert f["f_rvol_20d"] > 0
    # the fixture holds exactly 61 sessions up to the 26th: 60 sessions back is June 1
    assert f["f_ret_60d"] == pytest.approx(math.log(daily_close("2026-08-26") / daily_close("2026-06-01")))
    # an in-session release (15:16 ET) does not see the day's bar yet (and has 60 sessions only)
    rth = build_features(ctx_for("pre_5m", event(**{E.t0: T0_RTH, E.timing: "RTH"})), groups=["pre_price"])
    assert rth["f_ret_1d"] == pytest.approx(math.log(daily_close("2026-08-25") / daily_close("2026-08-24")))
    assert math.isnan(rth["f_ret_60d"])
    assert math.isnan(f["f_dist_52w_high"])  # a 52-week distance needs a year of bars
    bars = hl_bars()
    at = T0 - pd.Timedelta(minutes=5)
    p_now = float(bars.loc[bars[C.t_end] <= at, C.close].iloc[-1])
    p_60 = float(bars.loc[bars[C.t_end] <= at - pd.Timedelta(minutes=60), C.close].iloc[-1])
    p_close = float(bars.loc[bars[C.t_end] <= utc("2026-08-26 20:00"), C.close].iloc[-1])
    assert f["f_drift_60m"] == pytest.approx(math.log(p_now / p_60))
    assert f["f_gap_since_close"] == pytest.approx(math.log(p_now / p_close))
    ext = bars[(bars[C.t_end] > utc("2026-08-26 20:00")) & (bars[C.t_end] <= at)]
    base = bars[(bars[C.t_end] > utc("2026-08-25 20:00")) & (bars[C.t_end] <= utc("2026-08-26 20:00"))]
    assert f["f_ext_vol_ratio"] == pytest.approx(ext[C.volume].mean() / base[C.volume].mean())
    assert f["f_vol_30m_ratio"] > 0
    # a full year of synthetic daily bars makes the 52-week distances available
    days = pd.date_range("2025-08-01", periods=260, freq="B", tz="America/New_York").tz_convert("UTC")
    daily = pd.DataFrame({C.t: days, C.t_end: days + pd.Timedelta(days=1), C.open: 100.0, C.high: 120.0,
                          C.low: 80.0, C.close: 100.0, C.volume: 1.0})
    g = build_features(ctx_for("pre_5m", daily=daily, bars=None), groups=["pre_price"])
    assert g["f_dist_52w_high"] == pytest.approx(math.log(100 / 120))
    assert g["f_dist_52w_low"] == pytest.approx(math.log(100 / 80))
    assert g["f_ret_60d"] == 0.0 and math.isnan(g["f_drift_60m"])


def test_history_group_values():
    t0s = [utc(f"2025-{m:02d}-20 20:21") for m in (2, 5, 8, 11)] + [utc("2026-02-20 21:21")]
    r24 = [0.05, -0.02, 0.10, 0.01, -0.04]
    hist = pd.DataFrame({E.event_id: [f"NVDA:{i}" for i in range(5)], E.underlying: "NVDA", E.t0: t0s,
                         T.r("24h"): r24, T.continuation_30m: [1.0, -1.0, 1.0, np.nan, 1.0],
                         E.eps_surprise_pct: [2.0, -1.0, 6.0, 1.0, -3.0]})
    f = build_features(ctx_for("pre_5m", history=hist), groups=["history"])
    s = pd.Series(r24)
    assert f["f_hist_n"] == 5.0
    assert f["f_hist_r24_mean"] == pytest.approx(s.mean())
    assert f["f_hist_r24_std"] == pytest.approx(s.std(ddof=1))
    assert f["f_hist_r24_skew"] == pytest.approx(s.skew())
    assert f["f_hist_abs_r24_mean"] == pytest.approx(s.abs().mean())
    assert f["f_hist_up_rate"] == pytest.approx(0.6)
    assert f["f_hist_cont_rate"] == pytest.approx(0.75)
    assert [f[f"f_hist_last{i}_r24"] for i in (1, 2, 3, 4)] == [-0.04, 0.01, 0.10, -0.02]
    x = np.array([2.0, -1.0, 6.0, 1.0, -3.0])
    assert f["f_hist_surprise_beta"] == pytest.approx(np.cov(x, np.array(r24))[0, 1] / np.var(x, ddof=1))
    assert f["f_hist_eps_surprise_mean"] == pytest.approx(1.0)
    # the order of rows in the frame does not matter: "last" means latest t0
    g = build_features(ctx_for("pre_5m", history=hist.iloc[::-1].reset_index(drop=True)), groups=["history"])
    assert g["f_hist_last1_r24"] == -0.04
    # too little history for the higher moments
    h = build_features(ctx_for("pre_5m", history=hist.iloc[:1]), groups=["history"])
    assert h["f_hist_n"] == 1.0 and math.isnan(h["f_hist_r24_std"]) and math.isnan(h["f_hist_surprise_beta"])
    none = build_features(ctx_for("pre_5m", history=None), groups=["history"])
    assert math.isnan(none["f_hist_n"])


def test_market_group_values():
    daily = fmp_daily("SPY")
    vix = daily.copy()
    vix[C.close] = 20.0
    f = build_features(ctx_for("pre_5m", market_daily=daily, market_bars=hl_bars(),
                               extra={X_VIX_DAILY: vix, X_SECTOR_DAILY: daily}), groups=["market"])
    # the release-day session is complete at 16:16 ET (see test_pre_price_group_values)
    assert f["f_mkt_ret_1d"] == pytest.approx(math.log(daily_close("2026-08-26") / daily_close("2026-08-25")))
    assert f["f_sector_ret_5d"] == pytest.approx(math.log(daily_close("2026-08-26") / daily_close("2026-08-19")))
    assert f["f_vix_level"] == 20.0 and f["f_vix_chg_5d"] == 0.0
    assert not math.isnan(f["f_mkt_drift_60m"])
    g = build_features(ctx_for("pre_5m", market_daily=None), groups=["market"])
    assert all(math.isnan(v) for v in values(g).values())


def test_perp_state_group_values():
    fund = funding_frame()
    at = T0 - pd.Timedelta(minutes=5)
    settled = fund[fund["t"] + pd.Timedelta(minutes=1) <= at]
    snaps = pd.DataFrame({
        "t": [T0 - pd.Timedelta(hours=30), T0 - pd.Timedelta(hours=2)], "market": "xyz:NVDA",
        "funding": [1e-5, 2e-5], "open_interest": [1000.0, 1200.0], "premium": [1e-4, 3e-4],
        "mark_px": [205.0, 210.0], "oracle_px": [205.0, 210.0], "day_ntl_vlm": [5e7, 6e7],
    })
    pdaily = fmp_daily()
    f = build_features(ctx_for("pre_5m", perp_ctx=snaps, extra={X_FUNDING: fund, X_PERP_DAILY: pdaily,
                                                                X_MAX_LEVERAGE: 20}), groups=["perp_state"])
    assert f["f_funding_rate"] == pytest.approx(float(settled["funding_rate"].iloc[-1]))
    assert settled["t"].iloc[-1] == utc("2026-08-26 20:00")
    day = settled[settled["t"] > at - pd.Timedelta(days=1)]
    assert f["f_funding_mean_24h"] == pytest.approx(float(day["funding_rate"].mean()))
    assert f["f_premium"] == 3e-4 and f["f_oi_notional"] == pytest.approx(1200 * 210)
    assert f["f_oi_chg_24h"] == pytest.approx(math.log(1.2)) and f["f_day_ntl_vlm"] == 6e7
    sub = pdaily[pdaily[C.t_end] <= at].tail(30)
    assert f["f_perp_vol_30d"] == pytest.approx(float((sub[C.volume] * sub[C.close]).median()))
    assert f["f_max_leverage"] == 20.0
    assert f["f_listing_age_d"] == pytest.approx((T0 - utc("2025-11-12")) / pd.Timedelta(days=1))
    # a settlement exactly at the decision instant is not yet known
    exact = fund.copy()
    exact.loc[len(exact)] = {"market": "xyz:NVDA", "t": at, "funding_rate": 9.0, "premium": 9.0}
    g = build_features(ctx_for("pre_5m", extra={X_FUNDING: exact}), groups=["perp_state"])
    assert g["f_funding_rate"] == f["f_funding_rate"]
    # without ctx snapshots the premium comes from the funding history
    assert g["f_premium"] == pytest.approx(float(settled["premium"].iloc[-1]))
    none = build_features(ctx_for("pre_5m", ev=event(**{E.listing_start: None})), groups=["perp_state"])
    assert all(math.isnan(v) for v in values(none).values())
    # the listing age is point-in-time: a listing after the anchor is not known at the anchor
    # (the events module fills listing_start for releases before the perp existed, which would
    # otherwise give a negative age), and has_perp_at_t0 False means no perp to age
    for over in ({E.listing_start: at + pd.Timedelta(minutes=1)},
                 {E.listing_start: T0 + pd.Timedelta(days=100), E.has_perp_at_t0: True},
                 {E.has_perp_at_t0: False}):
        unknown = build_features(ctx_for("pre_5m", ev=event(**over)), groups=["perp_state"])
        assert math.isnan(unknown["f_listing_age_d"]), over
    same_day = build_features(ctx_for("pre_5m", ev=event(**{E.listing_start: at - pd.Timedelta(hours=1)})), groups=["perp_state"])
    assert same_day["f_listing_age_d"] == pytest.approx((T0 - (at - pd.Timedelta(hours=1))) / pd.Timedelta(days=1))
    # perp 1d candles without a close or a volume column: no notional, no exception
    for drop in (C.close, C.volume):
        partial = build_features(ctx_for("pre_5m", extra={X_PERP_DAILY: pdaily.drop(columns=[drop])}), groups=["perp_state"])
        assert math.isnan(partial["f_perp_vol_30d"]), drop


def test_surprise_group_values():
    f = build_features(ctx_for("post_1m"), groups=["surprise"])
    assert f["f_eps_surprise"] == pytest.approx((2.22 - 2.09) / 2.09 * 100)
    assert f["f_rev_surprise"] == pytest.approx((96221 - 92270.94) / 92270.94 * 100)
    assert f["f_eps_beat"] == 1.0 and f["f_sign_agree"] == 1.0 and f["f_n_estimates"] == 30.0
    assert f["f_eps_surprise_abs"] == pytest.approx(f["f_eps_surprise"])
    assert math.isnan(f["f_eps_surprise_z"])  # no history to standardise against
    hist = pd.DataFrame({E.event_id: [f"NVDA:{i}" for i in range(5)], E.underlying: "NVDA",
                         E.t0: [utc(f"2025-{m:02d}-20 20:21") for m in (1, 4, 7, 10, 12)],
                         E.eps_surprise_pct: [1.0, 2.0, 3.0, 4.0, 5.0], E.rev_surprise_pct: [1.0, 1.0, 1.0, 1.0, 1.0]})
    g = build_features(ctx_for("post_1m", history=hist), groups=["surprise"])
    assert g["f_eps_surprise_z"] == pytest.approx((f["f_eps_surprise"] - 3.0) / np.std([1, 2, 3, 4, 5], ddof=1))
    assert math.isnan(g["f_rev_surprise_z"])  # zero dispersion
    # a vendor surprise wins over the derived one; a miss and a disagreeing revenue sign
    miss = event(**{E.eps_surprise_pct: -2.5, E.rev_surprise_pct: 1.0})
    h = build_features(ctx_for("post_1m", miss), groups=["surprise"])
    assert h["f_eps_surprise"] == -2.5 and h["f_eps_beat"] == 0.0 and h["f_sign_agree"] == -1.0
    blank = event(**{E.eps_actual: None, E.eps_estimate: None, E.rev_actual: None, E.rev_estimate: None, E.n_estimates: None})
    b = build_features(ctx_for("post_1m", blank), groups=["surprise"])
    assert all(math.isnan(v) for v in values(b).values())
