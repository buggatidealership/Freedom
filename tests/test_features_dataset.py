"""Offline end-to-end tests of build_dataset and the feature loaders.

Hyperliquid is served by tests.fakes.FakeHyperliquidInfo (fixtures + a flat synthetic prefix so
the archive covers the whole day before the release); an archive is built from it with the real
archiver so the fine bars take the hl_archive route independent of the wall clock. FMP is a
get_json stub serving the committed fixtures for every symbol. No network.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd
import pytest

from freedom.data.archive import archive_markets, snapshot_ctx
from freedom.data.base import HttpClient
from freedom.data.hyperliquid import HyperliquidClient, to_ms
from freedom.features import (
    META_COLUMNS,
    SEASON,
    TARGET_MISSING,
    build_dataset,
    feature_columns,
    read_dataset,
)
from freedom.features.loaders import ContextLoader, sector_proxy
from freedom.schemas import SCHEMA_VERSION, C, D, E, T
from freedom.targets import compute_targets
from freedom.timeutil import to_utc
from tests.fakes import NVDA, FakeHyperliquidInfo, synth_candles
from tests.test_features import daily_close, funding_frame, hl_bars

FIX = Path(__file__).parent / "fixtures"
BASE = "https://financialmodelingprep.com"
FMP_FIXTURES = {
    "stable/historical-chart/1min": "historical-chart_1min_NVDA_20260826_extended.json",
    "stable/historical-chart/5min": "historical-chart_5min_NVDA_20260826_27_extended.json",
    "stable/historical-price-eod/full": "historical-price-eod_NVDA_20260601_0901.json",
    "stable/profile": "profile_NVDA.json",
}
NOW = pd.Timestamp("2026-08-29 00:30", tz="UTC")
T0 = to_utc("2026-08-26 20:21:19", assume_tz="UTC")
T0_MAY = to_utc("2026-05-20 20:21:00", assume_tz="UTC")
T0_AAPL = to_utc("2026-07-30 20:30:28", assume_tz="UTC")
DTS = ["pre_5m", "post_30m", "post_60m"]


class FakeFMPHttp:
    """Serves the FMP fixtures for any symbol and records every request."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def install(self, monkeypatch) -> FakeFMPHttp:
        fake = self

        def get_json(http_self, url, params=None, *, cache_ttl=None, weight=1.0, headers=None,
                     cache_params=None):
            path = url.removeprefix(BASE + "/")
            fake.calls.append({"path": path, "params": dict(params or {})})
            return json.loads((FIX / "fmp" / FMP_FIXTURES[path]).read_text())

        monkeypatch.setattr(HttpClient, "get_json", get_json)
        return self

    def symbols(self, path: str) -> list[str]:
        return [c["params"].get("symbol") for c in self.calls if c["path"] == path]


def flat_prefix(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Flat 5-minute candles (price 210.5, volume 1000) so the archive reaches a day back."""
    step = 300_000
    return [
        {"t": t, "T": t + step - 1, "s": NVDA, "i": "5m", "o": "210.5", "c": "210.5", "h": "210.6",
         "l": "210.4", "v": "1000.0", "n": 10}
        for t in range(to_ms(start), to_ms(end), step)
    ]


@pytest.fixture
def offline(settings, monkeypatch):
    fake = FakeHyperliquidInfo().install(monkeypatch)
    fixture_5m = fake.candles[(NVDA, "5m")]
    first = pd.Timestamp(fixture_5m[0]["t"], unit="ms", tz="UTC")
    fake.candles[(NVDA, "5m")] = flat_prefix(first - pd.Timedelta(days=1), first) + fixture_5m
    for m in ("xyz:SP500", "xyz:VIX"):
        fake.candles[(m, "1d")] = synth_candles(m, "1d", to_ms(pd.Timestamp("2026-04-01", tz="UTC")), 160)
    fmp = FakeFMPHttp().install(monkeypatch)  # after the HL fake: GET now serves FMP fixtures
    client = HyperliquidClient(settings)
    archive_markets(settings, [NVDA], ["5m"], client=client, now=NOW)
    snapshot_ctx(client, settings, "xyz", T0 - pd.Timedelta(hours=30))
    snapshot_ctx(client, settings, "xyz", T0 - pd.Timedelta(hours=6))
    return fake, fmp


def make_events() -> pd.DataFrame:
    common = {E.kind: "equity_us", E.t0_source: "sec_8k", E.t0_confidence: 0.95, E.timing: "AMC"}
    rows = [
        {E.event_id: "NVDA:2026-07", E.underlying: "NVDA", E.market: NVDA, E.t0: T0,
         E.eps_actual: 2.22, E.eps_estimate: 2.09, E.rev_actual: 96221000000.0,
         E.rev_estimate: 92270940000.0, E.n_estimates: 30, E.has_perp_at_t0: True,
         E.listing_start: to_utc("2025-11-12", assume_tz="UTC"), **common},
        {E.event_id: "NVDA:2026-04", E.underlying: "NVDA", E.market: NVDA, E.t0: T0_MAY,
         E.eps_actual: 1.87, E.eps_estimate: 1.76, E.rev_actual: 81615000000.0,
         E.rev_estimate: 78423370000.0, E.n_estimates: 28, E.has_perp_at_t0: True,
         E.listing_start: to_utc("2025-11-12", assume_tz="UTC"), **common},
        {E.event_id: "AAPL:2026-06", E.underlying: "AAPL", E.market: None, E.t0: T0_AAPL,
         E.eps_actual: 1.5, E.eps_estimate: 1.4, E.rev_actual: 9e10, E.rev_estimate: 8.9e10,
         E.n_estimates: 20, E.has_perp_at_t0: False, E.listing_start: pd.NaT, **common},
    ]
    return pd.DataFrame(rows)


def make_targets(events: pd.DataFrame) -> pd.DataFrame:
    bars = hl_bars()
    rows = []
    for _, ev in events.iterrows():
        path = bars if ev[E.event_id] == "NVDA:2026-07" else pd.DataFrame()
        rows.append(compute_targets(ev, path, None))
    return pd.DataFrame(rows)


def row(df: pd.DataFrame, event_id: str, d: str) -> pd.Series:
    hit = df[(df[D.event_id] == event_id) & (df[D.decision_time] == d)]
    assert len(hit) == 1, (event_id, d)
    return hit.iloc[0]


def test_build_dataset_offline(settings, offline):
    fake, fmp = offline
    events, targets = make_events(), make_targets(make_events())
    df = build_dataset(settings, events, targets, decision_times=DTS)

    assert len(df) == 9 and not df.duplicated([D.event_id, D.decision_time]).any()
    assert list(df.columns[:3]) == [D.event_id, D.decision_time, D.as_of]
    for col in (E.underlying, E.market, E.t0, E.t0_source, E.t0_confidence, E.kind, E.timing,
                E.has_perp_at_t0, "season", TARGET_MISSING, T.price_source, T.r("24h"), T.direction,
                T.continuation_30m, T.p0_time, T.h24_in_closure):
        assert col in df.columns, col
    fcols = feature_columns(df)
    assert len(fcols) > 60 and all(c + D.missing_suffix in df.columns for c in fcols)
    for c in fcols:  # companions are consistent even where a group was inadmissible
        assert (df[c + D.missing_suffix] == df[c].isna().astype(float)).all(), c

    r = row(df, "NVDA:2026-07", "post_30m")
    assert r[D.as_of] == T0 + pd.Timedelta(minutes=30)
    assert r["season"] == "2026Q3" and row(df, "NVDA:2026-04", "pre_5m")["season"] == "2026Q2"
    tg = targets.set_index(T.event_id).loc["NVDA:2026-07"]
    assert r["f_r_30m"] == pytest.approx(tg[T.r("30m")]) and r["f_r_5m"] == pytest.approx(tg[T.r("5m")])
    assert r[T.r("24h")] == pytest.approx(tg[T.r("24h")]) and r[T.price_source] == "hl_live"
    assert not r[TARGET_MISSING] and r[E.has_perp_at_t0]
    # the flat archive prefix reaches a day back, so the extended-hours volume has a baseline
    bars = hl_bars()
    close_2000 = float(bars.loc[bars[C.t_end] <= to_utc("2026-08-26 20:00", assume_tz="UTC"), C.close].iloc[-1])
    assert r["f_ext_vol_ratio"] > 1 and r["f_gap_since_close"] == pytest.approx(math.log(210.63 / close_2000))
    # an AMC release sees the release-day session (complete at 16:00 ET) in its daily returns
    assert r["f_ret_1d"] == pytest.approx(math.log(daily_close("2026-08-26") / daily_close("2026-08-25")))
    assert r["f_holiday_adjacent"] == 0.0
    assert r["f_mkt_ret_1d"] > 0 and r["f_vix_level"] > 0  # synthetic xyz:SP500 / xyz:VIX 1d candles
    assert not math.isnan(r["f_sector_ret_1d"])  # profile: Semiconductors -> SMH daily
    fund = funding_frame()
    settled = fund[fund["t"] == to_utc("2026-08-26 20:00", assume_tz="UTC")]
    assert r["f_funding_rate"] == pytest.approx(float(settled["funding_rate"].iloc[0]))
    assert r["f_premium"] == pytest.approx(-0.0000807419)  # ctx snapshot 6h before t0
    assert r["f_oi_chg_24h"] == 0.0 and r["f_oi_notional"] == pytest.approx(598277.336 * 216.72, rel=1e-6)
    assert r["f_max_leverage"] == 20.0 and r["f_listing_age_d"] == pytest.approx(287.85, abs=0.01)
    assert r["f_days_since_last_event"] == pytest.approx((T0 - T0_MAY) / pd.Timedelta(days=1))
    assert r["f_hist_n"] == 0.0  # the May event has no targets
    assert r["f_n_events_same_day"] == 1.0 and r["f_eps_surprise"] > 0
    r60 = row(df, "NVDA:2026-07", "post_60m")
    at_21 = fund[fund["t"] == to_utc("2026-08-26 21:00", assume_tz="UTC")]
    assert r60["f_premium_post"] == pytest.approx(float(at_21["premium"].iloc[0]))
    assert r60["f_r_60m"] == pytest.approx(tg[T.r("60m")]) and math.isnan(r["f_r_60m"])
    pre = row(df, "NVDA:2026-07", "pre_5m")
    assert math.isnan(pre["f_r_30m"]) and pre["f_r_30m" + D.missing_suffix] == 1.0
    assert math.isnan(pre["f_eps_surprise"]) and pre["f_drift_60m"] == r["f_drift_60m"]

    may = row(df, "NVDA:2026-04", "post_30m")
    assert may[TARGET_MISSING] and math.isnan(may["f_r_30m"]) and math.isnan(may["f_days_since_last_event"])
    assert math.isnan(may["f_ret_1d"]) and math.isnan(may["f_funding_rate"])  # daily fixture starts in June
    assert may["f_amc"] == 1.0 and may["f_eps_surprise"] > 0
    aapl = row(df, "AAPL:2026-06", "post_30m")
    assert aapl[TARGET_MISSING] and not aapl[E.has_perp_at_t0] and math.isnan(aapl["f_max_leverage"])
    assert not math.isnan(aapl["f_ret_5d"]) and aapl["f_amc"] == 1.0

    assert settings.dataset_path.exists()
    back, version = read_dataset(settings.dataset_path)
    assert version == SCHEMA_VERSION and back.shape == df.shape and list(back.columns) == list(df.columns)
    assert str(back[D.as_of].dt.tz) == "UTC"
    # daily bars are fetched once per symbol, not once per event or decision time
    assert sorted(fmp.symbols("stable/historical-price-eod/full")) == ["AAPL", "NVDA", "SMH", "SPY"]


def test_build_dataset_own_r24h_trap(settings, offline):
    events = make_events()
    targets = make_targets(events)
    clean = build_dataset(settings, events, targets, decision_times=["post_60m"], write=False)
    poisoned = targets.copy()
    own = poisoned[T.event_id] == "NVDA:2026-07"
    poisoned.loc[own, [T.r("24h"), T.direction, T.magnitude, T.continuation_30m]] = [5.0, 1.0, 5.0, 1.0]
    got = build_dataset(settings, events, poisoned, decision_times=["post_60m"], write=False)
    fcols = feature_columns(clean)
    pd.testing.assert_frame_equal(clean[[D.event_id, *fcols]], got[[D.event_id, *fcols]])
    assert row(got, "NVDA:2026-07", "post_60m")[T.r("24h")] == 5.0
    assert not settings.dataset_path.exists()


def test_build_dataset_defaults_groups_and_errors(settings, offline):
    events = make_events().iloc[:1]
    targets = make_targets(events)
    df = build_dataset(settings, events, targets, groups=["calendar", "reaction"], write=False)
    assert len(df) == 5 and set(df[D.decision_time]) == {"pre_5m", "post_1m", "post_15m", "post_30m", "post_60m"}
    fcols = feature_columns(df)
    assert "f_amc" in fcols and "f_r_30m" in fcols and "f_ret_1d" not in fcols
    assert (df[D.as_of] == df[E.t0] + df[D.decision_time].map(lambda d: pd.Timedelta(minutes={"pre_5m": -5, "post_1m": 1, "post_15m": 15, "post_30m": 30, "post_60m": 60}[d]))).all()
    with pytest.raises(ValueError):
        build_dataset(settings, events, targets, decision_times=["post_7m"], write=False)
    with pytest.raises(ValueError):
        build_dataset(settings, events, targets, groups=["text"], write=False)
    # no events: an empty frame with the full schema of the same build
    empty = build_dataset(settings, events.iloc[0:0], targets, groups=["calendar", "reaction"], write=False)
    assert empty.empty and list(empty.columns) == list(df.columns)
    assert str(empty[D.as_of].dt.tz) == "UTC" and str(empty[E.has_perp_at_t0].dtype) == "boolean"
    # the schema follows the requested decision times: a pre-only build has no post-only group
    pre_only = build_dataset(settings, events.iloc[0:0], targets, decision_times=["pre_5m"], write=False)
    assert "f_amc" in pre_only.columns and "f_ret_1d" in pre_only.columns and "f_r_30m" not in pre_only.columns
    assert list(pre_only.columns) == list(build_dataset(settings, events, targets, decision_times=["pre_5m"], write=False).columns)


def test_build_dataset_edge_cases(settings, offline, caplog):
    events, targets = make_events(), make_targets(make_events())
    # a duplicated decision time is built once
    df = build_dataset(settings, events.iloc[:1], targets, decision_times=["post_30m", "post_30m"], write=False)
    assert len(df) == 1 and df[D.decision_time].tolist() == ["post_30m"]
    # an event without underlying is skipped with a warning, never silently dropped
    bad = events.copy()
    bad.loc[2, E.underlying] = None
    with caplog.at_level(logging.WARNING, logger="freedom.features"):
        df = build_dataset(settings, bad, targets, decision_times=["pre_5m"], write=False)
    assert set(df[D.event_id]) == {"NVDA:2026-07", "NVDA:2026-04"}
    assert any("without underlying" in r.getMessage() and "AAPL:2026-06" in r.getMessage() for r in caplog.records)
    with pytest.raises(ValueError, match="required column"):
        build_dataset(settings, events.drop(columns=[E.underlying]), targets, decision_times=["pre_5m"], write=False)
    # slim events (no metadata columns) still yield the full metadata schema, as NA
    slim = events[[E.event_id, E.underlying, E.market, E.t0]].iloc[:1]
    df = build_dataset(settings, slim, targets, decision_times=["pre_5m"], write=False)
    for c in [*META_COLUMNS, SEASON]:
        assert c in df.columns, c
    r = df.iloc[0]
    assert pd.isna(r[E.has_perp_at_t0]) and pd.isna(r[E.t0_confidence]) and pd.isna(r[E.timing])
    assert r[SEASON] == "2026Q3" and str(df[E.has_perp_at_t0].dtype) == "boolean"
    assert r[E.t0] == T0 and r["f_amc"] == 1.0  # the timing falls back to the calendar


def test_context_loader_bar_sources(settings, offline):
    events = make_events()
    loader = ContextLoader(settings, events)
    aug, aug_mkt = loader.event_bars(events.iloc[0])
    assert aug[C.source].iloc[0] == "hl_archive" and aug[C.t].min() <= T0 - pd.Timedelta(hours=23)
    assert aug_mkt is None  # no archived xyz:SP500 fine candles in the fake
    may, _ = loader.event_bars(events.iloc[1])
    assert may is None  # no fine source covers May: features are missing, never wrong
    assert loader.daily_bars("NVDA")[C.t_end].max() <= NOW + pd.Timedelta(days=7)
    assert loader.max_leverage(NVDA) == 20.0 and loader.max_leverage("xyz:NOPE") is None
    assert loader.sector_etf("NVDA") == "SMH"
    ctx_rows = loader.perp_ctx(NVDA, T0)
    assert len(ctx_rows) == 2 and (ctx_rows["t"] < T0).all()
    ctx = loader.context_for(events.iloc[0], "post_30m", events=events, targets=make_targets(events))
    assert ctx.as_of == T0 + pd.Timedelta(minutes=30) and ctx.history[E.event_id].tolist() == ["NVDA:2026-04"]


def test_sector_proxy_mapping():
    assert sector_proxy({"sector": "Technology", "industry": "Semiconductors"}) == "SMH"
    assert sector_proxy({"sector": "Technology", "industry": "Software - Infrastructure"}) == "XLK"
    assert sector_proxy({"sector": "Healthcare", "industry": "Biotechnology"}) == "XBI"
    assert sector_proxy({"sector": "Energy"}) == "XLE"
    assert sector_proxy({"sector": "Unknown"}) is None and sector_proxy({}) is None and sector_proxy(None) is None
