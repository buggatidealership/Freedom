"""Offline tests for HyperliquidClient against tests/fixtures/hyperliquid (no network)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from freedom.data import hyperliquid as hl
from freedom.data.hyperliquid import HyperliquidClient
from freedom.schemas import C, PriceSource
from tests.fakes import NVDA, FakeHyperliquidInfo, synth_candles, synth_funding

T0 = pd.Timestamp("2026-08-24 00:00", tz="UTC")  # first bar of the 1h fixture
T0_MS = 1_787_529_600_000
H = pd.Timedelta(hours=1)
EXPECTED_CANDLE_COLUMNS = [C.market, C.interval, C.t, C.t_end, C.open, C.high, C.low, C.close,
                           C.volume, C.n_trades, C.source]


@pytest.fixture
def fake(monkeypatch) -> FakeHyperliquidInfo:
    return FakeHyperliquidInfo().install(monkeypatch)


@pytest.fixture
def client(settings, fake) -> HyperliquidClient:
    return HyperliquidClient(settings)


def assert_candle_frame(df: pd.DataFrame, market: str, interval: str, source: str) -> None:
    assert list(df.columns) == EXPECTED_CANDLE_COLUMNS
    for col in (C.t, C.t_end):
        assert str(df[col].dt.tz) == "UTC"
    for col in (C.open, C.high, C.low, C.close, C.volume):
        assert df[col].dtype == np.float64
    assert df[C.n_trades].dtype == np.int64
    # empty and non-empty frames carry identical dtypes (one datetime resolution for t and t_end)
    assert df.dtypes.equals(hl.empty_candles().dtypes), df.dtypes
    if len(df):
        assert (df[C.market] == market).all()
        assert (df[C.interval] == interval).all()
        assert (df[C.source] == source).all()
        assert df[C.t].is_monotonic_increasing and df[C.t].is_unique
        step = pd.Timedelta(milliseconds=hl.interval_ms(interval))
        assert (df[C.t_end] == df[C.t] + step).all()


# ---- helpers -------------------------------------------------------------------------------------
def test_epoch_helpers_and_weights():
    assert hl.to_ms(T0) == T0_MS
    assert hl.to_ms("2026-08-24 00:00") == T0_MS  # naive is taken as UTC
    assert hl.from_ms([T0_MS]).iloc[0] == T0
    assert str(hl.from_ms([T0_MS]).dtype) == hl.DATETIME_DTYPE == "datetime64[ns, UTC]"
    assert hl.candle_weight(0) == 20 and hl.candle_weight(1) == 21
    assert hl.candle_weight(5000) == 20 + math.ceil(5000 / 60)
    assert hl.funding_weight(48) == 20 + 3
    assert hl.interval_ms("1w") == 7 * 86_400_000
    with pytest.raises(ValueError):
        hl.interval_ms("7m")


# ---- metadata ----------------------------------------------------------------------------------
def test_perp_dexs_drops_main_dex(client, fake):
    dexs = client.perp_dexs()
    assert [d["name"] for d in dexs] == ["xyz", "flx", "vntl", "hyna", "km", "abcd", "cash",
                                         "para", "mkts", "io"]
    assert all(d is not None for d in dexs)
    (call,) = fake.calls
    assert call["body"] == {"type": "perpDexs"}
    assert call["weight"] == 20 and call["cache_ttl"] == 3600
    assert call["url"].endswith("/info")


def test_meta_request_shape(client, fake):
    meta = client.meta("xyz", cache_ttl=None)
    assert len(meta["universe"]) == 117
    assert fake.calls[-1]["body"] == {"type": "meta", "dex": "xyz"}
    assert fake.calls[-1]["cache_ttl"] is None


def test_meta_and_asset_ctxs_is_live_and_aligned(client, fake):
    meta, ctxs = client.meta_and_asset_ctxs("xyz")
    assert isinstance(meta, dict) and isinstance(ctxs, list)
    assert len(ctxs) == len(meta["universe"]) == 117
    idx = [a["name"] for a in meta["universe"]].index(NVDA)
    assert ctxs[idx]["markPx"] == "216.72"
    assert fake.calls[-1]["body"] == {"type": "metaAndAssetCtxs", "dex": "xyz"}
    assert fake.calls[-1]["cache_ttl"] is None  # never served from cache


def test_all_markets_excludes_delisted(client, fake):
    df = client.all_markets()
    assert list(df.columns) == ["market", "dex", "symbol", "max_leverage", "growth_mode",
                                "deployer_fee_scale", "only_isolated"]
    # xyz 103 + para 23 + io 4 + mkts 4 + hyna 6; flx/vntl/km/abcd/cash are fully delisted
    assert len(df) == 140
    assert df["market"].is_unique
    assert df.groupby("dex").size().to_dict() == {"hyna": 6, "io": 4, "mkts": 4, "para": 23,
                                                  "xyz": 103}
    row = df.set_index("market").loc[NVDA]
    assert row["dex"] == "xyz" and row["symbol"] == "NVDA"
    assert row["max_leverage"] == 20 and row["growth_mode"] is np.True_
    assert row["deployer_fee_scale"] == 1.0 and row["only_isolated"] is np.False_
    assert df["max_leverage"].dtype == np.int64 and df["growth_mode"].dtype == bool
    assert (df.set_index("market").loc["xyz:SHEIN", "only_isolated"]) is np.True_
    # one perpDexs + one meta per dex
    assert len(fake.calls_of("perpDexs")) == 1 and len(fake.calls_of("meta")) == 10


# ---- candles ------------------------------------------------------------------------------------
def test_candles_fixture_window_is_half_open(client, fake):
    end = T0 + pd.Timedelta(days=5)  # fixture holds 121 bars: T0 .. end inclusive
    df = client.candles(NVDA, "1h", T0, end)
    assert_candle_frame(df, NVDA, "1h", PriceSource.hl_live)
    assert len(df) == 120  # the bar starting exactly at `end` is excluded
    assert df[C.t].iloc[0] == T0 and df[C.t].iloc[-1] == end - H
    assert df[C.t_end].iloc[-1] == end
    assert df[C.close].iloc[0] == 216.83 and df[C.n_trades].iloc[0] == 3778
    assert df[C.volume].iloc[0] == pytest.approx(46220.416)
    (call,) = fake.calls
    assert call["body"]["req"] == {"coin": NVDA, "interval": "1h", "startTime": T0_MS,
                                   "endTime": hl.to_ms(end) - 1}
    assert call["weight"] == 20 + math.ceil(120 / 60)
    assert call["cache_ttl"] is None


def test_candles_drop_bar_containing_unaligned_start(client, fake):
    # The server returns the bar that contains startTime; the client must drop it (t < start).
    start, end = T0 + pd.Timedelta(minutes=1), T0 + pd.Timedelta(hours=5, minutes=30)
    df = client.candles(NVDA, "1h", start, end)
    assert list(df[C.t]) == [T0 + H * k for k in range(1, 6)]
    assert (df[C.t] >= start).all() and (df[C.t] < end).all()


def test_candles_5m_fixture(client, fake):
    start = pd.Timestamp("2026-08-26 18:00", tz="UTC")
    df = client.candles(NVDA, "5m", start, pd.Timestamp("2026-08-27 22:00", tz="UTC"))
    assert_candle_frame(df, NVDA, "5m", PriceSource.hl_live)
    assert len(df) == 336 and df[C.t].iloc[0] == start
    assert (df[C.t].diff().dropna() == pd.Timedelta(minutes=5)).all()


def test_candles_page_backwards_without_exceeding_5000(client, fake):
    n, step = 12_000, hl.INTERVAL_MS["1m"]
    fake.candles[("xyz:SYN", "1m")] = synth_candles("xyz:SYN", "1m", T0_MS, n)
    end = hl.from_ms([T0_MS + n * step]).iloc[0]
    df = client.candles("xyz:SYN", "1m", T0, end)
    assert_candle_frame(df, "xyz:SYN", "1m", PriceSource.hl_live)
    assert len(df) == n and df[C.t].iloc[0] == T0 and df[C.t_end].iloc[-1] == end
    calls = fake.calls_of("candleSnapshot")
    assert len(calls) == 3
    spans = [(c["body"]["req"]["endTime"] + 1 - c["body"]["req"]["startTime"]) // step for c in calls]
    assert spans == [5000, 5000, 2000]  # newest window first, never more than 5000 bars
    assert calls[0]["body"]["req"]["endTime"] == hl.to_ms(end) - 1
    assert calls[-1]["body"]["req"]["startTime"] == T0_MS
    assert [c["weight"] for c in calls] == [hl.candle_weight(5000), hl.candle_weight(5000),
                                            hl.candle_weight(2000)]


def test_candles_stop_at_server_horizon(client, fake):
    # Server holds only the newest 5000 bars of a 12000-bar request: one full page, then an
    # empty older page ends the paging.
    step = hl.INTERVAL_MS["1m"]
    fake.candles[("xyz:SYN", "1m")] = synth_candles("xyz:SYN", "1m", T0_MS + 7000 * step, 5000)
    end = hl.from_ms([T0_MS + 12_000 * step]).iloc[0]
    df = client.candles("xyz:SYN", "1m", T0, end)
    assert len(df) == 5000 and df[C.t].iloc[0] == hl.from_ms([T0_MS + 7000 * step]).iloc[0]
    assert len(fake.calls_of("candleSnapshot")) == 2


def test_candles_empty_frame_has_schema(client, fake):
    df = client.candles("xyz:NOPE", "1h", T0, T0 + pd.Timedelta(days=1))
    assert_candle_frame(df, "xyz:NOPE", "1h", PriceSource.hl_live)
    assert df.empty and len(fake.calls_of("candleSnapshot")) == 1
    # far in the past: also empty, and a single request per <=5000-bar window
    df = client.candles(NVDA, "1h", pd.Timestamp("2021-01-01", tz="UTC"),
                        pd.Timestamp("2021-01-02", tz="UTC"))
    assert df.empty


def test_candles_dedup_keeps_latest_version(client, fake):
    rows = synth_candles("xyz:SYN", "1h", T0_MS, 3)
    dup = dict(rows[1], c="999.0")  # same bar served twice, the later copy is the final one
    fake.candles[("xyz:SYN", "1h")] = rows + [dup]
    df = client.candles("xyz:SYN", "1h", T0, T0 + 3 * H)
    assert len(df) == 3 and df[C.t].is_unique
    assert df[C.close].iloc[1] == 999.0


def test_listing_start_from_first_daily_candle(client, fake, settings):
    listing = pd.Timestamp("2025-11-12", tz="UTC")
    fake.candles[(NVDA, "1d")] = synth_candles(NVDA, "1d", hl.to_ms(listing), 10)
    assert client.listing_start(NVDA) == listing
    (call,) = fake.calls_of("candleSnapshot")
    # one page from the HIP-3 era, not from 2020: the limiter is charged from the window length
    assert call["body"]["req"]["startTime"] == hl.to_ms(hl.LISTING_SEARCH_START)
    assert call["body"]["req"]["interval"] == "1d"
    assert call["weight"] < hl.candle_weight(2000)
    assert call["cache_ttl"] == settings.cache_ttl_seconds  # listing dates do not change
    assert client.listing_start("xyz:NOPE") is None


def test_listing_start_before_search_window_looks_further_back(client, fake):
    listed = pd.Timestamp("2024-06-01", tz="UTC")
    fake.candles[("xyz:OLD", "1d")] = synth_candles("xyz:OLD", "1d", hl.to_ms(listed), 2000)
    assert client.listing_start("xyz:OLD") == listed
    calls = fake.calls_of("candleSnapshot")
    assert [c["body"]["req"]["startTime"] for c in calls] == [
        hl.to_ms(hl.LISTING_SEARCH_START), hl.to_ms(hl.LISTING_SEARCH_FLOOR)]
    assert calls[1]["body"]["req"]["endTime"] == hl.to_ms(hl.LISTING_SEARCH_START) - 1


# ---- funding ------------------------------------------------------------------------------------
def test_funding_history_fixture(client, fake):
    start = pd.Timestamp("2026-08-26 00:00", tz="UTC")
    end = pd.Timestamp("2026-08-28 00:00", tz="UTC")
    df = client.funding_history(NVDA, start, end)
    assert list(df.columns) == ["market", "t", "funding_rate", "premium"]
    assert len(df) == 48 and str(df["t"].dt.tz) == "UTC"
    assert df.dtypes.equals(hl.empty_funding().dtypes), df.dtypes
    assert df["t"].is_monotonic_increasing and df["t"].is_unique
    assert (df["t"] >= start).all() and (df["t"] < end).all()
    # the server stamps the settlement block (48-120 ms after the hour); t is the hour itself
    assert (df["t"] == df["t"].dt.floor("h")).all()
    assert df["t"].iloc[0] == start
    assert df["funding_rate"].dtype == np.float64 and df["premium"].dtype == np.float64
    assert df["funding_rate"].iloc[0] == pytest.approx(0.00000625)
    assert df["premium"].iloc[0] == pytest.approx(0.0002191143)
    assert (df["market"] == NVDA).all()
    (call,) = fake.calls
    assert call["body"] == {"type": "fundingHistory", "coin": NVDA,
                            "startTime": hl.to_ms(start), "endTime": hl.to_ms(end)}
    assert call["weight"] == hl.funding_weight(48) and call["cache_ttl"] is None
    # half-open end: the 23:00 entry of the 27th is the last, the 28th 00:00 entry is out
    assert df["t"].iloc[-1] == end - H


def test_funding_frame_floors_to_hour_and_keeps_last_per_hour():
    raw = [{"time": T0_MS + 48, "fundingRate": "1e-6", "premium": "0"},
           {"time": T0_MS + 900_000, "fundingRate": "2e-6", "premium": "0"},  # same hour, later
           {"time": T0_MS + 3_600_000 + 120, "fundingRate": "3e-6", "premium": "0"}]
    df = hl.funding_frame("xyz:SYN", raw, 0, 10**14)
    assert list(df["t"]) == [T0, T0 + H]
    assert df["funding_rate"].tolist() == [2e-6, 3e-6]
    assert df.dtypes.equals(hl.empty_funding().dtypes)


def test_funding_history_pages_by_500(client, fake):
    n = 1200
    fake.funding["xyz:SYN"] = synth_funding("xyz:SYN", T0_MS, n)
    end = T0 + n * H
    df = client.funding_history("xyz:SYN", T0, end)
    assert len(df) == n and df["t"].is_unique and df["t"].is_monotonic_increasing
    calls = fake.calls_of("fundingHistory")
    assert len(calls) == 3
    # each page restarts one millisecond after the last entry received
    assert calls[1]["body"]["startTime"] == T0_MS + 499 * 3_600_000 + 40 + 1
    assert calls[2]["body"]["startTime"] == T0_MS + 999 * 3_600_000 + 40 + 1
    # weight is charged before the response is known, from the hours the window can hold
    # (capped at the page size); the last page starts 41 ms into hour 1000 so 201 hours remain
    assert [c["weight"] for c in calls] == [hl.funding_weight(500), hl.funding_weight(500),
                                            hl.funding_weight(201)]


def test_funding_history_empty_and_default_end(client, fake):
    df = client.funding_history("xyz:NOPE", T0)
    assert df.empty and list(df.columns) == ["market", "t", "funding_rate", "premium"]
    assert str(df["t"].dt.tz) == "UTC" and df.dtypes.equals(hl.empty_funding().dtypes)
    (call,) = fake.calls
    assert call["body"]["endTime"] > call["body"]["startTime"]  # end defaults to now
