"""Offline tests for the candle/funding/ctx archiver using the Hyperliquid fixtures."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from freedom.data import archive
from freedom.data.archive import archive_markets, load_archive
from freedom.data.hyperliquid import HyperliquidClient
from freedom.schemas import C, PriceSource
from tests.fakes import NVDA, FakeHyperliquidInfo, synth_candles

NOW = pd.Timestamp("2026-08-29 00:30", tz="UTC")  # just after the fixture windows
T0 = pd.Timestamp("2026-08-24 00:00", tz="UTC")
H = pd.Timedelta(hours=1)


@pytest.fixture
def fake(monkeypatch) -> FakeHyperliquidInfo:
    return FakeHyperliquidInfo().install(monkeypatch)


def summary_row(summary: pd.DataFrame, market: str, interval: str) -> pd.Series:
    rows = summary[(summary["market"] == market) & (summary["interval"] == interval)]
    assert len(rows) == 1, (market, interval)
    return rows.iloc[0]


def tmp_files(settings) -> list:
    return list(settings.archive_dir.rglob("*.tmp"))


def test_first_run_writes_every_file(settings, fake):
    summary = archive_markets(settings, [NVDA], ["1h", "5m"], now=NOW)
    assert list(summary.columns) == ["market", "interval", "rows_added", "first_t", "last_t",
                                     "rows_total", "error"]
    assert len(summary) == 4  # 1h, 5m, funding, ctx(xyz)
    assert summary["error"].isna().all()
    assert str(summary["first_t"].dt.tz) == "UTC" and str(summary["last_t"].dt.tz) == "UTC"

    r1h = summary_row(summary, NVDA, "1h")
    assert r1h["rows_added"] == 121 and r1h["rows_total"] == 121
    assert r1h["first_t"] == T0 and r1h["last_t"] == T0 + pd.Timedelta(days=5)
    r5m = summary_row(summary, NVDA, "5m")
    assert r5m["rows_added"] == 337
    rf = summary_row(summary, NVDA, "funding")
    assert rf["rows_added"] == 48
    assert rf["first_t"].floor("h") == pd.Timestamp("2026-08-26 00:00", tz="UTC")
    rc = summary_row(summary, "xyz", "ctx")
    assert rc["rows_added"] == 103 and rc["first_t"] == NOW

    root = settings.archive_dir
    assert (root / "candles" / "xyz_NVDA" / "1h.parquet").exists()
    assert (root / "candles" / "xyz_NVDA" / "5m.parquet").exists()
    assert (root / "candles" / "xyz_NVDA" / "funding.parquet").exists()
    assert (root / "ctx" / "xyz" / "2026-08-29.parquet").exists()
    assert tmp_files(settings) == []

    stored = pd.read_parquet(archive.candle_path(settings, NVDA, "1h"))
    assert list(stored.columns) == [C.market, C.interval, C.t, C.t_end, C.open, C.high, C.low,
                                    C.close, C.volume, C.n_trades, C.source]
    assert (stored[C.source] == PriceSource.hl_archive).all()
    assert str(stored[C.t].dt.tz) == "UTC" and stored[C.t].is_unique
    assert stored[C.t].is_monotonic_increasing

    funding = pd.read_parquet(archive.funding_path(settings, NVDA))
    assert list(funding.columns) == ["market", "t", "funding_rate", "premium"]
    assert funding["t"].is_unique and str(funding["t"].dt.tz) == "UTC"

    ctx = pd.read_parquet(archive.ctx_path(settings, "xyz", date(2026, 8, 29)))
    assert list(ctx.columns) == archive.CTX_COLUMNS
    assert len(ctx) == 103 and ctx["market"].is_unique  # delisted xyz markets are skipped
    nvda = ctx.set_index("market").loc[NVDA]
    assert nvda["mark_px"] == 216.72 and nvda["oracle_px"] == 216.74
    assert nvda["impact_bid"] == 216.708 and nvda["impact_ask"] == 216.737
    assert nvda["funding"] == pytest.approx(0.0000048249)
    assert nvda["t"] == NOW and nvda["dex"] == "xyz"

    # requests: the first pull asks for the server's whole 5000-bar horizon ending at the bar
    # in progress, in a single request per interval
    calls = fake.calls_of("candleSnapshot")
    assert len(calls) == 2
    req = next(c["body"]["req"] for c in calls if c["body"]["req"]["interval"] == "1h")
    end_ms = req["endTime"] + 1
    assert end_ms == archive.to_ms(NOW.floor("h") + H)
    assert end_ms - req["startTime"] == 5000 * 3_600_000


def test_rerun_is_idempotent_and_incremental(settings, fake):
    archive_markets(settings, [NVDA], ["1h"], now=NOW)
    n_calls = len(fake.calls)
    summary = archive_markets(settings, [NVDA], ["1h"], now=NOW)
    r = summary_row(summary, NVDA, "1h")
    assert r["rows_added"] == 0 and r["rows_total"] == 121
    assert summary_row(summary, NVDA, "funding")["rows_added"] == 0
    assert summary_row(summary, "xyz", "ctx")["rows_added"] == 0  # same (market, t) rows
    # the second pull starts at the last archived bar (refreshing it), not the whole horizon
    req = fake.calls_of("candleSnapshot")[-1]["body"]["req"]
    assert req["startTime"] == archive.to_ms(T0 + pd.Timedelta(days=5))
    assert len(fake.calls) == 2 * n_calls
    stored = pd.read_parquet(archive.candle_path(settings, NVDA, "1h"))
    assert len(stored) == 121 and stored[C.t].is_unique and tmp_files(settings) == []


def test_append_dedups_and_refreshes_partial_bar(settings, fake):
    archive_markets(settings, [NVDA], ["1h"], now=NOW)
    rows = fake.candles[(NVDA, "1h")]
    last_t = rows[-1]["t"]
    # the bar that was in progress gets its final values, and three new bars arrive
    rows[-1] = dict(rows[-1], c="123.45", n=99)
    rows.extend(synth_candles(NVDA, "1h", last_t + 3_600_000, 3))
    later = NOW + 3 * H
    summary = archive_markets(settings, [NVDA], ["1h"], now=later)
    r = summary_row(summary, NVDA, "1h")
    assert r["rows_added"] == 3 and r["rows_total"] == 124
    stored = pd.read_parquet(archive.candle_path(settings, NVDA, "1h"))
    assert len(stored) == 124 and stored[C.t].is_unique and stored[C.t].is_monotonic_increasing
    refreshed = stored.set_index(C.t).loc[pd.Timestamp(last_t, unit="ms", tz="UTC")]
    assert refreshed[C.close] == 123.45 and refreshed[C.n_trades] == 99
    assert (stored[C.t_end] == stored[C.t] + H).all()
    # ctx snapshots taken at different times on the same day accumulate in one file
    ctx = pd.read_parquet(archive.ctx_path(settings, "xyz", date(2026, 8, 29)))
    assert len(ctx) == 206 and not ctx.duplicated(["market", "t"]).any()


def test_http_failure_is_recorded_not_fatal(settings, fake):
    fake.fail_markets.add("xyz:BAD")
    summary = archive_markets(settings, [NVDA, "xyz:BAD"], ["1h"], now=NOW)
    bad = summary_row(summary, "xyz:BAD", "1h")
    assert "simulated network failure" in bad["error"] and bad["rows_added"] == 0
    assert pd.isna(bad["first_t"])
    assert summary_row(summary, NVDA, "1h")["rows_added"] == 121
    assert not archive.candle_path(settings, "xyz:BAD", "1h").exists()
    assert tmp_files(settings) == []


def test_unknown_interval_fails_before_any_request(settings, fake):
    with pytest.raises(ValueError):
        archive_markets(settings, [NVDA], ["1h", "7m"], now=NOW)
    assert fake.calls == []


def test_accepts_injected_client(settings, fake):
    client = HyperliquidClient(settings)
    summary = archive_markets(settings, [NVDA], ["5m"], client=client, now=NOW)
    assert summary_row(summary, NVDA, "5m")["rows_added"] == 337


def test_load_archive_window_is_half_open(settings, fake):
    archive_markets(settings, [NVDA], ["1h"], now=NOW)
    start, end = T0 + pd.Timedelta(days=1), T0 + pd.Timedelta(days=2)
    df = load_archive(settings, NVDA, "1h", start, end)
    assert list(df.columns) == [C.market, C.interval, C.t, C.t_end, C.open, C.high, C.low,
                                C.close, C.volume, C.n_trades, C.source]
    assert len(df) == 24
    assert df[C.t].iloc[0] == start and df[C.t].iloc[-1] == end - H
    assert df[C.t_end].iloc[-1] == end
    assert str(df[C.t].dt.tz) == "UTC" and str(df[C.t_end].dt.tz) == "UTC"
    assert (df[C.source] == PriceSource.hl_archive).all()
    assert (df[C.market] == NVDA).all() and (df[C.interval] == "1h").all()
    assert df[C.close].dtype == np.float64 and df[C.n_trades].dtype == np.int64
    assert df.index.tolist() == list(range(24))
    # naive bounds are taken as UTC
    assert len(load_archive(settings, NVDA, "1h", "2026-08-25", "2026-08-26")) == 24


def test_load_archive_missing_or_outside_window(settings, fake):
    empty = load_archive(settings, NVDA, "1h", T0, T0 + H)
    assert empty.empty and list(empty.columns)[2:4] == [C.t, C.t_end]
    assert str(empty[C.t].dt.tz) == "UTC" and empty[C.n_trades].dtype == np.int64
    archive_markets(settings, [NVDA], ["1h"], now=NOW)
    outside = load_archive(settings, NVDA, "1h", T0 - pd.Timedelta(days=30),
                           T0 - pd.Timedelta(days=29))
    assert outside.empty and list(outside.columns) == list(empty.columns)


def test_write_parquet_atomic_replaces_and_cleans_up(tmp_path):
    path = tmp_path / "x" / "y.parquet"
    archive.write_parquet_atomic(pd.DataFrame({"a": [1]}), path)
    archive.write_parquet_atomic(pd.DataFrame({"a": [2, 3]}), path)
    assert pd.read_parquet(path)["a"].tolist() == [2, 3]
    assert list(path.parent.iterdir()) == [path]
