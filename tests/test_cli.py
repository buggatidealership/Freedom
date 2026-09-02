"""CLI tests for the wired-up commands (offline, through the Hyperliquid fake)."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from freedom.cli import app
from freedom.schemas import U
from tests.fakes import NVDA, FakeHyperliquidInfo


@pytest.fixture
def fake(monkeypatch) -> FakeHyperliquidInfo:
    return FakeHyperliquidInfo().install(monkeypatch)


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv("FREEDOM_DATA_DIR", str(d))
    monkeypatch.setenv("COLUMNS", "200")  # rich: one line per summary row
    return d


def test_archive_uses_universe_markets(data_dir, fake):
    pd.DataFrame({U.market: [NVDA], U.dex: ["xyz"]}).to_parquet(data_dir / "universe.parquet",
                                                                 index=False)
    result = CliRunner().invoke(app, ["archive", "--intervals", "1h"])
    assert result.exit_code == 0, result.output
    assert "xyz:NVDA" in result.output and "funding" in result.output and "ctx" in result.output
    assert (data_dir / "archive" / "candles" / "xyz_NVDA" / "1h.parquet").exists()
    assert (data_dir / "archive" / "candles" / "xyz_NVDA" / "funding.parquet").exists()
    assert [c["body"]["req"]["interval"] for c in fake.calls_of("candleSnapshot")] == ["1h", "1d"]


def test_archive_markets_option_bypasses_universe(data_dir, fake):
    result = CliRunner().invoke(app, ["archive", "--intervals", "1h,5m", "--markets", NVDA])
    assert result.exit_code == 0, result.output
    assert (data_dir / "archive" / "candles" / "xyz_NVDA" / "5m.parquet").exists()


def test_archive_reports_errors_without_failing_the_run(data_dir, fake):
    fake.fail_markets.add("xyz:BAD")
    result = CliRunner().invoke(app, ["archive", "--intervals", "1h", "--markets",
                                      f"{NVDA},xyz:BAD"])
    assert result.exit_code == 0, result.output
    assert "ConnectError" in result.output and "reported an error" in result.output
    assert (data_dir / "archive" / "candles" / "xyz_NVDA" / "1h.parquet").exists()


def test_archive_strict_exits_4_after_archiving_the_other_items(data_dir, fake):
    fake.fail_markets.add("xyz:BAD")
    result = CliRunner().invoke(app, ["archive", "--strict", "--intervals", "1h", "--markets",
                                      f"{NVDA},xyz:BAD"])
    assert result.exit_code == 4, result.output
    # the full error string is printed (the table truncates it) and the good market was still archived
    assert "xyz:BAD 1h: ConnectError: simulated network failure" in result.output
    assert (data_dir / "archive" / "candles" / "xyz_NVDA" / "1h.parquet").exists()
    assert (data_dir / "archive" / "candles" / "xyz_NVDA" / "funding.parquet").exists()
    result = CliRunner().invoke(app, ["archive", "--strict", "--intervals", "1h", "--markets", NVDA])
    assert result.exit_code == 0, result.output  # a clean run is unaffected by --strict


def test_archive_without_universe_exits_with_message(data_dir, fake):
    result = CliRunner().invoke(app, ["archive"])
    assert result.exit_code != 0
    assert "universe" in result.output
    assert fake.calls == []
