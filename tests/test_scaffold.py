import pandas as pd
from typer.testing import CliRunner

from freedom import schemas
from freedom.cli import app
from freedom.timeutil import classify_timing, next_close_after, next_open_after, to_utc


def test_cli_help():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("universe", "archive", "events", "dataset", "evaluate", "optimize", "predict"):
        assert cmd in result.stdout


def test_to_utc_requires_zone_for_naive():
    import pytest

    with pytest.raises(ValueError):
        to_utc("2026-08-26 16:21")
    t = to_utc("2026-08-26 16:21", assume_tz=schemas.NY)
    assert str(t) == "2026-08-26 20:21:00+00:00"


def test_calendar_boundaries_and_timing():
    t0 = to_utc("2026-08-26 20:21", assume_tz=schemas.UTC)  # NVDA release, AMC
    assert classify_timing(t0) == schemas.Timing.amc
    assert next_open_after(t0) == to_utc("2026-08-27 13:30", assume_tz=schemas.UTC)
    assert next_close_after(t0) == to_utc("2026-08-27 20:00", assume_tz=schemas.UTC)
    sat = to_utc("2026-08-29 15:00", assume_tz=schemas.UTC)
    assert classify_timing(sat) == schemas.Timing.closed
    bmo = to_utc("2026-08-27 11:00", assume_tz=schemas.UTC)  # 07:00 NY
    assert classify_timing(bmo) == schemas.Timing.bmo
    rth = to_utc("2026-08-27 15:00", assume_tz=schemas.UTC)
    assert classify_timing(rth) == schemas.Timing.rth


def test_settings_reads_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("FMP_API_KEY", "abc")
    monkeypatch.setenv("FREEDOM_TAKER_FEE_BPS", "1.5")
    from freedom.config import Settings

    s = Settings(data_dir=tmp_path, _env_file=None)
    assert s.fmp_api_key == "abc" and s.taker_fee_bps == 1.5
    assert isinstance(pd.Timestamp.now(tz="UTC"), pd.Timestamp)
