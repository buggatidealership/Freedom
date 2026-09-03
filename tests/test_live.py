"""Live prediction mechanics (docs/design.md §10) with the events/features/models stubs faked."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import freedom.events as events_mod
import freedom.features as features_mod
import freedom.models as models_mod
from freedom import live
from freedom.features.groups import (
    X_FUNDING,
    X_LISTING_START,
    X_MAX_LEVERAGE,
    X_N_EVENTS_SAME_DAY,
    X_PERP_DAILY,
    X_SECTOR_DAILY,
    X_VIX_DAILY,
)
from freedom.schemas import C, D, E, U
from freedom.timeutil import to_utc

DAY = pd.Timestamp("2026-08-26")
T0_LIVE = to_utc("2026-08-26 20:21", assume_tz="UTC")
EVENT = "NVDA:2026-07"
# the `world` fixture fakes these; the tests that exercise the real ones restore them
REAL_EXPECTED_RELEASE_CLOCK = events_mod.expected_release_clock
REAL_BUILD_FEATURES = features_mod.build_features


class FakeModel(models_mod.BaseModel):
    @classmethod
    def load(cls, path):
        m = cls()
        m.feature_names_ = json.loads((path / "features.json").read_text())
        m.residual_q_ = (-0.05, 0.06)
        return m

    def fit(self, X, y_return, y_direction):
        return self

    def predict_proba_up(self, X):
        assert list(X.columns) == self.feature_names_
        return np.full(len(X), 0.62)

    def predict_return(self, X):
        return np.full(len(X), 0.01)

    def feature_importance(self):
        return pd.Series({"f_pre_price_x": 0.7, "f_calendar_weekday": 0.3})


def _bars(market: str, end: pd.Timestamp, n: int = 600) -> pd.DataFrame:
    t = pd.date_range(end=end.floor("min"), periods=n, freq="1min")
    return pd.DataFrame({C.market: market, C.interval: "1m", C.t: t, C.t_end: t + pd.Timedelta(minutes=1),
                         C.open: 100.0, C.high: 101.0, C.low: 99.0, C.close: 100.5, C.volume: 10.0,
                         C.n_trades: 5, C.source: "hl_live"})


def _daily(symbol: str, end: pd.Timestamp, n: int = 12) -> pd.DataFrame:
    """Session bars as the FMP loader labels them: t = New York midnight of the session date."""
    days = pd.bdate_range(end=end.tz_convert("America/New_York").normalize().tz_localize(None), periods=n)
    t = days.tz_localize("America/New_York").tz_convert("UTC")
    closes = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame({C.market: symbol, C.interval: "1d", C.t: t, C.t_end: t + pd.Timedelta(days=1),
                         C.open: closes - 0.5, C.high: closes + 1.0, C.low: closes - 1.0, C.close: closes,
                         C.volume: 1e6, C.n_trades: 0, C.source: "fmp_daily"})


class FakeHL:
    def __init__(self, latest: pd.Timestamp):
        self.latest, self.calls = latest, []

    def candles(self, market, interval, start, end, **kw):
        self.calls.append((market, interval, start, end))
        return _bars(market, self.latest)


class FakeFMP:
    def __init__(self, latest: pd.Timestamp):
        self.latest, self.calls = latest, []

    def intraday(self, symbol, interval, start_day, end_day, **kw):
        self.calls.append(("intraday", symbol))
        return _bars(symbol, self.latest).assign(source="fmp_intraday")

    def daily(self, symbol, start, end, **kw):
        self.calls.append(("daily", symbol))
        return _bars(symbol, self.latest, n=5)


class FakeHLWithFunding(FakeHL):
    """... plus the funding history the dataset loader falls back to without an archive."""

    def funding_history(self, market, start, end=None, **kw):
        t = pd.date_range(start.ceil("h"), end, freq="1h")
        return pd.DataFrame({"market": market, "t": t, "funding_rate": 1.25e-5, "premium": 1e-4})


class FakeFMPWithProfile(FakeFMP):
    """... plus the company profile the sector proxy comes from, and session-shaped daily bars."""

    def profile(self, symbol):
        return {"symbol": symbol, "sector": "Technology", "industry": "Semiconductors"}

    def daily(self, symbol, start, end, **kw):
        self.calls.append(("daily", symbol))
        return _daily(symbol, self.latest)


class FakeSEC:
    def earnings_filings(self, cik):
        return pd.DataFrame({"accession": ["a"], "form": ["8-K"],
                             "accepted": [to_utc("2026-08-26 20:21:19", assume_tz="UTC")], "items": ["2.02"]})


@pytest.fixture
def world(settings, monkeypatch):
    events = pd.DataFrame([{
        E.event_id: EVENT, E.underlying: "NVDA", E.market: "xyz:NVDA", E.cik: 1045810, E.kind: "equity_us",
        E.fiscal_period: "2026-07", E.report_date_ny: DAY.date(), E.t0: pd.NaT, E.timing: "AMC",
        E.t0_source: "calendar_flag", E.pending: False, E.flags: "upcoming", "t0_acceptance": pd.NaT,
        E.estimate_source: "consensus_snapshot", E.estimate_snapshot_time: to_utc("2026-08-25 12:00", assume_tz="UTC"),
        E.eps_estimate: 1.2, E.rev_estimate: 4.6e10, E.n_estimates: 30,
    }])
    events.to_parquet(settings.events_path, index=False)
    monkeypatch.setattr(events_mod, "load_events", lambda s: pd.read_parquet(s.events_path))
    # the events implementation returns free-text provenance as the clock's source
    monkeypatch.setattr(events_mod, "expected_release_clock",
                        lambda ev, u, before=None: ("16:05", "median of 3 sec_8k acceptances"))
    detector_calls: list[tuple] = []

    def detect_release_live(bars, day, **kw):
        detector_calls.append((bars, day, kw))
        return T0_LIVE if len(bars) else None

    monkeypatch.setattr(events_mod, "detect_release_live", detect_release_live)
    monkeypatch.setattr(events_mod, "upcoming_events", lambda s, days=14, **kw: events.iloc[0:0])
    contexts: list[features_mod.FeatureContext] = []

    def build_features(ctx, groups=None):
        contexts.append(ctx)
        return {"f_pre_price_x": 0.5, "f_pre_price_x__missing": 0.0, "f_calendar_weekday": float(ctx.as_of.weekday()),
                "f_calendar_weekday__missing": 0.0, "f_history_mean": np.nan, "f_history_mean__missing": 1.0}

    monkeypatch.setattr(features_mod, "build_features", build_features)
    monkeypatch.setattr(features_mod, "history_view", lambda ev, tg, u, as_of, h=24: ev.iloc[0:0])
    monkeypatch.setitem(models_mod.REGISTRY, "fake", FakeModel)
    for decision in ("pre_5m", "post_30m"):
        d = settings.models_dir / decision / "fake"
        d.mkdir(parents=True)
        (d / "model.json").write_text(json.dumps({"model_name": "fake", "dataset_sha256": "abcdef0123456789",
                                                  "n_events": 300, "decision_time": decision}))
        (d / "features.json").write_text(json.dumps(["f_pre_price_x", "f_pre_price_x__missing", "f_calendar_weekday",
                                                     "f_calendar_weekday__missing", "f_history_mean",
                                                     "f_history_mean__missing"]))
    return {"settings": settings, "contexts": contexts, "events": events, "detector_calls": detector_calls}


def _upcoming(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """The events implementation's upcoming_events frame (UPCOMING_COLUMNS): event_id is the
    events.parquet id when the table has the row, else None (the calendar knows no fiscal period)."""
    return pd.DataFrame([{
        E.event_id: None,
        E.underlying: u, E.market: f"xyz:{u}", E.kind: "equity_us", E.report_date_ny: pd.Timestamp(day).date(),
        "expected_t0": to_utc(f"{day} 20:05", assume_tz="UTC"), "expected_t0_source": "calendar default (AMC)",
        E.eps_estimate: 0.9, E.rev_estimate: 7.0e9, E.n_estimates: pd.NA, E.estimate_source: "fmp_calendar",
        E.estimate_snapshot_time: pd.NaT} for u, day in rows])


def test_pre_5m_as_of_comes_from_the_expected_release_clock(world):
    s = world["settings"]
    now = to_utc("2026-08-26 19:30", assume_tz="UTC")  # 15:30 New York
    hl = FakeHL(latest=now - pd.Timedelta(seconds=90))
    res = live.predict_event(s, event_id=EVENT, decision="pre_5m", now=now, hl=hl, fmp=FakeFMP(now))
    row = res["row"]
    assert row["t0_used"] == to_utc("2026-08-26 20:05", assume_tz="UTC")  # 16:05 NY
    assert row[D.as_of] == to_utc("2026-08-26 20:00", assume_tz="UTC")
    assert row["t0_source_live"] == "expected_sec_8k" and row["off_schedule"] is False  # a fixed stratum key ...
    assert "median of 3 sec_8k acceptances" in row["schedule_note"] and "16:05" in row["schedule_note"]  # ... detail here
    assert row[E.report_date_ny] == "2026-08-26"
    assert world["contexts"][-1].as_of == row[D.as_of] and world["contexts"][-1].decision_time == "pre_5m"
    assert row["p_up"] == 0.62 and row["r_lo"] == pytest.approx(-0.04) and row["r_hi"] == pytest.approx(0.07)
    assert row["model_id"] == "pre_5m/fake@abcdef01" and row["bar_source"] == "hyperliquid"
    assert row["input_lag_s_hyperliquid"] == 60.0  # now minus the end of the newest (whole-minute) bar
    assert row["n_features"] == 6 and row["n_features_missing"] == 1
    assert res["contributions"][0]["feature"] == "f_pre_price_x" and res["contributions"][0]["value"] == 0.5
    assert res["consensus"][E.estimate_source] == "consensus_snapshot"
    saved = pd.read_parquet(live.live_predictions_path(s))
    assert len(saved) == 1 and saved.loc[0, "model_id"] == row["model_id"]
    assert saved.loc[0, "f_pre_price_x"] == 0.5 and pd.isna(saved.loc[0, "t0_live"])


def test_pre_5m_without_an_acceptance_clock_uses_the_calendar_flag_label(world, monkeypatch):
    s = world["settings"]
    monkeypatch.setattr(events_mod, "expected_release_clock", lambda ev, u, before=None: None)
    now = to_utc("2026-08-26 19:30", assume_tz="UTC")
    res = live.predict_event(s, event_id=EVENT, decision="pre_5m", now=now, hl=FakeHL(now), fmp=FakeFMP(now), append=False)
    row = res["row"]
    assert row["t0_source_live"] == "expected_calendar_flag" and row["t0_used"] == to_utc("2026-08-26 20:05", assume_tz="UTC")
    assert "calendar-flag default for AMC" in row["schedule_note"]


def test_pre_5m_after_the_expected_release_is_off_schedule(world):
    s = world["settings"]
    now = to_utc("2026-08-26 20:30", assume_tz="UTC")
    res = live.predict_event(s, event_id=EVENT, decision="pre_5m", now=now, hl=FakeHL(now), fmp=FakeFMP(now), append=False)
    assert res["row"]["off_schedule"] is True and "after the expected release" in res["row"]["schedule_note"]
    assert not live.live_predictions_path(s).exists()


def test_post_30m_uses_the_live_detector_and_the_schedule_window(world):
    s = world["settings"]
    now = T0_LIVE + pd.Timedelta(minutes=31)
    res = live.predict_event(s, event_id=EVENT, decision="post_30m", now=now, hl=FakeHL(now), fmp=FakeFMP(now), sec=FakeSEC())
    row = res["row"]
    assert row["t0_live"] == T0_LIVE and row["t0_source_live"] == "detected"
    assert row[D.as_of] == T0_LIVE + pd.Timedelta(minutes=30) and row["off_schedule"] is False
    assert row["t0_actual"] == to_utc("2026-08-26 20:21:19", assume_tz="UTC") and row["t0_lag_s"] == 19.0
    assert row["sources_used"] == "hyperliquid;fmp;sec" and row["input_lag_s_sec"] > 0
    late = live.predict_event(s, event_id=EVENT, decision="post_30m", now=T0_LIVE + pd.Timedelta(minutes=40),
                              hl=FakeHL(now), fmp=FakeFMP(now), sec=FakeSEC())
    assert late["row"]["off_schedule"] is True and "not tradable" in late["row"]["schedule_note"]
    early = live.predict_event(s, event_id=EVENT, decision="post_30m", now=T0_LIVE + pd.Timedelta(minutes=28),
                               hl=FakeHL(now), fmp=FakeFMP(now), sec=FakeSEC())
    assert early["row"]["off_schedule"] is True
    assert len(pd.read_parquet(live.live_predictions_path(s))) == 3


def test_forming_candle_never_reaches_detector_features_or_lags(world):
    s = world["settings"]
    now = T0_LIVE + pd.Timedelta(minutes=31)
    hl = FakeHL(latest=now)  # the provider returns the candle that started at `now` and has not closed
    res = live.predict_event(s, event_id=EVENT, decision="post_30m", now=now, hl=hl, fmp=FakeFMP(now), sec=FakeSEC(),
                             append=False)
    assert res["row"]["input_lag_s_hyperliquid"] == 0.0  # measured from the newest CLOSED bar
    bars, _day, kw = world["detector_calls"][-1]
    assert kw["now"] == now and bars[C.t_end].max() == now
    assert world["contexts"][-1].bars[C.t_end].max() == now
    assert live.closed_bars(_bars("xyz:NVDA", now, n=1), now) is None  # nothing closed yet -> None, not an empty frame


def test_post_without_a_detected_release_raises(world, monkeypatch):
    s = world["settings"]
    monkeypatch.setattr(events_mod, "detect_release_live", lambda bars, day, **kw: None)
    now = T0_LIVE + pd.Timedelta(minutes=31)
    with pytest.raises(live.ReleaseNotDetected, match="no release detected"):
        live.predict_event(s, event_id=EVENT, decision="post_30m", now=now, hl=FakeHL(now), fmp=FakeFMP(now), sec=FakeSEC())


def test_falls_back_to_fmp_bars_without_a_perp_market(world):
    s = world["settings"]
    ev = world["events"].assign(**{E.market: None})
    ev.to_parquet(s.events_path, index=False)
    now = to_utc("2026-08-26 19:30", assume_tz="UTC")
    fmp = FakeFMP(now - pd.Timedelta(seconds=120))
    res = live.predict_event(s, event_id=EVENT, decision="pre_5m", now=now, hl=FakeHL(now), fmp=fmp, append=False)
    row = res["row"]
    assert row["bar_source"] == "fmp" and row["sources_used"] == "fmp"
    assert ("intraday", "NVDA") in fmp.calls and np.isnan(row["input_lag_s_hyperliquid"])
    assert row["input_lag_s_fmp"] == 60.0


def test_missing_model_and_unknown_event_name_the_command_to_run(world):
    s = world["settings"]
    now = to_utc("2026-08-26 19:30", assume_tz="UTC")
    with pytest.raises(live.ModelNotFound, match="freedom train"):
        live.predict_event(s, event_id=EVENT, decision="post_15m", now=now, hl=FakeHL(now), fmp=FakeFMP(now))
    with pytest.raises(live.EventNotFound, match="freedom upcoming"):
        live.predict_event(s, event_id="ZZZ:2026-07", decision="pre_5m", now=now, hl=FakeHL(now), fmp=FakeFMP(now))


def test_upcoming_event_ids_are_minted_from_the_report_date():
    assert live.derived_fiscal_period(pd.Timestamp("2026-08-26").date()) == "2026-06"
    assert live.derived_fiscal_period("2026-06-30") == "2026-03"  # the quarter ending on the report date is not over
    assert live.derived_fiscal_period("2026-01-05") == "2025-12"
    ids = live.with_event_ids(_upcoming([("AMD", "2026-09-15"), ("nvda", "2026-11-18")]))
    assert list(ids.columns)[0] == E.event_id and ids[E.event_id].tolist() == ["AMD:2026-06", "NVDA:2026-09"]
    given = live.with_event_ids(ids.assign(**{E.event_id: ["AMD:2026-07", None]}))  # an id the calendar knows wins
    assert given[E.event_id].tolist() == ["AMD:2026-07", "NVDA:2026-09"]


def test_upcoming_event_is_found_under_the_printed_id_or_its_bare_underlying(world, monkeypatch):
    s = world["settings"]
    monkeypatch.setattr(events_mod, "upcoming_events", lambda s, days=14, **kw: _upcoming([("AMD", "2026-09-15")]))
    now = to_utc("2026-09-15 19:30", assume_tz="UTC")
    kw = dict(decision="pre_5m", now=now, hl=FakeHL(now), fmp=FakeFMP(now), append=False)
    row = live.predict_event(s, event_id="AMD:2026-06", **kw)["row"]
    assert row[E.event_id] == "AMD:2026-06" and row[E.underlying] == "AMD" and row[E.market] == "xyz:AMD"
    assert row[E.report_date_ny] == "2026-09-15" and row["t0_used"] == to_utc("2026-09-15 20:05", assume_tz="UTC")
    assert live.predict_event(s, event_id="amd", **kw)["row"][E.event_id] == "AMD:2026-06"  # the resolved id is recorded
    with pytest.raises(live.EventNotFound, match="freedom upcoming"):
        live.predict_event(s, event_id="AMD:2026-03", **kw)
    monkeypatch.setattr(events_mod, "upcoming_events",
                        lambda s, days=14, **kw: _upcoming([("AMD", "2026-09-15"), ("AMD", "2026-09-29")]))
    with pytest.raises(live.EventNotFound, match="matches 2 upcoming events .*AMD:2026-06"):
        live.predict_event(s, event_id="AMD", **kw)


# ---- integrated review: schedule chain, replay tagging, loader parity, table ids ----------------------
def _sec_8k_row(event_id: str, day: str, acceptance_utc: str, **over) -> dict:
    acc = to_utc(acceptance_utc, assume_tz="UTC")
    row = {E.event_id: event_id, E.underlying: "NVDA", E.market: "xyz:NVDA", E.cik: 1045810, E.kind: "equity_us",
           E.fiscal_period: event_id.split(":")[1], E.report_date_ny: pd.Timestamp(day).date(), E.t0: acc,
           "t0_acceptance": acc, E.t0_source: "sec_8k", E.timing: "AMC", E.pending: False, E.flags: ""}
    row.update(over)
    return row


def test_pre_schedule_clock_ignores_the_event_itself_and_later_acceptances(world, monkeypatch):
    s = world["settings"]
    monkeypatch.setattr(events_mod, "expected_release_clock", REAL_EXPECTED_RELEASE_CLOCK)
    # New York clocks: 16:30 (Feb), 16:20 (May), the event's own 16:50 (Aug), a later 16:40 (Nov)
    rows = [_sec_8k_row("NVDA:2026-01", "2026-02-25", "2026-02-25 21:30:00"),
            _sec_8k_row("NVDA:2026-04", "2026-05-28", "2026-05-28 20:20:00"),
            _sec_8k_row("NVDA:2026-07", "2026-08-26", "2026-08-26 20:50:00"),
            _sec_8k_row("NVDA:2026-10", "2026-11-18", "2026-11-18 21:40:00")]
    events = pd.DataFrame(rows)
    now = to_utc("2026-08-26 20:00", assume_tz="UTC")  # 16:00 New York: replaying the pre_5m decision
    sched = live.pre_schedule(s, events.iloc[2], events, "pre_5m", now)
    # the median of the two admissible clocks (16:25), not of all four (16:35)
    assert sched.t0 == to_utc("2026-08-26 20:25", assume_tz="UTC") and sched.as_of == sched.t0 - pd.Timedelta(minutes=5)
    assert sched.t0_source == "expected_sec_8k" and "median of 2 sec_8k acceptances" in sched.note
    assert sched.off_schedule is False
    # without an admissible acceptance the resolved row's own t0 (16:50) must not seed the expectation
    own = pd.DataFrame(rows[2:])
    sched = live.pre_schedule(s, own.iloc[0], own, "pre_5m", now)
    assert sched.t0 == to_utc("2026-08-26 20:05", assume_tz="UTC") and sched.t0_source == "expected_calendar_flag"
    assert "calendar-flag default for AMC" in sched.note
    bmo = own.assign(**{E.timing: "BMO"})
    assert live.pre_schedule(s, bmo.iloc[0], bmo, "pre_5m", now).t0 == to_utc("2026-08-26 11:00", assume_tz="UTC")


def test_pre_schedule_uses_the_issuer_release_clock_after_the_median(world, monkeypatch):
    s = world["settings"]
    s.configs_dir = s.data_dir / "configs"
    s.configs_dir.mkdir()
    (s.configs_dir / "release_clock_overrides.yaml").write_text('NVDA: "14:00 Asia/Taipei"\n')
    events = world["events"]
    now = to_utc("2026-08-26 05:30", assume_tz="UTC")
    # the fixture's median acceptance clock (16:05 New York) still outranks the issuer clock
    sched = live.pre_schedule(s, events.iloc[0], events, "pre_5m", now)
    assert sched.t0_source == "expected_sec_8k" and sched.t0 == to_utc("2026-08-26 20:05", assume_tz="UTC")
    # without one the issuer's clock sets the schedule under its own stratum key
    monkeypatch.setattr(events_mod, "expected_release_clock", lambda ev, u, before=None: None)
    sched = live.pre_schedule(s, events.iloc[0], events, "pre_5m", now)
    assert sched.t0 == to_utc("2026-08-26 06:00", assume_tz="UTC")
    assert sched.as_of == to_utc("2026-08-26 05:55", assume_tz="UTC") and sched.off_schedule is False
    assert sched.t0_source == "expected_issuer_clock"
    assert "issuer release clock 14:00 Asia/Taipei (configs/release_clock_overrides.yaml)" in sched.note
    assert live.expected_t0_source_key("events table: issuer_clock") == "expected_issuer_clock"


def test_pre_5m_manual_override_on_the_table_row_wins(world):
    s = world["settings"]
    ev = world["events"].assign(**{E.t0: to_utc("2026-08-26 19:30", assume_tz="UTC"), E.t0_source: "manual"})
    ev.to_parquet(s.events_path, index=False)
    now = to_utc("2026-08-26 19:00", assume_tz="UTC")
    res = live.predict_event(s, event_id=EVENT, decision="pre_5m", now=now, hl=FakeHL(now), fmp=FakeFMP(now), append=False)
    row = res["row"]
    # the fixture's median clock says 16:05; the override (15:30 New York) wins
    assert row["t0_used"] == to_utc("2026-08-26 19:30", assume_tz="UTC") and row["t0_source_live"] == "expected_manual"
    assert row[D.as_of] == to_utc("2026-08-26 19:25", assume_tz="UTC") and row["off_schedule"] is False
    assert "events table: manual" in row["schedule_note"]


def test_replay_rows_are_tagged_and_run_at_stays_the_wall_clock(world):
    s = world["settings"]
    now = to_utc("2026-08-26 19:30", assume_tz="UTC")
    row = live.predict_event(s, event_id=EVENT, decision="pre_5m", now=now, hl=FakeHL(now), fmp=FakeFMP(now))["row"]
    assert row["replay"] is True and row["now_override"] == now and row["run_at"] > now
    saved = pd.read_parquet(live.live_predictions_path(s))
    assert bool(saved.loc[0, "replay"]) and saved.loc[0, "now_override"] == now and saved.loc[0, "run_at"] > now
    wall = pd.Timestamp.now(tz="UTC")
    row = live.predict_event(s, event_id=EVENT, decision="pre_5m", hl=FakeHL(wall), fmp=FakeFMP(wall), append=False)["row"]
    assert row["replay"] is False and pd.isna(row["now_override"]) and row["run_at"] >= wall


def test_live_features_use_the_dataset_loader_inputs(world, monkeypatch):
    s = world["settings"]
    contexts = []

    def build_features(ctx, groups=None):
        contexts.append(ctx)
        return REAL_BUILD_FEATURES(ctx, groups)

    monkeypatch.setattr(features_mod, "build_features", build_features)
    pd.DataFrame({U.market: ["xyz:NVDA"], U.dex: ["xyz"], U.max_leverage: [10]}).to_parquet(s.universe_path, index=False)
    now = to_utc("2026-08-26 19:30", assume_tz="UTC")
    hl, fmp = FakeHLWithFunding(now - pd.Timedelta(seconds=90)), FakeFMPWithProfile(now)
    res = live.predict_event(s, event_id=EVENT, decision="pre_5m", now=now, hl=hl, fmp=fmp, append=False)
    ctx = contexts[-1]
    extra = ctx.extra
    assert extra["sector_etf"] == "SMH" and ("daily", "SMH") in fmp.calls
    assert extra[X_MAX_LEVERAGE] == 10.0 and extra[X_N_EVENTS_SAME_DAY] == 1.0
    for key in (X_FUNDING, X_PERP_DAILY, X_VIX_DAILY, X_SECTOR_DAILY):
        assert extra[key] is not None and len(extra[key]), key
    assert extra[X_LISTING_START] == extra[X_PERP_DAILY][C.t].min()
    feats = res["features"]
    for key in ("f_funding_rate", "f_max_leverage", "f_perp_vol_30d", "f_sector_ret_1d", "f_n_events_same_day",
                "f_listing_age_d", "f_vix_level"):
        assert not np.isnan(feats[key]), key
    assert feats["f_max_leverage"] == 10.0 and feats["f_funding_rate"] == pytest.approx(1.25e-5)
    assert feats["f_sector_ret_1d"] == pytest.approx(np.log(111.0 / 110.0))
    # the groups that read these inputs agree with the dataset loader on the same event and instant
    from freedom.features.loaders import ContextLoader

    events = events_mod.load_events(s)
    ref = ContextLoader(s, events, hl=hl, fmp=fmp, now=ctx.as_of).context_for(
        ctx.event, "pre_5m", events=events, targets=None, as_of=ctx.as_of)
    ref_feats = REAL_BUILD_FEATURES(ref, groups=["perp_state", "market"])
    live_feats = REAL_BUILD_FEATURES(ctx, groups=["perp_state", "market"])
    for key, value in ref_feats.items():
        if key.startswith("f_n_events_same_day") or key.startswith("f_mkt_drift_60m"):
            continue  # the loader has no same-day count for a row without a t0; benchmark fine bars differ
        assert (np.isnan(value) and np.isnan(live_feats[key])) or value == pytest.approx(live_feats[key]), key


def test_a_calendar_hit_the_table_knows_is_predicted_from_the_table_row(world, monkeypatch):
    s = world["settings"]
    up = _upcoming([("NVDA", "2026-08-26")]).assign(**{E.event_id: [EVENT]})  # `freedom upcoming` prints the table id
    monkeypatch.setattr(events_mod, "upcoming_events", lambda s, days=14, **kw: up)
    now = to_utc("2026-08-26 19:30", assume_tz="UTC")
    row = live.predict_event(s, event_id="nvda", decision="pre_5m", now=now, hl=FakeHL(now), fmp=FakeFMP(now),
                             append=False)["row"]
    assert row[E.event_id] == EVENT and row[E.estimate_source] == "consensus_snapshot"  # the table row, not the calendar's
    assert row["t0_source_live"] == "expected_sec_8k" and "median of 3 sec_8k acceptances" in row["schedule_note"]
    assert live.with_event_ids(up)[E.event_id].tolist() == [EVENT]  # a table id is never re-minted
