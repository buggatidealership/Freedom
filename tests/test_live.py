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
from freedom.schemas import C, D, E
from freedom.timeutil import to_utc

DAY = pd.Timestamp("2026-08-26")
T0_LIVE = to_utc("2026-08-26 20:21", assume_tz="UTC")
EVENT = "NVDA:2026-07"


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


class FakeSEC:
    def earnings_filings(self, cik):
        return pd.DataFrame({"accession": ["a"], "form": ["8-K"],
                             "accepted": [to_utc("2026-08-26 20:21:19", assume_tz="UTC")], "items": ["2.02"]})


@pytest.fixture
def world(settings, monkeypatch):
    events = pd.DataFrame([{
        E.event_id: EVENT, E.underlying: "NVDA", E.market: "xyz:NVDA", E.cik: 1045810, E.kind: "equity_us",
        E.fiscal_period: "2026-07", E.report_date_ny: DAY.date(), E.t0: pd.NaT, E.timing: "AMC",
        E.estimate_source: "consensus_snapshot", E.estimate_snapshot_time: to_utc("2026-08-25 12:00", assume_tz="UTC"),
        E.eps_estimate: 1.2, E.rev_estimate: 4.6e10, E.n_estimates: 30,
    }])
    events.to_parquet(settings.events_path, index=False)
    monkeypatch.setattr(events_mod, "load_events", lambda s: pd.read_parquet(s.events_path))
    monkeypatch.setattr(events_mod, "expected_release_clock", lambda ev, u: ("16:05", "sec_8k"))
    monkeypatch.setattr(events_mod, "detect_release_live", lambda bars, day, **kw: T0_LIVE if len(bars) else None)
    monkeypatch.setattr(events_mod, "upcoming_events", lambda s, days=14: events.iloc[0:0])
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
    return {"settings": settings, "contexts": contexts, "events": events}


def test_pre_5m_as_of_comes_from_the_expected_release_clock(world):
    s = world["settings"]
    now = to_utc("2026-08-26 19:30", assume_tz="UTC")  # 15:30 New York
    hl = FakeHL(latest=now - pd.Timedelta(seconds=90))
    res = live.predict_event(s, event_id=EVENT, decision="pre_5m", now=now, hl=hl, fmp=FakeFMP(now))
    row = res["row"]
    assert row["t0_used"] == to_utc("2026-08-26 20:05", assume_tz="UTC")  # 16:05 NY
    assert row[D.as_of] == to_utc("2026-08-26 20:00", assume_tz="UTC")
    assert row["t0_source_live"] == "expected_sec_8k" and row["off_schedule"] is False
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
    with pytest.raises(LookupError, match="freedom upcoming"):
        live.predict_event(s, event_id="ZZZ:2026-07", decision="pre_5m", now=now, hl=FakeHL(now), fmp=FakeFMP(now))
