"""`freedom cards`: what is due, the wait, the dedupe against live rows, the retry on an
undetected release, the late-release re-run and the files written — with
events.upcoming_events and live.predict_event faked and the clock driven by the test."""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest
from typer.testing import CliRunner

import freedom.events as events_mod
import freedom.live as live_mod
from freedom import cards
from freedom.cli import app
from freedom.config import get_settings
from freedom.data.base import ProviderUnavailable
from freedom.schemas import E
from tests.test_cli_commands import _prediction  # the predict test's fake result shape

runner = CliRunner()
T0 = pd.Timestamp("2026-08-26 20:05", tz="UTC")  # the expected release, 16:05 New York
EVENT = "NVDA:2026-06"  # minted from the report date: the calendar row has no events.parquet id
MIN = pd.Timedelta(minutes=1)


@pytest.fixture
def dirs(monkeypatch, tmp_path):
    data, reports = tmp_path / "data", tmp_path / "reports"
    data.mkdir()
    monkeypatch.setenv("FREEDOM_DATA_DIR", str(data))
    monkeypatch.setenv("FREEDOM_REPORTS_DIR", str(reports))
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("COLUMNS", "250")
    return data, reports


def _upcoming(rows: list[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    """events.upcoming_events' frame for (underlying, expected_t0) pairs: event_id None as for
    a calendar-only event (with_event_ids mints <underlying>:<quarter before the report date>)."""
    return pd.DataFrame([{
        E.event_id: None, E.underlying: u, E.market: f"xyz:{u}", E.kind: "equity_us",
        E.report_date_ny: t0.tz_convert("America/New_York").date(), "expected_t0": t0,
        "expected_t0_source": "calendar default (AMC)", E.eps_estimate: 1.2, E.rev_estimate: 4.6e10,
        E.n_estimates: pd.NA, E.estimate_source: "fmp_calendar", E.estimate_snapshot_time: pd.NaT}
        for u, t0 in rows])


@pytest.fixture
def world(dirs, monkeypatch):
    """One NVDA event expected at T0. predict_event records its calls and returns the predict
    test's fake result for the requested pair (or the next queued outcome: a result to return or
    an exception to raise). time.sleep records its calls and advances the faked clock."""
    monkeypatch.setattr(events_mod, "upcoming_events", lambda s, days=14: _upcoming([("NVDA", T0)]))
    calls: list[dict] = []
    outcomes: list = []

    def predict_event(s, *, event_id, decision, model_name=None, now=None, append=True, **kw):
        calls.append({"event_id": event_id, "decision": decision, "now": now, "append": append})
        outcome = outcomes.pop(0) if outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        res = outcome or _prediction(replay=now is not None)
        res["row"][E.event_id], res["row"]["decision_time"] = event_id, decision
        res["card"]["event_id"], res["card"]["decision"] = event_id, decision
        return res

    monkeypatch.setattr(live_mod, "predict_event", predict_event)
    clock = {"now": T0 - 10 * MIN}
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] = clock["now"] + pd.Timedelta(seconds=seconds)

    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(cards, "utcnow", lambda: clock["now"])
    return {"calls": calls, "outcomes": outcomes, "clock": clock, "sleeps": sleeps, "dirs": dirs}


def test_nothing_due(world):
    world["clock"]["now"] = T0 - pd.Timedelta(hours=3)
    result = runner.invoke(app, ["cards"])
    assert result.exit_code == 0, result.output
    assert "no cards due in the next 45 minutes" in result.output
    assert world["calls"] == [] and world["sleeps"] == []
    assert not (world["dirs"][1] / "cards").exists()
    result = runner.invoke(app, ["cards", "--now", "2026-08-25T20:00:00Z", "--horizon-minutes", "30"])
    assert result.exit_code == 0 and "no cards due in the next 30 minutes" in result.output


def test_due_instants_window_and_order(world):
    s = get_settings()
    horizon = pd.Timedelta(minutes=45)
    due, skipped = cards.due_instants(s, now=T0 - 10 * MIN, horizon=horizon,
                                      decisions=["post_30m", "pre_10m", "post_15m"])
    assert skipped == []
    assert [(d.decision, d.instant) for d in due] == [("pre_10m", T0 - 10 * MIN), ("post_15m", T0 + 15 * MIN),
                                                      ("post_30m", T0 + 30 * MIN)]
    assert due[0].event_id == EVENT and due[0].market == "xyz:NVDA" and due[0].expected_t0 == T0
    # an instant up to 5 minutes past is still due (cron starts late); further past or beyond the horizon is not
    assert len(cards.due_instants(s, now=T0 - 5 * MIN, horizon=horizon, decisions=["pre_10m"])[0]) == 1
    assert cards.due_instants(s, now=T0 - 4 * MIN, horizon=horizon, decisions=["pre_10m"])[0] == []
    assert cards.due_instants(s, now=T0 - 56 * MIN, horizon=horizon, decisions=["pre_10m"])[0] == []
    assert len(cards.due_instants(s, now=T0 - 55 * MIN, horizon=horizon, decisions=["pre_10m"])[0]) == 1


def test_pre_10m_due_now_prints_and_writes_the_card(world):
    _, reports = world["dirs"]
    result = runner.invoke(app, ["cards", "--now", "2026-08-26T19:55:00Z", "--decisions", "pre_10m"])
    assert result.exit_code == 0, result.output
    assert world["calls"] == [{"event_id": EVENT, "decision": "pre_10m", "now": "2026-08-26T19:55:00Z",
                               "append": True}]
    assert world["sleeps"] == []  # a --now run never sleeps
    out = result.output
    assert "1 due in the next 45 minutes" in out and "REPLAY" in out
    assert "CARD pre_10m  NVDA:2026-06  (xyz:NVDA)" in out and "CALL: LONG" in out
    assert "NOT TRADEABLE: replay" in out and "+1.10 %" in out and "+0.210 (up)" in out
    assert "1 card(s) and 0 note(s)" in out
    md = reports / "cards" / "NVDA_2026-06__pre_10m.md"  # ':' is not a valid artifact path character
    js = reports / "cards" / "NVDA_2026-06__pre_10m.json"
    text = md.read_text(encoding="utf-8")
    assert text.startswith("## CALL: LONG — NVDA:2026-06 pre_10m (xyz:NVDA)")
    assert "p_up 0.62" in text and "edge +0.120 vs band ±0.10" in text
    assert "expected 24 h move **+1.10 %**" in text and "10/90 % band -4.00 % .. +7.00 %" in text
    assert "**NOT TRADEABLE**: replay" in text
    assert "| +0.210 (up) | `f_ret_5d` | 0.032 | the stock's return over the last 5 sessions |" in text
    assert "as_of 2026-08-26 20:00 UTC" in text and "model `pre_5m/linear@abcdef01`" in text
    assert "REPLAY: `now` was overridden to 2026-08-26 19:30 UTC" in text
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["card"]["call"] == "LONG" and payload["card"]["as_of"] == "2026-08-26T20:00:00+00:00"
    assert payload["row"]["event_id"] == EVENT and payload["row"]["decision_time"] == "pre_10m"
    assert payload["row"]["replay"] is True and payload["row"]["input_lag_s_fmp"] is None  # NaN -> null
    assert payload["instant"] == "2026-08-26T19:55:00+00:00" and payload["model_meta"]["n_events"] == 300
    index = (reports / "cards" / "index.md").read_text(encoding="utf-8")
    assert index.startswith("# Cards") and index.count("\n- ") == 1
    assert "NVDA:2026-06 pre_10m · CALL: LONG · p_up 0.62 · expected +1.10 % · NOT TRADEABLE (replay)" in index
    assert "[NVDA_2026-06__pre_10m.md](NVDA_2026-06__pre_10m.md)" in index


def test_dedupe_skips_pairs_with_a_live_row_but_not_replays(world):
    data, _ = world["dirs"]
    pd.DataFrame({E.event_id: [EVENT, EVENT, "AMD:2026-06"], "decision_time": ["pre_10m", "post_15m", "post_15m"],
                  "replay": [False, True, False]}).to_parquet(data / "live_predictions.parquet", index=False)
    result = runner.invoke(app, ["cards", "--now", "2026-08-26T19:55:00Z"])
    assert result.exit_code == 0, result.output
    assert [(c["event_id"], c["decision"]) for c in world["calls"]] == [(EVENT, "post_15m"), (EVENT, "post_30m")]
    assert "already predicted live (skipped): NVDA:2026-06 pre_10m" in result.output
    assert "2 due in the next 45 minutes" in result.output


def test_waits_until_the_instant_in_small_steps_unless_no_wait(world):
    _, reports = world["dirs"]
    world["clock"]["now"] = T0 - 12 * MIN  # the pre_10m instant is two minutes away
    result = runner.invoke(app, ["cards", "--decisions", "pre_10m"])
    assert result.exit_code == 0, result.output
    assert "waiting 2.0 min for NVDA:2026-06 pre_10m at 2026-08-26 19:55 UTC" in result.output
    assert world["sleeps"] == [30.0] * 4 and world["clock"]["now"] == T0 - 10 * MIN
    assert world["calls"] == [{"event_id": EVENT, "decision": "pre_10m", "now": None, "append": True}]
    assert "REPLAY" not in result.output
    text = (reports / "cards" / "NVDA_2026-06__pre_10m.md").read_text(encoding="utf-8")
    assert "tradeable: **yes**" in text and "REPLAY" not in text
    world["calls"].clear(), world["sleeps"].clear()
    world["clock"]["now"] = T0 - 12 * MIN
    result = runner.invoke(app, ["cards", "--decisions", "pre_10m", "--no-wait"])
    assert result.exit_code == 0 and world["sleeps"] == [] and len(world["calls"]) == 1


def test_undetected_release_is_retried_then_noted(world):
    _, reports = world["dirs"]
    world["clock"]["now"] = T0 + 15 * MIN  # the post_15m instant
    world["outcomes"].extend([live_mod.ReleaseNotDetected(f"no release detected yet for {EVENT}")] * 40)
    result = runner.invoke(app, ["cards", "--decisions", "post_15m"])
    assert result.exit_code == 0, result.output
    # attempts at 0, 1, ..., 15 minutes after the instant, one minute apart, then the note
    assert len(world["calls"]) == 16 and world["sleeps"] == [60.0] * 15
    assert world["clock"]["now"] == T0 + 30 * MIN
    assert "retrying in 60 s until 2026-08-26 20:35 UTC" in result.output
    assert "NOTE NVDA:2026-06 post_15m: no release detected: no release detected yet" in result.output
    assert "0 card(s) and 1 note(s)" in result.output
    md = reports / "cards" / "NVDA_2026-06__post_15m.md"
    text = md.read_text(encoding="utf-8")
    assert text.startswith("## NOTE: no release detected — NVDA:2026-06 post_15m (xyz:NVDA)")
    assert "decision instant 2026-08-26 20:20 UTC" in text and "no release detected yet" in text
    assert not (reports / "cards" / "NVDA_2026-06__post_15m.json").exists()
    assert "NOTE: no release detected" in (reports / "cards" / "index.md").read_text(encoding="utf-8")


def test_release_detected_on_a_retry_yields_the_card(world):
    _, reports = world["dirs"]
    world["clock"]["now"] = T0 + 15 * MIN
    world["outcomes"].extend([live_mod.ReleaseNotDetected("not yet")] * 2)
    result = runner.invoke(app, ["cards", "--decisions", "post_15m"])
    assert result.exit_code == 0, result.output
    assert len(world["calls"]) == 3 and world["sleeps"] == [60.0, 60.0]
    assert "CALL: LONG" in result.output and "1 card(s) and 0 note(s)" in result.output
    assert (reports / "cards" / "NVDA_2026-06__post_15m.json").exists()


def test_a_replay_or_no_wait_run_never_retries(world):
    world["outcomes"].append(live_mod.ReleaseNotDetected("not yet"))
    result = runner.invoke(app, ["cards", "--now", "2026-08-26T20:20:00Z", "--decisions", "post_15m"])
    assert result.exit_code == 0, result.output
    assert len(world["calls"]) == 1 and world["sleeps"] == [] and "NOTE" in result.output
    world["calls"].clear()
    world["clock"]["now"] = T0 + 15 * MIN
    world["outcomes"].append(live_mod.ReleaseNotDetected("not yet"))
    result = runner.invoke(app, ["cards", "--no-wait", "--decisions", "post_15m"])
    assert result.exit_code == 0 and len(world["calls"]) == 1 and world["sleeps"] == []


def test_a_late_release_reruns_the_post_card_at_its_as_of(world):
    _, reports = world["dirs"]
    world["clock"]["now"] = T0 + 15 * MIN
    early = _prediction()  # the detector saw the release three minutes late: off schedule, as_of ahead of now
    early["row"]["off_schedule"], early["row"]["as_of"] = True, T0 + 18 * MIN
    early["card"]["tradeable"], early["card"]["not_tradeable_because"] = False, ["off schedule"]
    world["outcomes"].append(early)
    result = runner.invoke(app, ["cards", "--decisions", "post_15m"])
    assert result.exit_code == 0, result.output
    assert len(world["calls"]) == 2 and sum(world["sleeps"]) == 180.0
    assert "the release came late" in result.output and "re-running at as_of 2026-08-26 20:23 UTC" in result.output
    assert "1 card(s) and 0 note(s)" in result.output
    assert "tradeable: **yes**" in (reports / "cards" / "NVDA_2026-06__post_15m.md").read_text(encoding="utf-8")


def test_a_failed_card_is_a_note_and_the_run_continues(world, monkeypatch):
    _, reports = world["dirs"]
    monkeypatch.setattr(events_mod, "upcoming_events", lambda s, days=14: _upcoming([("NVDA", T0), ("AMD", T0)]))
    world["outcomes"].append(RuntimeError("FMP is down"))
    result = runner.invoke(app, ["cards", "--now", "2026-08-26T19:55:00Z", "--decisions", "pre_10m"])
    assert result.exit_code == 0, result.output
    assert [c["event_id"] for c in world["calls"]] == ["AMD:2026-06", EVENT]  # same instant: by id
    assert "NOTE AMD:2026-06 pre_10m: prediction failed: RuntimeError: FMP is down" in result.output
    assert "CARD pre_10m  NVDA:2026-06" in result.output and "1 card(s) and 1 note(s)" in result.output
    assert "## NOTE: prediction failed" in (reports / "cards" / "AMD_2026-06__pre_10m.md").read_text(encoding="utf-8")
    index = (reports / "cards" / "index.md").read_text(encoding="utf-8")
    assert index.count("\n- ") == 2 and "NOTE: prediction failed: RuntimeError: FMP is down" in index


def test_prerequisites_and_bad_options_exit_2(world, monkeypatch):
    assert runner.invoke(app, ["cards", "--decisions", "noon"]).exit_code == 2

    def no_key(s, days=14):
        raise ProviderUnavailable("FMP_API_KEY is not set")

    monkeypatch.setattr(events_mod, "upcoming_events", no_key)
    result = runner.invoke(app, ["cards"])
    assert result.exit_code == 2 and "FMP_API_KEY" in result.output

    def no_universe(s, days=14):
        raise FileNotFoundError(f"{s.universe_path} missing")

    monkeypatch.setattr(events_mod, "upcoming_events", no_universe)
    result = runner.invoke(app, ["cards"])
    assert result.exit_code == 2 and "run `freedom universe` first" in result.output


def test_slug_and_markdown_helpers():
    assert cards.slug("NVDA:2026-07") == "NVDA_2026-07" and cards.slug("BRK.B:2026-06") == "BRK.B_2026-06"
    assert cards.fmt_pct(0.011) == "+1.10 %" and cards.fmt_pct(float("nan")) == "" and cards.fmt_value(0.62) == "0.62"
    res = _prediction()
    res["card"]["reasons"], res["card"]["tradeable"], res["card"]["not_tradeable_because"] = [], True, []
    text = cards.card_markdown(res)
    assert "no reasons: the direction head is untrained" in text and "tradeable: **yes**" in text
