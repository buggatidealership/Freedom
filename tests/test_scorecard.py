"""The forward scorecard: live cards graded against realised outcomes, nothing else."""

from __future__ import annotations

import json
import math

import pandas as pd
from typer.testing import CliRunner

from freedom import scorecard
from freedom.cli import app
from freedom.live import live_predictions_path
from freedom.schemas import D, E, T

runner = CliRunner()
NOW = pd.Timestamp("2026-09-12 12:00", tz="UTC")


def _live(event_id, decision, p_up, *, replay=False, off_schedule=False, call=None, forced=None):
    return {E.event_id: event_id, D.decision_time: decision, D.as_of: pd.Timestamp("2026-09-08 19:55", tz="UTC"),
            "p_up": p_up, "replay": replay, "off_schedule": off_schedule, "call": call, "forced_call": forced,
            "model_id": "pre_10m/lightgbm@abc"}


def _world(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    live = pd.DataFrame([
        _live("GME:2026-06", "pre_10m", 0.62, call="LONG", forced="LONG"),      # scored: up -> hit, banded hit
        {**_live("GME:2026-06", "post_30m", 0.45),                             # scored: NO TRADE band, forced SHORT -> miss
         D.as_of: pd.Timestamp("2026-09-08 20:35", tz="UTC")},
        _live("ORCL:2026-06", "pre_10m", 0.30, call="SHORT", forced="SHORT"),  # scored: down -> hit
        _live("BB:2026-06", "pre_10m", 0.55),                                  # pending: window not closed
        _live("COST:2026-06", "pre_10m", 0.52),                                # unlabelled: window closed, no label
        _live("NVDA:2026-07", "pre_10m", 0.62, replay=True),                   # excluded
        _live("MU:2026-08", "pre_10m", 0.70, off_schedule=True),               # excluded
        {**_live("AMD:2026-06", "pre_10m", 0.66, call="LONG", forced="LONG"),   # contaminated: made after the release
         D.as_of: pd.Timestamp("2026-09-09 20:10", tz="UTC")},
        {**_live("AMD:2026-06", "post_15m", 0.40),                             # premature: made before the release
         D.as_of: pd.Timestamp("2026-09-09 19:50", tz="UTC")},
    ])
    live.to_parquet(live_predictions_path(settings), index=False)
    targets = pd.DataFrame([{E.event_id: "GME:2026-06", T.r("24h"): 0.08}, {E.event_id: "ORCL:2026-06", T.r("24h"): -0.03},
                            {E.event_id: "AMD:2026-06", T.r("24h"): 0.05},
                            {E.event_id: "BB:2026-06", T.r("24h"): float("nan")},
                            {E.event_id: "COST:2026-06", T.r("24h"): float("nan")}])
    targets.to_parquet(settings.targets_path, index=False)
    events = pd.DataFrame([{E.event_id: "GME:2026-06", E.t0: pd.Timestamp("2026-09-08 20:05", tz="UTC"), E.t0_source: "sec_8k"},
                           {E.event_id: "ORCL:2026-06", E.t0: pd.Timestamp("2026-09-10 20:05", tz="UTC"), E.t0_source: "detected"},
                           {E.event_id: "BB:2026-06", E.t0: pd.Timestamp("2026-09-11 20:05", tz="UTC"), E.t0_source: "calendar_flag"},
                           {E.event_id: "COST:2026-06", E.t0: pd.Timestamp("2026-09-09 20:05", tz="UTC"), E.t0_source: "calendar_flag"},
                           {E.event_id: "AMD:2026-06", E.t0: pd.Timestamp("2026-09-09 20:02", tz="UTC"), E.t0_source: "sec_8k"}])
    events.to_parquet(settings.events_path, index=False)


def test_wilson_interval():
    assert scorecard.wilson(0, 0) == (float("nan"), float("nan")) or all(math.isnan(x) for x in scorecard.wilson(0, 0))
    lo, hi = scorecard.wilson(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0 and hi - lo > 0.4  # ten calls prove nothing
    lo, hi = scorecard.wilson(300, 500)
    assert 0.55 < lo < 0.6 < hi < 0.65


def test_scorecard_grades_only_live_on_schedule_cards(settings):
    _world(settings)
    sc = scorecard.build_scorecard(settings, now=NOW)
    assert sc["n_live_rows"] == 9
    assert sc["excluded"] == {"replay": 1, "off_schedule": 1, "no_probability": 0, "contaminated": 1, "premature": 1, "late": 0,
                              "duplicate": 0}
    order = {(r["event_id"], r["decision"]): r for r in sc["rows"] if r["status"] in ("contaminated", "premature")}
    assert order[("AMD:2026-06", "pre_10m")]["margin_min"] == 8.0 and order[("AMD:2026-06", "post_15m")]["margin_min"] == -12.0
    gme = next(r for r in sc["rows"] if r["event_id"] == "GME:2026-06" and r["decision"] == "pre_10m")
    assert gme["margin_min"] == -10.0  # ten minutes before the measured release
    pre = sc["by_decision"]["pre_10m"]
    assert pre["counted"] == 4 and pre["scored"] == 2 and pre["pending"] == 1 and pre["unlabelled"] == 1
    assert pre["forced_hit_rate"] == 1.0 and pre["banded_calls"] == 2 and pre["banded_hit_rate"] == 1.0
    assert abs(pre["banded_mean_signed_r_24h"] - (0.08 + 0.03) / 2) < 1e-9  # SHORT on -3 % is +3 %
    assert pre["base_rate_up"] == 0.5 and 0 < pre["brier"] < 0.25
    post = sc["by_decision"]["post_30m"]
    assert post["scored"] == 1 and post["forced_hit_rate"] == 0.0 and post["banded_calls"] == 0
    assert post["banded_hit_rate"] is None and post["forced_hit_90"][0] == 0.0
    assert sc["scored_total"] == 3 and sc["pending_total"] == 1 and sc["unlabelled_total"] == 1
    statuses = {(r["event_id"], r["decision"]): r["status"] for r in sc["rows"]}
    assert statuses[("BB:2026-06", "pre_10m")] == "pending" and statuses[("COST:2026-06", "pre_10m")] == "unlabelled"
    md = scorecard.scorecard_markdown(sc)
    assert "| pre_10m | 4 | 2 | 1 | 100 % |" in md and "GME:2026-06 | pre_10m" in md and "Awaiting an outcome" in md
    assert "Excluded for disclosure order: AMD:2026-06 pre_10m (contaminated, +8 min vs release)" in md


def test_score_command_writes_the_files(settings, monkeypatch):
    _world(settings)
    monkeypatch.setenv("FREEDOM_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("FREEDOM_REPORTS_DIR", str(settings.reports_dir))
    monkeypatch.setenv("COLUMNS", "250")
    monkeypatch.setattr(scorecard, "build_scorecard",
                        lambda s, now=None, _b=scorecard.build_scorecard: _b(s, now=NOW))
    result = runner.invoke(app, ["score"])
    assert result.exit_code == 0, result.output
    assert "3 scored, 1 pending, 1 unlabelled" in result.output
    md, js = settings.reports_dir / "scorecard.md", settings.data_dir / "scorecard.json"
    assert md.exists() and json.loads(js.read_text())["scored_total"] == 3
    # an empty record is a clean zero, not an error
    live_predictions_path(settings).unlink()
    result = runner.invoke(app, ["score"])
    assert result.exit_code == 0 and "no counted live cards yet" in result.output


def test_duplicate_rows_of_one_card_are_counted_once(settings):
    """Two overlapping card runs posted ORCL's post_30m card twice on 2026-09-10; the record keeps
    the earlier run and the scorecard excludes the later one as a duplicate."""
    _world(settings)
    live = pd.read_parquet(live_predictions_path(settings))
    first = {**_live("GME:2026-06", "post_30m", 0.45), D.as_of: pd.Timestamp("2026-09-08 20:35", tz="UTC"),
             "run_at": pd.Timestamp("2026-09-08 20:35:00", tz="UTC"), "posted_at": pd.Timestamp("2026-09-08 20:35:03", tz="UTC")}
    dup = {**first, "p_up": 0.80, "call": "LONG", "forced_call": "LONG", "model_id": "post_30m/lightgbm@later",
           "posted_at": pd.Timestamp("2026-09-08 20:35:45", tz="UTC")}
    live = live[~((live[E.event_id] == "GME:2026-06") & (live[D.decision_time] == "post_30m"))]
    pd.concat([live, pd.DataFrame([dup, first])], ignore_index=True).to_parquet(live_predictions_path(settings), index=False)
    sc = scorecard.build_scorecard(settings, now=NOW)
    assert sc["excluded"]["duplicate"] == 1
    rows = [r for r in sc["rows"] if r["event_id"] == "GME:2026-06" and r["decision"] == "post_30m"]
    assert len(rows) == 1 and rows[0]["p_up"] == 0.45 and rows[0]["forced_call"] == "SHORT"  # the earlier run stands
    assert sc["by_decision"]["post_30m"]["counted"] == 1
    assert "1 duplicates of an earlier run" in scorecard.scorecard_markdown(sc)


def test_off_schedule_attempt_is_superseded_by_the_on_schedule_rerun(settings):
    """A release detected after the scheduled instant: `freedom cards` records the first attempt
    off schedule and re-runs the card at its true as_of. The re-run is graded; the attempt counts
    as off schedule, not as a duplicate (ORCL 2026-09-10, both post cards)."""
    _world(settings)
    live = pd.read_parquet(live_predictions_path(settings))
    attempt = {**_live("GME:2026-06", "post_30m", 0.80, off_schedule=True, call="LONG", forced="LONG"),
               D.as_of: pd.Timestamp("2026-09-08 20:35", tz="UTC"), "run_at": pd.Timestamp("2026-09-08 20:35:01", tz="UTC")}
    rerun = {**_live("GME:2026-06", "post_30m", 0.45), D.as_of: pd.Timestamp("2026-09-08 20:37", tz="UTC"),
             "run_at": pd.Timestamp("2026-09-08 20:37:01", tz="UTC")}
    live = live[~((live[E.event_id] == "GME:2026-06") & (live[D.decision_time] == "post_30m"))]
    pd.concat([live, pd.DataFrame([attempt, rerun])], ignore_index=True).to_parquet(live_predictions_path(settings), index=False)
    sc = scorecard.build_scorecard(settings, now=NOW)
    rows = [r for r in sc["rows"] if r["event_id"] == "GME:2026-06" and r["decision"] == "post_30m"]
    assert len(rows) == 1 and rows[0]["p_up"] == 0.45 and rows[0]["status"] == "scored"
    assert sc["excluded"]["duplicate"] == 0 and sc["excluded"]["off_schedule"] == 2  # MU's pre card + the attempt
    assert sc["by_decision"]["post_30m"]["counted"] == 1
    kept, dropped = scorecard.dedupe_live_rows(pd.DataFrame([rerun, attempt, {**rerun, "p_up": 0.46}]))
    assert dropped == {"duplicate": 1, "off_schedule": 1, "replay": 0} and kept["p_up"].tolist() == [0.45]


def test_replay_rows_never_displace_live_rows(settings):
    """A `--now` replay appended to the record (append is the default) ranks below every live row:
    it neither displaces the live card nor blocks it as a duplicate."""
    _world(settings)
    live = pd.read_parquet(live_predictions_path(settings))
    replay = {**_live("ORCL:2026-06", "pre_10m", 0.90, replay=True, call="LONG", forced="LONG"),
              "run_at": pd.Timestamp("2026-09-09 12:00", tz="UTC")}  # a dry run the day before
    pd.concat([pd.DataFrame([replay]), live], ignore_index=True).to_parquet(live_predictions_path(settings), index=False)
    sc = scorecard.build_scorecard(settings, now=NOW)
    rows = [r for r in sc["rows"] if r["event_id"] == "ORCL:2026-06" and r["decision"] == "pre_10m"]
    assert len(rows) == 1 and rows[0]["p_up"] == 0.30 and rows[0]["status"] == "scored"
    assert sc["excluded"]["replay"] == 2 and sc["excluded"]["duplicate"] == 0


def test_late_post_card_is_excluded(settings):
    """A post card whose as_of is far later than the measured release plus its offset and fill lag
    was made on a mistimed detection: excluded as late, not graded as a post_k card."""
    _world(settings)
    live = pd.read_parquet(live_predictions_path(settings))
    late = {**_live("AMD:2026-06", "post_30m", 0.70, call="LONG", forced="LONG"),
            D.as_of: pd.Timestamp("2026-09-09 21:00", tz="UTC")}  # measured t0 20:02: 58 min for a 30-minute card
    pd.concat([live, pd.DataFrame([late])], ignore_index=True).to_parquet(live_predictions_path(settings), index=False)
    sc = scorecard.build_scorecard(settings, now=NOW)
    row = [r for r in sc["rows"] if r["event_id"] == "AMD:2026-06" and r["decision"] == "post_30m"]
    assert len(row) == 1 and row[0]["status"] == "late" and row[0]["margin_min"] == 58.0
    assert sc["excluded"]["late"] == 1 and "1 post cards made too long after it" in scorecard.scorecard_markdown(sc)


def test_live_import_keeps_an_attempt_and_its_rerun_apart(settings, tmp_path):
    """An off-schedule attempt and its on-schedule re-run share as_of and model; both import."""
    from freedom.live import import_live_rows

    _world(settings)
    rec = tmp_path / "rec.json"
    base = {"event_id": "BB:2026-08", "decision_time": "post_15m", "as_of": "2026-09-24T11:15:00Z",
            "model_id": "post_15m/lightgbm@a", "p_up": 0.6, "call": "LONG", "forced_call": "LONG", "replay": False,
            "recovered_from": "issue"}
    rec.write_text(json.dumps({"rows": [{**base, "run_at": "2026-09-24T11:13:02Z", "off_schedule": True},
                                        {**base, "run_at": "2026-09-24T11:15:01Z", "off_schedule": False}]}))
    assert import_live_rows(settings, rec) == (2, 0)
    assert import_live_rows(settings, rec) == (0, 2)


def test_live_import_is_idempotent_and_grades(settings, tmp_path):
    """Rows recovered from the posted cards join the record once; a second import adds nothing."""
    from freedom.live import import_live_rows

    _world(settings)
    rec = tmp_path / "live_recovery.json"
    rec.write_text(json.dumps({"rows": [
        {"event_id": "ORCL:2026-06", "decision_time": "post_30m", "as_of": "2026-09-10T20:40:00Z", "run_at": "2026-09-10T20:40:00Z",
         "posted_at": "2026-09-10T20:40:03Z", "replay": False, "off_schedule": False, "model_id": "post_30m/lightgbm@a",
         "p_up": 0.7145, "call": "LONG", "forced_call": "LONG", "recovered_from": "issue"},
        {"event_id": "ORCL:2026-06", "decision_time": "post_30m", "as_of": "2026-09-10T20:40:00Z", "run_at": "2026-09-10T20:40:00Z",
         "posted_at": "2026-09-10T20:40:45Z", "replay": False, "off_schedule": False, "model_id": "post_30m/lightgbm@b",
         "p_up": 0.6551, "call": "LONG", "forced_call": "LONG", "recovered_from": "issue"}]}))
    before = len(pd.read_parquet(live_predictions_path(settings)))
    assert import_live_rows(settings, rec) == (2, 0)
    assert import_live_rows(settings, rec) == (0, 2)
    live = pd.read_parquet(live_predictions_path(settings))
    assert len(live) == before + 2 and bool(live["recovered"].fillna(False).astype(bool).sum() == 2)
    sc = scorecard.build_scorecard(settings, now=NOW)
    row = [r for r in sc["rows"] if r["event_id"] == "ORCL:2026-06" and r["decision"] == "post_30m"]
    assert len(row) == 1 and row[0]["status"] == "scored" and row[0]["forced_hit"] is False  # r_24h -0.03: LONG wrong
    assert sc["excluded"]["duplicate"] == 1
    result = runner.invoke(app, ["live-import", str(rec)], env={"FREEDOM_DATA_DIR": str(settings.data_dir),
                                                                 "FREEDOM_REPORTS_DIR": str(settings.reports_dir)})
    assert result.exit_code == 0 and "0 row(s) added, 2 already present" in result.output
