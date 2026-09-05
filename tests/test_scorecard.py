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
        _live("GME:2026-06", "post_30m", 0.45),                                # scored: NO TRADE band, forced SHORT -> miss
        _live("ORCL:2026-06", "pre_10m", 0.30, call="SHORT", forced="SHORT"),  # scored: down -> hit
        _live("BB:2026-06", "pre_10m", 0.55),                                  # pending: window not closed
        _live("COST:2026-06", "pre_10m", 0.52),                                # unlabelled: window closed, no label
        _live("NVDA:2026-07", "pre_10m", 0.62, replay=True),                   # excluded
        _live("MU:2026-08", "pre_10m", 0.70, off_schedule=True),               # excluded
    ])
    live.to_parquet(live_predictions_path(settings), index=False)
    targets = pd.DataFrame([{E.event_id: "GME:2026-06", T.r("24h"): 0.08}, {E.event_id: "ORCL:2026-06", T.r("24h"): -0.03},
                            {E.event_id: "BB:2026-06", T.r("24h"): float("nan")},
                            {E.event_id: "COST:2026-06", T.r("24h"): float("nan")}])
    targets.to_parquet(settings.targets_path, index=False)
    events = pd.DataFrame([{E.event_id: "GME:2026-06", E.t0: pd.Timestamp("2026-09-08 20:05", tz="UTC")},
                           {E.event_id: "ORCL:2026-06", E.t0: pd.Timestamp("2026-09-10 20:05", tz="UTC")},
                           {E.event_id: "BB:2026-06", E.t0: pd.Timestamp("2026-09-11 20:05", tz="UTC")},
                           {E.event_id: "COST:2026-06", E.t0: pd.Timestamp("2026-09-09 20:05", tz="UTC")}])
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
    assert sc["n_live_rows"] == 7 and sc["excluded"] == {"replay": 1, "off_schedule": 1, "no_probability": 0}
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
