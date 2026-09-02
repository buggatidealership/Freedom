"""Every CLI command through typer's CliRunner with the module functions monkeypatched:
exit codes, the BudgetExhausted path, and the missing-artifact messages."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import freedom.eval as eval_mod
import freedom.events as events_mod
import freedom.features as features_mod
import freedom.live as live_mod
import freedom.optimize as optimize_mod
import freedom.targets as targets_mod
import freedom.universe as universe_mod
from freedom.cli import _summary_rows, app
from freedom.data.base import BudgetExhausted, DailyBudget, ProviderUnavailable
from freedom.schemas import D, E, T, U

runner = CliRunner()


@pytest.fixture
def dirs(monkeypatch, tmp_path):
    data, reports = tmp_path / "data", tmp_path / "reports"
    data.mkdir()
    monkeypatch.setenv("FREEDOM_DATA_DIR", str(data))
    monkeypatch.setenv("FREEDOM_REPORTS_DIR", str(reports))
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("COLUMNS", "250")
    return data, reports


def _universe() -> pd.DataFrame:
    return pd.DataFrame({
        U.market: ["xyz:NVDA", "xyz:GOLD", "zzz:FOO"], U.dex: ["xyz", "xyz", "zzz"],
        U.symbol: ["NVDA", "GOLD", "FOO"], U.kind: ["equity_us", "commodity", "other"],
        U.underlying: ["NVDA", None, None], U.cik: pd.array([1045810, None, None], dtype="Int64"),
        U.name: ["NVIDIA CORP", None, None], U.exclude_reason: [None, "kind=commodity", "kind=other"],
        U.verify: [False, False, True], U.in_event_universe: [True, False, False],
    })


def _events() -> pd.DataFrame:
    return pd.DataFrame({
        E.event_id: ["NVDA:2026-04", "NVDA:2026-07"], E.underlying: ["NVDA", "NVDA"],
        E.t0: pd.to_datetime(["2026-05-28 20:20", "2026-08-26 20:21"], utc=True),
        E.t0_source: ["sec_8k", "sec_8k"], E.t0_confidence: [0.95, 0.95], E.pending: [False, True],
    })


def _dataset() -> pd.DataFrame:
    return pd.DataFrame({D.event_id: ["a", "b"], D.decision_time: ["pre_5m", "post_30m"],
                         T.r("24h"): [0.01, -0.02], T.ar("24h"): [np.nan, -0.01],
                         "f_calendar_weekday": [1.0, 2.0], "f_calendar_weekday__missing": [0.0, 0.0]})


# ---- universe -----------------------------------------------------------------------------------------
def test_universe_prints_summary_and_verify_only_rows(dirs, monkeypatch):
    monkeypatch.setattr(universe_mod, "build_universe", lambda s, write=True: _universe())
    result = runner.invoke(app, ["universe"])
    assert result.exit_code == 0, result.output
    assert "equity_us" in result.output and "1 rows need verification" in result.output
    result = runner.invoke(app, ["universe", "--verify-only"])
    assert result.exit_code == 0, result.output
    assert "zzz:FOO" in result.output and "xyz:NVDA" not in result.output


def test_universe_without_fmp_is_fine_but_provider_failures_exit_2(dirs, monkeypatch):
    def boom(s, write=True):
        raise ProviderUnavailable("Hyperliquid is down")

    monkeypatch.setattr(universe_mod, "build_universe", boom)
    result = runner.invoke(app, ["universe"])
    assert result.exit_code == 2 and "Hyperliquid is down" in result.output


# ---- events -------------------------------------------------------------------------------------------
def test_events_summary_and_options(dirs, monkeypatch):
    calls = {}

    def build(s, *, underlyings=None, since=None, write=True):
        calls["underlyings"], calls["since"] = underlyings, since
        return _events()

    monkeypatch.setattr(events_mod, "build_events", build)
    result = runner.invoke(app, ["events", "--since", "2024-01-01", "--underlyings", "NVDA,AAPL"])
    assert result.exit_code == 0, result.output
    assert calls["underlyings"] == ["NVDA", "AAPL"] and calls["since"] == pd.Timestamp("2024-01-01", tz="UTC")
    assert "sec_8k" in result.output and "pending" in result.output


def test_events_budget_exhausted_exits_2_with_a_clear_message(dirs, monkeypatch):
    def build(s, **kw):
        raise BudgetExhausted("fmp: daily budget of 240 requests exhausted (240 used).")

    monkeypatch.setattr(events_mod, "build_events", build)
    result = runner.invoke(app, ["events"])
    assert result.exit_code == 2
    assert "budget exhausted" in result.output and "pending" in result.output
    assert "freedom events" in result.output and "FREEDOM_FMP_DAILY_BUDGET" in result.output


def test_events_missing_key_exits_2(dirs, monkeypatch):
    def build(s, **kw):
        raise ProviderUnavailable("FMP_API_KEY is not set; see .env.example")

    monkeypatch.setattr(events_mod, "build_events", build)
    result = runner.invoke(app, ["events"])
    assert result.exit_code == 2 and "FMP_API_KEY" in result.output


# ---- dataset ------------------------------------------------------------------------------------------
def test_dataset_without_events_names_the_events_command(dirs, monkeypatch):
    def load(s):
        raise FileNotFoundError(f"{s.events_path} missing")

    monkeypatch.setattr(events_mod, "load_events", load)
    result = runner.invoke(app, ["dataset"])
    assert result.exit_code == 2 and "run `freedom events` first" in result.output


def test_dataset_builds_targets_then_features(dirs, monkeypatch):
    order = []
    monkeypatch.setattr(events_mod, "load_events", lambda s: _events())

    def targets(s, ev, *, write=True, **kw):
        order.append("targets")
        return pd.DataFrame({T.event_id: ev[E.event_id], T.price_source: ["hl_archive", None]})

    def dataset(s, ev, tg, *, decision_times=None, groups=None, write=True):
        order.append(("dataset", tuple(decision_times)))
        return _dataset()

    monkeypatch.setattr(targets_mod, "build_targets", targets)
    monkeypatch.setattr(features_mod, "build_dataset", dataset)
    result = runner.invoke(app, ["dataset", "--decision-times", "pre_5m,post_30m"])
    assert result.exit_code == 0, result.output
    assert order == ["targets", ("dataset", ("pre_5m", "post_30m"))]
    assert "hl_archive" in result.output and "post_30m" in result.output
    result = runner.invoke(app, ["dataset", "--decision-times", "pre_5m,post_7m"])
    assert result.exit_code == 2 and "post_7m" in result.output


# ---- evaluate -----------------------------------------------------------------------------------------
def test_evaluate_without_dataset_names_the_dataset_command(dirs):
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 2 and "run `freedom dataset` first" in result.output


def _cell(n: int, brier: float, accuracy: float, comparison: dict | None = None) -> dict:
    return {"n": n, "n_direction": n, "n_return": n, "accuracy": accuracy, "balanced_accuracy": accuracy,
            "brier": brier, "log_loss": 0.69, "mae": 0.031, "rmse": 0.042, "spearman_ic": 0.05,
            "ci": {"accuracy": [0.45, 0.6], "brier": [0.23, 0.26]}, "mde": {"brier": 0.02, "accuracy": 0.08},
            "comparison": comparison, "calibration": None}


def _eval_summary() -> dict:
    """The shape eval.runner.evaluate returns: metrics nested as results[decision_time][model]["subsets"][subset]."""
    comparison = {"brier": {"baseline": "zero", "baseline_value": 0.25, "model_value": 0.241, "improvement": 0.009,
                            "ci": [0.001, 0.017], "n": 120, "mde": 0.02, "verdict": "improves"}}
    trading = lambda sharpe, pnl: {"fixed": {"sharpe_like": sharpe, "max_drawdown": -0.05, "n_trades": 100,  # noqa: E731
                                             "mean_pnl": {"point": pnl, "ci": [pnl - 0.001, pnl + 0.001]}}}
    return {"run_id": "20260902T000000Z-abcd1234", "created_at": "2026-09-02T00:00:00+00:00", "final": False,
            "target": "r_24h", "schema_version": 2, "dataset_sha256": "abcd1234" * 8, "n_rows": 600, "n_events": 300,
            "git": {"sha": "deadbeef", "dirty": False}, "settings": {"holdout_season": "2026Q3"}, "config_hash": "ff" * 32,
            "versions": {"lightgbm": "4.7.0"}, "decision_times": ["pre_5m"], "models": ["zero", "linear"],
            "baselines": ["zero"],
            "holdout": {"season": "2026Q3", "scorings_before": 0, "scorings_after": 0, "scored_now": False, "n_events": 40},
            "folds": {"pre_5m": []}, "skipped_seasons": {"pre_5m": []}, "cohorts": {"pre_5m": {"n_events": 300}},
            "best_baseline": {"pre_5m": {"headline": {"brier": {"model": "zero", "value": 0.25}}}},
            "results": {"pre_5m": {
                "zero": {"is_baseline": True, "subsets": {"all": _cell(300, 0.25, 0.5), "headline": _cell(120, 0.25, 0.5)},
                         "residual_band": {"q10": -0.05, "q90": 0.06, "n": 300, "coverage": 0.8, "n_with_band": 300},
                         "trading": trading(0.0, 0.0)},
                "linear": {"is_baseline": False,
                           "subsets": {"all": _cell(300, 0.244, 0.53), "headline": _cell(120, 0.241, 0.55, comparison)},
                           "residual_band": {"q10": -0.04, "q90": 0.05, "n": 300, "coverage": 0.81, "n_with_band": 300},
                           "trading": trading(0.8, 0.0012)}}},
            "holdout_results": None, "sizings": ["fixed", "by_confidence"], "n_boot": 1000, "notes": ["a note"]}


def test_summary_rows_flatten_the_nested_eval_results():
    rows = _summary_rows(_eval_summary())
    assert list(rows) == ["pre_5m"] and [r["model"] for r in rows["pre_5m"]] == ["zero", "linear"]
    linear = rows["pre_5m"][1]
    assert linear["subset"] == "headline" and linear["n"] == 120 and linear["brier"] == 0.241
    assert linear["baseline"] == "zero" and linear["verdict"] == "improves" and linear["Δbrier vs baseline"] == 0.009
    assert linear["mean net bp (fixed)"] == pytest.approx(12.0) and rows["pre_5m"][0]["baseline"] == "(is baseline)"
    assert _summary_rows({"run_id": "x"}) == {}


def test_evaluate_prints_leaderboard_and_final_holdout_count(dirs, monkeypatch):
    data, reports = dirs
    _dataset().to_parquet(data / "dataset.parquet", index=False)
    seen = {}

    def evaluate(s, ds, *, model_names, decision_times, final=False, run_id=None, target="r_24h"):
        seen.update(models=model_names, dts=decision_times, final=final, target=target)
        return _eval_summary()

    monkeypatch.setattr(eval_mod, "evaluate", evaluate)
    result = runner.invoke(app, ["evaluate", "--models", "zero,linear", "--decision-times", "pre_5m", "--target", "ar_24h"])
    assert result.exit_code == 0, result.output
    assert seen == {"models": ["zero", "linear"], "dts": ["pre_5m"], "final": False, "target": "ar_24h"}
    out = result.output
    assert "pre_5m" in out and "linear" in out and "headline" in out and "0.241" in out and "improves" in out
    assert "abcd1234" in out and str(reports / "20260902T000000Z-abcd1234" / "leaderboard.md") in out
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "holdout_scorings.jsonl").write_text('{"a":1}\n{"b":2}\n')
    result = runner.invoke(app, ["evaluate", "--final"])
    assert result.exit_code == 0 and seen["final"] is True and "scored 2 time(s)" in result.output


# ---- optimize -----------------------------------------------------------------------------------------
def test_optimize_runs_one_study_per_decision_time(dirs, monkeypatch):
    data, _ = dirs
    _dataset().to_parquet(data / "dataset.parquet", index=False)
    studies = []

    def run_study(s, ds, *, decision_time, n_trials, objective="brier", timeout_seconds=None):
        studies.append((decision_time, n_trials, objective))
        return {"study": f"freedom_{decision_time}_{objective}", "n_trials": n_trials, "best_value": 0.24,
                "best_params": {"model": "linear", "target": "r_24h", "train_window_seasons": 4},
                "baseline_name": "base_rate", "baseline_value": 0.25, "improvement": 0.01, "p_noise": 0.2,
                "n_events": 300, "n_folds": 5, "report_dir": str(s.reports_dir / "optimize" / decision_time)}

    monkeypatch.setattr(optimize_mod, "run_study", run_study)
    result = runner.invoke(app, ["optimize", "--decision-times", "pre_5m,post_30m", "--n-trials", "7", "--objective", "log_loss"])
    assert result.exit_code == 0, result.output
    assert studies == [("pre_5m", 7, "log_loss"), ("post_30m", 7, "log_loss")]
    assert "freedom_pre_5m_log_loss" in result.output and "base_rate" in result.output
    assert runner.invoke(app, ["optimize", "--objective", "sharpe"]).exit_code == 2
    assert runner.invoke(app, ["optimize", "--decision-times", "noon"]).exit_code == 2


def test_optimize_without_dataset_or_rows_exits_2(dirs, monkeypatch):
    data, _ = dirs
    result = runner.invoke(app, ["optimize"])
    assert result.exit_code == 2 and "run `freedom dataset` first" in result.output
    _dataset().to_parquet(data / "dataset.parquet", index=False)

    def run_study(s, ds, **kw):
        raise ValueError("dataset has no rows for decision time 'post_60m'; run `freedom dataset --decision-times post_60m` first")

    monkeypatch.setattr(optimize_mod, "run_study", run_study)
    result = runner.invoke(app, ["optimize", "--decision-times", "post_60m"])
    assert result.exit_code == 2 and "freedom dataset --decision-times post_60m" in result.output


# ---- train --------------------------------------------------------------------------------------------
def test_train_saves_and_prints_model_metadata(dirs, monkeypatch):
    data, _ = dirs
    result = runner.invoke(app, ["train"])  # no dataset yet: the message names the command
    assert result.exit_code == 2 and "run `freedom dataset` first" in result.output
    _dataset().to_parquet(data / "dataset.parquet", index=False)

    def train_final(s, ds, *, model_name, decision_time, target="r_24h"):
        d = s.models_dir / decision_time / model_name
        d.mkdir(parents=True)
        (d / "model.json").write_text(json.dumps({"model_name": model_name, "decision_time": decision_time,
                                                  "n_events": 280, "dataset_sha256": "ff00" * 16}))
        return object()

    monkeypatch.setattr(eval_mod, "train_final", train_final)
    result = runner.invoke(app, ["train", "--model", "linear", "--decision-time", "pre_5m"])
    assert result.exit_code == 0, result.output
    assert "280" in result.output and str(data / "models" / "pre_5m" / "linear") in result.output
    assert runner.invoke(app, ["train", "--decision-time", "noon"]).exit_code == 2


# ---- predict ------------------------------------------------------------------------------------------
def _prediction() -> dict:
    as_of = pd.Timestamp("2026-08-26 20:00", tz="UTC")
    row = {E.event_id: "NVDA:2026-07", E.market: "xyz:NVDA", "decision_time": "pre_5m", "as_of": as_of,
           "t0_used": as_of + pd.Timedelta(minutes=5), "t0_source_live": "expected_sec_8k", "off_schedule": False,
           "p_up": 0.62, "r_hat": 0.011, "r_lo": -0.04, "r_hi": 0.07, "magnitude_hat": 0.011,
           "model_id": "pre_5m/linear@abcdef01", "sources_used": "hyperliquid", "bar_source": "hyperliquid",
           "input_lag_s_hyperliquid": 75.0, "input_lag_s_fmp": np.nan, "input_lag_s_sec": np.nan,
           "n_features": 12, "n_features_missing": 2}
    return {"row": row, "features": {}, "contributions": [{"feature": "f_pre_price_x", "importance": 0.7, "value": 0.5}],
            "schedule": {"note": "on schedule"}, "model_meta": {"n_events": 300, "trained_at": "2026-09-01T00:00:00Z"},
            "consensus": {E.estimate_source: "consensus_snapshot", E.estimate_snapshot_time: as_of,
                          E.eps_estimate: 1.2, E.rev_estimate: 4.6e10, E.n_estimates: 30}}


def test_predict_prints_the_prediction(dirs, monkeypatch):
    seen = {}

    def predict_event(s, *, event_id, decision, model_name=None, now=None, append=True, **kw):
        seen.update(event_id=event_id, decision=decision, model_name=model_name, now=now, append=append)
        return _prediction()

    monkeypatch.setattr(live_mod, "predict_event", predict_event)
    result = runner.invoke(app, ["predict", "--event", "NVDA:2026-07", "--decision", "pre_5m", "--model", "linear",
                                 "--now", "2026-08-26T19:30:00Z"])
    assert result.exit_code == 0, result.output
    assert seen == {"event_id": "NVDA:2026-07", "decision": "pre_5m", "model_name": "linear",
                    "now": "2026-08-26T19:30:00Z", "append": True}
    assert "0.62" in result.output and "consensus_snapshot" in result.output and "f_pre_price_x" in result.output
    assert "live_predictions.parquet" in result.output
    assert runner.invoke(app, ["predict", "--event", "x", "--decision", "noon"]).exit_code == 2


def test_predict_failure_modes_have_distinct_exit_codes(dirs, monkeypatch):
    def not_detected(s, **kw):
        raise live_mod.ReleaseNotDetected("no release detected yet for NVDA:2026-07")

    monkeypatch.setattr(live_mod, "predict_event", not_detected)
    result = runner.invoke(app, ["predict", "--event", "NVDA:2026-07"])
    assert result.exit_code == 3 and "nothing to predict yet" in result.output

    def no_model(s, **kw):
        raise live_mod.ModelNotFound("data/models/post_30m/linear has no trained model: run `freedom train --model linear --decision-time post_30m` first")

    monkeypatch.setattr(live_mod, "predict_event", no_model)
    result = runner.invoke(app, ["predict", "--event", "NVDA:2026-07"])
    assert result.exit_code == 2 and "freedom train" in result.output

    def no_event(s, **kw):
        raise live_mod.EventNotFound("event 'ZZZ:2026-07' is neither in events.parquet nor upcoming: run `freedom events` or `freedom upcoming`")

    monkeypatch.setattr(live_mod, "predict_event", no_event)
    result = runner.invoke(app, ["predict", "--event", "ZZZ:2026-07"])
    assert result.exit_code == 2 and "freedom upcoming" in result.output

    def key_error(s, **kw):  # a data-shape bug: not a missing prerequisite, no "run X first" hint
        raise KeyError("r_24h")

    monkeypatch.setattr(live_mod, "predict_event", key_error)
    result = runner.invoke(app, ["predict", "--event", "NVDA:2026-07"])
    assert result.exit_code == 1 and isinstance(result.exception, KeyError) and "first" not in result.output

    def no_events_table(s, **kw):
        raise FileNotFoundError(f"{s.events_path} missing")

    monkeypatch.setattr(live_mod, "predict_event", no_events_table)
    result = runner.invoke(app, ["predict", "--event", "NVDA:2026-07"])
    assert result.exit_code == 2 and "run `freedom events` first" in result.output


# ---- upcoming -----------------------------------------------------------------------------------------
def test_upcoming_lists_events_with_the_id_predict_takes(dirs, monkeypatch):
    def upcoming(s, days=14):  # the events implementation's columns: no event_id
        return pd.DataFrame({E.underlying: ["NVDA"], E.market: ["xyz:NVDA"], E.kind: ["equity_us"],
                             E.report_date_ny: [pd.Timestamp("2026-11-18").date()],
                             "expected_t0": [pd.Timestamp("2026-11-18 21:05", tz="UTC")],
                             "expected_t0_source": ["calendar default (AMC)"], E.eps_estimate: [1.3]})

    monkeypatch.setattr(events_mod, "upcoming_events", upcoming)
    result = runner.invoke(app, ["upcoming", "--days", "30"])
    assert result.exit_code == 0 and "NVDA:2026-09" in result.output and "30 days" in result.output
    monkeypatch.setattr(events_mod, "upcoming_events", lambda s, days=14: pd.DataFrame())
    assert "no universe events" in runner.invoke(app, ["upcoming"]).output

    def no_key(s, days=14):
        raise ProviderUnavailable("FMP_API_KEY is not set")

    monkeypatch.setattr(events_mod, "upcoming_events", no_key)
    assert runner.invoke(app, ["upcoming"]).exit_code == 2


# ---- status -------------------------------------------------------------------------------------------
def test_status_reports_keys_budgets_artifacts_and_archive(dirs):
    data, reports = dirs
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "missing: run `freedom universe`" in result.output and "present" in result.output
    # budget used today, archive coverage, artifact rows, holdout scorings
    DailyBudget("fmp", 240, data).consume(3)
    _universe().to_parquet(data / "universe.parquet", index=False)
    from freedom.config import get_settings
    from freedom.data.archive import candle_path, write_parquet_atomic

    s = get_settings()
    t = pd.date_range("2026-08-20", periods=48, freq="1h", tz="UTC")
    write_parquet_atomic(pd.DataFrame({"t": t, "t_end": t + pd.Timedelta(hours=1), "close": 1.0}),
                         candle_path(s, "xyz:NVDA", "1h"))
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "holdout_scorings.jsonl").write_text('{"a":1}\n')
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "48" in result.output and "xyz_NVDA" in result.output and "2026-08-21 23:00" in result.output
    assert "universe" in result.output and "ok" in result.output
    assert "holdout scorings so far" in result.output
