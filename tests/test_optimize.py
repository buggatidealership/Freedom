"""Optuna study on a tiny synthetic dataset with fake eval functions and models patched in.

The eval and models modules are implemented elsewhere; here they are replaced by minimal
deterministic stand-ins that honour their docstrings (fold shapes, metric dicts, the
BaseModel interface) so the study mechanics can be tested offline in seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import freedom.eval as eval_mod
import freedom.models as models_mod
from freedom import optimize as opt
from freedom.schemas import D, E, T, season_of

SEASONS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]
HOLDOUT = "2026Q3"


def _season_start(season: str) -> pd.Timestamp:
    year, q = int(season[:4]), int(season[-1])
    return pd.Timestamp(year=year, month=(q - 1) * 3 + 1, day=1, tz="UTC")


def make_dataset(n_per_season: int = 12, decision_times=("pre_5m", "post_30m"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for si, season in enumerate(SEASONS):
        for i in range(n_per_season):
            t0 = _season_start(season) + pd.Timedelta(days=2 * i, hours=20, minutes=30)
            signal = rng.normal()
            r24 = 0.02 * signal + rng.normal(scale=0.03)
            for d in decision_times:
                post = d.startswith("post")
                rows.append({
                    D.event_id: f"U{i % 5}:{season}-{i}", E.underlying: f"U{i % 5}", D.decision_time: d,
                    D.as_of: t0, E.t0: t0, E.t0_confidence: 0.95 if i % 6 else 0.5, E.t0_source: "sec_8k",
                    E.has_perp_at_t0: si >= 4, "season": season,
                    T.r("24h"): r24, T.ar("24h"): r24 - 0.001, T.direction: float(np.sign(r24)),
                    "f_calendar_weekday": float(t0.weekday()), "f_calendar_weekday__missing": 0.0,
                    "f_pre_price_signal": signal, "f_pre_price_signal__missing": 0.0,
                    "f_history_mean_r24": rng.normal(scale=0.01),
                    "f_surprise_eps_pct": (signal + rng.normal()) if post else np.nan,
                    "f_surprise_eps_pct__missing": 0.0 if post else 1.0,
                    "f_reaction_r_15m": (0.5 * r24 + rng.normal(scale=0.01)) if post else np.nan,
                })
    return pd.DataFrame(rows)


# ---- stand-ins --------------------------------------------------------------------------------
class FakeLinear(models_mod.BaseModel):
    """Least squares of the return on the features; p_up from a steep logistic of r_hat."""

    def fit(self, X, y_return, y_direction):
        self.feature_names_ = list(X.columns)
        A = np.c_[X.fillna(0.0).to_numpy(float), np.ones(len(X))]
        self.coef_, *_ = np.linalg.lstsq(A, y_return.to_numpy(float), rcond=None)
        return self

    def predict_return(self, X):
        A = np.c_[X[self.feature_names_].fillna(0.0).to_numpy(float), np.ones(len(X))]
        return A @ self.coef_

    def predict_proba_up(self, X):
        return 1.0 / (1.0 + np.exp(-40.0 * self.predict_return(X)))


class FakeZero(models_mod.BaseModel):
    def fit(self, X, y_return, y_direction):
        self.feature_names_ = list(X.columns)
        return self

    def predict_return(self, X):
        return np.zeros(len(X))

    def predict_proba_up(self, X):
        return np.full(len(X), 0.5)


class FakeBaseRate(models_mod.BaseModel):
    def fit(self, X, y_return, y_direction):
        self.feature_names_ = list(X.columns)
        self.p_ = float((y_direction > 0).mean())
        self.r_ = float(y_return.mean())
        return self

    def predict_return(self, X):
        return np.full(len(X), self.r_)

    def predict_proba_up(self, X):
        return np.full(len(X), self.p_)


class Broken(models_mod.BaseModel):
    def fit(self, X, y_return, y_direction):
        raise RuntimeError("simulated fit failure")

    def predict_return(self, X):
        raise AssertionError

    def predict_proba_up(self, X):
        raise AssertionError


def fake_folds(events, *, min_train, embargo_days, holdout_season):
    seasons = sorted({season_of(t) for t in events[E.t0]})
    labels = events[E.t0].map(season_of)
    folds, holdout = [], None
    for s in seasons:
        test = events.index[labels == s]
        train = events.index[events[E.t0] < _season_start(s) - pd.Timedelta(days=embargo_days)]
        if s == holdout_season:
            holdout = eval_mod.Fold(fold=-1, train_idx=train, test_idx=test, test_season=s)
            continue
        if len(train) < min_train:
            continue
        folds.append(eval_mod.Fold(fold=len(folds), train_idx=train, test_idx=test, test_season=s))
    return folds, holdout


def fake_classification_metrics(p_up, y_dir):
    p, y = np.asarray(p_up, float), np.asarray(y_dir, float)
    keep = y != 0
    p, y = np.clip(p[keep], 1e-6, 1 - 1e-6), (y[keep] > 0).astype(float)
    hit = (p > 0.5) == (y > 0)
    return {"accuracy": float(hit.mean()), "balanced_accuracy": float(0.5 * (hit[y > 0].mean() + hit[y == 0].mean())),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "n": int(len(y))}


def fake_regression_metrics(r_hat, r_true):
    a, b = pd.Series(np.asarray(r_hat, float)), pd.Series(np.asarray(r_true, float))
    return {"mae": float((a - b).abs().mean()), "rmse": float(np.sqrt(((a - b) ** 2).mean())),
            "spearman_ic": float(a.corr(b, method="spearman")), "n": int(len(a))}


@pytest.fixture
def patched(monkeypatch):
    for name, cls in {"linear": FakeLinear, "lightgbm": FakeLinear, "ensemble": FakeLinear,
                      "zero": FakeZero, "base_rate": FakeBaseRate}.items():
        monkeypatch.setitem(models_mod.REGISTRY, name, cls)
    monkeypatch.setattr(eval_mod, "walk_forward_folds", fake_folds)
    monkeypatch.setattr(eval_mod, "classification_metrics", fake_classification_metrics)
    monkeypatch.setattr(eval_mod, "regression_metrics", fake_regression_metrics)
    return monkeypatch


@pytest.fixture
def small_settings(settings):
    return settings.model_copy(update={"min_train_events": 20, "holdout_season": HOLDOUT, "embargo_days": 2})


# ---- tests ------------------------------------------------------------------------------------------
def test_feature_groups_respect_the_decision_phase():
    cols = make_dataset(n_per_season=2).columns
    pre = opt.feature_groups(cols, "pre_5m")
    post = opt.feature_groups(cols, "post_30m")
    assert set(pre) == {"calendar", "pre_price", "history"}
    assert set(post) == {"calendar", "pre_price", "history", "surprise", "reaction"}
    assert pre["calendar"] == ["f_calendar_weekday", "f_calendar_weekday__missing"]


def test_prepare_rows_drops_holdout_and_low_confidence(small_settings):
    rows = opt.prepare_rows(small_settings, make_dataset(), "pre_5m")
    assert HOLDOUT not in set(rows["season"])
    assert (rows[E.t0_confidence] >= small_settings.min_t0_confidence).all()
    assert rows.index.equals(pd.RangeIndex(len(rows)))
    assert (rows[D.decision_time] == "pre_5m").all()
    with pytest.raises(ValueError, match="freedom dataset"):
        opt.prepare_rows(small_settings, make_dataset(), "post_60m")


def test_training_window_is_floored_at_min_train(patched, small_settings):
    rows = opt.prepare_rows(small_settings, make_dataset(), "pre_5m")
    folds = opt.make_folds(small_settings, rows)
    last = folds[-1]
    one = opt.training_rows(rows, last, 1, min_train=5)
    assert one["season"].nunique() == 1 and len(one) == 10  # 12 per season minus 2 low-confidence
    floored = opt.training_rows(rows, last, 1, min_train=25)
    assert len(floored) == 25 and floored[E.t0].max() == rows.loc[last.train_idx, E.t0].max()
    assert len(opt.training_rows(rows, last, 99, min_train=5)) == len(last.train_idx)


def test_run_study_persists_reports_and_never_scores_the_holdout(patched, small_settings):
    seen: list[pd.DataFrame] = []
    real = eval_mod.walk_forward_folds

    def spy(events, **kw):
        seen.append(events)
        return real(events, **kw)

    patched.setattr(eval_mod, "walk_forward_folds", spy)
    ds = make_dataset()
    res = opt.run_study(small_settings, ds, decision_time="pre_5m", n_trials=6, objective="brier")

    assert res["study"] == "freedom_pre_5m_brier" and small_settings.optuna_db.exists()
    assert res["n_trials"] >= 1 and res["n_folds"] >= 2
    assert HOLDOUT not in res["seasons"] and res["holdout_season"] == HOLDOUT
    assert all(HOLDOUT not in set(frame["season"]) for frame in seen)
    assert set(res["groups"]) == {"calendar", "pre_price", "history"}
    # decision time and the confidence floor are not search dimensions
    forbidden = {"decision_time", "min_t0_confidence", "holdout_season"}
    assert not forbidden & set(res["best_params"])
    board = opt.leaderboard(small_settings, "pre_5m", "brier")
    assert board["state"].eq("complete").sum() == res["n_trials"]
    assert not any(forbidden & set(json.loads(p)) for p in board["params"])
    assert board.loc[board["rank"] == 1, "value"].iloc[0] == pytest.approx(res["best_value"])
    # baseline comparison and noise probability
    assert res["baseline_name"] in {"zero", "base_rate"}
    assert res["improvement"] == pytest.approx(res["baseline_value"] - res["best_value"])
    assert 0.0 <= res["p_noise"] <= 1.0
    # reports
    out = small_settings.reports_dir / "optimize" / "freedom_pre_5m_brier"
    text = (out / "leaderboard.md").read_text()
    assert "never scored" in text and f"holdout {HOLDOUT}" in text and "p_noise" in text
    best = json.loads((out / "best_params.json").read_text())
    assert best["best_params"] == res["best_params"] and best["test_set_hash"] == res["test_set_hash"]
    assert best["n_trials"] == res["n_trials"]


def test_resume_adds_trials_to_the_same_study(patched, small_settings):
    ds = make_dataset()
    first = opt.run_study(small_settings, ds, decision_time="post_30m", n_trials=3)
    second = opt.run_study(small_settings, ds, decision_time="post_30m", n_trials=3)
    assert second["n_trials"] + second["n_pruned"] + second["n_failed"] == 6
    assert second["test_set_hash"] == first["test_set_hash"]
    assert len(opt.leaderboard(small_settings, "post_30m")) == 6


def test_resumed_study_on_a_different_test_set_aborts(patched, small_settings):
    ds = make_dataset()
    opt.run_study(small_settings, ds, decision_time="pre_5m", n_trials=2)
    smaller = ds[ds[D.event_id] != "U1:2026Q1-1"]
    with pytest.raises(opt.TestSetMismatch, match="different test set"):
        opt.run_study(small_settings, smaller, decision_time="pre_5m", n_trials=2)


def test_trial_with_a_drifting_test_set_aborts_the_study(patched, small_settings):
    calls = {"n": 0}

    def drifting(events, **kw):
        folds, holdout = fake_folds(events, **kw)
        calls["n"] += 1
        if calls["n"] > 1:  # the first call is the study-level reference; later ones drop a test event
            f = folds[-1]
            folds[-1] = eval_mod.Fold(f.fold, f.train_idx, f.test_idx[:-1], f.test_season)
        return folds, holdout

    patched.setattr(eval_mod, "walk_forward_folds", drifting)
    with pytest.raises(opt.TestSetMismatch, match="study aborted"):
        opt.run_study(small_settings, make_dataset(), decision_time="pre_5m", n_trials=3)


def test_failed_trials_are_recorded_not_fatal(patched, small_settings):
    patched.setitem(models_mod.REGISTRY, "lightgbm", Broken)
    patched.setitem(models_mod.REGISTRY, "ensemble", Broken)
    res = opt.run_study(small_settings, make_dataset(), decision_time="post_30m", n_trials=8)
    board = opt.leaderboard(small_settings, "post_30m")
    assert res["n_failed"] >= 1 and (board["state"] == "fail").sum() == res["n_failed"]
    assert board.loc[board["state"] == "fail", "error"].str.contains("simulated fit failure").all()
    assert "simulated fit failure" in (small_settings.reports_dir / "optimize" / "freedom_post_30m_brier" / "leaderboard.md").read_text()


def test_leaderboard_without_a_study_names_the_command(small_settings):
    with pytest.raises(FileNotFoundError, match="freedom optimize"):
        opt.leaderboard(small_settings, "pre_5m")


def test_p_noise_is_low_for_a_clear_edge_and_high_for_none(patched):
    rng = np.random.default_rng(1)
    n = 200
    y = rng.choice([-1.0, 1.0], size=n)
    frame = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "fold": 0,
                          "test_season": np.repeat(["a", "b", "c", "d"], n // 4),
                          "p_up": np.where(y > 0, 0.8, 0.2), "r_hat": 0.0, "r_true": y * 0.01, "direction_true": y})
    base = frame.assign(p_up=0.5)
    assert opt.p_noise_bootstrap(frame, base, "brier", n=200) == 0.0
    assert opt.p_noise_bootstrap(base, base, "brier", n=200) == 1.0
