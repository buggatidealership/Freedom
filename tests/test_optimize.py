"""Optuna study on a tiny synthetic dataset with fake eval functions, models and feature
groups patched in.

The eval, models and features modules are implemented elsewhere; here they are replaced by
minimal deterministic stand-ins that honour their contracts (fold shapes, metric dicts, the
BaseModel interface and constructor signatures, the group registry with its declared keys) so
the study mechanics can be tested offline in seconds. The tests marked with the `needs_real_*`
markers run only once the real registries are importable and check the contracts against them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest
from lightgbm.basic import _ConfigAliases

import freedom.eval as eval_mod
import freedom.features as features_mod
import freedom.models as models_mod
from freedom import optimize as opt
from freedom.schemas import D, E, T, season_of

SEASONS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]
HOLDOUT = "2026Q3"

# The v1 groups (docs/design.md §6) with their admissibility and a subset of the keys the
# features module declares for them (features.groups.GROUP_KEYS): column f_<key>.
FAKE_GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "calendar": (("pre", "post"), ("weekday", "amc")),
    "pre_price": (("pre", "post"), ("ret_1d", "drift_60m")),
    "history": (("pre", "post"), ("hist_n", "hist_r24_mean")),
    "market": (("pre", "post"), ("mkt_ret_1d",)),
    "perp_state": (("pre", "post"), ("funding_rate",)),
    "surprise": (("post",), ("eps_surprise",)),
    "reaction": (("post",), ("r_15m", "r_now")),
}
REAL_MODELS = all(f in models_mod.REGISTRY and models_mod.REGISTRY[f].__module__.startswith("freedom.models")
                  for f in ("linear", "lightgbm", "ensemble"))
REAL_FEATURES = bool(features_mod.REGISTRY) and bool(getattr(getattr(features_mod, "groups", None), "GROUP_KEYS", None))
needs_real_models = pytest.mark.skipif(not REAL_MODELS, reason="the models registry holds no real linear/lightgbm/ensemble here")
needs_real_features = pytest.mark.skipif(not REAL_FEATURES, reason="the features registry is not populated here")


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
                    "f_weekday": float(t0.weekday()), "f_weekday__missing": 0.0,
                    "f_amc": 1.0, "f_amc__missing": 0.0,
                    "f_ret_1d": signal, "f_ret_1d__missing": 0.0,
                    "f_hist_r24_mean": rng.normal(scale=0.01),
                    "f_eps_surprise": (signal + rng.normal()) if post else np.nan,
                    "f_eps_surprise__missing": 0.0 if post else 1.0,
                    "f_r_15m": (0.5 * r24 + rng.normal(scale=0.01)) if post else np.nan,
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


class StrictLinear(FakeLinear):
    """The constructor of models.linear.LinearModel: grids `alphas` / `Cs` (a one-element grid
    is a fixed value), `cv_folds`, and a TypeError for any other keyword."""

    def __init__(self, *, seed: int = 7, alphas=(1.0, 10.0, 100.0), Cs=(0.01, 0.1, 1.0), cv_folds: int = 5, **params):
        if params:
            raise TypeError(f"linear: unknown parameter(s) {sorted(params)}")
        super().__init__(seed=seed, alphas=tuple(alphas), Cs=tuple(Cs), cv_folds=cv_folds)


class StrictLightGBM(FakeLinear):
    """The constructor of models.lgbm.LightGBMModel: the round count and the early-stopping
    patience are arguments, everything else goes into LightGBM's params dict — where an alias
    of num_iterations / early_stopping_round would override them, so it must never arrive."""

    FORBIDDEN = frozenset(_ConfigAliases.get("num_iterations")) | frozenset(_ConfigAliases.get("early_stopping_round"))

    def __init__(self, *, seed: int = 7, num_boost_round: int = 500, early_stopping_rounds: int = 30,
                 valid_fraction: float = 0.2, **lgb_params):
        bad = sorted(self.FORBIDDEN & lgb_params.keys())
        if bad:
            raise TypeError(f"lightgbm: {bad} reached the params dict")
        super().__init__(seed=seed, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                         valid_fraction=valid_fraction, **lgb_params)
        self.num_boost_round, self.early_stopping_rounds = num_boost_round, early_stopping_rounds
        self.lgb_params = dict(lgb_params)  # what lgb.train would receive as `params`


class StrictEnsemble(FakeLinear):
    """models.ensemble.Ensemble's constructor: members by registry name built with member_params."""

    def __init__(self, *, seed: int = 7, members=("linear", "lightgbm"), member_params=None, **params):
        super().__init__(seed=seed, members=tuple(members), member_params=member_params, **params)
        self.members_ = [models_mod.make_model(m, seed=seed, **(member_params or {}).get(m, {})) for m in members]


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
def fake_features(monkeypatch):
    """The v1 groups registered with their admissibility and declared keys (FAKE_GROUPS)."""
    monkeypatch.setattr(features_mod, "REGISTRY", {name: ((lambda ctx: {}), adm) for name, (adm, _) in FAKE_GROUPS.items()})
    monkeypatch.setattr(features_mod, "groups",
                        SimpleNamespace(GROUP_KEYS={name: keys for name, (_, keys) in FAKE_GROUPS.items()}),
                        raising=False)
    return monkeypatch


@pytest.fixture
def patched(fake_features):
    monkeypatch = fake_features
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


# ---- feature groups ---------------------------------------------------------------------------------
def test_feature_groups_come_from_the_declared_keys_and_respect_the_phase(fake_features):
    cols = make_dataset(n_per_season=2).columns
    pre = opt.feature_groups(cols, "pre_5m")
    post = opt.feature_groups(cols, "post_30m")
    assert set(pre) == {"calendar", "pre_price", "history"}
    assert set(post) == {"calendar", "pre_price", "history", "surprise", "reaction"}
    assert pre["calendar"] == ["f_amc", "f_amc__missing", "f_weekday", "f_weekday__missing"]
    assert post["reaction"] == ["f_r_15m"] and post["surprise"] == ["f_eps_surprise", "f_eps_surprise__missing"]
    # keys never share a common prefix with their group name: a naming convention cannot do this
    assert "f_hist_r24_mean" in pre["history"] and "f_ret_1d" in pre["pre_price"]


def test_feature_column_outside_the_registry_is_an_error_not_a_group(fake_features):
    cols = [*make_dataset(n_per_season=1).columns, "f_bogus_thing", "f_bogus_thing__missing"]
    with pytest.raises(ValueError, match="f_bogus_thing.*freedom dataset"):
        opt.feature_groups(cols, "pre_5m")
    fake_features.setattr(features_mod, "REGISTRY", {})
    with pytest.raises(ValueError, match="registry is empty"):
        opt.feature_groups(cols, "pre_5m")


def test_group_without_declared_keys_or_with_shared_keys_is_an_error(fake_features):
    fake_features.setitem(features_mod.REGISTRY, "text", ((lambda ctx: {}), ("post",)))
    with pytest.raises(ValueError, match=r"\['text'\] declare no output keys"):
        opt.group_keys()
    fake_features.setitem(features_mod.groups.GROUP_KEYS, "text", ("weekday",))
    with pytest.raises(ValueError, match="'weekday' is declared by both"):
        opt.group_keys()


@needs_real_features
def test_real_feature_registry_attributes_every_declared_column():
    keys = opt.group_keys()
    assert set(keys) == set(features_mod.REGISTRY) and all(keys[g] for g in keys)
    cols = [f"{D.feature_prefix}{k}{s}" for g in keys for k in keys[g] for s in ("", D.missing_suffix)]
    pre, post = opt.feature_groups(cols, "pre_5m"), opt.feature_groups(cols, "post_30m")
    assert set(post) == set(keys)
    assert set(pre) == {g for g, (_, adm) in features_mod.REGISTRY.items() if "pre" in adm}
    assert {"surprise", "reaction"}.isdisjoint(pre)
    for g, group_cols in post.items():
        assert len(group_cols) == 2 * len(keys[g])
        assert all(c[len(D.feature_prefix):].removesuffix(D.missing_suffix) in keys[g] for c in group_cols)


# ---- model knobs ------------------------------------------------------------------------------------
def test_lightgbm_alias_table_matches_the_installed_lightgbm():
    assert opt.LGB_NUM_ITERATIONS_ALIASES == frozenset(_ConfigAliases.get("num_iterations")) - {"num_boost_round"}
    with pytest.raises(ValueError, match="num_boost_round"):
        opt.model_kwargs("lightgbm", {"model": "lightgbm", "lightgbm.n_estimators": 100})


@pytest.mark.parametrize("family", opt.FAMILIES)
def test_suggested_params_are_the_models_constructor_knobs(fake_features, family):
    fake_features.setitem(models_mod.REGISTRY, "linear", StrictLinear)
    fake_features.setitem(models_mod.REGISTRY, "lightgbm", StrictLightGBM)
    fake_features.setitem(models_mod.REGISTRY, "ensemble", StrictEnsemble)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    study.enqueue_trial({"model": family})
    params = opt.suggest(study.ask(), ["calendar", "pre_price"], (T.r("24h"),), n_seasons=4)
    cfg = opt.TrialConfig.from_params(params, ["calendar", "pre_price"])
    model = opt.build_model(cfg, seed=7)  # StrictLinear / StrictLightGBM raise on a stray keyword
    members = {family: model} if family != "ensemble" else dict(zip(model.params["members"], model.members_, strict=True))
    if family in ("linear", "ensemble"):
        lin = members["linear"]
        assert lin.params["alphas"] == (params["linear.alpha"],) and lin.params["Cs"] == (params["linear.C"],)
        assert not {"alpha", "C"} & lin.params.keys()
    if family in ("lightgbm", "ensemble"):
        lgb = members["lightgbm"]
        assert lgb.num_boost_round == params["lightgbm.num_boost_round"]
        assert lgb.early_stopping_rounds == params["lightgbm.early_stopping_rounds"]
        assert lgb.lgb_params["num_leaves"] == params["lightgbm.num_leaves"]
        assert not StrictLightGBM.FORBIDDEN & lgb.lgb_params.keys()


def _fit_predict(params: dict, X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    cfg = opt.TrialConfig.from_params({**params, "target": T.r("24h"), "train_window_seasons": 2}, [])
    m = opt.build_model(cfg, seed=7).fit(X, y, np.sign(y))
    return np.asarray(m.predict_return(X), float), np.asarray(m.predict_proba_up(X), float)


@needs_real_models
def test_real_models_change_with_the_suggested_params():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(160, 3)), columns=["f_ret_1d", "f_weekday", "f_hist_r24_mean"])
    y = pd.Series(0.03 * X["f_ret_1d"] + 0.01 * X["f_hist_r24_mean"] + rng.normal(scale=0.02, size=len(X)))
    weak = _fit_predict({"model": "linear", "linear.alpha": 0.1, "linear.C": 10.0}, X, y)
    strong = _fit_predict({"model": "linear", "linear.alpha": 1000.0, "linear.C": 1e-3}, X, y)
    assert not np.allclose(weak[0], strong[0]) and not np.allclose(weak[1], strong[1])
    assert np.abs(strong[0] - y.mean()).max() < np.abs(weak[0] - y.mean()).max()  # stronger L2 shrinks harder
    base = {"model": "lightgbm", "lightgbm.num_leaves": 4, "lightgbm.min_data_in_leaf": 20,
            "lightgbm.feature_fraction": 1.0, "lightgbm.bagging_fraction": 1.0, "lightgbm.learning_rate": 0.1,
            "lightgbm.early_stopping_rounds": 500, "lightgbm.lambda_l2": 0.01}
    one = _fit_predict({**base, "lightgbm.num_boost_round": 1}, X, y)
    many = _fit_predict({**base, "lightgbm.num_boost_round": 300}, X, y)
    assert not np.allclose(one[0], many[0]) and not np.allclose(one[1], many[1])
    assert np.abs(many[0] - y).mean() < np.abs(one[0] - y).mean()  # more rounds fit the training data better
    ens = _fit_predict({**base, "lightgbm.num_boost_round": 300, "linear.alpha": 0.1, "linear.C": 10.0, "model": "ensemble"}, X, y)
    assert np.allclose(ens[0], 0.5 * (weak[0] + many[0]))


# ---- rows, windows, study ---------------------------------------------------------------------------
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


def test_share_true_tolerates_the_nullable_boolean_dtype():
    assert opt.share_true(pd.Series([True, None, False], dtype="boolean")) == pytest.approx(1 / 3)
    assert opt.share_true(pd.Series([True, np.nan, True])) == pytest.approx(2 / 3)


def test_run_study_persists_reports_and_never_scores_the_holdout(patched, small_settings):
    seen: list[pd.DataFrame] = []
    real = eval_mod.walk_forward_folds

    def spy(events, **kw):
        seen.append(events)
        return real(events, **kw)

    patched.setattr(eval_mod, "walk_forward_folds", spy)
    ds = make_dataset()
    ds[E.has_perp_at_t0] = ds[E.has_perp_at_t0].astype("boolean")
    ds.loc[ds.index[:3], E.has_perp_at_t0] = pd.NA  # as build_dataset writes it: nullable, possibly NA
    res = opt.run_study(small_settings, ds, decision_time="pre_5m", n_trials=6, objective="brier")

    assert res["study"] == "freedom_pre_5m_brier" and small_settings.optuna_db.exists()
    assert res["n_trials"] >= 1 and res["n_folds"] >= 2
    assert HOLDOUT not in res["seasons"] and res["holdout_season"] == HOLDOUT
    assert all(HOLDOUT not in set(frame["season"]) for frame in seen)
    assert set(res["groups"]) == {"calendar", "pre_price", "history"}
    assert 0.0 <= res["has_perp_share"] <= 1.0
    # decision time and the confidence floor are not search dimensions
    forbidden = {"decision_time", "min_t0_confidence", "holdout_season"}
    assert not forbidden & set(res["best_params"])
    assert not {"lightgbm.n_estimators", "linear.alphas"} & set(res["best_params"])
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
