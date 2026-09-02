"""Optuna study over models, features and target variant — one study per decision time.

Decision time and the t0-confidence floor are NOT search dimensions (docs/design.md §9). The
objective is the walk-forward metric on folds that exclude the pinned holdout season; the
holdout is never scored here. Every trial is scored on the identical event-id set for its
decision time; the study aborts if a trial's test set differs (hash recorded).

How a trial is scored
* The dataset rows for the decision time are filtered once (t0_confidence >= min_t0_confidence,
  a non-NaN r_24h label, holdout season dropped) and frozen for the whole study.
* Folds come from eval.walk_forward_folds and are rebuilt inside every trial; the sha256 of
  (fold, sorted test event ids) is compared with the hash stored on the study (`test_set_hash`)
  so that neither a drifting fold builder nor a resumed study on a different dataset can mix
  test sets.
* The search space changes only what the model is trained on: family and hyper-parameters,
  which admissible feature groups are used, the training target (`r_24h` or `ar_24h`) and the
  number of seasons in the training window (floored at settings.min_train_events). Scoring is
  always against the fixed headline labels `r_24h` / `direction_24h`, so every trial's value is
  comparable with every other trial and with the baselines.
* Baselines (models registry, docs/design.md §7) are scored on the same folds; the best one per
  objective is the reference. `p_noise` is the share of season-block bootstrap resamples in
  which the best trial does not beat that baseline (paired, same events). It does not correct
  for the best-of-N selection over trials, and the leaderboard says so.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from .. import eval as eval_mod
from .. import features as features_mod
from .. import models as models_mod
from ..config import Settings
from ..schemas import DECISION_TIMES, D, E, P, T, season_of

log = logging.getLogger(__name__)

FAMILIES: tuple[str, ...] = ("linear", "lightgbm", "ensemble")
ENSEMBLE_MEMBERS: tuple[str, ...] = ("linear", "lightgbm")
TARGET_VARIANTS: tuple[str, ...] = (T.r("24h"), T.ar("24h"))
LABEL_RETURN = T.r("24h")
LABEL_DIRECTION = T.direction
# objective -> (metric family, optuna direction)
OBJECTIVES: dict[str, tuple[str, str]] = {
    "brier": ("classification", "minimize"),
    "log_loss": ("classification", "minimize"),
    "accuracy": ("classification", "maximize"),
    "balanced_accuracy": ("classification", "maximize"),
    "mae": ("regression", "minimize"),
    "rmse": ("regression", "minimize"),
    "spearman_ic": ("regression", "maximize"),
}
BASELINES: tuple[str, ...] = ("zero", "base_rate", "historical_mean", "hist_abs_mean", "vol_scaled",
                              "sign_of_reaction", "always_extends", "surprise_sign")
# Feature groups of docs/design.md §6 and the phases at which they are admissible; the
# features registry (when populated) is layered on top of this table.
DESIGN_GROUPS: dict[str, tuple[str, ...]] = {
    "calendar": ("pre", "post"), "pre_price": ("pre", "post"), "history": ("pre", "post"),
    "market": ("pre", "post"), "perp_state": ("pre", "post"),
    "surprise": ("post",), "reaction": ("post",),
}
MIN_WINDOW_SEASONS = 2
N_BOOTSTRAP = 1000
TEST_SET_ATTR = "test_set_hash"
SEASON_COL = "season"


class TestSetMismatch(RuntimeError):
    """A trial (or a resumed study) was about to be scored on a different test set."""


class TrialFailed(RuntimeError):
    """A trial's model could not be fitted or scored; recorded on the trial, study continues."""


class _ZeroModel(models_mod.BaseModel):
    """Fallback `zero` baseline (p = 0.5, r = 0) used only when the registry has none."""

    name = "zero"

    def fit(self, X, y_return, y_direction):
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba_up(self, X):
        return np.full(len(X), 0.5)

    def predict_return(self, X):
        return np.zeros(len(X))


# ---- search space -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialConfig:
    model: str
    model_params: dict
    groups: tuple[str, ...]
    target: str
    window_seasons: int  # training window in seasons; n_seasons means the full expanding window
    label: str = ""  # free-text (baseline name) for reporting

    @classmethod
    def from_params(cls, params: dict, groups: list[str]) -> TrialConfig:
        family = params["model"]
        lin = {k.split(".", 1)[1]: v for k, v in params.items() if k.startswith("linear.")}
        lgb = {k.split(".", 1)[1]: v for k, v in params.items() if k.startswith("lightgbm.")}
        if family == "linear":
            mp: dict = lin
        elif family == "lightgbm":
            mp = lgb
        else:
            mp = {"members": list(ENSEMBLE_MEMBERS), "member_params": {"linear": lin, "lightgbm": lgb}}
        used = tuple(g for g in groups if params.get(f"use_{g}", False))
        return cls(model=family, model_params=mp, groups=used, target=params["target"],
                   window_seasons=int(params["train_window_seasons"]))


def suggest(trial: optuna.Trial, groups: list[str], targets: tuple[str, ...], n_seasons: int) -> dict:
    """Sample one configuration. Parameter names are flat (`linear.alpha`, `use_calendar`, ...)
    so best_params.json is self-describing; TrialConfig.from_params turns them into a model."""
    family = trial.suggest_categorical("model", list(FAMILIES))
    if family in ("linear", "ensemble"):
        trial.suggest_float("linear.alpha", 0.1, 1000.0, log=True)  # ridge L2
        trial.suggest_float("linear.C", 1e-3, 10.0, log=True)  # logistic inverse L2
    if family in ("lightgbm", "ensemble"):
        trial.suggest_int("lightgbm.num_leaves", 2, 7)
        trial.suggest_int("lightgbm.min_data_in_leaf", 20, 60)
        trial.suggest_float("lightgbm.feature_fraction", 0.4, 1.0)
        trial.suggest_float("lightgbm.bagging_fraction", 0.5, 1.0)
        trial.suggest_float("lightgbm.learning_rate", 0.01, 0.2, log=True)
        trial.suggest_int("lightgbm.n_estimators", 50, 400)
        trial.suggest_float("lightgbm.lambda_l2", 1e-3, 10.0, log=True)
    for g in groups:
        trial.suggest_categorical(f"use_{g}", [True, False])
    trial.suggest_categorical("target", list(targets))
    trial.suggest_int("train_window_seasons", MIN_WINDOW_SEASONS, max(MIN_WINDOW_SEASONS, n_seasons))
    return dict(trial.params)


def group_phases() -> dict[str, tuple[str, ...]]:
    """Design table overlaid with whatever the features registry declares."""
    out = dict(DESIGN_GROUPS)
    for name, (_fn, admissible) in features_mod.REGISTRY.items():
        out[name] = tuple(admissible)
    return out


def feature_groups(columns, decision_time: str) -> dict[str, list[str]]:
    """Feature columns (f_<group>_..., including __missing companions) by group, keeping only
    the groups admissible at the decision time's phase. Columns of a group unknown to both the
    design table and the registry are kept under their first token: the dataset builder only
    writes admissible groups, so an unknown group cannot be a post-only one leaking into pre."""
    phase = features_mod.phase_of(decision_time)
    phases = group_phases()
    names = sorted(phases, key=len, reverse=True)
    prefix = D.feature_prefix
    out: dict[str, list[str]] = {}
    for col in columns:
        if not str(col).startswith(prefix):
            continue
        stem = str(col)[len(prefix):]
        group = next((g for g in names if stem == g or stem.startswith(g + "_")), None)
        if group is None:
            group = stem.split("_", 1)[0]
        elif phase not in phases[group]:
            continue
        out.setdefault(group, []).append(str(col))
    return {g: sorted(cols) for g, cols in sorted(out.items())}


# ---- data preparation --------------------------------------------------------------------------
def prepare_rows(settings: Settings, dataset: pd.DataFrame, decision_time: str) -> pd.DataFrame:
    """Rows of one decision time that can be scored: confidence floor applied, r_24h present,
    holdout season removed. Sorted by t0 with a fresh 0..n-1 index (folds index into it)."""
    if decision_time not in DECISION_TIMES:
        raise ValueError(f"unknown decision time {decision_time!r}; choose from {sorted(DECISION_TIMES)}")
    if D.decision_time not in dataset.columns:
        raise ValueError(f"dataset lacks the {D.decision_time!r} column; run `freedom dataset` first")
    rows = dataset[dataset[D.decision_time] == decision_time].copy()
    if rows.empty:
        raise ValueError(f"dataset has no rows for decision time {decision_time!r}; "
                         f"run `freedom dataset --decision-times {decision_time}` first")
    if E.t0_confidence in rows.columns:
        rows = rows[rows[E.t0_confidence].fillna(0.0) >= settings.min_t0_confidence]
    rows = rows[rows[LABEL_RETURN].notna()]
    if SEASON_COL not in rows.columns:
        rows[SEASON_COL] = rows[E.t0].map(season_of)
    if LABEL_DIRECTION not in rows.columns:
        rows[LABEL_DIRECTION] = np.sign(rows[LABEL_RETURN].astype(float))
    if settings.holdout_season:
        rows = rows[rows[SEASON_COL] != settings.holdout_season]  # never scored here
    if rows.empty:
        raise ValueError(f"no scorable events at {decision_time} outside the holdout season "
                         f"{settings.holdout_season}")
    rows = rows.sort_values([E.t0, D.event_id]).reset_index(drop=True)
    return rows


def available_targets(rows: pd.DataFrame) -> tuple[str, ...]:
    """Target variants with at least one non-NaN value (ar_24h needs a benchmark path)."""
    return tuple(t for t in TARGET_VARIANTS if t in rows.columns and rows[t].notna().any()) or (LABEL_RETURN,)


def make_folds(settings: Settings, rows: pd.DataFrame) -> list[eval_mod.Fold]:
    folds, _holdout = eval_mod.walk_forward_folds(rows, min_train=settings.min_train_events,
                                                 embargo_days=settings.embargo_days,
                                                 holdout_season=settings.holdout_season)
    if not folds:
        raise ValueError(f"no walk-forward fold has {settings.min_train_events} training events; "
                         "more history is needed before optimisation is meaningful")
    return list(folds)


def test_set_hash(rows: pd.DataFrame, folds: list[eval_mod.Fold]) -> str:
    """sha256 over (fold, sorted test event ids) — the identity of what a trial is scored on."""
    parts = []
    for f in folds:
        ids = sorted(str(x) for x in rows.loc[f.test_idx, D.event_id])
        parts.append(f"{f.fold}:{f.test_season}:" + ",".join(ids))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def dataset_hash(rows: pd.DataFrame) -> str:
    cols = [c for c in (D.event_id, E.t0, LABEL_RETURN) if c in rows.columns]
    h = pd.util.hash_pandas_object(rows[cols], index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


def training_rows(rows: pd.DataFrame, fold: eval_mod.Fold, window_seasons: int, min_train: int) -> pd.DataFrame:
    """The fold's training rows restricted to the last `window_seasons` seasons, floored at
    `min_train` most recent events so a short window never drops below the fold minimum."""
    train = rows.loc[fold.train_idx]
    seasons = sorted(train[SEASON_COL].unique())
    if window_seasons <= 0 or window_seasons >= len(seasons):
        return train
    sub = train[train[SEASON_COL].isin(seasons[-window_seasons:])]
    if len(sub) < min_train:
        sub = train.sort_values(E.t0).tail(min_train)
    return sub


# ---- scoring ------------------------------------------------------------------------------------
def build_model(cfg: TrialConfig, seed: int) -> models_mod.BaseModel:
    if cfg.model == "zero" and "zero" not in models_mod.REGISTRY:
        return _ZeroModel(seed=seed)
    return models_mod.make_model(cfg.model, seed=seed, **cfg.model_params)


def oos_predictions(settings: Settings, rows: pd.DataFrame, groups: dict[str, list[str]],
                    cfg: TrialConfig, folds: list[eval_mod.Fold]) -> pd.DataFrame:
    """Out-of-sample predictions of `cfg` over `folds` (schemas.P columns + season)."""
    cols = [c for g in cfg.groups for c in groups.get(g, [])]
    out = []
    for fold in folds:
        train = training_rows(rows, fold, cfg.window_seasons, settings.min_train_events)
        y = train[cfg.target].astype(float)
        keep = y.notna()
        train, y = train[keep], y[keep]
        if train.empty:
            raise TrialFailed(f"fold {fold.fold}: no training rows with a {cfg.target} label")
        model = build_model(cfg, settings.random_seed)
        model.fit(train[cols], y, np.sign(y))
        test = rows.loc[fold.test_idx]
        p_up = np.asarray(model.predict_proba_up(test[cols]), dtype=float).reshape(-1)
        r_hat = np.asarray(model.predict_return(test[cols]), dtype=float).reshape(-1)
        if len(p_up) != len(test) or len(r_hat) != len(test):
            raise TrialFailed(f"fold {fold.fold}: model returned {len(p_up)}/{len(r_hat)} predictions for {len(test)} rows")
        out.append(pd.DataFrame({
            P.event_id: test[D.event_id].to_numpy(), P.fold: fold.fold, P.test_season: fold.test_season,
            P.p_up: p_up, P.r_hat: r_hat,
            P.r_true: test[LABEL_RETURN].astype(float).to_numpy(),
            P.direction_true: test[LABEL_DIRECTION].astype(float).to_numpy(),
        }))
    return pd.concat(out, ignore_index=True)


def score(preds: pd.DataFrame, objective: str) -> float:
    kind, _direction = OBJECTIVES[objective]
    if kind == "classification":
        m = eval_mod.classification_metrics(preds[P.p_up], preds[P.direction_true])
    else:
        m = eval_mod.regression_metrics(preds[P.r_hat], preds[P.r_true])
    return float(m[objective])


def improvement_of(value: float, baseline: float, objective: str) -> float:
    """Signed improvement of `value` over `baseline`: positive means better, whatever the
    objective's direction."""
    return (baseline - value) if OBJECTIVES[objective][1] == "minimize" else (value - baseline)


def p_noise_bootstrap(best: pd.DataFrame, base: pd.DataFrame, objective: str, *,
                      n: int = N_BOOTSTRAP, seed: int = 7) -> float:
    """Share of paired bootstrap resamples in which the best trial fails to beat the baseline.
    Blocks are test seasons when there are at least three of them, single events otherwise."""
    m = best.merge(base[[P.event_id, P.p_up, P.r_hat]], on=P.event_id, suffixes=("", "_base"))
    if len(m) != len(best) or len(m) != len(base):
        raise TestSetMismatch("best trial and baseline were scored on different event sets")
    seasons = m[P.test_season].to_numpy()
    blocks = [np.flatnonzero(seasons == s) for s in np.unique(seasons)]
    if len(blocks) < 3:
        blocks = [np.array([i]) for i in range(len(m))]
    rng = np.random.default_rng(seed)
    base_cols = pd.DataFrame({P.p_up: m[P.p_up + "_base"], P.r_hat: m[P.r_hat + "_base"],
                              P.r_true: m[P.r_true], P.direction_true: m[P.direction_true]})
    best_cols = m[[P.p_up, P.r_hat, P.r_true, P.direction_true]]
    worse = 0
    done = 0
    for _ in range(n):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        try:
            v_best = score(best_cols.iloc[idx], objective)
            v_base = score(base_cols.iloc[idx], objective)
        except ValueError:  # a degenerate resample (one class only) is skipped, not counted
            continue
        if not (math.isfinite(v_best) and math.isfinite(v_base)):
            continue
        done += 1
        if improvement_of(v_best, v_base, objective) <= 0:
            worse += 1
    return worse / done if done else float("nan")


# ---- study --------------------------------------------------------------------------------------
def study_name(decision_time: str, objective: str) -> str:
    return f"freedom_{decision_time}_{objective}"


def storage_url(settings: Settings) -> str:
    settings.optuna_db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{settings.optuna_db}"


def report_dir(settings: Settings, decision_time: str, objective: str) -> Path:
    return settings.reports_dir / "optimize" / study_name(decision_time, objective)


def baseline_scores(settings: Settings, rows: pd.DataFrame, groups: dict[str, list[str]],
                    folds: list[eval_mod.Fold], objective: str) -> dict[str, tuple[float, pd.DataFrame]]:
    """name -> (value, predictions) for every registered baseline that runs on these folds
    (plus the built-in zero model when the registry has no `zero`)."""
    all_groups = tuple(groups)
    out: dict[str, tuple[float, pd.DataFrame]] = {}
    names = [b for b in BASELINES if b in models_mod.REGISTRY]
    if "zero" not in names:
        names.insert(0, "zero")
    for name in names:
        cfg = TrialConfig(model=name, model_params={}, groups=all_groups, target=LABEL_RETURN,
                          window_seasons=0, label=name)
        try:
            preds = oos_predictions(settings, rows, groups, cfg, folds)
            value = score(preds, objective)
        except Exception as exc:  # a baseline that cannot run here is reported, not fatal
            log.warning("baseline %s skipped: %s", name, exc)
            continue
        if math.isfinite(value):
            out[name] = (value, preds)
    return out


def best_baseline(scores: dict[str, tuple[float, pd.DataFrame]], objective: str) -> tuple[str, float, pd.DataFrame] | None:
    if not scores:
        return None
    minimize = OBJECTIVES[objective][1] == "minimize"
    name = min(scores, key=lambda k: scores[k][0]) if minimize else max(scores, key=lambda k: scores[k][0])
    return name, scores[name][0], scores[name][1]


def run_study(settings: Settings, dataset: pd.DataFrame, *, decision_time: str, n_trials: int,
              objective: str = "brier", timeout_seconds: int | None = None) -> dict:
    """Search space: model family (linear, lightgbm, ensemble) and hyper-parameters, feature
    groups on/off (only groups admissible at the decision time), target variant (r_24h vs
    ar_24h), training-window length in seasons. Persists to settings.optuna_db under study name
    f"freedom_{decision_time}_{objective}". Writes reports/optimize/<study>/leaderboard.md and
    best_params.json; returns {study, best_value, best_params, n_trials, baseline_value,
    improvement, p_noise} where p_noise is the bootstrap probability that the improvement over
    the best baseline is noise."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}")
    direction = OBJECTIVES[objective][1]
    rows = prepare_rows(settings, dataset, decision_time)
    groups = feature_groups(rows.columns, decision_time)
    groups = {g: c for g, c in groups.items() if rows[c].notna().any().any()}
    if not groups:
        raise ValueError(f"no feature columns ({D.feature_prefix}*) admissible at {decision_time} in the dataset")
    targets = available_targets(rows)
    folds = make_folds(settings, rows)
    n_seasons = int(rows[SEASON_COL].nunique())
    ref_hash = test_set_hash(rows, folds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    name = study_name(decision_time, objective)
    study = optuna.create_study(study_name=name, storage=storage_url(settings), direction=direction,
                                sampler=optuna.samplers.TPESampler(seed=settings.random_seed),
                                load_if_exists=True)
    if study.trials:  # resumed: derive the seed so the first proposals are not repeated
        study.sampler = optuna.samplers.TPESampler(seed=settings.random_seed + len(study.trials))
    stored = study.user_attrs.get(TEST_SET_ATTR)
    if stored is None:
        study.set_user_attr(TEST_SET_ATTR, ref_hash)
        study.set_user_attr("decision_time", decision_time)
        study.set_user_attr("objective", objective)
        study.set_user_attr("holdout_season", settings.holdout_season)
    elif stored != ref_hash:
        raise TestSetMismatch(
            f"study {name} in {settings.optuna_db} was built on a different test set "
            f"(stored {stored[:12]}, now {ref_hash[:12]}): the dataset or the fold settings changed. "
            f"Delete the study (optuna.delete_study) or use another objective name before continuing.")

    def objective_fn(trial: optuna.Trial) -> float:
        trial_folds = make_folds(settings, rows)
        h = test_set_hash(rows, trial_folds)
        trial.set_user_attr(TEST_SET_ATTR, h)
        if h != study.user_attrs[TEST_SET_ATTR]:
            raise TestSetMismatch(f"trial {trial.number} would be scored on a different test set "
                                  f"({h[:12]} != {study.user_attrs[TEST_SET_ATTR][:12]}); study aborted")
        params = suggest(trial, list(groups), targets, n_seasons)
        cfg = TrialConfig.from_params(params, list(groups))
        trial.set_user_attr("groups", list(cfg.groups))
        if not cfg.groups:
            raise optuna.TrialPruned("no feature group selected")
        try:
            preds = oos_predictions(settings, rows, groups, cfg, trial_folds)
            value = score(preds, objective)
        except TestSetMismatch:
            raise
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise TrialFailed(str(exc)) from exc
        if not math.isfinite(value):
            trial.set_user_attr("error", f"non-finite {objective}")
            raise TrialFailed(f"non-finite {objective}")
        trial.set_user_attr("n_test", int(len(preds)))
        trial.set_user_attr("n_folds", len(trial_folds))
        return value

    study.optimize(objective_fn, n_trials=n_trials, timeout=timeout_seconds, catch=(TrialFailed,))

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    failed = sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)
    scores = baseline_scores(settings, rows, groups, folds, objective)
    base = best_baseline(scores, objective)
    result: dict = {
        "study": name, "decision_time": decision_time, "objective": objective, "direction": direction,
        "n_trials": len(complete), "n_pruned": pruned, "n_failed": failed,
        "n_events": int(len(rows)), "n_folds": len(folds), "n_seasons": n_seasons,
        "seasons": sorted(rows[SEASON_COL].unique().tolist()),
        "holdout_season": settings.holdout_season, "test_set_hash": ref_hash,
        "dataset_hash": dataset_hash(rows), "groups": list(groups), "targets": list(targets),
        "has_perp_share": float(rows[E.has_perp_at_t0].astype(bool).mean()) if E.has_perp_at_t0 in rows.columns else float("nan"),
        "best_value": None, "best_params": None, "best_trial": None,
        "baseline_name": base[0] if base else None, "baseline_value": base[1] if base else None,
        "baselines": {k: v[0] for k, v in scores.items()},
        "improvement": None, "p_noise": None,
        "report_dir": str(report_dir(settings, decision_time, objective)),
    }
    if complete:
        best = study.best_trial
        result.update(best_value=float(best.value), best_params=dict(best.params), best_trial=best.number)
        if base is not None:
            result["improvement"] = improvement_of(float(best.value), base[1], objective)
            cfg = TrialConfig.from_params(best.params, list(groups))
            best_preds = oos_predictions(settings, rows, groups, cfg, folds)
            result["p_noise"] = p_noise_bootstrap(best_preds, base[2], objective, seed=settings.random_seed)
    write_reports(settings, study, result)
    return result


# ---- reports --------------------------------------------------------------------------------------
def trials_frame(study: optuna.Study) -> pd.DataFrame:
    """One row per trial: rank (complete trials only), number, state, value, model, target,
    window, groups, params (json), n_test, error."""
    minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
    rows = []
    for t in study.trials:
        p = t.params
        rows.append({
            "trial": t.number, "state": t.state.name.lower(),
            "value": float(t.value) if t.value is not None else np.nan,
            "model": p.get("model"), "target": p.get("target"),
            "train_window_seasons": p.get("train_window_seasons"),
            "groups": "+".join(t.user_attrs.get("groups", [])),
            "params": json.dumps({k: v for k, v in p.items() if k in ("model", "target", "train_window_seasons") or "." in k}, sort_keys=True, default=str),
            "n_test": t.user_attrs.get("n_test"), "error": t.user_attrs.get("error"),
        })
    df = pd.DataFrame(rows, columns=["trial", "state", "value", "model", "target", "train_window_seasons",
                                     "groups", "params", "n_test", "error"])
    if df.empty:
        df.insert(0, "rank", pd.Series(dtype="Int64"))
        return df
    order = df["value"].rank(method="first", ascending=minimize)
    df.insert(0, "rank", order.where(df["state"] == "complete").astype("Int64"))
    return df.sort_values(["rank", "trial"], na_position="last").reset_index(drop=True)


def _fmt(v: object, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def leaderboard_markdown(study: optuna.Study, result: dict) -> str:
    d, obj = result["decision_time"], result["objective"]
    lines = [f"# freedom optimize — {d}, objective {obj} ({result['direction']})", ""]
    lines.append(f"- study `{result['study']}`: {result['n_trials']} complete trials, "
                 f"{result['n_pruned']} pruned, {result['n_failed']} failed")
    seasons = result["seasons"]
    lines.append(f"- scored on {result['n_events']} events over {result['n_folds']} walk-forward folds "
                 f"(seasons {seasons[0]} … {seasons[-1]}); holdout {result['holdout_season']} excluded "
                 f"and never scored; test-set hash `{result['test_set_hash'][:12]}`, "
                 f"dataset hash `{result['dataset_hash'][:12]}`")
    lines.append(f"- subset: t0_confidence ≥ floor, r_24h present, all price sources; "
                 f"has_perp_at_t0 share {_fmt(result['has_perp_share'], 2)}")
    lines.append(f"- admissible feature groups at {d}: {', '.join(result['groups'])}; "
                 f"target variants: {', '.join(result['targets'])}")
    if result["best_value"] is not None:
        bp = result["best_params"]
        used = [g for g in result["groups"] if bp.get(f"use_{g}")]
        lines.append(f"- best trial #{result['best_trial']}: {obj} {_fmt(result['best_value'])} "
                     f"({bp.get('model')}, target {bp.get('target')}, window {bp.get('train_window_seasons')} seasons, "
                     f"groups {'+'.join(used) or 'none'})")
    else:
        lines.append("- no trial completed; nothing to rank")
    if result["baseline_name"] is not None:
        lines.append(f"- best baseline: {result['baseline_name']} {_fmt(result['baseline_value'])} → improvement "
                     f"{_fmt(result['improvement'])} with {result['n_trials']} trials; "
                     f"p_noise = {_fmt(result['p_noise'], 3)} ({N_BOOTSTRAP} paired season-block bootstraps)")
        lines.append("- read p_noise as: the chance the best trial's edge over that baseline on these "
                     "events is a resampling accident. It does not correct for choosing the best of "
                     f"{result['n_trials']} trials; only `freedom evaluate --final` scores the holdout.")
        lines.append("")
        lines.append("| baseline | " + obj + " |")
        lines.append("|---|---|")
        for k, v in sorted(result["baselines"].items(), key=lambda kv: kv[1],
                           reverse=result["direction"] == "maximize"):
            lines.append(f"| {k} | {_fmt(v)} |")
    else:
        lines.append("- no baseline could be scored (models registry empty?); improvement not available")
    lines.append("")
    lines.append(f"| rank | trial | {obj} | model | target | window | groups | params | state |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in trials_frame(study).itertuples(index=False):
        rank = "" if pd.isna(r.rank) else str(int(r.rank))
        state = r.state if r.error is None else f"{r.state}: {r.error}"
        lines.append(f"| {rank} | {r.trial} | {_fmt(r.value)} | {r.model or ''} | {r.target or ''} | "
                     f"{'' if r.train_window_seasons is None else r.train_window_seasons} | {r.groups} | "
                     f"`{r.params}` | {state} |")
    lines.append("")
    return "\n".join(lines)


def write_reports(settings: Settings, study: optuna.Study, result: dict) -> Path:
    out = report_dir(settings, result["decision_time"], result["objective"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.md").write_text(leaderboard_markdown(study, result), encoding="utf-8")
    payload = {k: v for k, v in result.items() if k != "report_dir"}
    payload["written_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
    payload["optuna_db"] = str(settings.optuna_db)
    (out / "best_params.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                                          encoding="utf-8")
    return out


def load_study(settings: Settings, decision_time: str, objective: str = "brier") -> optuna.Study:
    name = study_name(decision_time, objective)
    hint = f"run `freedom optimize --decision-times {decision_time} --objective {objective}` first"
    if not settings.optuna_db.exists():
        raise FileNotFoundError(f"{settings.optuna_db} not found; {hint}")
    try:
        return optuna.load_study(study_name=name, storage=storage_url(settings))
    except KeyError as exc:
        raise FileNotFoundError(f"no study {name} in {settings.optuna_db}; {hint}") from exc


def leaderboard(settings: Settings, decision_time: str, objective: str = "brier") -> pd.DataFrame:
    """Trials of the persisted study ranked by value (see trials_frame)."""
    return trials_frame(load_study(settings, decision_time, objective))
