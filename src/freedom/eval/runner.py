"""Orchestration of `freedom evaluate` and `freedom train` (docs/design.md §8, §10).

`evaluate` runs the walk-forward for every (model, decision_time) on folds that exclude the
pinned holdout season, scores every cell (subset x metric) with bootstrap intervals, the
minimum detectable improvement at that n and a paired comparison against the best baseline,
runs the cost-aware simulation for every sizing rule, and writes reports/<run_id>/.
`train_final` fits one model on all non-holdout events and saves it with its provenance.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import features as features_mod
from .. import models as models_mod
from ..config import Settings
from ..features.groups import NON_POINT_IN_TIME, NON_POINT_IN_TIME_KEYS
from ..models import MIN_TRAIN_ROWS
from ..schemas import DECISION_TIMES, SCHEMA_VERSION, D, E, P, T, T0Source
from .folds import (
    HOLDOUT_FOLD,
    Fold,
    season_end,
    season_start,
    seasons_of,
    t0_utc,
    walk_forward_folds,
)
from .metrics import (
    EPS,
    MDE_METRICS,
    bootstrap_ci,
    bootstrap_distribution,
    brier_scores,
    calibration_table,
    choose_blocks,
    classification_metrics,
    hit_scores,
    min_detectable_improvement,
    paired_mde,
    paired_se,
    regression_metrics,
    residual_band,
    spearman,
)
from .report import (
    append_holdout_log,
    config_hash,
    count_holdout_scorings,
    dataset_sha256,
    git_info,
    last_holdout_scoring,
    library_versions,
    make_run_id,
    public_settings,
    to_jsonable,
    utcnow,
    write_reports,
)
from .sim import (
    CAPITAL_RULE,
    FUNDING_ARCHIVE,
    SIZINGS,
    archive_funding_loader,
    as_utc,
    loader_paths,
    memoised_bar_index,
    memoised_funding,
    portfolio_metrics,
    simulate_rows,
)

log = logging.getLogger(__name__)

DEFAULT_BASELINES = frozenset({"zero", "base_rate", "historical_mean", "hist_abs_mean", "vol_scaled",
                               "sign_of_reaction", "always_extends", "surprise_sign"})
COMPARED_METRICS = ("accuracy", "brier", "log_loss", "spearman_ic", "mae", "magnitude_mae")
HIGHER_IS_BETTER = {"accuracy": True, "balanced_accuracy": True, "brier": False, "log_loss": False,
                    "spearman_ic": True, "mae": False, "rmse": False, "magnitude_mae": False,
                    "magnitude_ic": True}
SCORE_COLUMNS = {"accuracy": "hit", "brier": "brier", "log_loss": "ll", "mae": "ae", "magnitude_mae": "mag_ae"}
MDE_PAIRED = "paired_se"  # MDE from the paired comparison's own standard error
MDE_UPPER_BOUND = "closed_form_upper_bound"  # no comparison: the unpaired closed form (conservative)
VERDICT_IDENTICAL = "identical_to_baseline"
VERDICT_UNTRAINED = "untrained"
FALLBACK = "fallback_direction"  # prediction column: the model's direction head used the base rate  # zero paired SE: the model reproduced the baseline exactly
MAX_NONFINITE_P_SHARE = 0.1  # a model returning more non-finite p_up than this is rejected
HEADLINE_SOURCES = frozenset({T0Source.sec_8k.value, T0Source.manual.value, T0Source.detected.value})
STRATA = (E.t0_source, E.kind, E.timing)
CALIBRATION_SUBSETS = ("all", "headline")
# the trading simulation runs once over every row; its statistics (and the paired PnL comparison
# against the best baseline) are computed per subset so the leaderboard's trading columns describe
# the same rows as the metric cell beside them ("all" = every simulated row, as trades.parquet)
TRADING_SUBSETS = ("all", "headline")
META_COLUMNS = [E.t0, E.t0_source, E.t0_confidence, E.kind, E.timing, E.has_perp_at_t0, E.market,
                E.estimate_source, "season", "price_source", "target_missing"]
ESTIMATE_SOURCE_MISSING = "missing"  # estimate_source label of an event with no consensus provenance
ESTIMATE_SOURCE_UNAVAILABLE = "unavailable"  # the dataset (and events calendar) carry no such column
Y, DIR, TRAINABLE = "_y", "_dir", "_trainable"


class HoldoutNotReady(RuntimeError):
    """evaluate(final=True) refuses: the holdout season cannot be scored honestly yet."""


def baseline_names() -> frozenset[str]:
    """Models counted as baselines (the models package may publish its own set)."""
    return frozenset(getattr(models_mod, "BASELINES", DEFAULT_BASELINES))


# ---- dataset preparation -----------------------------------------------------------------------
def _truthy(v: Any) -> bool:
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return bool(v)


def prepare_dataset(dataset: pd.DataFrame, target: str, settings: Settings) -> pd.DataFrame:
    """Copy of the features.build_dataset frame with UTC t0, a recomputed season, coerced
    metadata and the private columns _y (target), _dir (direction label) and _trainable
    (t0_confidence >= min_t0_confidence, target present, not target_missing)."""
    if dataset is None or len(dataset) == 0:
        raise ValueError("dataset is empty")
    for col in (D.event_id, D.decision_time, E.t0, target):
        if col not in dataset.columns:
            raise ValueError(f"dataset lacks the {col!r} column")
    df = dataset.copy()
    df[E.t0] = t0_utc(df)
    df["season"] = seasons_of(df[E.t0])
    for col, default in ((E.t0_source, "unknown"), (E.kind, "unknown"), (E.timing, "unknown"),
                         (E.market, None), (T.price_source, None), (E.estimate_source, None)):
        if col not in df.columns:
            df[col] = default
    if E.t0_confidence not in df.columns:
        df[E.t0_confidence] = np.nan
    df[E.t0_confidence] = pd.to_numeric(df[E.t0_confidence], errors="coerce")
    df[E.has_perp_at_t0] = df[E.has_perp_at_t0].map(_truthy) if E.has_perp_at_t0 in df.columns else False
    df[E.has_perp_at_t0] = df[E.has_perp_at_t0].astype(bool)
    df[Y] = pd.to_numeric(df[target], errors="coerce")
    if "target_missing" in df.columns:
        df["target_missing"] = df["target_missing"].map(_truthy).astype(bool)
    else:
        df["target_missing"] = df[Y].isna()
    if E.pending in df.columns:
        df[E.pending] = df[E.pending].map(_truthy).astype(bool)
    sign = np.sign(df[Y])
    if target == T.r("24h") and T.direction in df.columns:
        given = pd.to_numeric(df[T.direction], errors="coerce")
        df[DIR] = given.where(given.notna(), sign)
    else:
        df[DIR] = sign
    df[TRAINABLE] = ((df[E.t0_confidence] >= settings.min_t0_confidence) & ~df["target_missing"]
                     & df[Y].notna()).astype(bool)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith(D.feature_prefix)]


def non_point_in_time_in_scope(feats: list[str], decision_time: str) -> dict[str, str]:
    """{group: reason} for the features.groups.NON_POINT_IN_TIME groups that are admissible at
    `decision_time` and have at least one of their non-point-in-time keys among the feature
    columns `feats` (design §5, §6). The learners here are fitted on every feature column, so
    a group in scope was consumed by every trained model of that decision time."""
    if not feats:
        return {}
    admissible = set(features_mod.admissible_groups(decision_time))
    present = set(map(str, feats))
    out: dict[str, str] = {}
    for group, reason in NON_POINT_IN_TIME.items():
        if group not in admissible:
            continue
        keys = NON_POINT_IN_TIME_KEYS.get(group, ())
        if any(f"{D.feature_prefix}{k}" in present for k in keys):
            out[group] = reason
    return out


def estimate_source_counts(rows: pd.DataFrame) -> dict[str, int]:
    """{estimate_source: n} over `rows` (design §5: fmp_final / nasdaq_final are the vendor's
    final consensus, consensus_snapshot is point-in-time); events without a value count as
    'missing', and a frame without the column reports every row as 'unavailable'."""
    if E.estimate_source not in rows.columns:
        return {ESTIMATE_SOURCE_UNAVAILABLE: int(len(rows))} if len(rows) else {}

    def label(v: Any) -> str:
        try:
            if v is None or pd.isna(v):  # None, NaN, NaT and pd.NA alike
                return ESTIMATE_SOURCE_MISSING
        except (TypeError, ValueError):
            pass
        return str(v)

    labels = rows[E.estimate_source].map(label)
    return {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}


def attach_estimate_source(df: pd.DataFrame, events: pd.DataFrame | None) -> tuple[pd.DataFrame, bool]:
    """(df, filled): `df[estimate_source]` filled from the events calendar (event_id ->
    estimate_source) when the dataset was built before the column joined features.META_COLUMNS
    (column absent or every value None); a dataset that carries values is left alone."""
    if events is None or E.estimate_source not in events.columns or E.event_id not in events.columns:
        return df, False
    if E.estimate_source in df.columns and df[E.estimate_source].notna().any():
        return df, False
    cal = events[[E.event_id, E.estimate_source]].dropna(subset=[E.event_id]).copy()
    cal[E.event_id] = cal[E.event_id].astype(str)
    lookup = cal.drop_duplicates(E.event_id, keep="last").set_index(E.event_id)[E.estimate_source]
    df = df.copy()
    df[E.estimate_source] = df[D.event_id].astype(str).map(lookup)
    return df, True


def _X(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    if not feats:
        return pd.DataFrame(index=df.index)
    return df[feats].apply(pd.to_numeric, errors="coerce").astype(float)


def _direction_target(train: pd.DataFrame) -> pd.Series:
    """The direction label as the model sees it: sign(direction_24h) in {-1, 0, +1}, passed
    through unchanged (a zero move is 'no direction', never 'up'); NaN -> 0."""
    return pd.Series(np.sign(train[DIR].to_numpy(dtype=float)), index=train.index).fillna(0.0)


def _validate_predictions(name: str, p: np.ndarray, r: np.ndarray, mag: np.ndarray,
                          n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check the model output: p_up must be a probability (values outside [0, 1] are clipped
    with a warning; more than MAX_NONFINITE_P_SHARE non-finite values reject the model),
    r_hat is warned about when non-finite, magnitude falls back to |r_hat| where it is missing
    and is made non-negative."""
    if len(p) != n or len(r) != n or len(mag) != n:
        raise ValueError(f"model {name!r} returned {len(p)}/{len(r)}/{len(mag)} predictions for {n} rows")
    if n == 0:
        return p, r, mag
    bad_p = ~np.isfinite(p)
    if bad_p.mean() > MAX_NONFINITE_P_SHARE:
        raise ValueError(f"model {name!r} returned {int(bad_p.sum())} non-finite p_up out of {n} rows "
                         f"(more than {MAX_NONFINITE_P_SHARE:.0%}); models must return a probability per row")
    out_of_range = ~bad_p & ((p < 0.0) | (p > 1.0))
    if out_of_range.any():
        log.warning("model %r returned %d p_up outside [0, 1] (min %.4f, max %.4f); clipping",
                    name, int(out_of_range.sum()), float(np.nanmin(p)), float(np.nanmax(p)))
        p = np.where(bad_p, p, np.clip(p, 0.0, 1.0))
    bad_r = ~np.isfinite(r)
    if bad_r.any():
        log.warning("model %r returned %d non-finite r_hat out of %d rows", name, int(bad_r.sum()), n)
    bad_m = ~np.isfinite(mag)
    if bad_m.any():
        mag = np.where(bad_m, np.abs(r), mag)
    if np.any(mag[np.isfinite(mag)] < 0):
        log.warning("model %r returned negative magnitude forecasts; using their absolute value", name)
        mag = np.abs(mag)
    return p, r, mag


def _fit_predict(name: str, seed: int, train: pd.DataFrame, test: pd.DataFrame,
                 feats: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit `name` on `train` and predict `test`: (p_up, r_hat, magnitude_hat, model). The
    magnitude comes from model.predict_magnitude (BaseModel default: |r_hat|)."""
    model = models_mod.make_model(name, seed=seed)
    model.fit(_X(train, feats), train[Y].astype(float), _direction_target(train))
    X_te = _X(test, feats)
    p = np.asarray(model.predict_proba_up(X_te), dtype=float).reshape(-1)
    r = np.asarray(model.predict_return(X_te), dtype=float).reshape(-1)
    mag = np.asarray(model.predict_magnitude(X_te), dtype=float).reshape(-1)
    p, r, mag = _validate_predictions(name, p, r, mag, len(test))
    return p, r, mag, model


# ---- walk-forward ------------------------------------------------------------------------------
def _fold_plan(sub: pd.DataFrame, settings: Settings) -> tuple[list[Fold], Fold | None, list[dict], list[dict]]:
    # ask the builder for every season and apply the trainable-row minimum here, so seasons
    # that cannot be tested are listed as skipped instead of vanishing
    folds, holdout = walk_forward_folds(sub, min_train=0,
                                        embargo_days=settings.embargo_days,
                                        holdout_season=settings.holdout_season)
    usable, info, skipped = [], [], []
    for f in folds:
        n_train = int(sub.loc[f.train_idx, TRAINABLE].sum())
        if n_train < settings.min_train_events:
            skipped.append({"test_season": f.test_season, "n_train_trainable": n_train,
                            "reason": f"fewer than {settings.min_train_events} trainable events"})
            continue
        usable.append(Fold(len(usable), f.train_idx, f.test_idx, f.test_season))
        info.append({"fold": len(usable) - 1, "test_season": f.test_season, "n_train": n_train,
                     "n_test": int(len(f.test_idx))})
    return usable, holdout, info, skipped


def _walk_forward(sub: pd.DataFrame, feats: list[str], folds: list[Fold], name: str, settings: Settings,
                  *, holdout: Fold | None) -> pd.DataFrame:
    """Out-of-sample predictions (schemas.P columns + event metadata) for every test fold and,
    when `holdout` is given, for the holdout fold (fold = HOLDOUT_FOLD). r_lo/r_hi for a fold
    come from the residuals of the walk-forward folds whose test season *precedes* it (NaN for
    the first); the holdout likewise uses only the folds whose test season starts before the
    holdout season, never a season after it (the dataset holds post-holdout seasons between a
    season closing and the human edit that advances holdout_season)."""
    frames: list[pd.DataFrame] = []
    residuals: dict[str, np.ndarray] = {}  # test season -> finite out-of-sample residuals of that fold
    plan = list(folds) + ([holdout] if holdout is not None else [])
    for fold in plan:
        train = sub.loc[fold.train_idx]
        train = train[train[TRAINABLE]]
        test = sub.loc[fold.test_idx]
        p, r, mag, model = _fit_predict(name, settings.random_seed, train, test, feats)
        fell_back = "direction" in getattr(model, "fallback_heads_", set())
        cutoff = season_start(fold.test_season)
        use = [v for season, v in residuals.items() if season_start(season) < cutoff]
        pooled = np.concatenate(use) if use else np.array([], dtype=float)
        q10, q90 = residual_band(np.zeros(len(pooled)), pooled) if len(pooled) else (math.nan, math.nan)
        frame = test[[D.event_id, D.decision_time, *META_COLUMNS]].copy()
        frame[P.model] = name
        frame[P.fold] = int(fold.fold)
        frame[P.test_season] = fold.test_season
        frame[P.p_up] = p
        frame[P.r_hat] = r
        frame[P.magnitude_hat] = mag
        frame[P.r_lo] = r + q10
        frame[P.r_hi] = r + q90
        frame[P.r_true] = test[Y].to_numpy(dtype=float)
        frame[P.direction_true] = test[DIR].to_numpy(dtype=float)
        frame[FALLBACK] = bool(fell_back)
        frames.append(frame)
        if fold.fold != HOLDOUT_FOLD:
            res = frame[P.r_true].to_numpy(dtype=float) - r
            residuals[fold.test_season] = res[np.isfinite(res)]
    cols = [P.event_id, P.decision_time, P.model, P.fold, P.test_season, P.p_up, P.r_hat, P.magnitude_hat,
            P.r_lo, P.r_hi, P.r_true, P.direction_true, FALLBACK, *META_COLUMNS]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    return out[cols]


# ---- scoring ---------------------------------------------------------------------------------------
def _scores(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-event scores indexed by event_id: hit, brier, ll, ae, mag_ae (|magnitude_hat| vs
    |r_true|) plus the raw columns and the bootstrap block labels season / day (UTC day of t0)."""
    p = preds[P.p_up].to_numpy(dtype=float)
    y = preds[P.direction_true].to_numpy(dtype=float)
    yb = (y > 0).astype(float)
    pc = np.clip(p, EPS, 1 - EPS)
    ll = -(yb * np.log(pc) + (1 - yb) * np.log(1 - pc))
    ll[~(np.isfinite(p) & np.isfinite(y) & (y != 0))] = np.nan
    r_hat = preds[P.r_hat].to_numpy(dtype=float)
    r_true = preds[P.r_true].to_numpy(dtype=float)
    if P.magnitude_hat in preds.columns:
        mag = preds[P.magnitude_hat].to_numpy(dtype=float)
    else:
        mag = np.abs(r_hat)
    out = pd.DataFrame({"hit": hit_scores(p, y), "brier": brier_scores(p, y), "ll": ll,
                        "ae": np.abs(r_true - r_hat), "mag_ae": np.abs(np.abs(r_true) - mag),
                        P.p_up: p, P.r_hat: r_hat, P.magnitude_hat: mag, P.r_true: r_true,
                        P.direction_true: y, "season": preds[P.test_season].to_numpy(),
                        "day": _day_labels(preds)},
                       index=pd.Index(preds[P.event_id].astype(str), name=P.event_id))
    return out[~out.index.duplicated(keep="first")]


def _day_labels(frame: pd.DataFrame) -> np.ndarray:
    """UTC day of t0 as strings (bootstrap block labels); 'NaT' where t0 is missing."""
    if E.t0 not in frame.columns or len(frame) == 0:
        return np.full(len(frame), "NaT", dtype=object)
    return as_utc(frame[E.t0]).dt.strftime("%Y-%m-%d").fillna("NaT").to_numpy(dtype=object)


def _blocks(frame: pd.DataFrame) -> tuple[str, pd.Series | None]:
    """Bootstrap resampling scheme for a scores slice: season blocks when at least MIN_BLOCKS
    seasons are present, else UTC-day-of-t0 blocks, else iid rows (metrics.choose_blocks)."""
    return choose_blocks([("block:season", frame["season"]), ("block:day", frame["day"])])


def subset_masks(preds: pd.DataFrame, *, min_t0_confidence: float) -> dict[str, pd.Series]:
    """Named boolean masks over a predictions frame: all (scorable rows), has_perp_at_t0,
    headline (perp, confident, non-calendar t0 source) and one per t0_source / kind / timing."""
    scorable = preds[P.r_true].notna()
    perp = preds[E.has_perp_at_t0].astype(bool)
    conf_ok = pd.to_numeric(preds[E.t0_confidence], errors="coerce") >= min_t0_confidence
    src = preds[E.t0_source].astype(str)
    masks = {"all": scorable, "has_perp_at_t0": scorable & perp,
             "headline": scorable & perp & conf_ok & src.isin(HEADLINE_SOURCES)}
    for col in STRATA:
        if col not in preds.columns:
            continue
        values = preds[col].astype(str).where(preds[col].notna(), None)
        for v in sorted(x for x in values.dropna().unique()):
            masks[f"{col}={v}"] = scorable & (values == v)
    return masks


def _cell(preds: pd.DataFrame, scores: pd.DataFrame, *, n_boot: int, seed: int, with_calibration: bool) -> dict[str, Any]:
    cm = classification_metrics(preds[P.p_up], preds[P.direction_true])
    rm = regression_metrics(preds[P.r_hat], preds[P.r_true])
    mag_col = preds[P.magnitude_hat] if P.magnitude_hat in preds.columns else preds[P.r_hat].abs()
    mm = regression_metrics(mag_col, preds[P.r_true].abs())
    cell: dict[str, Any] = {"n": int(len(preds)), "n_direction": cm["n"], "n_return": rm["n"],
                            "n_magnitude": mm["n"]}
    cell.update({k: v for k, v in cm.items() if k != "n"})
    cell.update({k: v for k, v in rm.items() if k != "n"})
    cell["magnitude_mae"], cell["magnitude_ic"] = mm["mae"], mm["spearman_ic"]
    scheme, block = _blocks(scores)
    cell["resampling"] = scheme
    ci: dict[str, list[float]] = {}
    for metric, col in (("accuracy", "hit"), ("brier", "brier")):
        s = scores[col].dropna()
        if len(s):
            _, lo, hi = bootstrap_ci(s, lambda v: float(v.mean()), n=n_boot,
                                     block=block.loc[s.index] if block is not None else None, seed=seed)
            ci[metric] = [lo, hi]
    cell["ci"] = ci
    cell["mde"] = {}
    cell["mde_source"] = {}
    cell["comparison"] = None
    cell["calibration"] = calibration_table(preds[P.p_up], preds[P.direction_true]) if with_calibration else None
    return cell


def _best_baselines(cells: dict[str, dict[str, dict]], baselines: frozenset[str]) -> dict[str, dict[str, dict]]:
    """{subset: {metric: {'model', 'value'}}} over the baseline models present."""
    out: dict[str, dict[str, dict]] = {}
    subsets = sorted({s for name in cells for s in cells[name]})
    for subset in subsets:
        out[subset] = {}
        for metric in COMPARED_METRICS:
            best = None
            for name, per_subset in cells.items():
                if name not in baselines or subset not in per_subset:
                    continue
                v = per_subset[subset].get(metric)
                if v is None or not np.isfinite(v) or per_subset[subset]["n"] == 0:
                    continue
                if best is None or (v > best["value"] if HIGHER_IS_BETTER[metric] else v < best["value"]):
                    best = {"model": name, "value": float(v)}
            if best is not None:
                out[subset][metric] = best
    return out


def verdict(metric: str, lo: float | None, hi: float | None, mde: float | None, n: int) -> str:
    """'improves' when the paired interval excludes 0 from above; 'worse' when it lies below 0;
    'not_predictable' (Brier/accuracy only) when it excludes the MDE; else 'inconclusive at n'."""
    if lo is not None and np.isfinite(lo) and lo > 0:
        return "improves"
    if hi is not None and np.isfinite(hi) and hi < 0:
        return "worse"
    if metric in MDE_METRICS and hi is not None and mde is not None and np.isfinite(hi) and np.isfinite(mde) and hi < mde:
        return "not_predictable"
    return f"inconclusive at n = {n}"


def _compare(sm: pd.DataFrame, sb: pd.DataFrame, ids: pd.Index, metric: str, *, n_boot: int,
             seed: int) -> dict[str, Any] | None:
    """Paired bootstrap of the improvement of model scores `sm` over baseline scores `sb` on the
    events `ids` (blocks chosen by `_blocks`, recorded as 'resampling'). Improvement is signed
    so that positive is better. `se` is the empirical standard error of the mean paired
    difference (metrics.paired_se; the bootstrap SD of the improvement for spearman_ic), the
    input of the paired MDE."""
    common = ids.intersection(sm.index).intersection(sb.index)
    if len(common) == 0:
        return None
    a, b = sm.loc[common], sb.loc[common]
    se = math.nan
    if metric == "spearman_ic":
        keep = np.isfinite(a[P.r_true].to_numpy(dtype=float))
        a, b = a[keep], b[keep]
        if len(a) < 3:
            return None
        pos = pd.Series(np.arange(len(a)), index=a.index)
        rm, rb, y = a[P.r_hat].to_numpy(dtype=float), b[P.r_hat].to_numpy(dtype=float), a[P.r_true].to_numpy(dtype=float)

        def stat(v: pd.Series) -> float:
            i = v.to_numpy(dtype=int)
            return spearman(rm[i], y[i]) - spearman(rb[i], y[i])

        values = pos
    else:
        col = SCORE_COLUMNS[metric]
        sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
        diff = sign * (a[col].astype(float) - b[col].astype(float))
        keep = diff.notna()
        diff, a = diff[keep], a[keep]
        if len(diff) == 0:
            return None
        values = diff
        se = paired_se(diff)

        def stat(v: pd.Series) -> float:
            return float(v.mean())

    scheme, block = _blocks(a)
    point = float(stat(values))
    dist = bootstrap_distribution(values, stat, n=n_boot, block=block, seed=seed)
    finite = dist[np.isfinite(dist)]
    if len(finite):
        lo, hi = (float(x) for x in np.percentile(finite, [2.5, 97.5]))
        p_noise = float(np.mean(finite <= 0))
        se_boot = float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan
    else:
        lo = hi = p_noise = se_boot = math.nan
    if metric == "spearman_ic":
        se = se_boot
    return {"improvement": point, "ci": [lo, hi], "p_noise": p_noise, "n": int(len(values)), "se": se,
            "se_bootstrap": se_boot, "resampling": scheme}


def _score_block(blocks: dict[str, pd.DataFrame], *, settings: Settings, baselines: frozenset[str],
                 n_boot: int, bar_index: Callable, funding_fn: Callable | None) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Score one block of predictions per model (walk-forward or holdout): cells per subset,
    best baseline per (subset, metric), comparisons, MDE, residual bands and trading."""
    seed = settings.random_seed
    scores = {name: _scores(preds) for name, preds in blocks.items()}
    masks = {name: subset_masks(preds, min_t0_confidence=settings.min_t0_confidence) for name, preds in blocks.items()}
    cells: dict[str, dict[str, dict]] = {}
    for name, preds in blocks.items():
        cells[name] = {}
        for subset, mask in masks[name].items():
            part = preds[mask]
            cells[name][subset] = _cell(part, scores[name].loc[scores[name].index.intersection(part[P.event_id].astype(str))],
                                        n_boot=n_boot, seed=seed, with_calibration=subset in CALIBRATION_SUBSETS)
    best = _best_baselines(cells, baselines)
    for name, preds in blocks.items():
        for subset, cell in cells[name].items():
            ids = pd.Index(preds.loc[masks[name][subset], P.event_id].astype(str))
            comparison: dict[str, Any] = {}
            if name not in baselines:
                for metric in COMPARED_METRICS:
                    base = best.get(subset, {}).get(metric)
                    if base is None:
                        continue
                    cmp = _compare(scores[name], scores[base["model"]], ids, metric, n_boot=n_boot, seed=seed)
                    if cmp is None:
                        continue
                    cmp.update({"baseline": base["model"], "baseline_value": base["value"], "model_value": cell.get(metric)})
                    # the MDE of the test actually run: from the paired comparison's own standard error
                    mde = paired_mde(cmp["se"]) if metric in MDE_METRICS else None
                    mde_source = MDE_PAIRED if metric in MDE_METRICS else None
                    identical = metric in MDE_METRICS and (mde is None or not np.isfinite(mde) or mde <= 0)
                    if identical:
                        # identical predictions (e.g. a model that fell back to the base rate)
                        # give a zero paired SE; report the unpaired bound and say so
                        mde = min_detectable_improvement(cmp["n"], metric, cmp["baseline_value"])
                        mde_source = MDE_UPPER_BOUND
                    cmp["mde"] = mde
                    cmp["mde_source"] = mde_source
                    cmp["verdict"] = (VERDICT_IDENTICAL if identical
                                      else verdict(metric, cmp["ci"][0], cmp["ci"][1], mde, cmp["n"]))
                    comparison[metric] = cmp
            cell["comparison"] = comparison or None
            for metric in MDE_METRICS:
                cmp = comparison.get(metric)
                if cmp is not None and cmp["mde"] is not None and np.isfinite(cmp["mde"]):
                    cell["mde"][metric] = cmp["mde"]
                    cell["mde_source"][metric] = cmp.get("mde_source", MDE_PAIRED)
                else:  # no paired comparison (a baseline, or no baseline present): the closed-form upper bound
                    base = best.get(subset, {}).get(metric)
                    base_value = base["value"] if base else cell.get(metric)
                    cell["mde"][metric] = min_detectable_improvement(cell["n_direction"], metric, base_value)
                    cell["mde_source"][metric] = MDE_UPPER_BOUND

    # trading simulation: one pass per model over every row (trades.parquet keeps them all, with
    # a `headline` flag), then statistics and the paired PnL comparison per TRADING_SUBSETS entry
    # so that a headline row never carries the PnL of pre-listing FMP-proxy prints simulated at
    # perp fees or of default-clock (calendar_flag / timing_unknown) t0s
    trade_frames: list[pd.DataFrame] = []
    trading: dict[str, dict[str, dict[str, dict]]] = {}  # model -> subset -> sizing -> stats
    pnl_by_event: dict[tuple[str, str, str], pd.Series] = {}
    for name, preds in blocks.items():
        trades = simulate_rows(preds, bar_index, settings=settings, funding=funding_fn, sizings=SIZINGS,
                               threshold=settings.trade_threshold, target_vol=settings.target_vol)
        headline_ids = pd.Index(preds.loc[masks[name]["headline"], P.event_id].astype(str))
        trades["headline"] = trades[P.event_id].astype(str).isin(headline_ids).to_numpy()
        trade_frames.append(trades)
        trading[name] = {}
        for subset in TRADING_SUBSETS:
            ids = None if subset == "all" else pd.Index(preds.loc[masks[name][subset], P.event_id].astype(str))
            trading[name][subset] = {}
            for sizing in SIZINGS:
                t = trades[trades["sizing"] == sizing]
                if ids is not None:
                    t = t[t[P.event_id].astype(str).isin(ids)]
                stats, pnl = _trading_stats(t, subset=subset, settings=settings, n_boot=n_boot, seed=seed)
                trading[name][subset][sizing] = stats
                pnl_by_event[(name, subset, sizing)] = pnl
    for subset in TRADING_SUBSETS:
        for sizing in SIZINGS:
            best_b = None
            for name in blocks:
                if name in baselines:
                    v = trading[name][subset][sizing]["mean_pnl"]["point"]
                    if v is not None and np.isfinite(v) and (best_b is None or v > best_b[1]):
                        best_b = (name, v)
            for name in blocks:
                if name in baselines or best_b is None:
                    continue
                a, b = pnl_by_event[(name, subset, sizing)], pnl_by_event[(best_b[0], subset, sizing)]
                common = a.index.intersection(b.index)
                if len(common) == 0:
                    continue
                diff = a.loc[common] - b.loc[common]
                scheme, block = _blocks(scores[name][["season", "day"]].reindex(common).fillna("?"))
                dist = bootstrap_distribution(diff, lambda v: float(v.mean()), n=n_boot, block=block, seed=seed)
                finite = dist[np.isfinite(dist)]
                lo, hi = (float(x) for x in np.percentile(finite, [2.5, 97.5])) if len(finite) else (math.nan, math.nan)
                trading[name][subset][sizing]["comparison"] = {
                    "baseline": best_b[0], "improvement": float(diff.mean()), "ci": [lo, hi],
                    "p_noise": float(np.mean(finite <= 0)) if len(finite) else math.nan,
                    "se": paired_se(diff), "resampling": scheme,
                    "verdict": verdict("mean_pnl", lo, hi, None, int(len(diff)))}

    per_model: dict[str, Any] = {}
    for name, preds in blocks.items():
        res = preds[preds[P.r_true].notna()]
        q10, q90 = residual_band(res[P.r_hat], res[P.r_true])
        banded = res[res[P.r_lo].notna() & res[P.r_hi].notna()]
        coverage = float(((banded[P.r_true] >= banded[P.r_lo]) & (banded[P.r_true] <= banded[P.r_hi])).mean()) if len(banded) else math.nan
        # `trading` is the every-row ("all") simulation, the shape the CLI table reads;
        # `trading_subsets` holds the same statistics per TRADING_SUBSETS entry
        per_model[name] = {"is_baseline": name in baselines, "subsets": cells[name],
                           "residual_band": {"q10": q10, "q90": q90, "n": int(len(res)), "coverage": coverage,
                                             "n_with_band": int(len(banded))},
                           "trading": trading[name]["all"], "trading_subsets": trading[name]}
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return per_model, best, trades_all


def _trading_stats(t: pd.DataFrame, *, subset: str, settings: Settings, n_boot: int,
                   seed: int) -> tuple[dict[str, Any], pd.Series]:
    """Portfolio metrics, bootstrap intervals of the per-event mean net PnL and hit rate,
    untraded reasons and funding shares of one (model, sizing, subset) slice of simulated
    trades; also the per-event PnL series (0 for untraded events) the paired comparison uses."""
    pm = portfolio_metrics(t, gross_exposure_cap=settings.gross_exposure_cap)
    traded = t[t["traded"]]
    pnl = pd.Series(traded["pnl"].to_numpy(dtype=float), index=pd.Index(traded[P.event_id].astype(str)))
    labels = pd.DataFrame({"season": traded[P.test_season].to_numpy(), "day": _day_labels(traded)}, index=pnl.index)
    scheme, block = _blocks(labels)
    stats: dict[str, Any] = dict(pm)
    stats["subset"] = subset
    stats["n_events"] = int(t[P.event_id].nunique())
    stats["resampling"] = scheme
    stats["mean_pnl"] = _ci_dict(pnl, lambda v: float(v.mean()), block, n_boot, seed)
    stats["hit_rate"] = _ci_dict(pnl, lambda v: float((v > 0).mean()), block, n_boot, seed)
    stats["untraded_reasons"] = {str(k): int(v) for k, v in t.loc[~t["traded"], "untraded_reason"].value_counts().items()}
    with_funding = traded["funding_source"] == FUNDING_ARCHIVE
    stats["funding_share_events"] = float(with_funding.mean()) if len(traded) else math.nan
    abs_pnl = traded["pnl"].abs()
    stats["funding_share_abs_pnl"] = float(abs_pnl[with_funding].sum() / abs_pnl.sum()) if len(traded) and abs_pnl.sum() > 0 else math.nan
    stats["comparison"] = None
    all_ids = pd.Index(t[P.event_id].astype(str))
    return stats, pnl.reindex(all_ids[~all_ids.duplicated()]).fillna(0.0)


def _ci_dict(values: pd.Series, stat: Callable[[pd.Series], float], block: pd.Series | None, n_boot: int,
             seed: int) -> dict[str, float]:
    if len(values) == 0:
        return {"point": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    point, lo, hi = bootstrap_ci(values, stat, n=n_boot, block=block, seed=seed)
    return {"point": point, "lo": lo, "hi": hi, "n": int(len(values))}


# ---- holdout guard ------------------------------------------------------------------------------
def check_holdout_ready(df: pd.DataFrame, settings: Settings, now: pd.Timestamp, *,
                        events: pd.DataFrame | None = None) -> pd.DataFrame:
    """The holdout rows, or HoldoutNotReady when no holdout season is pinned, the season is not
    closed yet (now < start of the next season + horizon: a dataset built mid-season cannot
    contain the events still scheduled in it), the dataset has no holdout events, any holdout
    event has t0 + horizon in the future, any holdout row is pending or target_missing without
    a resolved price path, or -- when the events calendar `events` (events.parquet: event_id,
    t0) is given -- an event scheduled in the holdout season is missing from the dataset.

    A holdout event whose fine path was chosen and whose p0 resolved but whose +24h label is
    NaN is unobservable by construction (design §2: the +24h checkpoint had no bar within the
    staleness limit, as for the FMP proxy's 04:00-19:55 ET session when t0 + 24h falls in an
    XNYS closure, or a corporate action inside [P0, t0 + 24h]); no rebuild can fill it, so it
    never blocks the run. Such event ids are returned in `hold.attrs['unobservable_24h']` and
    the scorer leaves them out of the holdout cells."""
    season = settings.holdout_season
    if not season:
        raise HoldoutNotReady("no holdout_season is pinned in settings; nothing to score")
    horizon = pd.Timedelta(hours=settings.horizon_hours)
    closes = season_end(season) + horizon
    if now < closes:
        raise HoldoutNotReady(f"the holdout season {season} is not closed: events scheduled in it can still have "
                              f"t0 + {settings.horizon_hours}h in the future until {closes.isoformat()} "
                              f"(now {now.isoformat()})")
    hold = df[df["season"] == season]
    if hold.empty:
        raise HoldoutNotReady(f"the dataset has no events in the holdout season {season}")
    future = hold[hold[E.t0] + horizon > now]
    if len(future):
        latest = future[E.t0].max()
        raise HoldoutNotReady(f"{future[D.event_id].nunique()} holdout event(s) have t0 + {settings.horizon_hours}h in the "
                              f"future (latest t0 {latest.isoformat()}); the season {season} is not closed")
    missing = hold["target_missing"].astype(bool)
    pending = hold[E.pending].astype(bool) if E.pending in hold.columns else pd.Series(False, index=hold.index)
    resolved = hold[T.price_source].notna()
    if T.p0 in hold.columns:
        resolved &= pd.to_numeric(hold[T.p0], errors="coerce").notna()
    unobservable = missing & resolved & ~pending
    blocking = (missing | pending) & ~unobservable
    if blocking.any():
        n = int(hold.loc[blocking, D.event_id].nunique())
        raise HoldoutNotReady(f"{n} holdout event(s) have missing or pending targets; complete the dataset first")
    hold = hold.copy()
    hold.attrs["unobservable_24h"] = sorted(hold.loc[unobservable, D.event_id].astype(str).unique())
    if events is not None and len(events):
        for col in (E.event_id, E.t0):
            if col not in events.columns:
                raise ValueError(f"events calendar lacks the {col!r} column")
        scheduled = events.loc[seasons_of(t0_utc(events)) == season, E.event_id].astype(str)
        absent = sorted(set(scheduled) - set(hold[D.event_id].astype(str)))
        if absent:
            shown = ", ".join(absent[:5]) + (", ..." if len(absent) > 5 else "")
            raise HoldoutNotReady(f"{len(absent)} event(s) scheduled in the holdout season {season} are not in the "
                                  f"dataset ({shown}); rebuild the dataset before scoring the holdout")
    return hold


def _dataset_hash(settings: Settings, dataset: pd.DataFrame, dataset_path: Path | str | None) -> tuple[str, str]:
    """(sha256, source): the parquet file's bytes when `dataset_path` is given or
    settings.dataset_path exists (design §8: run_id = <ts>-<sha256(dataset.parquet)[:8]>, so
    the CLI and optimize see the same id), else the content hash of the frame."""
    path = Path(dataset_path) if dataset_path is not None else (settings.dataset_path if settings.dataset_path.exists() else None)
    if path is not None:
        return dataset_sha256(path), f"file:{path}"
    return dataset_sha256(dataset), "content"


# ---- public entry points --------------------------------------------------------------------------
def evaluate(settings: Settings, dataset: pd.DataFrame, *, model_names: list[str],
             decision_times: list[str], final: bool = False, run_id: str | None = None,
             target: str = "r_24h", paths: Callable[[str], pd.DataFrame | None] | None = None,
             funding: Callable[[str], pd.DataFrame | None] | None = None, n_boot: int = 1000,
             now: pd.Timestamp | None = None, events: pd.DataFrame | None = None,
             dataset_path: Path | str | None = None) -> dict:
    """Walk-forward for each (model, decision_time) on folds that exclude the holdout season;
    with final=True additionally scores the holdout once and logs it. Writes
    reports/<run_id>/{summary.json, predictions.parquet, trades.parquet, leaderboard.md} and
    returns the summary dict (metrics per model/decision_time/subset, trading sim, bootstrap
    intervals, paired comparison vs best baseline, residual bands, provenance).

    `paths(event_id)` supplies the fine bars for the simulation (default: targets.loaders through
    the archive / live candles / FMP); `funding(market)` the archived hourly funding (default:
    the archive). `n_boot` bootstrap replicates per interval; `now` overrides the clock used by
    the final-run guard. `events` is the earnings calendar (events.parquet; default: read from
    settings.events_path when it exists) that a final run cross-checks for holdout-season
    events missing from the dataset and that supplies `estimate_source` per event when the
    dataset predates that column. Trading statistics come per subset (`trading_subsets`, see
    TRADING_SUBSETS; `trading` is the every-row entry) and the summary marks the non-point-in-time
    feature groups in scope (features.groups.NON_POINT_IN_TIME) with the estimate_source
    breakdown of the trainable events. `dataset` must be the content of `dataset_path` (default:
    settings.dataset_path when it exists), whose file bytes give the dataset hash in run_id;
    without a file the frame's content hash is used and summary['dataset_hash_source'] says so.
    Provider errors from the default bar loader (budget exhausted, provider unavailable)
    propagate: a run never silently reports trades on a partially fetched set of events."""
    now = now or utcnow()
    for name in model_names:
        if name not in models_mod.REGISTRY:
            raise KeyError(f"unknown model {name!r}; available: {models_mod.available_models()}")
    for d in decision_times:
        if d not in DECISION_TIMES:
            raise ValueError(f"unknown decision_time {d!r}; expected one of {sorted(DECISION_TIMES)}")
    if not model_names or not decision_times:
        raise ValueError("model_names and decision_times must be non-empty")
    df = prepare_dataset(dataset, target, settings)
    ds_hash, hash_source = _dataset_hash(settings, dataset, dataset_path)
    run_id = run_id or make_run_id(ds_hash, now)
    if events is None and settings.events_path.exists():
        events = pd.read_parquet(settings.events_path)
    df, filled = attach_estimate_source(df, events)
    # a dataset built before estimate_source joined features.META_COLUMNS, with no calendar to
    # fill it, reports the provenance of its consensus inputs as 'unavailable'
    estimate_source_known = E.estimate_source in dataset.columns or filled
    holdout_unobservable: list[str] = []
    if final:
        hold = check_holdout_ready(df, settings, now, events=events)
        holdout_unobservable = list(hold.attrs.get("unobservable_24h", []))
    scorings_before = count_holdout_scorings(settings.holdout_log_path)
    bar_index = memoised_bar_index(paths if paths is not None else loader_paths(settings, df))
    funding_fn = memoised_funding(funding if funding is not None else archive_funding_loader(settings))
    baselines = baseline_names()
    feats = feature_columns(df)

    results: dict[str, Any] = {}
    holdout_results: dict[str, Any] = {}
    best_baseline: dict[str, Any] = {}
    folds_info: dict[str, Any] = {}
    skipped: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    pred_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for d in decision_times:
        sub_all = df[df[D.decision_time] == d].drop_duplicates(D.event_id).reset_index(drop=True)
        # rows without a realised target (upcoming events, or no bars) can be neither trained
        # on nor scored; keeping them would create phantom test folds for future seasons
        sub = sub_all[sub_all[Y].notna()].reset_index(drop=True)
        n_unscorable = int(len(sub_all) - len(sub))
        if sub.empty:
            raise ValueError(f"the dataset has no scorable rows for decision_time {d!r}")
        folds, holdout, info, skip = _fold_plan(sub, settings)
        # seasons whose every event is unscorable disappear from the plan; list them as skipped
        # so the report shows where the data ran out instead of silently narrowing
        from .folds import seasons_of

        gone = sorted(set(seasons_of(sub_all[E.t0]).dropna()) - set(seasons_of(sub[E.t0]).dropna()))
        for label in gone:
            if label == settings.holdout_season:
                continue
            before = int((sub[TRAINABLE] & (seasons_of(sub[E.t0]) < label)).sum())
            skip.append({"test_season": label, "n_train_trainable": before, "reason": "no scorable events"})
        skip.sort(key=lambda x: x["test_season"])
        if not folds:
            raise ValueError(f"{d}: no season has {settings.min_train_events} trainable events before it "
                             f"(embargo {settings.embargo_days} d); {len(sub)} events in the dataset")
        if final and (holdout is None or int(sub.loc[holdout.train_idx, TRAINABLE].sum()) < settings.min_train_events):
            raise HoldoutNotReady(f"{d}: fewer than {settings.min_train_events} trainable events before the holdout season")
        folds_info[d], skipped[d] = info, skip
        extras[d] = {"n_events": int(len(sub)), "n_unscorable_excluded": n_unscorable,
                     "n_trainable": int(sub[TRAINABLE].sum()),
                     "n_has_perp": int(sub[E.has_perp_at_t0].sum()),
                     "n_low_confidence": int((sub[E.t0_confidence] < settings.min_t0_confidence).sum()),
                     "n_target_missing": int(sub["target_missing"].sum())}
        if T.continuation in sub.columns and T.r("24h") in sub.columns:
            extras[d]["continuation_dead_band_n"] = int((sub[T.r("24h")].notna() & sub[T.continuation].isna()).sum())
        # design §5/§6: the inputs that were not knowable at t0, and how many trainable events
        # carry a vendor-final consensus rather than a point-in-time snapshot
        extras[d]["non_point_in_time_groups"] = sorted(non_point_in_time_in_scope(feats, d))
        trainable_rows = sub[sub[TRAINABLE]]
        extras[d]["estimate_source"] = (estimate_source_counts(trainable_rows) if estimate_source_known
                                        else {ESTIMATE_SOURCE_UNAVAILABLE: int(len(trainable_rows))})
        blocks: dict[str, pd.DataFrame] = {}
        hold_blocks: dict[str, pd.DataFrame] = {}
        for name in model_names:
            preds = _walk_forward(sub, feats, folds, name, settings, holdout=holdout if final else None)
            pred_frames.append(preds)
            blocks[name] = preds[preds[P.fold] != HOLDOUT_FOLD].reset_index(drop=True)
            if final:
                hold_blocks[name] = preds[preds[P.fold] == HOLDOUT_FOLD].reset_index(drop=True)
        per_model, best, trades = _score_block(blocks, settings=settings, baselines=baselines, n_boot=n_boot,
                                               bar_index=bar_index, funding_fn=funding_fn)
        # a learner whose direction head fell back to the base rate in every fold (fewer than
        # MIN_TRAIN_ROWS usable rows) was never trained: its comparison says nothing about
        # predictability, so label it instead of judging it
        max_train = max((int(sub.loc[f.train_idx, TRAINABLE].sum()) for f in folds), default=0)
        for name, res in per_model.items():
            if res.get("is_baseline") or name not in blocks or FALLBACK not in blocks[name].columns:
                continue
            if len(blocks[name]) and bool(blocks[name][FALLBACK].all()):
                res["untrained"] = True
                for cell in res["subsets"].values():
                    for cmp in (cell.get("comparison") or {}).values():
                        if isinstance(cmp, dict) and "verdict" in cmp:
                            cmp["verdict"] = (f"{VERDICT_UNTRAINED} (direction head fell back to the base rate "
                                              f"in every fold: fewer than {MIN_TRAIN_ROWS} usable rows)")
        extras[d]["max_train_rows"] = max_train
        results[d], best_baseline[d] = per_model, best
        trade_frames.append(trades.assign(block="walk_forward"))
        if final:
            per_model_h, best_h, trades_h = _score_block(hold_blocks, settings=settings, baselines=baselines,
                                                         n_boot=n_boot, bar_index=bar_index, funding_fn=funding_fn)
            holdout_results[d] = {"models": per_model_h, "best_baseline": best_h}
            trade_frames.append(trades_h.assign(block="holdout"))

    predictions = pd.concat(pred_frames, ignore_index=True)
    trades_all = pd.concat([t for t in trade_frames if len(t)], ignore_index=True) if any(len(t) for t in trade_frames) else pd.DataFrame()
    git = git_info()
    non_pit = {g: reason for d in decision_times for g, reason in non_point_in_time_in_scope(feats, d).items()}
    notes = _notes(results, extras, scorings_before, settings, model_names=model_names, baselines=baselines,
                   holdout_unobservable=holdout_unobservable)
    summary: dict[str, Any] = {
        "run_id": run_id, "created_at": now, "final": bool(final), "target": target,
        "schema_version": SCHEMA_VERSION, "dataset_sha256": ds_hash, "dataset_hash_source": hash_source,
        "n_rows": int(len(dataset)),
        "n_events": int(df[D.event_id].nunique()), "git": git, "settings": public_settings(settings),
        "config_hash": config_hash(settings), "versions": library_versions(),
        "decision_times": list(decision_times), "models": list(model_names),
        "baselines": sorted(b for b in model_names if b in baselines),
        "holdout": {"season": settings.holdout_season, "scorings_before": scorings_before,
                    "scorings_after": scorings_before, "scored_now": bool(final),
                    "n_events": int((df["season"] == settings.holdout_season)[df[D.decision_time] == decision_times[0]].sum()) if settings.holdout_season else 0,
                    "n_unobservable_24h": len(holdout_unobservable), "unobservable_24h": holdout_unobservable},
        "folds": folds_info, "skipped_seasons": skipped, "cohorts": extras,
        "non_point_in_time_groups": non_pit,
        "best_baseline": best_baseline, "results": results,
        "holdout_results": holdout_results if final else None,
        "sizings": list(SIZINGS), "trading_subsets": list(TRADING_SUBSETS), "n_boot": int(n_boot),
        "capital_rule": CAPITAL_RULE,
        "mde_sources": {MDE_PAIRED: "MDE = (z_0.975 + z_0.8) * SE of the mean paired score difference vs the best baseline",
                        MDE_UPPER_BOUND: "no paired comparison: closed-form unpaired bound, conservative (larger)"},
        "resampling": "block bootstrap by season with at least 5 seasons, else by UTC day of t0, else iid rows; "
                      "recorded per cell as 'resampling'",
        "notes": notes,
    }
    if final:
        append_holdout_log(settings.holdout_log_path, {
            "timestamp": now, "run_id": run_id, "git_commit": git["sha"], "git_dirty": git["dirty"],
            "dataset_sha256": ds_hash, "models": list(model_names), "decision_times": list(decision_times),
            "holdout_season": settings.holdout_season, "target": target})
        summary["holdout"]["scorings_after"] = count_holdout_scorings(settings.holdout_log_path)
    summary = to_jsonable(summary)
    write_reports(settings, run_id, summary, predictions, trades_all)
    return summary


def _notes(results: dict[str, Any], extras: dict[str, Any], scorings_before: int, settings: Settings, *,
           model_names: list[str], baselines: frozenset[str], holdout_unobservable: list[str]) -> list[str]:
    notes = [f"holdout season {settings.holdout_season} had been scored {scorings_before} time(s) before this run; "
             "discount any holdout number accordingly"]
    if holdout_unobservable:
        shown = ", ".join(holdout_unobservable[:5]) + (", ..." if len(holdout_unobservable) > 5 else "")
        notes.append(f"{len(holdout_unobservable)} holdout event(s) have a price path but no +24h label by construction "
                     "(the +24h checkpoint had no bar within the staleness limit: the FMP proxy with t0 + 24h in an "
                     "XNYS closure, or a corporate action inside [P0, t0 + 24h]; design §2) and are excluded from "
                     f"the holdout cells: {shown}")
    # every learner is fitted on every feature column, so a non-point-in-time group in scope was
    # consumed by every trained model (and by surprise_sign, which reads the surprise itself)
    consumers = [m for m in model_names if m not in baselines or m == "surprise_sign"]
    for d, cohort in extras.items():
        groups = cohort.get("non_point_in_time_groups") or []
        if not groups or not consumers:
            continue
        reasons = "; ".join(f"{g} ({NON_POINT_IN_TIME.get(g, 'not point-in-time')})" for g in groups)
        notes.append(f"{d}: non-point-in-time inputs in scope: {reasons}; estimate_source of the trainable events "
                     f"{cohort.get('estimate_source') or {}}; learners here use every feature column, so "
                     f"{', '.join(consumers)} consumed them and their cells are not evidence about point-in-time predictability")
    for d, per_model in results.items():
        n_cells = n_inconclusive = n_identical = 0
        for res in per_model.values():
            for cell in res["subsets"].values():
                cmp = (cell.get("comparison") or {}).get("brier")
                if cmp is None:
                    continue
                n_cells += 1
                n_inconclusive += str(cmp["verdict"]).startswith("inconclusive")
                n_identical += str(cmp["verdict"]) == VERDICT_IDENTICAL or str(cmp["verdict"]).startswith(VERDICT_UNTRAINED)
        n_perp = extras.get(d, {}).get("n_has_perp", 0)
        if n_cells:
            note = (f"{d}: {n_inconclusive} of {n_cells} Brier comparisons are inconclusive at their n"
                    + (f" and {n_identical} come from learners that were never trained (every fold below "
                       f"{MIN_TRAIN_ROWS} training rows, so they reproduce the base rate)" if n_identical else "")
                    + f"; the perp-era cohort (has_perp_at_t0) holds {n_perp} events")
            if 2 * (n_inconclusive + n_identical) >= n_cells:
                note += (", so with listings only since Nov 2025 this report is mostly inconclusive, "
                         "as expected for early runs")
            notes.append(note)
    return notes


def train_final(settings: Settings, dataset: pd.DataFrame, *, model_name: str, decision_time: str,
                target: str = "r_24h", dataset_path: Path | str | None = None) -> object:
    """Fit on all non-holdout events that pass the headline filters (min_t0_confidence,
    has_perp_at_t0 when enough events exist), attach the residual band from a walk-forward pass,
    save under settings.models_dir/<decision_time>/<model_name>/ with model.json (decision_time,
    dataset_sha256, git sha, config hash, trained_at, n_events, filters, holdout reference) and
    return the model. The dataset hash follows the same rule as `evaluate` (file bytes of
    `dataset_path` / settings.dataset_path when present, else the frame's content hash)."""
    if model_name not in models_mod.REGISTRY:
        raise KeyError(f"unknown model {model_name!r}; available: {models_mod.available_models()}")
    if decision_time not in DECISION_TIMES:
        raise ValueError(f"unknown decision_time {decision_time!r}")
    df = prepare_dataset(dataset, target, settings)
    sub = df[df[D.decision_time] == decision_time].drop_duplicates(D.event_id).reset_index(drop=True)
    if settings.holdout_season:
        sub = sub[sub["season"] != settings.holdout_season]
    trainable = sub[sub[TRAINABLE]].reset_index(drop=True)
    perp = trainable[trainable[E.has_perp_at_t0]].reset_index(drop=True)
    use_perp = len(perp) >= settings.min_train_events
    train = perp if use_perp else trainable
    if train.empty:
        raise ValueError(f"no trainable events for {decision_time} (min_t0_confidence={settings.min_t0_confidence})")
    feats = feature_columns(df)

    # residual band from an honest walk-forward over the training cohort (or, failing that, all
    # trainable events); in-sample residuals only as a flagged last resort
    residuals: list[np.ndarray] = []
    source = "walk_forward"
    for cohort, label in ((train, "walk_forward"), (trainable, "walk_forward_all_events")):
        folds, _ = walk_forward_folds(cohort, min_train=settings.min_train_events,
                                      embargo_days=settings.embargo_days, holdout_season=settings.holdout_season)
        for fold in folds:
            tr, te = cohort.loc[fold.train_idx], cohort.loc[fold.test_idx]
            _, r, _, _ = _fit_predict(model_name, settings.random_seed, tr, te, feats)
            res = te[Y].to_numpy(dtype=float) - r
            residuals.append(res[np.isfinite(res)])
        if residuals and sum(len(r) for r in residuals) > 0:
            source = label
            break
    model = models_mod.make_model(model_name, seed=settings.random_seed)
    model.fit(_X(train, feats), train[Y].astype(float), _direction_target(train))
    if not residuals or sum(len(r) for r in residuals) == 0:
        source = "in_sample"
        r_in = np.asarray(model.predict_return(_X(train, feats)), dtype=float).reshape(-1)
        residuals = [train[Y].to_numpy(dtype=float) - r_in]
    pooled = np.concatenate(residuals)
    q10, q90 = residual_band(np.zeros(len(pooled)), pooled)
    model.residual_q_ = (q10, q90)
    if not getattr(model, "feature_names_", None):
        model.feature_names_ = list(feats)

    out_dir = settings.models_dir / decision_time / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save(out_dir)
    except NotImplementedError:
        import joblib

        joblib.dump(model, out_dir / "model.joblib")
    git = git_info()
    log_path = settings.holdout_log_path
    ds_hash, hash_source = _dataset_hash(settings, dataset, dataset_path)
    meta = {
        "model": model_name, "decision_time": decision_time, "target": target,
        "dataset_sha256": ds_hash, "dataset_hash_source": hash_source, "git_sha": git["sha"], "git_dirty": git["dirty"],
        "config_hash": config_hash(settings), "trained_at": utcnow(), "n_events": int(len(train)),
        "filters": {"min_t0_confidence": settings.min_t0_confidence, "has_perp_at_t0": bool(use_perp),
                    "holdout_season_excluded": settings.holdout_season, "target_present": True},
        "residual_band": {"q10": q10, "q90": q90, "source": source, "n": int(len(pooled))},
        "holdout": {"season": settings.holdout_season, "scorings": count_holdout_scorings(log_path),
                    "last_scoring": last_holdout_scoring(log_path)},
        "feature_names": list(model.feature_names_), "seed": settings.random_seed,
        "schema_version": SCHEMA_VERSION, "versions": library_versions(),
    }
    (out_dir / "model.json").write_text(json.dumps(to_jsonable(meta), indent=2, sort_keys=True))
    return model
