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
from typing import Any

import numpy as np
import pandas as pd

from .. import models as models_mod
from ..config import Settings
from ..schemas import DECISION_TIMES, SCHEMA_VERSION, D, E, P, T, T0Source
from .folds import HOLDOUT_FOLD, Fold, seasons_of, t0_utc, walk_forward_folds
from .metrics import (
    EPS,
    MDE_METRICS,
    bootstrap_ci,
    bootstrap_distribution,
    brier_scores,
    calibration_table,
    classification_metrics,
    hit_scores,
    min_detectable_improvement,
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
    FUNDING_ARCHIVE,
    SIZINGS,
    archive_funding_loader,
    loader_paths,
    memoised_bar_index,
    memoised_funding,
    portfolio_metrics,
    simulate_rows,
)

log = logging.getLogger(__name__)

DEFAULT_BASELINES = frozenset({"zero", "base_rate", "historical_mean", "hist_abs_mean", "vol_scaled",
                               "sign_of_reaction", "always_extends", "surprise_sign"})
COMPARED_METRICS = ("accuracy", "brier", "log_loss", "spearman_ic", "mae")
HIGHER_IS_BETTER = {"accuracy": True, "balanced_accuracy": True, "brier": False, "log_loss": False,
                    "spearman_ic": True, "mae": False, "rmse": False}
HEADLINE_SOURCES = frozenset({T0Source.sec_8k.value, T0Source.manual.value, T0Source.detected.value})
STRATA = (E.t0_source, E.kind, E.timing)
CALIBRATION_SUBSETS = ("all", "headline")
META_COLUMNS = [E.t0, E.t0_source, E.t0_confidence, E.kind, E.timing, E.has_perp_at_t0, E.market,
                "season", "price_source", "target_missing"]
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
                         (E.market, None), (T.price_source, None)):
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


def _X(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    if not feats:
        return pd.DataFrame(index=df.index)
    return df[feats].apply(pd.to_numeric, errors="coerce").astype(float)


def _fit_predict(name: str, seed: int, train: pd.DataFrame, test: pd.DataFrame,
                 feats: list[str]) -> tuple[np.ndarray, np.ndarray, Any]:
    model = models_mod.make_model(name, seed=seed)
    y = train[Y].astype(float)
    direction = pd.Series(np.where(train[DIR].fillna(0) >= 0, 1.0, -1.0), index=train.index)
    model.fit(_X(train, feats), y, direction)
    X_te = _X(test, feats)
    p = np.asarray(model.predict_proba_up(X_te), dtype=float).reshape(-1)
    r = np.asarray(model.predict_return(X_te), dtype=float).reshape(-1)
    if len(p) != len(test) or len(r) != len(test):
        raise ValueError(f"model {name!r} returned {len(p)}/{len(r)} predictions for {len(test)} rows")
    return p, r, model


# ---- walk-forward ------------------------------------------------------------------------------
def _fold_plan(sub: pd.DataFrame, settings: Settings) -> tuple[list[Fold], Fold | None, list[dict], list[dict]]:
    folds, holdout = walk_forward_folds(sub, min_train=settings.min_train_events,
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
    come from the residuals of the *earlier* folds only (NaN for the first); the holdout uses
    all walk-forward residuals."""
    frames: list[pd.DataFrame] = []
    residuals: list[np.ndarray] = []
    plan = list(folds) + ([holdout] if holdout is not None else [])
    for fold in plan:
        train = sub.loc[fold.train_idx]
        train = train[train[TRAINABLE]]
        test = sub.loc[fold.test_idx]
        p, r, _ = _fit_predict(name, settings.random_seed, train, test, feats)
        pooled = np.concatenate(residuals) if residuals else np.array([], dtype=float)
        q10, q90 = residual_band(np.zeros(len(pooled)), pooled) if len(pooled) else (math.nan, math.nan)
        frame = test[[D.event_id, D.decision_time, *META_COLUMNS]].copy()
        frame[P.model] = name
        frame[P.fold] = int(fold.fold)
        frame[P.test_season] = fold.test_season
        frame[P.p_up] = p
        frame[P.r_hat] = r
        frame[P.r_lo] = r + q10
        frame[P.r_hi] = r + q90
        frame[P.r_true] = test[Y].to_numpy(dtype=float)
        frame[P.direction_true] = test[DIR].to_numpy(dtype=float)
        frames.append(frame)
        if fold.fold != HOLDOUT_FOLD:
            res = frame[P.r_true].to_numpy(dtype=float) - r
            residuals.append(res[np.isfinite(res)])
    cols = [P.event_id, P.decision_time, P.model, P.fold, P.test_season, P.p_up, P.r_hat, P.r_lo, P.r_hi,
            P.r_true, P.direction_true, *META_COLUMNS]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    return out[cols]


# ---- scoring ---------------------------------------------------------------------------------------
def _scores(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-event scores indexed by event_id: hit, brier, ll, ae plus the raw columns."""
    p = preds[P.p_up].to_numpy(dtype=float)
    y = preds[P.direction_true].to_numpy(dtype=float)
    yb = (y > 0).astype(float)
    pc = np.clip(p, EPS, 1 - EPS)
    ll = -(yb * np.log(pc) + (1 - yb) * np.log(1 - pc))
    ll[~(np.isfinite(p) & np.isfinite(y) & (y != 0))] = np.nan
    r_hat = preds[P.r_hat].to_numpy(dtype=float)
    r_true = preds[P.r_true].to_numpy(dtype=float)
    out = pd.DataFrame({"hit": hit_scores(p, y), "brier": brier_scores(p, y), "ll": ll,
                        "ae": np.abs(r_true - r_hat), P.p_up: p, P.r_hat: r_hat, P.r_true: r_true,
                        P.direction_true: y, "season": preds[P.test_season].to_numpy()},
                       index=pd.Index(preds[P.event_id].astype(str), name=P.event_id))
    return out[~out.index.duplicated(keep="first")]


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
    cell: dict[str, Any] = {"n": int(len(preds)), "n_direction": cm["n"], "n_return": rm["n"]}
    cell.update({k: v for k, v in cm.items() if k != "n"})
    cell.update({k: v for k, v in rm.items() if k != "n"})
    ci: dict[str, list[float]] = {}
    for metric, col in (("accuracy", "hit"), ("brier", "brier")):
        s = scores[col].dropna()
        if len(s):
            _, lo, hi = bootstrap_ci(s, lambda v: float(v.mean()), n=n_boot,
                                     block=scores.loc[s.index, "season"], seed=seed)
            ci[metric] = [lo, hi]
    cell["ci"] = ci
    cell["mde"] = {}
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
    events `ids` (block by season). Improvement is signed so that positive is better."""
    common = ids.intersection(sm.index).intersection(sb.index)
    if len(common) == 0:
        return None
    a, b = sm.loc[common], sb.loc[common]
    season = a["season"]
    if metric == "spearman_ic":
        keep = np.isfinite(a[P.r_true].to_numpy(dtype=float))
        a, b, season = a[keep], b[keep], season[keep]
        if len(a) < 3:
            return None
        pos = pd.Series(np.arange(len(a)), index=a.index)
        rm, rb, y = a[P.r_hat].to_numpy(dtype=float), b[P.r_hat].to_numpy(dtype=float), a[P.r_true].to_numpy(dtype=float)

        def stat(v: pd.Series) -> float:
            i = v.to_numpy(dtype=int)
            return spearman(rm[i], y[i]) - spearman(rb[i], y[i])

        values = pos
    else:
        col = {"accuracy": "hit", "brier": "brier", "log_loss": "ll", "mae": "ae"}[metric]
        sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
        diff = sign * (a[col].astype(float) - b[col].astype(float))
        keep = diff.notna()
        diff, season = diff[keep], season[keep]
        if len(diff) == 0:
            return None
        values = diff

        def stat(v: pd.Series) -> float:
            return float(v.mean())

    point = float(stat(values))
    dist = bootstrap_distribution(values, stat, n=n_boot, block=season, seed=seed)
    finite = dist[np.isfinite(dist)]
    if len(finite):
        lo, hi = (float(x) for x in np.percentile(finite, [2.5, 97.5]))
        p_noise = float(np.mean(finite <= 0))
    else:
        lo = hi = p_noise = math.nan
    return {"improvement": point, "ci": [lo, hi], "p_noise": p_noise, "n": int(len(values))}


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
            for metric in MDE_METRICS:
                base = best.get(subset, {}).get(metric)
                base_value = base["value"] if base else cell.get(metric)
                cell["mde"][metric] = min_detectable_improvement(cell["n_direction"], metric, base_value)
            if name in baselines:
                continue
            comparison: dict[str, Any] = {}
            for metric in COMPARED_METRICS:
                base = best.get(subset, {}).get(metric)
                if base is None:
                    continue
                cmp = _compare(scores[name], scores[base["model"]], ids, metric, n_boot=n_boot, seed=seed)
                if cmp is None:
                    continue
                cmp.update({"baseline": base["model"], "baseline_value": base["value"], "model_value": cell.get(metric)})
                mde = cell["mde"].get(metric) if metric in MDE_METRICS else None
                cmp["mde"] = mde
                cmp["verdict"] = verdict(metric, cmp["ci"][0], cmp["ci"][1], mde, cmp["n"])
                comparison[metric] = cmp
            cell["comparison"] = comparison or None

    # trading simulation, all sizing rules in one pass per model
    trade_frames: list[pd.DataFrame] = []
    trading: dict[str, dict[str, dict]] = {}
    pnl_by_event: dict[tuple[str, str], pd.Series] = {}
    for name, preds in blocks.items():
        trades = simulate_rows(preds, bar_index, settings=settings, funding=funding_fn, sizings=SIZINGS,
                               threshold=0.0, target_vol=0.03)
        trade_frames.append(trades)
        trading[name] = {}
        for sizing in SIZINGS:
            t = trades[trades["sizing"] == sizing]
            pm = portfolio_metrics(t, gross_exposure_cap=settings.gross_exposure_cap)
            traded = t[t["traded"]]
            pnl = pd.Series(traded["pnl"].to_numpy(dtype=float), index=pd.Index(traded[P.event_id].astype(str)))
            season = pd.Series(traded[P.test_season].to_numpy(), index=pnl.index)
            stats: dict[str, Any] = dict(pm)
            stats["mean_pnl"] = _ci_dict(pnl, lambda v: float(v.mean()), season, n_boot, seed)
            stats["hit_rate"] = _ci_dict(pnl, lambda v: float((v > 0).mean()), season, n_boot, seed)
            stats["untraded_reasons"] = {str(k): int(v) for k, v in t.loc[~t["traded"], "untraded_reason"].value_counts().items()}
            with_funding = traded["funding_source"] == FUNDING_ARCHIVE
            stats["funding_share_events"] = float(with_funding.mean()) if len(traded) else math.nan
            abs_pnl = traded["pnl"].abs()
            stats["funding_share_abs_pnl"] = float(abs_pnl[with_funding].sum() / abs_pnl.sum()) if len(traded) and abs_pnl.sum() > 0 else math.nan
            stats["comparison"] = None
            trading[name][sizing] = stats
            all_ids = pd.Index(t[P.event_id].astype(str))
            pnl_by_event[(name, sizing)] = pnl.reindex(all_ids[~all_ids.duplicated()]).fillna(0.0)
    for sizing in SIZINGS:
        best_b = None
        for name in blocks:
            if name in baselines:
                v = trading[name][sizing]["mean_pnl"]["point"]
                if v is not None and np.isfinite(v) and (best_b is None or v > best_b[1]):
                    best_b = (name, v)
        for name in blocks:
            if name in baselines or best_b is None:
                continue
            a, b = pnl_by_event[(name, sizing)], pnl_by_event[(best_b[0], sizing)]
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            diff = a.loc[common] - b.loc[common]
            season = scores[name]["season"].reindex(common).fillna("?")
            dist = bootstrap_distribution(diff, lambda v: float(v.mean()), n=n_boot, block=season, seed=seed)
            finite = dist[np.isfinite(dist)]
            lo, hi = (float(x) for x in np.percentile(finite, [2.5, 97.5])) if len(finite) else (math.nan, math.nan)
            trading[name][sizing]["comparison"] = {"baseline": best_b[0], "improvement": float(diff.mean()),
                                                   "ci": [lo, hi], "p_noise": float(np.mean(finite <= 0)) if len(finite) else math.nan,
                                                   "verdict": verdict("mean_pnl", lo, hi, None, int(len(diff)))}

    per_model: dict[str, Any] = {}
    for name, preds in blocks.items():
        res = preds[preds[P.r_true].notna()]
        q10, q90 = residual_band(res[P.r_hat], res[P.r_true])
        banded = res[res[P.r_lo].notna() & res[P.r_hi].notna()]
        coverage = float(((banded[P.r_true] >= banded[P.r_lo]) & (banded[P.r_true] <= banded[P.r_hi])).mean()) if len(banded) else math.nan
        per_model[name] = {"is_baseline": name in baselines, "subsets": cells[name],
                           "residual_band": {"q10": q10, "q90": q90, "n": int(len(res)), "coverage": coverage,
                                             "n_with_band": int(len(banded))},
                           "trading": trading[name]}
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return per_model, best, trades_all


def _ci_dict(values: pd.Series, stat: Callable[[pd.Series], float], block: pd.Series, n_boot: int,
             seed: int) -> dict[str, float]:
    if len(values) == 0:
        return {"point": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    point, lo, hi = bootstrap_ci(values, stat, n=n_boot, block=block, seed=seed)
    return {"point": point, "lo": lo, "hi": hi, "n": int(len(values))}


# ---- holdout guard ------------------------------------------------------------------------------
def check_holdout_ready(df: pd.DataFrame, settings: Settings, now: pd.Timestamp) -> pd.DataFrame:
    """The holdout rows, or HoldoutNotReady when no holdout season is pinned, the dataset has
    no holdout events, any holdout event has t0 + horizon in the future, or any holdout row is
    target_missing / pending."""
    season = settings.holdout_season
    if not season:
        raise HoldoutNotReady("no holdout_season is pinned in settings; nothing to score")
    hold = df[df["season"] == season]
    if hold.empty:
        raise HoldoutNotReady(f"the dataset has no events in the holdout season {season}")
    horizon = pd.Timedelta(hours=settings.horizon_hours)
    future = hold[hold[E.t0] + horizon > now]
    if len(future):
        latest = future[E.t0].max()
        raise HoldoutNotReady(f"{future[D.event_id].nunique()} holdout event(s) have t0 + {settings.horizon_hours}h in the "
                              f"future (latest t0 {latest.isoformat()}); the season {season} is not closed")
    missing = hold["target_missing"].astype(bool)
    if E.pending in hold.columns:
        missing |= hold[E.pending].astype(bool)
    if missing.any():
        n = int(hold.loc[missing, D.event_id].nunique())
        raise HoldoutNotReady(f"{n} holdout event(s) have missing or pending targets; complete the dataset first")
    return hold


# ---- public entry points --------------------------------------------------------------------------
def evaluate(settings: Settings, dataset: pd.DataFrame, *, model_names: list[str],
             decision_times: list[str], final: bool = False, run_id: str | None = None,
             target: str = "r_24h", paths: Callable[[str], pd.DataFrame | None] | None = None,
             funding: Callable[[str], pd.DataFrame | None] | None = None, n_boot: int = 1000,
             now: pd.Timestamp | None = None) -> dict:
    """Walk-forward for each (model, decision_time) on folds that exclude the holdout season;
    with final=True additionally scores the holdout once and logs it. Writes
    reports/<run_id>/{summary.json, predictions.parquet, trades.parquet, leaderboard.md} and
    returns the summary dict (metrics per model/decision_time/subset, trading sim, bootstrap
    intervals, paired comparison vs best baseline, residual bands, provenance).

    `paths(event_id)` supplies the fine bars for the simulation (default: targets.loaders through
    the archive / live candles / FMP); `funding(market)` the archived hourly funding (default:
    the archive). `n_boot` bootstrap replicates per interval; `now` overrides the clock used by
    the final-run guard."""
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
    ds_hash = dataset_sha256(dataset)
    run_id = run_id or make_run_id(ds_hash, now)
    if final:
        check_holdout_ready(df, settings, now)
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
        sub = df[df[D.decision_time] == d].drop_duplicates(D.event_id).reset_index(drop=True)
        if sub.empty:
            raise ValueError(f"the dataset has no rows for decision_time {d!r}")
        folds, holdout, info, skip = _fold_plan(sub, settings)
        if not folds:
            raise ValueError(f"{d}: no season has {settings.min_train_events} trainable events before it "
                             f"(embargo {settings.embargo_days} d); {len(sub)} events in the dataset")
        if final and (holdout is None or int(sub.loc[holdout.train_idx, TRAINABLE].sum()) < settings.min_train_events):
            raise HoldoutNotReady(f"{d}: fewer than {settings.min_train_events} trainable events before the holdout season")
        folds_info[d], skipped[d] = info, skip
        extras[d] = {"n_events": int(len(sub)), "n_trainable": int(sub[TRAINABLE].sum()),
                     "n_has_perp": int(sub[E.has_perp_at_t0].sum()),
                     "n_low_confidence": int((sub[E.t0_confidence] < settings.min_t0_confidence).sum()),
                     "n_target_missing": int(sub["target_missing"].sum())}
        if T.continuation in sub.columns and T.r("24h") in sub.columns:
            extras[d]["continuation_dead_band_n"] = int((sub[T.r("24h")].notna() & sub[T.continuation].isna()).sum())
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
    notes = _notes(results, extras, scorings_before, settings)
    summary: dict[str, Any] = {
        "run_id": run_id, "created_at": now, "final": bool(final), "target": target,
        "schema_version": SCHEMA_VERSION, "dataset_sha256": ds_hash, "n_rows": int(len(dataset)),
        "n_events": int(df[D.event_id].nunique()), "git": git, "settings": public_settings(settings),
        "config_hash": config_hash(settings), "versions": library_versions(),
        "decision_times": list(decision_times), "models": list(model_names),
        "baselines": sorted(b for b in model_names if b in baselines),
        "holdout": {"season": settings.holdout_season, "scorings_before": scorings_before,
                    "scorings_after": scorings_before, "scored_now": bool(final),
                    "n_events": int((df["season"] == settings.holdout_season)[df[D.decision_time] == decision_times[0]].sum()) if settings.holdout_season else 0},
        "folds": folds_info, "skipped_seasons": skipped, "cohorts": extras,
        "best_baseline": best_baseline, "results": results,
        "holdout_results": holdout_results if final else None,
        "sizings": list(SIZINGS), "n_boot": int(n_boot), "notes": notes,
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


def _notes(results: dict[str, Any], extras: dict[str, Any], scorings_before: int, settings: Settings) -> list[str]:
    notes = [f"holdout season {settings.holdout_season} had been scored {scorings_before} time(s) before this run; "
             "discount any holdout number accordingly"]
    for d, per_model in results.items():
        n_cells = n_inconclusive = 0
        for res in per_model.values():
            for cell in res["subsets"].values():
                cmp = (cell.get("comparison") or {}).get("brier")
                if cmp is None:
                    continue
                n_cells += 1
                n_inconclusive += str(cmp["verdict"]).startswith("inconclusive")
        n_perp = extras.get(d, {}).get("n_has_perp", 0)
        if n_cells:
            note = (f"{d}: {n_inconclusive} of {n_cells} Brier comparisons are inconclusive at their n; "
                    f"the perp-era cohort (has_perp_at_t0) holds {n_perp} events")
            if 2 * n_inconclusive >= n_cells:
                note += (", so with listings only since Nov 2025 this report is mostly inconclusive, "
                         "as expected for early runs")
            notes.append(note)
    return notes


def train_final(settings: Settings, dataset: pd.DataFrame, *, model_name: str, decision_time: str,
                target: str = "r_24h") -> object:
    """Fit on all non-holdout events that pass the headline filters (min_t0_confidence,
    has_perp_at_t0 when enough events exist), attach the residual band from a walk-forward pass,
    save under settings.models_dir/<decision_time>/<model_name>/ with model.json (decision_time,
    dataset_sha256, git sha, config hash, trained_at, n_events, filters, holdout reference) and
    return the model."""
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
            _, r, _ = _fit_predict(model_name, settings.random_seed, tr, te, feats)
            res = te[Y].to_numpy(dtype=float) - r
            residuals.append(res[np.isfinite(res)])
        if residuals and sum(len(r) for r in residuals) > 0:
            source = label
            break
    model = models_mod.make_model(model_name, seed=settings.random_seed)
    direction = pd.Series(np.where(train[DIR].fillna(0) >= 0, 1.0, -1.0), index=train.index)
    model.fit(_X(train, feats), train[Y].astype(float), direction)
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
    meta = {
        "model": model_name, "decision_time": decision_time, "target": target,
        "dataset_sha256": dataset_sha256(dataset), "git_sha": git["sha"], "git_dirty": git["dirty"],
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
