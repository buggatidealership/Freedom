"""`freedom` command line. Every command is thin: parse args, call a module, print a table.

Exit codes: 0 success; 2 a prerequisite is missing (an artifact another command writes, an API
key, an exhausted daily budget, too few events for a walk-forward fold) — the message names
the command to run or the knob to change first; 3 nothing to predict yet (no release detected
on the live bars); 4 `freedom archive --strict` finished but at least one item errored or lost
bars past the server horizon (see the error column). Anything else (a KeyError from a dataset
without target columns, an unknown model name, ...) is a bug or a data-shape problem and
propagates with its traceback instead of a misleading "run X first" hint.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings, get_settings
from .errors import EventNotFound

app = typer.Typer(help="Post-earnings price-action harness for Hyperliquid equity perpetuals.")
console = Console()

EXIT_PREREQUISITE = 2
EXIT_NOTHING_TO_PREDICT = 3
EXIT_ARCHIVE_INCOMPLETE = 4

# `freedom evaluate --models` default: every registered baseline (eval.runner.DEFAULT_BASELINES,
# so the magnitude forecasts are scored against hist_abs_mean / vol_scaled and the
# continuation question against always_extends / surprise_sign) plus the two learners.
DEFAULT_EVALUATE_MODELS = ("zero,base_rate,historical_mean,hist_abs_mean,vol_scaled,sign_of_reaction,"
                           "always_extends,surprise_sign,linear,lightgbm")

# artifact file/dir name -> the command that creates it (used to complete error messages)
ARTIFACT_COMMANDS: dict[str, str] = {
    "universe.parquet": "universe", "events.parquet": "events", "targets.parquet": "dataset",
    "dataset.parquet": "dataset", "models": "train", "optuna.db": "optimize",
    "live_predictions.parquet": "predict", "archive": "archive",
}


def _csv(arg: str) -> list[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def _blank(v: object) -> bool:
    import pandas as pd

    return v is None or v is pd.NaT or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _cell(v: object) -> str:
    import pandas as pd

    if _blank(v):
        return ""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-3 or abs(v) >= 1e6 else f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def _print_frame(df, title: str) -> None:
    """A DataFrame as a rich table: timestamps to the minute, missing values blank."""
    table = Table(title=title)
    for col in df.columns:
        table.add_column(str(col))
    for row in df.itertuples(index=False):
        table.add_row(*(_cell(v) for v in row))
    console.print(table)


def _print_rows(rows: list[dict], title: str) -> None:
    import pandas as pd

    _print_frame(pd.DataFrame(rows), title)


def _print_kv(pairs: dict, title: str) -> None:
    table = Table(title=title, show_header=False)
    table.add_column("key")
    table.add_column("value")
    for k, v in pairs.items():
        table.add_row(str(k), _cell(v))
    console.print(table)


def _fail(message: str, code: int = EXIT_PREREQUISITE) -> None:
    console.print(message, style="red", markup=False)
    raise typer.Exit(code=code)


def _hint(message: str) -> str:
    """Append `run freedom <cmd> first` to a missing-artifact message that lacks one."""
    if "freedom " in message:
        return message
    for name, cmd in ARTIFACT_COMMANDS.items():
        if name in message:
            return f"{message}: run `freedom {cmd}` first"
    return message


def _require(path: Path, cmd: str) -> None:
    if not path.exists():
        _fail(f"{path} not found: run `freedom {cmd}` first")


@contextmanager
def _guard() -> Iterator[None]:
    """Turn a module's prerequisite failures into a message and exit code 2: budgets, provider
    availability, a missing artifact (FileNotFoundError, which live.ModelNotFound extends) and
    an unknown event id (errors.EventNotFound). Other LookupErrors — KeyError, IndexError — are
    programming or data-shape errors and propagate with a traceback."""
    from .data.base import BudgetExhausted, ProviderUnavailable

    try:
        yield
    except BudgetExhausted as exc:
        _fail(f"daily budget exhausted: {exc}\nRerun the same command after the budget resets "
              "(UTC midnight) or raise FREEDOM_FMP_DAILY_BUDGET / FREEDOM_ALPHAVANTAGE_DAILY_BUDGET.")
    except ProviderUnavailable as exc:
        _fail(f"provider unavailable: {exc}")
    except (FileNotFoundError, EventNotFound) as exc:
        _fail(_hint(str(exc)))


def _decision_times(arg: str) -> list[str]:
    from .schemas import DECISION_TIMES

    dts = _csv(arg)
    bad = [d for d in dts if d not in DECISION_TIMES]
    if bad or not dts:
        _fail(f"unknown decision time(s) {bad}: choose from {', '.join(DECISION_TIMES)}")
    return dts


def _load_dataset(s: Settings):
    import pandas as pd

    _require(s.dataset_path, "dataset")
    return pd.read_parquet(s.dataset_path)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


@app.callback()
def _root() -> None:
    pass


# ---- universe ---------------------------------------------------------------------------------------
@app.command()
def universe(verify_only: bool = typer.Option(False, help="Print only rows needing human verification")) -> None:
    """Pull Hyperliquid markets, classify them, write data/universe.parquet."""
    from . import universe as universe_mod
    from .schemas import U

    s = get_settings()
    with _guard():
        u = universe_mod.build_universe(s, write=not verify_only)  # a report only: leave the file alone
    report = universe_mod.verification_report(u)
    if verify_only:
        _print_frame(report, title=f"{len(report)} markets need human verification "
                                   f"(edit {s.universe_overrides_path})")
        return
    by_kind = u.groupby(U.kind).agg(markets=(U.market, "size"), in_event_universe=(U.in_event_universe, "sum"))
    by_kind = by_kind.reset_index().sort_values("in_event_universe", ascending=False)
    _print_frame(by_kind, title=f"freedom universe: {len(u)} markets, "
                                f"{int(u[U.in_event_universe].sum())} in the event universe -> {s.universe_path}")
    if len(report):
        console.print(f"{len(report)} rows need verification: run `freedom universe --verify-only`",
                      style="yellow", markup=False)


# ---- archive ------------------------------------------------------------------------------------------
@app.command()
def archive(
    intervals: str = typer.Option("1m,5m,15m,1h", help="Comma-separated candle intervals"),
    markets: str = typer.Option("", help="Comma-separated markets; default: every market in data/universe.parquet"),
    strict: bool = typer.Option(False, "--strict", help="Exit 4 when any item reported an error or a coverage "
                                                        "gap (for scheduled runs)"),
) -> None:
    """Archive recent candles and funding for every universe market (run at least every 3 days)."""
    import pandas as pd

    from .data import archive as archive_mod
    from .schemas import U

    s = get_settings()
    if markets:
        names = _csv(markets)
    elif s.universe_path.exists():
        names = pd.read_parquet(s.universe_path)[U.market].dropna().unique().tolist()
    else:
        console.print(f"{s.universe_path} not found: run `freedom universe` first or pass --markets",
                      style="red", markup=False)
        raise typer.Exit(code=2)
    summary = archive_mod.archive_markets(s, names, _csv(intervals))
    _print_frame(summary, title=f"freedom archive: {len(names)} markets -> {s.archive_dir}")
    n_problems = int(summary["error"].notna().sum())
    if n_problems:
        # the table truncates the error column; the full strings are what a cron log needs
        for r in summary[summary["error"].notna()].itertuples(index=False):
            console.print(f"{r.market} {r.interval}: {r.error}", style="yellow", markup=False)
        console.print(f"{n_problems} of {len(summary)} items reported an error or a coverage gap "
                      "(see the error column).", style="yellow")
        if strict:
            raise typer.Exit(code=EXIT_ARCHIVE_INCOMPLETE)


# ---- events -------------------------------------------------------------------------------------------
@app.command()
def events(since: str = typer.Option("2022-01-01", help="Earliest report date (UTC)"),
           underlyings: str = typer.Option("", help="Comma-separated subset")) -> None:
    """Build the earnings event table with resolved release times."""
    from . import events as events_mod
    from .data.base import BudgetExhausted
    from .schemas import E
    from .timeutil import to_utc

    s = get_settings()
    subset = _csv(underlyings) or None
    with _guard():
        try:
            df = events_mod.build_events(s, underlyings=subset, since=to_utc(since, assume_tz="UTC"))
        except BudgetExhausted as exc:
            _fail(f"daily budget exhausted while building events: {exc}\n"
                  f"Rows resolved so far were written to {s.events_path} and the remaining ones are "
                  "marked pending=True. Rerun `freedom events` after the budget resets (UTC midnight) "
                  "or raise FREEDOM_FMP_DAILY_BUDGET; cached responses are reused.")
    n_pending = int(df[E.pending].fillna(False).astype(bool).sum()) if E.pending in df.columns else 0
    n_low = int((df[E.t0_confidence].fillna(0) < s.min_t0_confidence).sum()) if E.t0_confidence in df.columns else 0
    by_source = (df[E.t0_source].fillna("unresolved").value_counts().rename_axis(E.t0_source)
                 .reset_index(name="events")) if E.t0_source in df.columns else None
    if by_source is not None:
        _print_frame(by_source, title=f"freedom events: {len(df)} events for {df[E.underlying].nunique()} "
                                      f"underlyings -> {s.events_path}")
    _print_kv({"pending (budget stopped before these)": n_pending,
               f"below min_t0_confidence {s.min_t0_confidence} (kept, not trained on)": n_low,
               "first t0": df[E.t0].min() if E.t0 in df.columns else None,
               "last t0": df[E.t0].max() if E.t0 in df.columns else None}, title="events summary")


# ---- dataset ------------------------------------------------------------------------------------------
@app.command()
def dataset(decision_times: str = typer.Option("pre_5m,post_15m,post_30m")) -> None:
    """Compute targets and features; write data/dataset.parquet."""
    from . import events as events_mod
    from . import features as features_mod
    from . import targets as targets_mod
    from .schemas import D, T

    s = get_settings()
    dts = _decision_times(decision_times)
    with _guard():
        ev = events_mod.load_events(s)
        tg = targets_mod.build_targets(s, ev)
        ds = features_mod.build_dataset(s, ev, tg, decision_times=dts)
    r24 = T.r("24h")
    rows = []
    for d in dts:
        sub = ds[ds[D.decision_time] == d] if D.decision_time in ds.columns else ds
        rows.append({"decision_time": d, "rows": len(sub),
                     "with r_24h": int(sub[r24].notna().sum()) if r24 in sub.columns else None,
                     "with ar_24h": int(sub[T.ar("24h")].notna().sum()) if T.ar("24h") in sub.columns else None,
                     "features": sum(c.startswith(D.feature_prefix) and not c.endswith(D.missing_suffix)
                                     for c in sub.columns)})
    _print_rows(rows, title=f"freedom dataset: {len(ds)} rows -> {s.dataset_path}")
    if T.price_source in tg.columns:
        src = tg[T.price_source].fillna("none (targets NaN)").value_counts().rename_axis("price_source")
        _print_frame(src.reset_index(name="events"), title=f"targets: {len(tg)} events -> {s.targets_path}")


# ---- evaluate -----------------------------------------------------------------------------------------
def _summary_rows(summary: dict) -> dict[str, list[dict]]:
    """decision time -> one row per model from the evaluation summary, whose metrics are nested
    as results[decision_time][model]["subsets"][subset] (eval.runner). Like the written
    leaderboard, a row shows the headline subset when it has events, else all events, with the
    paired comparison against the best baseline and the fixed-sizing trading result."""
    out: dict[str, list[dict]] = {}
    for d, per_model in (summary.get("results") or {}).items():
        rows = []
        for model, res in (per_model or {}).items():
            subsets = res.get("subsets") or {}
            subset = "headline" if (subsets.get("headline") or {}).get("n") else "all"
            cell = subsets.get(subset) or {}
            comp = (cell.get("comparison") or {}).get("brier") or {}
            trading = (res.get("trading") or {}).get("fixed") or {}
            mean_pnl = (trading.get("mean_pnl") or {}).get("point")
            rows.append({"model": model, "subset": subset, "n": cell.get("n"),
                         "accuracy": cell.get("accuracy"), "brier": cell.get("brier"),
                         "log_loss": cell.get("log_loss"), "spearman_ic": cell.get("spearman_ic"),
                         "mae": cell.get("mae"),
                         "Δbrier vs baseline": comp.get("improvement"),
                         "baseline": comp.get("baseline") or ("(is baseline)" if res.get("is_baseline") else None),
                         "verdict": comp.get("verdict"), "sharpe (fixed)": trading.get("sharpe_like"),
                         "mean net bp (fixed)": mean_pnl * 1e4 if isinstance(mean_pnl, int | float) else None})
        out[d] = rows
    return out


def _print_summary(summary: dict, title: str, report_dir: Path | None = None) -> None:
    """One table per decision time (eval's nested results flattened), then the run's scalars
    and the path of the leaderboard eval wrote."""
    tables = _summary_rows(summary)
    for d, rows in tables.items():
        if rows:
            _print_rows(rows, title=f"{title}: {d}")
    scalars = {k: v for k, v in summary.items() if isinstance(v, str | int | float | bool | type(None))}
    if report_dir is not None:
        scalars["leaderboard"] = report_dir / "leaderboard.md"
    if scalars or not tables:
        _print_kv(scalars or {"keys": ", ".join(summary)}, title=title)


@app.command()
def evaluate(models: str = typer.Option(DEFAULT_EVALUATE_MODELS),
             decision_times: str = typer.Option("pre_5m,post_30m"),
             final: bool = typer.Option(False, "--final", help="Also score the pinned holdout season once (logged)"),
             target: str = typer.Option("r_24h", help="Training target: r_24h or ar_24h")) -> None:
    """Walk-forward evaluation with cost-aware simulation; writes reports/<run_id>/."""
    from . import eval as eval_mod

    s = get_settings()
    dts = _decision_times(decision_times)
    ds = _load_dataset(s)
    with _guard():
        try:
            summary = eval_mod.evaluate(s, ds, model_names=_csv(models), decision_times=dts, final=final,
                                        target=target)
        except eval_mod.HoldoutNotReady as exc:
            _fail(f"holdout not ready: {exc}")
        except ValueError as exc:  # no fold / no scorable rows: the same prerequisite `optimize` maps to exit 2
            _fail(f"{exc}\nWith a small dataset lower FREEDOM_MIN_TRAIN_EVENTS (default {s.min_train_events}; "
                  "the first-run reports used 12) or add underlyings / extend `freedom events --since`.")
    run_id = summary.get("run_id")
    _print_summary(summary, title="freedom evaluate" + (" --final" if final else ""),
                   report_dir=s.reports_dir / str(run_id) if run_id else None)
    if final:
        console.print(f"holdout season {s.holdout_season} has now been scored {_count_lines(s.holdout_log_path)} "
                      f"time(s) ({s.holdout_log_path}); discount it accordingly.", style="yellow", markup=False)


# ---- optimize -----------------------------------------------------------------------------------------
@app.command()
def optimize(decision_times: str = typer.Option("pre_5m,post_30m"),
             n_trials: int = typer.Option(50),
             objective: str = typer.Option("brier"),
             timeout_seconds: int | None = typer.Option(None, help="Per-study wall-clock limit")) -> None:
    """Optuna search over models, features, target variant and training window: one study per decision time; never scores the holdout."""
    from . import optimize as optimize_mod

    s = get_settings()
    dts = _decision_times(decision_times)
    if objective not in optimize_mod.OBJECTIVES:
        _fail(f"unknown objective {objective!r}: choose from {', '.join(optimize_mod.OBJECTIVES)}")
    ds = _load_dataset(s)
    rows = []
    for d in dts:
        with _guard():
            try:
                res = optimize_mod.run_study(s, ds, decision_time=d, n_trials=n_trials, objective=objective,
                                             timeout_seconds=timeout_seconds)
            except (ValueError, optimize_mod.TestSetMismatch) as exc:
                _fail(f"{d}: {exc}")
        bp = res.get("best_params") or {}
        rows.append({"decision_time": d, "study": res["study"], "trials": res["n_trials"],
                     f"best {objective}": res["best_value"], "model": bp.get("model"),
                     "target": bp.get("target"), "window": bp.get("train_window_seasons"),
                     "best baseline": res["baseline_name"], "baseline value": res["baseline_value"],
                     "improvement": res["improvement"], "p_noise": res["p_noise"],
                     "events": res["n_events"], "folds": res["n_folds"]})
        console.print(f"{d}: leaderboard -> {res['report_dir']}/leaderboard.md", markup=False)
    _print_rows(rows, title=f"freedom optimize: objective {objective}, holdout {s.holdout_season} never scored")


# ---- train --------------------------------------------------------------------------------------------
@app.command()
def train(model: str = typer.Option("lightgbm"), decision_time: str = typer.Option("post_30m"),
          target: str = typer.Option("r_24h", help="Training target: r_24h or ar_24h")) -> None:
    """Fit a model on all non-holdout events and save it under data/models/<decision_time>/<model>/."""
    import json

    from . import eval as eval_mod

    s = get_settings()
    _decision_times(decision_time)
    ds = _load_dataset(s)
    with _guard():
        eval_mod.train_final(s, ds, model_name=model, decision_time=decision_time, target=target)
    path = s.models_dir / decision_time / model
    meta_path = path / "model.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    _print_kv({"saved to": path, **{k: v for k, v in meta.items() if isinstance(v, str | int | float | bool)}},
              title=f"freedom train: {model} @ {decision_time}")


# ---- predict ------------------------------------------------------------------------------------------
def _print_prediction(res: dict) -> None:
    from .schemas import E

    row, sched = res["row"], res["schedule"]
    head = {"event": row[E.event_id], "market": row[E.market], "decision": row["decision_time"],
            "as_of": row["as_of"], "t0 used": row["t0_used"], "t0 source": row["t0_source_live"],
            "off_schedule": row["off_schedule"], "schedule": sched["note"],
            "p_up": row["p_up"], "expected r_24h": row["r_hat"],
            "10/90 % band": f"{_cell(row['r_lo'])} .. {_cell(row['r_hi'])}",
            "magnitude": row["magnitude_hat"], "model_id": row["model_id"],
            "sources": row["sources_used"], "bar source": row["bar_source"],
            "input lag s (hyperliquid / fmp / sec)": f"{_cell(row['input_lag_s_hyperliquid'])} / "
                                                     f"{_cell(row['input_lag_s_fmp'])} / {_cell(row['input_lag_s_sec'])}",
            "features (missing)": f"{row['n_features']} ({row['n_features_missing']})"}
    _print_kv(head, title="freedom predict")
    if row["off_schedule"]:
        console.print("OFF SCHEDULE: this row is recorded but must not be traded.", style="yellow")
    if row.get("replay"):
        console.print(f"REPLAY: `now` was overridden to {_cell(row.get('now_override'))}; the row is recorded "
                      "with replay=True and does not count as a live prediction.", style="yellow", markup=False)
    if res["contributions"]:
        _print_rows(res["contributions"], title="top feature contributions (importance-ranked)")
    _print_kv(res["consensus"], title="consensus provenance")
    meta = res["model_meta"]
    record = {k: meta[k] for k in ("n_events", "trained_at", "dataset_sha256", "holdout_reference",
                                   "walk_forward", "residual_band") if k in meta}
    if record:
        _print_kv(record, title=f"model record for {row['decision_time']}")


@app.command()
def predict(event: str = typer.Option(..., "--event", help="event_id, e.g. NVDA:2026-07 (see `freedom upcoming`)"),
            decision: str = typer.Option("post_30m", "--decision", help="pre_5m | post_1m | post_15m | post_30m | post_60m"),
            model: str | None = typer.Option(None, "--model", help="Trained model name under data/models/<decision>/"),
            now: str | None = typer.Option(None, "--now", help="Override the current instant (UTC ISO) for replays"),
            no_append: bool = typer.Option(False, "--no-append", help="Do not append to data/live_predictions.parquet")) -> None:
    """Live prediction for one event at one decision time; appends to data/live_predictions.parquet."""
    from . import live

    s = get_settings()
    _decision_times(decision)
    with _guard():
        try:
            res = live.predict_event(s, event_id=event, decision=decision, model_name=model, now=now,
                                     append=not no_append)
        except live.ReleaseNotDetected as exc:
            _fail(f"nothing to predict yet: {exc}", code=EXIT_NOTHING_TO_PREDICT)
    _print_prediction(res)
    if not no_append:
        console.print(f"appended to {live.live_predictions_path(s)}", markup=False)


# ---- upcoming -----------------------------------------------------------------------------------------
@app.command()
def upcoming(days: int = typer.Option(14)) -> None:
    """List upcoming earnings events in the event universe with the event_id `freedom predict --event` takes."""
    from . import events as events_mod
    from . import live

    s = get_settings()
    with _guard():
        df = events_mod.upcoming_events(s, days=days)
    if df is None or len(df) == 0:
        console.print(f"no universe events in the next {days} days", markup=False)
        return
    _print_frame(live.with_event_ids(df), title=f"freedom upcoming: {len(df)} events in the next {days} days")


# ---- status -------------------------------------------------------------------------------------------
def _archive_coverage(s: Settings) -> list[dict]:
    """Per interval: markets archived, rows, earliest first bar, newest last bar and the market
    whose last bar is oldest (the one falling behind)."""
    import pandas as pd

    from .data.archive import CANDLES_SUBDIR, archived_rows

    root = s.archive_dir / CANDLES_SUBDIR
    per_interval: dict[str, list[tuple[str, int, pd.Timestamp, pd.Timestamp]]] = {}
    if root.exists():
        for market_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(market_dir.glob("*.parquet")):
                n = archived_rows(f)
                if n == 0:
                    continue
                t = pd.to_datetime(pd.read_parquet(f, columns=["t"])["t"], utc=True)
                per_interval.setdefault(f.stem, []).append((market_dir.name, n, t.min(), t.max()))
    rows = []
    for interval, items in sorted(per_interval.items()):
        lagging = min(items, key=lambda x: x[3])
        rows.append({"interval": interval, "markets": len(items), "rows": sum(i[1] for i in items),
                     "first bar": min(i[2] for i in items), "newest bar": max(i[3] for i in items),
                     "most stale market": f"{lagging[0]} @ {_cell(lagging[3])}"})
    return rows


@app.command()
def status() -> None:
    """Show configured keys, budgets used today, archive coverage and dataset sizes."""
    from .data.archive import archived_rows
    from .data.base import DailyBudget

    s = get_settings()
    budgets = {"fmp": (s.fmp_api_key, s.fmp_daily_budget), "alphavantage": (s.alphavantage_api_key, s.alphavantage_daily_budget)}
    key_rows = [{"provider": name, "key": "present" if key else "missing",
                 "used today": DailyBudget(name, limit, s.data_dir).used_today(), "daily budget": limit}
                for name, (key, limit) in budgets.items()]
    key_rows.append({"provider": "anthropic", "key": "present" if s.anthropic_api_key else "missing",
                     "used today": None, "daily budget": None})
    key_rows.append({"provider": "hyperliquid / sec / nasdaq", "key": "not needed", "used today": None,
                     "daily budget": None})
    _print_rows(key_rows, title=f"freedom status: data_dir {s.data_dir}, reports_dir {s.reports_dir}")
    artifacts = {"universe": (s.universe_path, "universe"), "events": (s.events_path, "events"),
                 "targets": (s.targets_path, "dataset"), "dataset": (s.dataset_path, "dataset"),
                 "live predictions": (s.data_dir / "live_predictions.parquet", "predict")}
    art_rows = [{"artifact": name, "path": path, "rows": archived_rows(path) if path.exists() else None,
                 "status": "ok" if path.exists() else f"missing: run `freedom {cmd}`"}
                for name, (path, cmd) in artifacts.items()]
    art_rows.append({"artifact": "optuna studies", "path": s.optuna_db, "rows": None,
                     "status": "ok" if s.optuna_db.exists() else "missing: run `freedom optimize`"})
    trained = sorted(str(p.relative_to(s.models_dir)) for p in s.models_dir.glob("*/*/model.json")) if s.models_dir.exists() else []
    art_rows.append({"artifact": "trained models", "path": s.models_dir, "rows": len(trained),
                     "status": ", ".join(p.rsplit("/", 1)[0] for p in trained) or "none: run `freedom train`"})
    _print_rows(art_rows, title="artifacts")
    coverage = _archive_coverage(s)
    if coverage:
        _print_rows(coverage, title=f"archive coverage ({s.archive_dir})")
    else:
        console.print(f"archive empty: run `freedom archive` ({s.archive_dir})", markup=False)
    _print_kv({"holdout season": s.holdout_season,
               "holdout scorings so far": _count_lines(s.holdout_log_path),
               "min_t0_confidence": s.min_t0_confidence, "random_seed": s.random_seed},
              title="evaluation settings")


if __name__ == "__main__":
    app()
