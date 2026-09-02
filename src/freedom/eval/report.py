"""Provenance, JSON conversion and report files for `freedom evaluate`.

reports/<run_id>/summary.json is plain JSON (numpy scalars, timestamps and NaN converted);
predictions.parquet holds every out-of-sample prediction; trades.parquet every simulated
trade; leaderboard.md the human-readable table per decision time.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings

SECRET_FIELDS = {"fmp_api_key", "alphavantage_api_key", "anthropic_api_key"}
SUMMARY_FILE = "summary.json"
PREDICTIONS_FILE = "predictions.parquet"
TRADES_FILE = "trades.parquet"
LEADERBOARD_FILE = "leaderboard.md"


def utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def dataset_sha256(dataset: pd.DataFrame) -> str:
    """Deterministic content hash of the dataset frame (column names + row hashes)."""
    h = hashlib.sha256()
    h.update("|".join(map(str, dataset.columns)).encode())
    try:
        rows = pd.util.hash_pandas_object(dataset, index=False).to_numpy()
        h.update(np.ascontiguousarray(rows).tobytes())
    except TypeError:  # unhashable cell types: fall back to a textual dump
        h.update(dataset.to_json(date_format="iso", orient="records").encode())
    return h.hexdigest()


def make_run_id(dataset_hash: str, now: pd.Timestamp | None = None) -> str:
    """<UTC yyyymmddTHHMMSSZ>-<sha256(dataset)[:8]>."""
    ts = (now or utcnow()).tz_convert("UTC")
    return f"{ts.strftime('%Y%m%dT%H%M%SZ')}-{dataset_hash[:8]}"


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    """{'sha': <commit or 'unknown'>, 'dirty': bool | None} from `git rev-parse` when available.
    Defaults to the package's own directory so a checkout is found from any working directory;
    an installed wheel reports 'unknown'."""
    cwd = cwd or Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                             timeout=5, check=False)
        if sha.returncode != 0:
            return {"sha": "unknown", "dirty": None}
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
                                capture_output=True, text=True, timeout=5, check=False)
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return {"sha": sha.stdout.strip(), "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"sha": "unknown", "dirty": None}


def public_settings(settings: Settings) -> dict[str, Any]:
    """Settings without API keys, JSON-ready (paths as strings)."""
    return settings.model_dump(mode="json", exclude=SECRET_FIELDS)


def config_hash(settings: Settings) -> str:
    return hashlib.sha256(json.dumps(public_settings(settings), sort_keys=True).encode()).hexdigest()


def library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in ("freedom", "pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "optuna"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "unknown"
    return out


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars and containers to JSON-serialisable values.
    NaN and NaT become None; timestamps become ISO strings; frames become record lists."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (pd.Timestamp, datetime)):
        if pd.isna(obj):
            return None
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    if obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.to_dict("records")]
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, (list, tuple, set, frozenset, np.ndarray, pd.Index)):
        return [to_jsonable(v) for v in list(obj)]
    if hasattr(obj, "item"):
        return to_jsonable(obj.item())
    return str(obj)


def count_holdout_scorings(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def append_holdout_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")


def last_holdout_scoring(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


# ---- leaderboard -----------------------------------------------------------------------------
def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "–"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}f}"
    return str(v)


def _fmt_ci(ci: Any, digits: int = 3) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return ""
    return f" [{_fmt(ci[0], digits)}, {_fmt(ci[1], digits)}]"


def leaderboard_markdown(summary: dict[str, Any]) -> str:
    """Markdown leaderboard: one table per decision time on the headline subset (fallback:
    all events), with the comparison against the best baseline and the MDE at that n."""
    hold = summary.get("holdout") or {}
    git = summary.get("git") or {}
    lines = [f"# Leaderboard — run {summary.get('run_id')}", ""]
    lines.append(f"dataset sha256 `{summary.get('dataset_sha256')}` · git `{git.get('sha')}`"
                 f"{' (dirty)' if git.get('dirty') else ''} · target `{summary.get('target')}` · "
                 f"holdout {hold.get('season')} scored {hold.get('scorings_after', 0)} time(s)"
                 f"{' · FINAL: holdout scored in this run' if summary.get('final') else ''}")
    lines.append("")
    lines.append("Comparisons are against the best baseline per metric (named per row). A cell "
                 "reads `improves` only when the paired bootstrap interval excludes 0 from above, "
                 "`worse` when it lies below 0, `not_predictable` only when it excludes the minimum "
                 "detectable improvement (MDE) at that n, and `inconclusive at n = …` otherwise.")
    lines.append("")
    for d, per_model in (summary.get("results") or {}).items():
        lines.append(f"## {d}")
        lines.append("")
        lines.append("| model | subset | n | accuracy [95% CI] | brier [95% CI] | IC | MAE | Δbrier vs best baseline [95% CI] | MDE (brier) | verdict | sharpe (fixed) | mean net (bp, fixed) |")
        lines.append("|---|---|---:|---|---|---:|---:|---|---:|---|---:|---:|")
        for model, res in per_model.items():
            subsets = res.get("subsets") or {}
            subset = "headline" if subsets.get("headline", {}).get("n") else "all"
            cell = subsets.get(subset) or {}
            comp = (cell.get("comparison") or {}).get("brier")
            trading = (res.get("trading") or {}).get("fixed") or {}
            mean_pnl = (trading.get("mean_pnl") or {}).get("point")
            if comp:
                delta = f"{_fmt(comp.get('improvement'), 4)}{_fmt_ci(comp.get('ci'), 4)} vs {comp.get('baseline')}"
                verdict = comp.get("verdict") or ""
            else:
                delta, verdict = ("baseline" if res.get("is_baseline") else "–"), ""
            lines.append(
                f"| {model} | {subset} | {_fmt(cell.get('n'))} | {_fmt(cell.get('accuracy'))}"
                f"{_fmt_ci((cell.get('ci') or {}).get('accuracy'))} | {_fmt(cell.get('brier'), 4)}"
                f"{_fmt_ci((cell.get('ci') or {}).get('brier'), 4)} | {_fmt(cell.get('spearman_ic'))} | "
                f"{_fmt(cell.get('mae'), 4)} | {delta} | {_fmt((cell.get('mde') or {}).get('brier'), 4)} | "
                f"{verdict} | {_fmt(trading.get('sharpe_like'), 2)} | "
                f"{_fmt(mean_pnl * 1e4 if isinstance(mean_pnl, (int, float)) and mean_pnl is not None else None, 1)} |")
        lines.append("")
    notes = summary.get("notes") or []
    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {n}" for n in notes)
        lines.append("")
    return "\n".join(lines)


def write_reports(settings: Settings, run_id: str, summary: dict[str, Any], predictions: pd.DataFrame,
                  trades: pd.DataFrame) -> Path:
    out_dir = settings.reports_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUMMARY_FILE).write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    predictions.to_parquet(out_dir / PREDICTIONS_FILE, index=False)
    trades.to_parquet(out_dir / TRADES_FILE, index=False)
    (out_dir / LEADERBOARD_FILE).write_text(leaderboard_markdown(summary))
    return out_dir
