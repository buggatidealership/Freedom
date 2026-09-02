"""Walk-forward evaluation, metrics, cost-aware simulation, bootstrap and reports.

Decisions from review (docs/design.md §8): pinned holdout season excluded from all folds and
scored only by `evaluate(..., final=True)` which appends to settings.holdout_log_path; fills at
the OPEN of the first bar starting at or after the signal instant (never the signal bar's
close) with fill_lag recorded and trades beyond max_fill_lag_minutes dropped; execution cost
per leg = slippage_floor_bps + slippage_range_coeff * range_bps(execution bar) + taker fee;
funding accrued only with archived coverage (funding_source recorded); equal_split capital
rule with a gross exposure cap; metrics reported for has_perp_at_t0 (headline) and all events,
stratified by t0_source, kind, timing; run_id = <UTC ts>-<sha256(dataset)[:8]>.
Every cell reports n and the minimum detectable improvement over the best baseline; the
summary labels a cell "inconclusive" unless the bootstrap interval excludes it. Sizing
variants: fixed, by_confidence, by_magnitude, magnitude_gate. `final=True` refuses while any
holdout-season event has t0 + 24h in the future or targets pending.

Layout: folds.py (seasons, expanding folds, holdout), metrics.py (metrics, calibration,
residual band, MDE, bootstrap), sim.py (fills, costs, funding, sizing, equal_split
portfolio), report.py (provenance, JSON, leaderboard, files), runner.py (evaluate,
train_final). Everything public is re-exported here.
"""

from __future__ import annotations

from .folds import HOLDOUT_FOLD, Fold, season_start, seasons_of, walk_forward_folds
from .metrics import (
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
    dataset_sha256,
    git_info,
    leaderboard_markdown,
    make_run_id,
    to_jsonable,
    write_reports,
)
from .runner import (
    DEFAULT_BASELINES,
    HoldoutNotReady,
    baseline_names,
    check_holdout_ready,
    evaluate,
    prepare_dataset,
    subset_masks,
    train_final,
    verdict,
)
from .sim import (
    FUNDING_ARCHIVE,
    FUNDING_NONE,
    SIZINGS,
    TRADE_COLUMNS,
    archive_funding_loader,
    equal_split_weights,
    fill_price,
    funding_sum,
    leg_cost_bps,
    loader_paths,
    memoised_funding,
    portfolio_metrics,
    position_size,
    prepare_funding,
    simulate,
)

__all__ = [
    "DEFAULT_BASELINES", "FUNDING_ARCHIVE", "FUNDING_NONE", "HOLDOUT_FOLD", "SIZINGS", "TRADE_COLUMNS",
    "Fold", "HoldoutNotReady", "archive_funding_loader", "baseline_names", "bootstrap_ci",
    "bootstrap_distribution", "brier_scores", "calibration_table", "check_holdout_ready",
    "classification_metrics", "dataset_sha256", "equal_split_weights", "evaluate", "fill_price",
    "funding_sum", "git_info", "hit_scores", "leaderboard_markdown", "leg_cost_bps", "loader_paths",
    "make_run_id", "memoised_funding", "min_detectable_improvement", "portfolio_metrics", "position_size",
    "prepare_dataset", "prepare_funding", "regression_metrics", "residual_band", "season_start", "seasons_of",
    "simulate", "spearman", "subset_masks", "to_jsonable", "train_final", "verdict",
    "walk_forward_folds", "write_reports",
]
