"""Walk-forward evaluation, metrics, cost-aware simulation, bootstrap and reports.

Decisions from review (docs/design.md §8): pinned holdout season excluded from all folds and
scored only by `evaluate(..., final=True)` which appends to settings.holdout_log_path; fills at
the OPEN of the first bar starting at or after the signal instant (never the signal bar's
close) with fill_lag recorded and trades beyond max_fill_lag_minutes dropped; execution cost
per leg = slippage_floor_bps + slippage_range_coeff * range_bps(execution bar) + taker fee;
funding accrued only with archived coverage (funding_source recorded); equal_split capital
rule with a gross exposure cap (sim.CAPITAL_RULE: a constant weight per position = cap / peak
concurrency over its interval); metrics reported for has_perp_at_t0 (headline) and all
events, stratified by t0_source, kind, timing; run_id = <UTC ts>-<sha256(dataset.parquet)[:8]>
(content hash of the frame only when no file exists; summary['dataset_hash_source'] says which).
Every cell reports n, the minimum detectable improvement over the best baseline derived from
the paired comparison's own standard error (the closed form is only a labelled upper bound
where no comparison exists) and the bootstrap resampling scheme: blocks by season with at
least MIN_BLOCKS seasons, else by UTC day of t0, else iid rows, so a single-season block (the
holdout) never yields a zero-width interval. The summary labels a cell "inconclusive" unless
the interval excludes the MDE. Sizing variants: fixed, by_confidence, by_magnitude,
magnitude_gate, the latter two driven by the model's `predict_magnitude` forecast
(predictions column magnitude_hat, scored as magnitude MAE against the best baseline).
`final=True` refuses until the holdout season is closed (now >= next season start + horizon),
while any holdout-season event has t0 + 24h in the future or targets pending, or when the
events calendar lists a holdout-season event the dataset lacks. Provider aborts (budget
exhausted, unavailable) from the bar loader propagate instead of becoming untraded events.

Layout: folds.py (seasons, expanding folds, holdout), metrics.py (metrics, calibration,
residual band, MDE, bootstrap), sim.py (fills, costs, funding, sizing, equal_split
portfolio), report.py (provenance, JSON, leaderboard, files), runner.py (evaluate,
train_final). Everything public is re-exported here.
"""

from __future__ import annotations

from .folds import HOLDOUT_FOLD, Fold, season_end, season_start, seasons_of, walk_forward_folds
from .metrics import (
    MIN_BLOCKS,
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
    dataset_sha256,
    git_info,
    leaderboard_markdown,
    make_run_id,
    to_jsonable,
    write_reports,
)
from .runner import (
    DEFAULT_BASELINES,
    TRADING_SUBSETS,
    HoldoutNotReady,
    baseline_names,
    check_holdout_ready,
    estimate_source_counts,
    evaluate,
    non_point_in_time_in_scope,
    prepare_dataset,
    subset_masks,
    train_final,
    verdict,
)
from .sim import (
    CAPITAL_RULE,
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
    "CAPITAL_RULE", "DEFAULT_BASELINES", "FUNDING_ARCHIVE", "FUNDING_NONE", "HOLDOUT_FOLD", "MIN_BLOCKS",
    "SIZINGS", "TRADE_COLUMNS", "TRADING_SUBSETS", "Fold", "HoldoutNotReady", "archive_funding_loader",
    "baseline_names", "bootstrap_ci", "bootstrap_distribution", "brier_scores", "calibration_table",
    "check_holdout_ready", "choose_blocks", "classification_metrics", "dataset_sha256", "equal_split_weights",
    "estimate_source_counts", "evaluate", "fill_price", "funding_sum", "git_info", "hit_scores",
    "leaderboard_markdown", "leg_cost_bps", "loader_paths", "make_run_id", "memoised_funding",
    "min_detectable_improvement", "non_point_in_time_in_scope", "paired_mde", "paired_se",
    "portfolio_metrics", "position_size", "prepare_dataset", "prepare_funding", "regression_metrics",
    "residual_band", "season_end", "season_start", "seasons_of", "simulate", "spearman", "subset_masks",
    "to_jsonable", "train_final", "verdict", "walk_forward_folds", "write_reports",
]
