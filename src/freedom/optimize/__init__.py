"""Optuna study over models, features, decision time and target variant."""

from __future__ import annotations

import pandas as pd

from ..config import Settings


def run_study(settings: Settings, dataset: pd.DataFrame, *, n_trials: int, objective: str = "brier",
              study_name: str = "freedom", timeout_seconds: int | None = None) -> dict:
    """Objective evaluated on walk-forward folds excluding the holdout season; the best trial is
    then scored once on the holdout and both numbers are written to reports/optimize/<study>/."""
    raise NotImplementedError


def leaderboard(settings: Settings, study_name: str = "freedom") -> pd.DataFrame:
    raise NotImplementedError
