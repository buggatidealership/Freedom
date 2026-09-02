"""Optuna study over models, features and target variant — one study per decision time.

Decision time and the t0-confidence floor are NOT search dimensions (docs/design.md §9). The
objective is the walk-forward metric on folds that exclude the pinned holdout season; the
holdout is never scored here. Every trial is scored on the identical event-id set for its
decision time; the study aborts if a trial's test set differs (hash recorded).
"""

from __future__ import annotations

import pandas as pd

from ..config import Settings


def run_study(settings: Settings, dataset: pd.DataFrame, *, decision_time: str, n_trials: int,
              objective: str = "brier", timeout_seconds: int | None = None) -> dict:
    """Search space: model family (linear, lightgbm, ensemble) and hyper-parameters, feature
    groups on/off (only groups admissible at the decision time), target variant (r_24h vs
    ar_24h), training-window length in seasons. Persists to settings.optuna_db under study name
    f"freedom_{decision_time}_{objective}". Writes reports/optimize/<study>/leaderboard.md and
    best_params.json; returns {study, best_value, best_params, n_trials, baseline_value,
    improvement, p_noise} where p_noise is the bootstrap probability that the improvement over
    the best baseline is noise."""
    raise NotImplementedError


def leaderboard(settings: Settings, decision_time: str, objective: str = "brier") -> pd.DataFrame:
    raise NotImplementedError
