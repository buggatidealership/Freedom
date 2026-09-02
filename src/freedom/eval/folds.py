"""Walk-forward folds by earnings season (docs/design.md §8).

A season is the calendar quarter of `t0` in UTC (schemas.season_of). Folds are expanding:
the test fold is one season, the training set is every event released before that season's
first instant minus the embargo, so same-day events of other names are never on both sides.
The pinned holdout season is excluded from every fold and returned separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..schemas import UTC, E, season_of

HOLDOUT_FOLD = -1  # `Fold.fold` of the holdout fold; walk-forward folds are numbered from 0


@dataclass(frozen=True)
class Fold:
    fold: int
    train_idx: pd.Index
    test_idx: pd.Index
    test_season: str  # e.g. "2026Q2"


def season_start(season: str) -> pd.Timestamp:
    """First instant (UTC) of a season label such as '2026Q3'."""
    year, quarter = season.split("Q")
    q = int(quarter)
    if q not in (1, 2, 3, 4):
        raise ValueError(f"bad season label {season!r}")
    return pd.Timestamp(year=int(year), month=3 * (q - 1) + 1, day=1, tz=UTC)


def t0_utc(events: pd.DataFrame) -> pd.Series:
    """The t0 column as tz-aware UTC (NaT preserved)."""
    return pd.to_datetime(events[E.t0], utc=True)


def seasons_of(t0: pd.Series) -> pd.Series:
    """schemas.season_of applied row-wise; None where t0 is NaT."""
    return t0.map(lambda t: season_of(t) if pd.notna(t) else None).astype(object)


def walk_forward_folds(events: pd.DataFrame, *, min_train: int, embargo_days: int,
                       holdout_season: str | None) -> tuple[list[Fold], Fold | None]:
    """Expanding-window folds by season (schemas.season_of(t0)). Test fold = one season;
    train = all events with t0 < season_start - embargo_days. The holdout season (if it exists
    in `events`) is returned separately and never appears in the fold list. `events` must
    carry event_id and t0; folds index into it. Seasons with fewer than min_train prior events
    are skipped."""
    for col in (E.event_id, E.t0):
        if col not in events.columns:
            raise ValueError(f"events frame needs a {col!r} column")
    t0 = t0_utc(events)
    season = seasons_of(t0)
    valid = t0.notna()
    if holdout_season:
        is_holdout = (season == holdout_season) & valid
    else:
        is_holdout = pd.Series(False, index=events.index)
    embargo = pd.Timedelta(days=embargo_days)

    folds: list[Fold] = []
    for label in sorted(season[valid & ~is_holdout].dropna().unique()):
        start = season_start(label)
        train = valid & ~is_holdout & (t0 < start - embargo)
        if int(train.sum()) < min_train:
            continue
        test = valid & (season == label)
        folds.append(Fold(len(folds), events.index[train], events.index[test], str(label)))

    holdout: Fold | None = None
    if holdout_season and bool(is_holdout.any()):
        start = season_start(holdout_season)
        train = valid & ~is_holdout & (t0 < start - embargo)
        holdout = Fold(HOLDOUT_FOLD, events.index[train], events.index[is_holdout], holdout_season)
    return folds, holdout
