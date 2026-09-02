"""Feature groups with as_of discipline.

Every group is a pure function `(ctx: FeatureContext) -> dict[str, float | None]` registered
with `@feature_group(name, admissible=("pre", "post"))`. `FeatureContext.as_of` is the decision
instant; any input bar with t_end > as_of must be excluded by the group (tests include a
look-ahead trap). Feature columns are written as schemas.D.feature_prefix + name; missing values
get a companion <name>__missing indicator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from ..config import Settings

GroupFn = Callable[["FeatureContext"], dict[str, float | None]]
REGISTRY: dict[str, tuple[GroupFn, tuple[str, ...]]] = {}


def feature_group(name: str, admissible: tuple[str, ...] = ("pre", "post")):
    def deco(fn: GroupFn) -> GroupFn:
        REGISTRY[name] = (fn, admissible)
        return fn
    return deco


@dataclass
class FeatureContext:
    event: pd.Series  # schemas.E row
    as_of: pd.Timestamp  # UTC decision instant
    decision_time: str  # key of schemas.DECISION_TIMES
    bars: pd.DataFrame | None = None  # fine bars (schemas.C) for the instrument, truncated to as_of
    daily: pd.DataFrame | None = None  # daily bars of the underlying, truncated to as_of
    market_bars: pd.DataFrame | None = None  # benchmark fine bars, truncated
    market_daily: pd.DataFrame | None = None
    history: pd.DataFrame | None = None  # this underlying's past events + targets with t0 < as_of
    perp_ctx: dict | None = None  # funding/premium/OI snapshot at or before as_of
    extra: dict = field(default_factory=dict)


def build_features(ctx: FeatureContext, groups: list[str] | None = None) -> dict[str, float | None]:
    """Run admissible groups (pre-only groups are skipped at post decision times only if they
    are marked so; post-only groups are skipped at pre decision times)."""
    raise NotImplementedError


def build_dataset(settings: Settings, events: pd.DataFrame, targets: pd.DataFrame,
                  *, decision_times: list[str] | None = None, groups: list[str] | None = None,
                  write: bool = True) -> pd.DataFrame:
    """One row per (event, decision_time) with schemas.D columns + f_* features + targets."""
    raise NotImplementedError
