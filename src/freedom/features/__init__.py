"""Feature groups with as_of discipline.

Every group is a pure function `(ctx: FeatureContext) -> dict[str, float | None]` registered
with `@feature_group(name, admissible=("pre", "post"))`. `FeatureContext.as_of` is the decision
instant; any input bar with t_end > as_of must be excluded by the group (tests include a
look-ahead trap). Feature columns are written as schemas.D.feature_prefix + name; missing values
get a companion <name>__missing indicator.

as_of gates the harness's own event/target store too: `ctx.history` must come from
`history_view(events, targets, underlying, as_of)` (rows with t0 + horizon <= as_of) and
`build_features` asserts that invariant. Second trap required in tests: at post_60m, setting the
event's own r_24h to +5.0 must leave every feature unchanged.

Groups in v1 (docs/design.md §6): calendar, pre_price, history, market, perp_state (pre);
surprise, reaction (post). Text features are deferred.
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
    bars: pd.DataFrame | None = None  # fine bars (schemas.C) for the instrument; may extend past as_of, groups must cut
    daily: pd.DataFrame | None = None  # daily bars of the underlying
    market_bars: pd.DataFrame | None = None  # benchmark fine bars
    market_daily: pd.DataFrame | None = None
    history: pd.DataFrame | None = None  # from history_view(): events+targets with t0 + horizon <= as_of
    perp_ctx: pd.DataFrame | None = None  # archived ctx snapshots (funding, premium, OI) with t <= as_of
    horizon_hours: int = 24
    extra: dict = field(default_factory=dict)


def phase_of(decision_time: str) -> str:
    """'pre' for negative offsets, 'post' otherwise (schemas.DECISION_TIMES)."""
    from ..schemas import DECISION_TIMES

    return "pre" if DECISION_TIMES[decision_time] < 0 else "post"


def history_view(events: pd.DataFrame, targets: pd.DataFrame, underlying: str,
                 as_of: pd.Timestamp, horizon_hours: int = 24) -> pd.DataFrame:
    """Same-underlying event rows joined with their targets, restricted to
    t0 + horizon_hours <= as_of. This is the ONLY way history reaches a feature group."""
    raise NotImplementedError


def build_features(ctx: FeatureContext, groups: list[str] | None = None) -> dict[str, float | None]:
    """Run the registered groups admissible at phase_of(ctx.decision_time). Asserts the history
    invariant. Output keys carry the schemas.D.feature_prefix and a __missing companion."""
    raise NotImplementedError


def build_dataset(settings: Settings, events: pd.DataFrame, targets: pd.DataFrame,
                  *, decision_times: list[str] | None = None, groups: list[str] | None = None,
                  write: bool = True) -> pd.DataFrame:
    """One row per (event, decision_time) with schemas.D columns, f_* features, the target
    columns, and event metadata needed downstream (t0, t0_source, kind, timing, has_perp_at_t0,
    season). Rows for events whose targets are all NaN are kept (they are still usable for
    intermediate checkpoints) but flagged. Bars are loaded through targets.loaders so the
    price source matches the targets."""
    raise NotImplementedError
