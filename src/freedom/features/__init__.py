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
surprise, reaction (post). Text features are deferred. The groups live in `features.groups`
(imported at the bottom of this module so that importing the package fills REGISTRY); the
provider plumbing that fills a FeatureContext lives in `features.loaders`.

Conventions every group follows (see groups.py):
* bars are cut by END time: a bar with t_end > the cut instant is never used;
* "pre" groups are anchored at min(as_of, t0 - p0 buffer), so at a post decision time they
  carry the same pre-release value as at pre_5m and never contain the reaction;
* group output keys are short snake_case WITHOUT the f_ prefix; None means unavailable.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..schemas import CHECKPOINTS, DECISION_TIMES, SCHEMA_VERSION, D, E, T, season_of
from ..targets import P0_BUFFER_MINUTES_SEC_8K
from ..timeutil import to_utc

log = logging.getLogger(__name__)

GroupFn = Callable[["FeatureContext"], dict[str, float | None]]
REGISTRY: dict[str, tuple[GroupFn, tuple[str, ...]]] = {}

PHASES = ("pre", "post")
TARGET_MISSING = "target_missing"  # dataset column: True when r_24h is NaN
SEASON = "season"
# event metadata carried into the dataset next to the features (docs/design.md §6, §8);
# estimate_source says whether the surprise inputs were point-in-time (consensus_snapshot) or the
# vendor's final value, so evaluate / optimize can report the breakdown (groups.NON_POINT_IN_TIME)
META_COLUMNS: list[str] = [E.underlying, E.market, E.t0, E.t0_source, E.t0_confidence, E.kind,
                           E.timing, E.has_perp_at_t0, E.estimate_source]


def feature_group(name: str, admissible: tuple[str, ...] = ("pre", "post")):
    bad = set(admissible) - set(PHASES)
    if bad or not admissible:
        raise ValueError(f"feature group {name!r}: admissible must be a non-empty subset of {PHASES}, got {admissible}")

    def deco(fn: GroupFn) -> GroupFn:
        REGISTRY[name] = (fn, tuple(admissible))
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
    # Settings.p0_buffer_minutes_sec_8k: the P0 back-off for 8-K times, shared with the targets
    p0_buffer_minutes_sec_8k: float = P0_BUFFER_MINUTES_SEC_8K
    extra: dict = field(default_factory=dict)


def phase_of(decision_time: str) -> str:
    """'pre' for negative offsets, 'post' otherwise (schemas.DECISION_TIMES)."""
    from ..schemas import DECISION_TIMES

    return "pre" if DECISION_TIMES[decision_time] < 0 else "post"


def admissible_groups(decision_time: str) -> list[str]:
    """Registered group names admissible at `decision_time`, in registration order."""
    phase = phase_of(decision_time)
    return [name for name, (_, adm) in REGISTRY.items() if phase in adm]


def decision_as_of(t0: pd.Timestamp, decision_time: str) -> pd.Timestamp:
    """The decision instant for a release at `t0`: t0 + DECISION_TIMES[d] minutes (UTC)."""
    if decision_time not in DECISION_TIMES:
        raise ValueError(f"unknown decision time {decision_time!r}; expected one of {list(DECISION_TIMES)}")
    return to_utc(t0, assume_tz="UTC") + pd.Timedelta(minutes=DECISION_TIMES[decision_time])


def _utc_series(s: pd.Series) -> pd.Series:
    """Datetime column as tz-aware UTC (naive values are taken as UTC, NaT stays NaT)."""
    out = pd.to_datetime(s, utc=True, errors="coerce")
    return out


def history_view(events: pd.DataFrame, targets: pd.DataFrame, underlying: str,
                 as_of: pd.Timestamp, horizon_hours: int = 24) -> pd.DataFrame:
    """Same-underlying event rows joined with their targets, restricted to
    t0 + horizon_hours <= as_of. This is the ONLY way history reaches a feature group."""
    as_of = to_utc(as_of, assume_tz="UTC")
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=[E.event_id, E.underlying, E.t0])
    ev = events[events[E.underlying] == underlying]
    if len(ev) == 0:
        return ev.iloc[0:0].reset_index(drop=True)
    t0 = _utc_series(ev[E.t0])
    settled = (t0 + pd.Timedelta(hours=horizon_hours)) <= as_of  # NaT compares False
    ev = ev.loc[settled.to_numpy()]
    if targets is not None and len(targets) and T.event_id in targets.columns:
        tg = targets.drop_duplicates(subset=T.event_id, keep="last")
        overlap = [c for c in tg.columns if c in ev.columns and c != T.event_id]
        out = ev.merge(tg.drop(columns=overlap), on=T.event_id, how="left")
    else:
        out = ev.copy()
    out = out.assign(**{E.t0: _utc_series(out[E.t0])})
    return out.sort_values(E.t0, kind="mergesort").reset_index(drop=True)


def _check_history(history: pd.DataFrame | None, as_of: pd.Timestamp, horizon_hours: int) -> None:
    if history is None or len(history) == 0:
        return
    if E.t0 not in history.columns:
        raise AssertionError("ctx.history has no t0 column; build it with history_view()")
    t0 = _utc_series(history[E.t0])
    settled = (t0 + pd.Timedelta(hours=horizon_hours)) <= to_utc(as_of, assume_tz="UTC")
    if t0.isna().any() or not bool(settled.all()):
        bad = history.loc[~settled.to_numpy() | t0.isna().to_numpy(), E.event_id].tolist()[:5]
        raise AssertionError(
            f"ctx.history leaks: rows whose t0 + {horizon_hours}h is after as_of={as_of} "
            f"(e.g. {bad}); build history with history_view(events, targets, underlying, as_of)")


def _as_float(name: str, v: object) -> float:
    if v is None:
        return math.nan
    if isinstance(v, bool | np.bool_):
        return float(v)
    if isinstance(v, pd.Timestamp | pd.Timedelta | str | bytes):
        raise TypeError(f"feature {name!r} must be numeric or None, got {type(v).__name__}")
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"feature {name!r} must be numeric or None, got {v!r}") from exc


def build_features(ctx: FeatureContext, groups: list[str] | None = None) -> dict[str, float | None]:
    """Run the registered groups admissible at phase_of(ctx.decision_time). Asserts the history
    invariant. Output keys carry the schemas.D.feature_prefix and a __missing companion."""
    phase = phase_of(ctx.decision_time)
    if groups is not None:
        unknown = [g for g in groups if g not in REGISTRY]
        if unknown:
            raise ValueError(f"unknown feature group(s) {unknown}; registered: {list(REGISTRY)}")
    _check_history(ctx.history, ctx.as_of, ctx.horizon_hours)
    out: dict[str, float | None] = {}
    for name, (fn, adm) in REGISTRY.items():
        if groups is not None and name not in groups:
            continue
        if phase not in adm:
            log.debug("group %s is not admissible at %s (%s)", name, ctx.decision_time, phase)
            continue
        feats = fn(ctx)
        for key, raw in feats.items():
            col = D.feature_prefix + key
            if col in out:
                raise ValueError(f"feature name {key!r} produced twice (group {name})")
            val = _as_float(key, raw)
            out[col] = val
            out[col + D.missing_suffix] = 1.0 if math.isnan(val) else 0.0
    return out


# ---- dataset -------------------------------------------------------------------------------------
def feature_columns(df: pd.DataFrame) -> list[str]:
    """The f_* value columns of a dataset (companions excluded)."""
    return [c for c in df.columns
            if c.startswith(D.feature_prefix) and not c.endswith(D.missing_suffix)]


def fill_missing_companions(df: pd.DataFrame) -> pd.DataFrame:
    """Make every f_<name>__missing equal to f_<name>.isna(); rows built at a decision time
    where a group was inadmissible get NaN + 1.0 like any other missing value."""
    fcols = feature_columns(df)
    if not fcols:
        return df.copy()
    vals = df[fcols].apply(pd.to_numeric, errors="coerce").astype("float64")
    comps = vals.isna().astype("float64")
    comps.columns = [c + D.missing_suffix for c in fcols]
    keep = [c for c in df.columns if c not in set(fcols) and c not in set(comps.columns)]
    return pd.concat([df[keep], vals, comps], axis=1)


def target_columns(targets: pd.DataFrame | None) -> list[str]:
    """Every schemas.T column (in schema order) present in `targets`."""
    wanted = [T.p0, T.p0_time, T.p0_staleness_min, T.price_source, T.price_interval,
              T.price_market, T.horizon_actual_h, T.h24_in_closure]
    for cp in CHECKPOINTS:
        wanted += [T.r(cp), T.ar(cp), T.p(cp), T.t(cp), T.s(cp)]
    wanted += [T.direction, T.magnitude, T.continuation_15m, T.continuation_30m, T.label_reason]
    if targets is None:
        return []
    return [c for c in wanted if c in targets.columns]


def write_dataset(df: pd.DataFrame, path: Path) -> None:
    """Parquet with `schema_version` (schemas.SCHEMA_VERSION) in the file metadata, written
    atomically (tmp file then replace)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"schema_version"] = str(SCHEMA_VERSION).encode()
    meta[b"freedom_artifact"] = b"dataset"
    table = table.replace_schema_metadata(meta)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def read_dataset(path: Path) -> tuple[pd.DataFrame, int | None]:
    """(frame, schema_version from the parquet metadata or None when absent)."""
    import pyarrow.parquet as pq

    meta = pq.read_metadata(path).metadata or {}
    raw = meta.get(b"schema_version")
    return pd.read_parquet(path), (int(raw.decode()) if raw else None)


def _event_meta(events: pd.DataFrame) -> pd.DataFrame:
    """event_id + every META_COLUMNS column (NA when `events` lacks it, so downstream filters
    on has_perp_at_t0 / t0_confidence never KeyError) + season."""
    cols = [c for c in META_COLUMNS if c in events.columns]
    meta = events[[E.event_id, *cols]].drop_duplicates(subset=E.event_id, keep="last").copy()
    for c in META_COLUMNS:
        if c not in meta.columns:
            meta[c] = pd.NaT if c == E.t0 else None
    meta = meta[[E.event_id, *META_COLUMNS]]
    meta[E.t0] = _utc_series(meta[E.t0])
    meta[SEASON] = [season_of(t) if pd.notna(t) else None for t in meta[E.t0]]
    meta[E.has_perp_at_t0] = meta[E.has_perp_at_t0].astype("boolean")
    return meta


def _dataset_columns(targets: pd.DataFrame | None, dts: list[str], groups: list[str] | None) -> list[str]:
    """Column order of a dataset built for `dts`/`groups`: lead columns, event metadata,
    target_missing, the features of every group admissible at one of `dts`, their __missing
    companions, then the target columns. Used for the empty dataset so its schema matches."""
    from .groups import GROUP_KEYS

    phases = {phase_of(d) for d in dts}
    fcols = [D.feature_prefix + k for name, (_, adm) in REGISTRY.items()
             if (groups is None or name in groups) and phases & set(adm)
             for k in GROUP_KEYS.get(name, ())]
    comps = [c + D.missing_suffix for c in fcols]
    meta = [c for c in META_COLUMNS if c != E.event_id] + [SEASON]
    return [D.event_id, D.decision_time, D.as_of, *meta, TARGET_MISSING, *fcols, *comps,
            *target_columns(targets)]


def _empty_dataset(targets: pd.DataFrame | None, dts: list[str], groups: list[str] | None) -> pd.DataFrame:
    cols = _dataset_columns(targets, dts, groups)
    dtypes = {D.as_of: "datetime64[ns, UTC]", E.t0: "datetime64[ns, UTC]",
              E.has_perp_at_t0: "boolean", TARGET_MISSING: "bool"}
    dtypes.update({c: "float64" for c in cols if c.startswith(D.feature_prefix)})
    return pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, "object")) for c in cols})


def build_dataset(settings: Settings, events: pd.DataFrame, targets: pd.DataFrame,
                  *, decision_times: list[str] | None = None, groups: list[str] | None = None,
                  write: bool = True) -> pd.DataFrame:
    """One row per (event, decision_time) with schemas.D columns, f_* features, the target
    columns, and event metadata needed downstream (t0, t0_source, kind, timing, has_perp_at_t0,
    season; NA where `events` lacks the column). Rows for events whose targets are all NaN are
    kept (they are still usable for intermediate checkpoints) but flagged. Events without t0 or
    without underlying are skipped with a warning; duplicate decision times are built once.
    Bars are loaded through targets.loaders so the price source matches the targets."""
    from .loaders import ContextLoader

    dts = list(dict.fromkeys(decision_times)) if decision_times is not None else list(DECISION_TIMES)
    for d in dts:
        if d not in DECISION_TIMES:
            raise ValueError(f"unknown decision time {d!r}; expected one of {list(DECISION_TIMES)}")
    if groups is not None:
        unknown = [g for g in groups if g not in REGISTRY]
        if unknown:
            raise ValueError(f"unknown feature group(s) {unknown}; registered: {list(REGISTRY)}")
    settings.ensure_dirs()
    if targets is None:
        targets = pd.DataFrame(columns=[T.event_id])
    events = events.reset_index(drop=True)
    if len(events):
        required = [c for c in (E.event_id, E.underlying) if c not in events.columns]
        if required:
            raise ValueError(f"events lack required column(s) {required}")
        no_underlying = events[E.underlying].isna().to_numpy()
        if no_underlying.any():
            log.warning("%d event(s) without underlying skipped: %s", int(no_underlying.sum()),
                        events.loc[no_underlying, E.event_id].tolist()[:5])
            events = events.loc[~no_underlying].reset_index(drop=True)
    horizon = int(settings.horizon_hours)
    loader = ContextLoader(settings, events)

    rows: list[dict] = []
    n_underlyings = int(events[E.underlying].nunique()) if len(events) else 0
    for i, (underlying, group_ev) in enumerate(events.groupby(E.underlying, sort=True), start=1):
        log.info("features %d/%d %s: %d events x %d decision times", i, n_underlyings,
                 underlying, len(group_ev), len(dts))
        uinputs = loader.underlying_inputs(str(underlying))
        for _, ev in group_ev.iterrows():
            t0_raw = ev.get(E.t0)
            if t0_raw is None or pd.isna(t0_raw):
                log.warning("%s: no t0, skipped", ev.get(E.event_id))
                continue
            t0 = to_utc(t0_raw, assume_tz="UTC")
            ev = ev.copy()
            ev[E.t0] = t0
            bars, market_bars = loader.event_bars(ev)
            einputs = loader.event_inputs(ev, uinputs, bars)
            for d in dts:
                as_of = decision_as_of(t0, d)
                hist = history_view(group_ev, targets, str(underlying), as_of, horizon)
                ctx = loader.context(ev, d, as_of, history=hist, bars=bars, market_bars=market_bars,
                                     uinputs=uinputs, einputs=einputs)
                feats = build_features(ctx, groups)
                rows.append({D.event_id: ev[E.event_id], D.decision_time: d, D.as_of: as_of, **feats})
        log.info("features %s done (%d rows so far)", underlying, len(rows))

    if not rows:
        out = _empty_dataset(targets, dts, groups)
    else:
        out = pd.DataFrame(rows)
        out[D.as_of] = _utc_series(out[D.as_of])
        out = fill_missing_companions(out)
        meta = _event_meta(events)
        out = out.merge(meta, on=E.event_id, how="left")
        tcols = target_columns(targets)
        if tcols:
            tg = targets[[T.event_id, *tcols]].drop_duplicates(subset=T.event_id, keep="last")
            out = out.merge(tg, on=T.event_id, how="left")
        r24 = T.r("24h")
        if r24 in out.columns:
            out[TARGET_MISSING] = pd.to_numeric(out[r24], errors="coerce").isna()
        else:
            out[TARGET_MISSING] = True
        if T.h24_in_closure in out.columns:
            out[T.h24_in_closure] = out[T.h24_in_closure].astype("boolean")
        fcols = feature_columns(out)
        comp = [c + D.missing_suffix for c in fcols]
        lead = [D.event_id, D.decision_time, D.as_of]
        meta_cols = [c for c in out.columns if c in set(meta.columns) - {E.event_id}]
        rest = [c for c in out.columns if c not in set(lead + meta_cols + fcols + comp + [TARGET_MISSING])]
        out = out[lead + meta_cols + [TARGET_MISSING] + fcols + comp + rest]
    if write:
        write_dataset(out, settings.dataset_path)
        log.info("dataset: %d rows, %d features -> %s", len(out), len(feature_columns(out)),
                 settings.dataset_path)
    return out


from . import groups as groups  # noqa: E402  (registers the v1 groups into REGISTRY)

__all__ = [
    "REGISTRY", "FeatureContext", "GroupFn", "META_COLUMNS", "SEASON", "TARGET_MISSING",
    "admissible_groups", "build_dataset", "build_features", "decision_as_of", "feature_columns",
    "feature_group", "fill_missing_companions", "groups", "history_view", "phase_of",
    "read_dataset", "target_columns", "write_dataset",
]
