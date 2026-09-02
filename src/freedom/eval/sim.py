"""Cost-aware trading simulation and the equal_split portfolio rule (docs/design.md §8).

Fill rule: a fill at instant x is the OPEN of the first bar whose start t >= x, so it is
always at or after the signal and never the close of the bar that contains the signal. The
lag (bar start - x) is recorded; a fill later than settings.max_fill_lag_minutes is not taken.
Execution cost per leg (bps) = slippage_floor_bps + slippage_range_coeff * range_bps of the
execution bar + taker_fee_bps, with range_bps = (high - low) / open * 1e4.
Funding is accrued from the archived hourly series only when the perp existed at t0 and every
settlement hour inside (entry, exit] is archived; longs pay positive funding, shorts receive
it. Returns are simple returns: gross = side * (exit_fill / entry_fill - 1).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..schemas import DECISION_TIMES, C, E, P
from ..timeutil import to_utc

SIZINGS = ("fixed", "by_confidence", "by_magnitude", "magnitude_gate")
FUNDING_ARCHIVE = "archive"
FUNDING_NONE = "none"
MAX_POSITION = 1.0  # a single position never holds more than its equal-split share

TRADE_COLUMNS: list[str] = [
    P.event_id, P.decision_time, P.model, P.fold, P.test_season, "sizing", E.t0, E.market,
    E.has_perp_at_t0, "signal_time", "exit_signal_time", P.p_up, P.r_hat, "side", "size",
    "entry_fill", "entry_fill_time", "fill_lag_min", "entry_range_bps",
    "exit_fill", "exit_fill_time", "exit_fill_lag_min", "exit_range_bps",
    "cost_bps", "funding_bps", "funding_source", "gross_return", "net_return", "pnl",
    "traded", "untraded_reason",
]


def as_utc(s: pd.Series) -> pd.Series:
    """Datetime column as tz-aware UTC without the per-element scan pd.to_datetime runs on
    columns that are already datetimes."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    return pd.to_datetime(s, utc=True)


class BarIndex:
    """Sorted fine bars of one event with an int64 start-time index, so repeated fills are
    binary searches instead of a sort per call."""

    def __init__(self, bars: pd.DataFrame):
        b = bars.sort_values(C.t, kind="mergesort").reset_index(drop=True)
        t = as_utc(b[C.t])
        self.t = t
        self.t_ns = t.dt.as_unit("ns").astype("int64").to_numpy()
        self.open = b[C.open].to_numpy(dtype=float)
        self.high = b[C.high].to_numpy(dtype=float)
        self.low = b[C.low].to_numpy(dtype=float)

    def __len__(self) -> int:
        return len(self.t_ns)

    def fill(self, when: pd.Timestamp, max_lag: pd.Timedelta) -> tuple[tuple[float, pd.Timestamp, float] | None, str | None]:
        """((open, bar_start, range_bps), None) or (None, reason)."""
        if len(self) == 0:
            return None, "no_bars"
        when = to_utc(when)
        idx = int(np.searchsorted(self.t_ns, np.int64(when.value), side="left"))
        if idx >= len(self):
            return None, "no_bar_after_signal"
        start = pd.Timestamp(self.t.iloc[idx])
        if start - when > max_lag:
            return None, "fill_lag"
        o = self.open[idx]
        if not (o > 0):
            return None, "bad_bar"
        rng = (self.high[idx] - self.low[idx]) / o * 1e4
        return (float(o), start, float(rng) if np.isfinite(rng) else math.nan), None


def fill_price(bars: pd.DataFrame, when: pd.Timestamp, *, max_lag: pd.Timedelta) -> tuple[float, pd.Timestamp, float] | None:
    """(open, bar_start, range_bps) of the first bar with t >= when, or None if that bar
    starts more than max_lag after `when` or does not exist."""
    if bars is None or len(bars) == 0:
        return None
    hit, _ = BarIndex(bars).fill(when, max_lag)
    return hit


def leg_cost_bps(settings: Settings, range_bps: float) -> float:
    """slippage_floor_bps + slippage_range_coeff * range_bps + taker_fee_bps (per leg)."""
    return settings.slippage_floor_bps + settings.slippage_range_coeff * range_bps + settings.taker_fee_bps


def prepare_funding(fund: pd.DataFrame | pd.Series | None) -> pd.Series | None:
    """Archived funding frame (t, funding_rate) -> rate series indexed by settlement hour
    (t floored to the hour, duplicates keep the last); None when unusable. An already
    prepared series passes through."""
    if isinstance(fund, pd.Series):
        return fund
    if fund is None or len(fund) == 0 or "t" not in fund.columns or "funding_rate" not in fund.columns:
        return None
    t = as_utc(fund["t"]).dt.floor("h")
    rates = pd.Series(fund["funding_rate"].to_numpy(dtype=float), index=t)
    return rates[~rates.index.duplicated(keep="last")].sort_index()


def memoised_funding(funding: Callable[[str], pd.DataFrame | None]) -> Callable[[str], pd.Series | None]:
    """Wrap a funding(market) callable so each market is read and prepared once."""
    cache: dict[str, pd.Series | None] = {}

    def get(market: str) -> pd.Series | None:
        if market not in cache:
            cache[market] = prepare_funding(funding(market))
        return cache[market]

    return get


def funding_sum(fund: pd.DataFrame | pd.Series | None, entry_time: pd.Timestamp,
                exit_time: pd.Timestamp) -> float | None:
    """Sum of archived hourly funding rates over the settlement hours h with
    entry_time < h <= exit_time; None unless every one of those hours is archived.
    `fund` is the archived frame or the series from `prepare_funding`."""
    start = entry_time.floor("h") + pd.Timedelta(hours=1)
    end = exit_time.floor("h")
    if start > end:
        return 0.0  # no settlement inside the interval
    rates = fund if isinstance(fund, pd.Series) else prepare_funding(fund)
    if rates is None:
        return None
    hit = rates.reindex(pd.date_range(start, end, freq="h"))
    if hit.isna().any():
        return None
    return float(hit.sum())


def position_size(sizing: str, conf: float, r_hat: float, *, target_vol: float,
                  round_trip_cost: float) -> tuple[float, str | None]:
    """Size in [0, 1] for one sizing rule and the reason when the rule declines the trade."""
    if sizing == "fixed":
        return 1.0, None
    if sizing == "by_confidence":
        return float(min(MAX_POSITION, 2.0 * abs(conf))), None
    mag = abs(r_hat) if r_hat is not None and np.isfinite(r_hat) else math.nan
    if sizing == "by_magnitude":
        if not np.isfinite(mag) or mag <= 0:
            return MAX_POSITION, None
        return float(min(MAX_POSITION, target_vol / mag)), None
    if sizing == "magnitude_gate":
        if not np.isfinite(mag) or mag <= round_trip_cost:
            return 0.0, "magnitude_gate"
        return 1.0, None
    raise ValueError(f"unknown sizing {sizing!r}; expected one of {SIZINGS}")


def _get(rec: dict, key: str, default: Any = None) -> Any:
    v = rec.get(key, default)
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    return v


def _bool(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return bool(v)


def simulate_rows(predictions: pd.DataFrame, bar_index: Callable[[str], BarIndex | None], *,
                  settings: Settings, funding: Callable[[str], pd.DataFrame | None] | None,
                  sizings: tuple[str, ...], threshold: float, target_vol: float) -> pd.DataFrame:
    """One pass over the predictions computing fills, costs and funding once per row, then one
    output row per (row, sizing). This is what `simulate` and `evaluate` share."""
    for s in sizings:
        if s not in SIZINGS:
            raise ValueError(f"unknown sizing {s!r}; expected one of {SIZINGS}")
    max_lag = pd.Timedelta(minutes=settings.max_fill_lag_minutes)
    horizon = pd.Timedelta(hours=settings.horizon_hours)
    fund_cache: dict[str, pd.Series | None] = {}
    out: list[dict] = []
    for rec in predictions.to_dict("records"):
        base: dict[str, Any] = {c: None for c in TRADE_COLUMNS}
        for c in (P.event_id, P.decision_time, P.model, P.fold, P.test_season, E.market):
            base[c] = rec.get(c)
        base[E.t0] = rec.get(E.t0)
        base[E.has_perp_at_t0] = _bool(rec.get(E.has_perp_at_t0))
        base[P.p_up] = _get(rec, P.p_up, math.nan)
        base[P.r_hat] = _get(rec, P.r_hat, math.nan)
        base.update({"side": 0, "size": 0.0, "fill_lag_min": math.nan, "exit_fill_lag_min": math.nan,
                     "entry_range_bps": math.nan, "exit_range_bps": math.nan, "cost_bps": math.nan,
                     "funding_bps": 0.0, "funding_source": FUNDING_NONE, "gross_return": math.nan,
                     "net_return": math.nan, "pnl": math.nan, "traded": False, "untraded_reason": None,
                     "entry_fill": math.nan, "exit_fill": math.nan, "entry_fill_time": pd.NaT,
                     "exit_fill_time": pd.NaT, "signal_time": pd.NaT, "exit_signal_time": pd.NaT})

        def emit(reason: str | None, common: dict[str, Any]) -> None:
            for s in sizings:
                row = dict(common)
                row["sizing"] = s
                if reason is not None:
                    row["traded"] = False
                    row["untraded_reason"] = reason
                out.append(row)

        p_up, r_hat = base[P.p_up], base[P.r_hat]
        t0 = rec.get(E.t0)
        d = rec.get(P.decision_time)
        if d not in DECISION_TIMES:
            raise ValueError(f"unknown decision_time {d!r} for event {rec.get(P.event_id)!r}")
        if t0 is None or pd.isna(t0):
            emit("no_t0", base)
            continue
        t0 = to_utc(t0)
        signal_time = t0 + pd.Timedelta(minutes=DECISION_TIMES[d])
        exit_signal = t0 + horizon
        base["signal_time"], base["exit_signal_time"] = signal_time, exit_signal
        if not np.isfinite(p_up):
            emit("no_signal", base)
            continue
        conf = p_up - 0.5
        side = int(np.sign(conf))
        if side == 0:
            emit("no_signal", base)
            continue
        if abs(conf) < threshold:
            emit("below_threshold", base)
            continue
        base["side"] = side
        bars = bar_index(str(rec.get(P.event_id)))
        if bars is None or len(bars) == 0:
            emit("no_bars", base)
            continue
        entry, reason = bars.fill(signal_time, max_lag)
        if entry is None:
            emit(f"entry_{reason}", base)
            continue
        base["entry_fill"], base["entry_fill_time"], base["entry_range_bps"] = entry
        base["fill_lag_min"] = (entry[1] - signal_time) / pd.Timedelta(minutes=1)
        exit_, reason = bars.fill(exit_signal, max_lag)
        if exit_ is None:
            emit(f"exit_{reason}", base)
            continue
        base["exit_fill"], base["exit_fill_time"], base["exit_range_bps"] = exit_
        base["exit_fill_lag_min"] = (exit_[1] - exit_signal) / pd.Timedelta(minutes=1)
        cost = leg_cost_bps(settings, entry[2]) + leg_cost_bps(settings, exit_[2])
        if not np.isfinite(cost):
            emit("bad_bar", base)
            continue
        base["cost_bps"] = cost
        # funding: archived hourly rates, only when the perp existed at t0 and the archive covers the interval
        market = rec.get(E.market)
        if base[E.has_perp_at_t0] and funding is not None and isinstance(market, str) and market:
            if market not in fund_cache:
                fund_cache[market] = prepare_funding(funding(market))
            total = funding_sum(fund_cache[market], entry[1], exit_[1])
            if total is not None:
                base["funding_bps"] = side * total * 1e4  # positive = paid by this position
                base["funding_source"] = FUNDING_ARCHIVE
        gross = side * (exit_[0] / entry[0] - 1.0)
        net = gross - (cost + base["funding_bps"]) / 1e4
        base["gross_return"], base["net_return"] = gross, net
        for s in sizings:
            row = dict(base)
            row["sizing"] = s
            size, why = position_size(s, conf, r_hat, target_vol=target_vol, round_trip_cost=cost / 1e4)
            row["size"] = size
            if why is not None or size <= 0:
                row["traded"], row["untraded_reason"] = False, why or "zero_size"
            else:
                row["traded"], row["pnl"] = True, size * net
            out.append(row)
    trades = pd.DataFrame(out, columns=TRADE_COLUMNS)
    for col in ("entry_fill_time", "exit_fill_time", "signal_time", "exit_signal_time", E.t0):
        trades[col] = as_utc(trades[col])
    trades["traded"] = trades["traded"].astype(bool)
    return trades


def memoised_bar_index(paths: Callable[[str], pd.DataFrame | None]) -> Callable[[str], BarIndex | None]:
    cache: dict[str, BarIndex | None] = {}

    def get(event_id: str) -> BarIndex | None:
        if event_id not in cache:
            bars = paths(event_id)
            cache[event_id] = BarIndex(bars) if bars is not None and len(bars) else None
        return cache[event_id]

    return get


def simulate(predictions: pd.DataFrame, paths: Callable[[str], pd.DataFrame | None], *,
             settings: Settings, funding: Callable[[str], pd.DataFrame | None] | None = None,
             sizing: str = "fixed", threshold: float = 0.0, target_vol: float = 0.03) -> pd.DataFrame:
    """Per-event trades from a predictions frame (schemas.P columns + t0, decision_time,
    has_perp_at_t0, market). `paths(event_id)` returns the event's fine bars; `funding(market)`
    returns archived hourly funding. Returns one row per prediction with side, entry/exit fill,
    fill_lag_min, cost_bps, funding_bps, funding_source, gross_return, net_return, traded."""
    return simulate_rows(predictions, memoised_bar_index(paths), settings=settings, funding=funding,
                         sizings=(sizing,), threshold=threshold, target_vol=target_vol)


def equal_split_weights(entry_ns: np.ndarray, exit_ns: np.ndarray, size: np.ndarray, *,
                        cap: float) -> tuple[np.ndarray, float]:
    """Capital share of each position under equal_split: cap * size / max n_open over the
    position's closed [entry, exit] interval, with n_open(t) = #(entry <= t) - #(exit < t).
    Dividing by the peak concurrency over the interval guarantees that the summed exposure of
    the open positions never exceeds `cap` at any instant. Returns (weights, max exposure)."""
    n = len(entry_ns)
    if n == 0:
        return np.array([], dtype=float), 0.0
    times = np.unique(np.concatenate([entry_ns, exit_ns]))
    sorted_entry, sorted_exit = np.sort(entry_ns), np.sort(exit_ns)
    n_open = np.searchsorted(sorted_entry, times, side="right") - np.searchsorted(sorted_exit, times, side="left")
    weights = np.empty(n, dtype=float)
    for i in range(n):
        m = (times >= entry_ns[i]) & (times <= exit_ns[i])
        weights[i] = cap * min(float(size[i]), MAX_POSITION) / max(int(n_open[m].max()), 1)
    exposure = np.zeros(len(times), dtype=float)
    for i in range(n):
        exposure += weights[i] * ((times >= entry_ns[i]) & (times <= exit_ns[i]))
    return weights, float(exposure.max()) if len(exposure) else 0.0


def portfolio_metrics(trades: pd.DataFrame, *, gross_exposure_cap: float = 1.0) -> dict[str, float]:
    """equal_split capital rule over overlapping [entry, exit] intervals; daily PnL series keyed
    by UTC exit date; sharpe_like, max_drawdown, turnover, n_trades, n_untraded.

    Pass the trades of ONE (model, decision_time, sizing) slice. sharpe_like =
    mean / std * sqrt(252) of the daily PnL series (NaN with fewer than two days);
    max_drawdown is the largest peak-to-trough fall of cumulative PnL (>= 0); turnover is the
    total notional traded per unit capital (entry + exit legs), turnover_daily divides it by
    the calendar span in days."""
    n_untraded = int((~trades["traded"].astype(bool)).sum()) if len(trades) else 0
    out: dict[str, float] = {"n_trades": 0, "n_untraded": n_untraded, "n_days": 0, "total_return": 0.0,
                             "mean_daily_pnl": math.nan, "sharpe_like": math.nan, "max_drawdown": 0.0,
                             "turnover": 0.0, "turnover_daily": 0.0, "max_gross_exposure": 0.0,
                             "hit_rate": math.nan, "mean_pnl_per_trade": math.nan}
    if len(trades) == 0:
        return out
    t = trades[trades["traded"].astype(bool)]
    t = t[t["entry_fill_time"].notna() & t["exit_fill_time"].notna() & np.isfinite(t["net_return"].astype(float))]
    if len(t) == 0:
        return out
    entry_ns = as_utc(t["entry_fill_time"]).dt.as_unit("ns").astype("int64").to_numpy()
    exit_ns = as_utc(t["exit_fill_time"]).dt.as_unit("ns").astype("int64").to_numpy()
    size = t["size"].astype(float).to_numpy()
    weights, max_exposure = equal_split_weights(entry_ns, exit_ns, size, cap=gross_exposure_cap)
    pnl = weights * t["net_return"].astype(float).to_numpy()
    exit_day = as_utc(t["exit_fill_time"]).dt.floor("D")
    daily = pd.Series(pnl, index=exit_day.to_numpy()).groupby(level=0).sum().sort_index()
    cum = daily.cumsum()
    drawdown = float((cum.cummax() - cum).max()) if len(cum) else 0.0
    std = float(daily.std(ddof=1)) if len(daily) > 1 else math.nan
    sharpe = float(daily.mean() / std * math.sqrt(252)) if len(daily) > 1 and std > 0 else math.nan
    first_day = as_utc(t["entry_fill_time"]).min().floor("D")
    span_days = max(int((daily.index.max() - first_day) / pd.Timedelta(days=1)) + 1, 1)
    turnover = float(2.0 * weights.sum())
    out.update({"n_trades": int(len(t)), "n_days": int(len(daily)), "total_return": float(pnl.sum()),
                "mean_daily_pnl": float(daily.mean()), "sharpe_like": sharpe, "max_drawdown": drawdown,
                "turnover": turnover, "turnover_daily": turnover / span_days,
                "max_gross_exposure": max_exposure, "hit_rate": float(np.mean(pnl > 0)),
                "mean_pnl_per_trade": float(pnl.mean())})
    return out


# ---- default data callables (used by evaluate when nothing is injected) ---------------------
def archive_funding_loader(settings: Settings) -> Callable[[str], pd.DataFrame | None]:
    """funding(market) -> archived hourly funding frame (market, t, funding_rate, premium) or None."""
    from ..data.archive import funding_path, read_parquet_or_none

    cache: dict[str, pd.DataFrame | None] = {}

    def load(market: str) -> pd.DataFrame | None:
        if market not in cache:
            cache[market] = read_parquet_or_none(funding_path(settings, market))
        return cache[market]

    return load


def loader_paths(settings: Settings, events: pd.DataFrame, *, benchmark_market: str = "xyz:SP500",
                 benchmark_equity: str = "SPY") -> Callable[[str], pd.DataFrame | None]:
    """paths(event_id) -> the event's fine bars read through targets.loaders (archive, live
    candles, then FMP), so the simulation uses the same price source as the targets. Provider
    clients are created lazily on the first call."""
    first = events.drop_duplicates(E.event_id).set_index(E.event_id)
    state: dict[str, Any] = {}

    def clients() -> tuple[Any, Any]:
        if "hl" not in state:
            from ..data.base import ProviderUnavailable
            from ..data.hyperliquid import HyperliquidClient

            state["hl"] = HyperliquidClient(settings)
            try:
                from ..data.fmp import FMPClient

                state["fmp"] = FMPClient(settings)
            except ProviderUnavailable:
                state["fmp"] = None
        return state["hl"], state["fmp"]

    def load(event_id: str) -> pd.DataFrame | None:
        from ..targets.loaders import load_event_bars

        if event_id not in first.index:
            return None
        ev = first.loc[event_id].copy()
        ev[E.event_id] = event_id
        if E.underlying not in ev.index or ev.get(E.underlying) is None:
            ev[E.underlying] = event_id.split(":", 1)[0]
        hl, fmp = clients()
        try:
            path, _ = load_event_bars(settings, ev, hl=hl, fmp=fmp, benchmark_market=benchmark_market,
                                      benchmark_equity=benchmark_equity)
        except Exception:  # a missing path only means an untraded event
            return None
        return path if path is not None and len(path) else None

    return load
