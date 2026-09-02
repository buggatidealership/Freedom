"""Assemble FeatureContext inputs for build_dataset (and for live prediction).

Sources, in the order tried:

* fine bars: `targets.loaders.load_event_bars` (archive -> live Hyperliquid 1m/5m -> FMP
  1-minute extended-hours bars), so the reaction features see exactly the path the targets
  were computed on; when no fine source covers the whole 24h window (the targets are NaN)
  the raw perp or equity bars around the release are still used, because features only need
  the hours before and just after t0;
* daily bars of the underlying and of its sector-proxy ETF: `FMPClient.daily`, then
  `NasdaqClient.daily` when FMP is unavailable or empty -- one request per symbol for the whole
  dataset span, cached on disk by the clients;
* benchmark: `xyz:SP500` 1d perp candles when the event's path is a perp and the candles
  reach back MIN_BENCHMARK_DAYS before t0, else SPY daily (FMP/Nasdaq) -- mirroring the
  abnormal-return benchmark of the targets; VIX: `xyz:VIX` 1d perp candles -- measured
  2026-09-02: that market is delisted (`isDelisted: true`, no candles at any interval), so the
  VIX features are reported missing until another source (e.g. FMP `^VIX` EOD, not measured) is
  wired in here;
* perp state: `candles/<market>/funding.parquet` from the archive (else live fundingHistory),
  the ctx snapshots under `archive_dir/ctx/<dex>/`, the market's 1d candles for the 30-day
  notional volume, and maxLeverage from universe.parquet or the dex meta -- the CURRENT cap,
  not the one in force at t0 (Hyperliquid publishes no leverage history; see groups.PERP_KEYS).

Every provider failure is logged and yields a None input (the features become missing);
a dataset build never stops because one symbol could not be loaded. All windows end at or
before the instants the groups cut at, but the groups cut again regardless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ..config import Settings
from ..data.archive import CTX_SUBDIR, funding_path, read_parquet_or_none
from ..data.base import ProviderUnavailable
from ..data.hyperliquid import HyperliquidClient
from ..schemas import NY, UTC, C, E, PriceSource, U
from ..targets.loaders import _equity_bars, _perp_bars, load_event_bars
from ..timeutil import to_utc
from . import FeatureContext, history_view
from .groups import (
    X_FUNDING,
    X_LISTING_START,
    X_MAX_LEVERAGE,
    X_N_EVENTS_SAME_DAY,
    X_PERP_DAILY,
    X_SECTOR_DAILY,
    X_VIX_DAILY,
    value,
)

log = logging.getLogger(__name__)

BENCHMARK_MARKET = "xyz:SP500"
VIX_MARKET = "xyz:VIX"
BENCHMARK_EQUITY = "SPY"
DAILY_LOOKBACK = pd.Timedelta(days=400)  # 52-week high/low needs a year of sessions
PERP_LOOKBACK = pd.Timedelta(days=3)  # funding / ctx snapshots before t0
POST_WINDOW = pd.Timedelta(hours=2)  # funding / ctx snapshots after t0 (decision times end at +60m)
MIN_BENCHMARK_DAYS = 30  # a perp benchmark must reach this far back before t0 to be used
PERP_SOURCES = {PriceSource.hl_archive.value, PriceSource.hl_live.value}

# FMP profile sector -> liquid sector ETF; industry overrides for the two groups the design
# names explicitly (semis -> SMH, biotech -> XBI).
SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK", "Healthcare": "XLV", "Financial Services": "XLF",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Communication Services": "XLC",
    "Industrials": "XLI", "Energy": "XLE", "Basic Materials": "XLB", "Real Estate": "XLRE",
    "Utilities": "XLU",
}
INDUSTRY_ETF: dict[str, str] = {
    "Semiconductors": "SMH", "Semiconductor Equipment & Materials": "SMH", "Biotechnology": "XBI",
}


def sector_proxy(profile: dict | None) -> str | None:
    """ETF symbol for an FMP profile (industry first, then sector); None when unknown."""
    if not profile:
        return None
    industry = profile.get("industry")
    if industry and industry in INDUSTRY_ETF:
        return INDUSTRY_ETF[industry]
    sector = profile.get("sector")
    return SECTOR_ETF.get(sector) if sector else None


@dataclass
class UnderlyingInputs:
    underlying: str
    daily: pd.DataFrame | None = None
    sector_etf: str | None = None
    sector_daily: pd.DataFrame | None = None


@dataclass
class EventInputs:
    market_daily: pd.DataFrame | None = None
    vix_daily: pd.DataFrame | None = None
    perp_daily: pd.DataFrame | None = None
    funding: pd.DataFrame | None = None
    perp_ctx: pd.DataFrame | None = None
    max_leverage: float | None = None
    listing_start: pd.Timestamp | None = None
    extra: dict = field(default_factory=dict)


def _t0_series(events: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(events[E.t0], utc=True, errors="coerce") if E.t0 in events.columns else pd.Series(dtype="datetime64[ns, UTC]")


def _same_day_counts(events: pd.DataFrame) -> dict:
    """event_id -> number of events (any underlying) on the same New York date. This is
    calendar knowledge (the schedule is public before any release), not an outcome, which is
    why it may reach a pre feature without going through history_view."""
    if len(events) == 0:
        return {}
    t0 = _t0_series(events)
    day = t0.dt.tz_convert(NY).dt.date
    counts = day.map(day.value_counts())
    return {eid: float(n) for eid, n, ok in zip(events[E.event_id], counts, t0.notna(), strict=True) if ok}


def _utc_frame(df: pd.DataFrame | None, cols: tuple[str, ...]) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


class ContextLoader:
    """Loads and caches everything the feature groups need for the events of one dataset.

    Clients are created lazily from `settings` unless injected (tests pass fakes). `fmp=None`
    with no key means: no daily bars from FMP (Nasdaq is tried) and no FMP intraday proxy."""

    def __init__(self, settings: Settings, events: pd.DataFrame, *, hl: HyperliquidClient | None = None,
                 fmp=None, nasdaq=None, benchmark_market: str = BENCHMARK_MARKET,
                 vix_market: str = VIX_MARKET, benchmark_equity: str = BENCHMARK_EQUITY,
                 sector_proxies: bool = True, now: pd.Timestamp | None = None):
        self.settings = settings
        self.hl = hl if hl is not None else HyperliquidClient(settings)
        if fmp is None:
            try:
                from ..data.fmp import FMPClient

                fmp = FMPClient(settings)
            except ProviderUnavailable as exc:
                log.warning("FMP unavailable (%s): daily bars come from Nasdaq, no FMP intraday proxy", exc)
                fmp = None
        self.fmp = fmp
        self._nasdaq = nasdaq
        self.benchmark_market, self.vix_market, self.benchmark_equity = benchmark_market, vix_market, benchmark_equity
        self.sector_proxies = sector_proxies
        self.now = to_utc(now, assume_tz=UTC) if now is not None else pd.Timestamp.now(tz=UTC)
        t0 = _t0_series(events).dropna()
        if len(t0):
            self.span_lo = t0.min() - DAILY_LOOKBACK
            self.span_hi = t0.max() + pd.Timedelta(days=2)
        else:
            self.span_lo, self.span_hi = self.now - DAILY_LOOKBACK, self.now
        # completed sessions are cached as immutable by the clients; never ask for today
        yesterday_ny = (self.now.tz_convert(NY).normalize() - pd.Timedelta(days=1)).tz_convert(UTC)
        self.span_hi = min(self.span_hi, yesterday_ny)
        self.n_same_day = _same_day_counts(events)
        self._daily: dict[str, pd.DataFrame | None] = {}
        self._candles: dict[tuple[str, str], pd.DataFrame | None] = {}
        self._ctx: dict[str, pd.DataFrame | None] = {}
        self._leverage: dict[str, float | None] = {}
        self._sector: dict[str, str | None] = {}
        self._universe: pd.DataFrame | None = None
        self._universe_loaded = False

    # ---- daily bars -------------------------------------------------------------------------
    @property
    def nasdaq(self):
        if self._nasdaq is None:
            try:
                from ..data.nasdaq import NasdaqClient

                self._nasdaq = NasdaqClient(self.settings)
            except Exception as exc:  # no network stack / misconfiguration: stay without it
                log.warning("Nasdaq client unavailable: %s", exc)
                self._nasdaq = False
        return self._nasdaq or None

    def daily_bars(self, symbol: str) -> pd.DataFrame | None:
        """Daily bars of `symbol` over the dataset span: FMP, then Nasdaq; None when neither."""
        if symbol in self._daily:
            return self._daily[symbol]
        out = None
        if self.fmp is not None:
            try:
                df = self.fmp.daily(symbol, self.span_lo, self.span_hi)
                out = df if df is not None and len(df) else None
            except Exception as exc:
                log.warning("FMP daily %s failed: %s", symbol, exc)
        if out is None and self.nasdaq is not None:
            try:
                df = self.nasdaq.daily(symbol, self.span_lo, self.span_hi)
                out = df if df is not None and len(df) else None
            except Exception as exc:
                log.warning("Nasdaq daily %s failed: %s", symbol, exc)
        if out is None:
            log.warning("no daily bars for %s", symbol)
        self._daily[symbol] = _utc_frame(out, (C.t, C.t_end))
        return self._daily[symbol]

    def perp_candles(self, market: str, interval: str = "1d") -> pd.DataFrame | None:
        """Hyperliquid candles of `market` over the dataset span (one fetch per build)."""
        key = (market, interval)
        if key in self._candles:
            return self._candles[key]
        out = None
        try:
            df = self.hl.candles(market, interval, self.span_lo, self.span_hi + pd.Timedelta(days=1),
                                 cache_ttl=self.settings.cache_ttl_seconds)
            out = df if df is not None and len(df) else None
        except Exception as exc:
            log.warning("Hyperliquid %s %s candles failed: %s", market, interval, exc)
        self._candles[key] = out
        return out

    def benchmark_daily(self, t0: pd.Timestamp, *, perp_path: bool) -> pd.DataFrame | None:
        """xyz:SP500 1d candles for a perp path that they reach back far enough for, else SPY."""
        if perp_path:
            bench = self.perp_candles(self.benchmark_market, "1d")
            if bench is not None and bench[C.t].min() <= t0 - pd.Timedelta(days=MIN_BENCHMARK_DAYS):
                return bench
        return self.daily_bars(self.benchmark_equity)

    def vix_daily(self) -> pd.DataFrame | None:
        return self.perp_candles(self.vix_market, "1d")

    # ---- sector proxy -----------------------------------------------------------------------
    def sector_etf(self, underlying: str) -> str | None:
        if underlying in self._sector:
            return self._sector[underlying]
        etf = None
        if self.sector_proxies and self.fmp is not None:
            try:
                etf = sector_proxy(self.fmp.profile(underlying))
            except Exception as exc:
                log.warning("FMP profile %s failed: %s", underlying, exc)
        self._sector[underlying] = etf
        return etf

    # ---- perp state -------------------------------------------------------------------------
    def funding(self, market: str, t0: pd.Timestamp) -> pd.DataFrame | None:
        """Hourly funding around t0 from the archive, else the live fundingHistory."""
        lo, hi = t0 - PERP_LOOKBACK, t0 + POST_WINDOW
        df = read_parquet_or_none(funding_path(self.settings, market))
        if df is not None and len(df):
            t = pd.to_datetime(df["t"], utc=True, errors="coerce")
            hit = df.loc[((t >= lo) & (t < hi)).to_numpy()]
            if len(hit):
                return _utc_frame(hit, ("t",))
        try:
            live = self.hl.funding_history(market, lo, min(hi, self.now), cache_ttl=self.settings.cache_ttl_seconds)
        except Exception as exc:
            log.warning("fundingHistory %s failed: %s", market, exc)
            return None
        return _utc_frame(live, ("t",)) if live is not None and len(live) else None

    def ctx_snapshots(self, market: str) -> pd.DataFrame | None:
        """Archived metaAndAssetCtxs rows for `market` (all days), sorted by t."""
        dex = market.split(":", 1)[0] if ":" in market else ""
        if dex not in self._ctx:
            root = self.settings.archive_dir / CTX_SUBDIR / dex
            frames = []
            if root.is_dir():
                for p in sorted(root.glob("*.parquet")):
                    try:
                        frames.append(pd.read_parquet(p))
                    except Exception as exc:
                        log.warning("ctx snapshot %s unreadable: %s", p, exc)
            all_rows = pd.concat(frames, ignore_index=True) if frames else None
            self._ctx[dex] = _utc_frame(all_rows, ("t",))
        rows = self._ctx[dex]
        if rows is None or "market" not in rows.columns:
            return None
        hit = rows.loc[(rows["market"] == market).to_numpy()]
        return hit.sort_values("t", kind="mergesort").reset_index(drop=True) if len(hit) else None

    def perp_ctx(self, market: str, t0: pd.Timestamp) -> pd.DataFrame | None:
        snaps = self.ctx_snapshots(market)
        if snaps is None:
            return None
        t = snaps["t"]
        hit = snaps.loc[((t >= t0 - PERP_LOOKBACK) & (t < t0 + POST_WINDOW)).to_numpy()]
        return hit.reset_index(drop=True) if len(hit) else None

    def _universe_rows(self) -> pd.DataFrame | None:
        if not self._universe_loaded:
            self._universe_loaded = True
            path = self.settings.universe_path
            if path.exists():
                try:
                    self._universe = pd.read_parquet(path)
                except Exception as exc:
                    log.warning("universe.parquet unreadable: %s", exc)
        return self._universe

    def max_leverage(self, market: str) -> float | None:
        if market in self._leverage:
            return self._leverage[market]
        lev = None
        uni = self._universe_rows()
        if uni is not None and U.market in uni.columns and U.max_leverage in uni.columns:
            hit = uni.loc[(uni[U.market] == market).to_numpy(), U.max_leverage]
            if len(hit) and pd.notna(hit.iloc[0]):
                lev = float(hit.iloc[0])
        if lev is None and ":" in market:
            dex = market.split(":", 1)[0]
            try:
                for asset in self.hl.meta(dex).get("universe", []):
                    if asset.get("name") == market and asset.get("maxLeverage") is not None:
                        lev = float(asset["maxLeverage"])
                        break
            except Exception as exc:
                log.warning("meta %s failed: %s", dex, exc)
        self._leverage[market] = lev
        return lev

    # ---- per underlying / per event ------------------------------------------------------------
    def underlying_inputs(self, underlying: str) -> UnderlyingInputs:
        u = UnderlyingInputs(underlying=underlying, daily=self.daily_bars(underlying))
        u.sector_etf = self.sector_etf(underlying)
        if u.sector_etf:
            u.sector_daily = self.daily_bars(u.sector_etf)
        return u

    def event_bars(self, event: pd.Series) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """(fine bars, benchmark fine bars) through targets.loaders; when no source covers the
        whole target window, the raw perp or equity bars around the release (targets stay NaN)."""
        try:
            path, mpath = load_event_bars(self.settings, event, hl=self.hl, fmp=self.fmp,
                                          benchmark_market=self.benchmark_market,
                                          benchmark_equity=self.benchmark_equity, now=self.now)
        except Exception as exc:
            log.warning("%s: load_event_bars failed: %s", value(event, E.event_id), exc)
            path, mpath = pd.DataFrame(), None
        if path is not None and len(path):
            return path, mpath
        return self._fallback_bars(event), None

    def _fallback_bars(self, event: pd.Series) -> pd.DataFrame | None:
        t0 = to_utc(event[E.t0], assume_tz=UTC)
        lo = t0 - pd.Timedelta(days=1)
        hi = t0 + pd.Timedelta(hours=self.settings.horizon_hours) + pd.Timedelta(hours=2)
        if lo > self.now:
            # the whole [t0 - 1d, t0 + horizon + 2h] window is in the future: no bars exist yet,
            # and asking would spend a provider request (cached only for the live TTL) on every
            # run. `lo` rather than `t0` keeps live `predict` fetching the pre-release bars.
            return None
        market = value(event, E.market)
        if isinstance(market, str) and market:
            try:
                perp = _perp_bars(self.settings, self.hl, market, lo, hi)
                if perp is not None and len(perp) and perp[C.t].min() <= t0 - pd.Timedelta(hours=1):
                    return perp
            except Exception as exc:
                log.warning("%s: perp bars failed: %s", value(event, E.event_id), exc)
        underlying = value(event, E.underlying)
        if self.fmp is not None and isinstance(underlying, str):
            try:
                eq = _equity_bars(self.fmp, underlying, lo, hi)
                if eq is not None and len(eq):
                    return eq
            except Exception as exc:
                log.warning("%s: equity bars failed: %s", value(event, E.event_id), exc)
        return None

    def event_inputs(self, event: pd.Series, uinputs: UnderlyingInputs,
                     bars: pd.DataFrame | None) -> EventInputs:
        t0 = to_utc(event[E.t0], assume_tz=UTC)
        perp_path = bars is not None and len(bars) > 0 and C.source in bars.columns \
            and str(bars[C.source].iloc[0]) in PERP_SOURCES
        ei = EventInputs(market_daily=self.benchmark_daily(t0, perp_path=perp_path),
                         vix_daily=self.vix_daily())
        market = value(event, E.market)
        has_perp = value(event, E.has_perp_at_t0)
        if isinstance(market, str) and market and has_perp is not False:
            ei.perp_daily = self.perp_candles(market, "1d")
            ei.funding = self.funding(market, t0)
            ei.perp_ctx = self.perp_ctx(market, t0)
            ei.max_leverage = self.max_leverage(market)
            if ei.perp_daily is not None and value(event, E.listing_start) is None:
                ei.listing_start = pd.Timestamp(ei.perp_daily[C.t].min())
        return ei

    def context(self, event: pd.Series, decision_time: str, as_of: pd.Timestamp, *,
                history: pd.DataFrame | None, bars: pd.DataFrame | None,
                market_bars: pd.DataFrame | None, uinputs: UnderlyingInputs,
                einputs: EventInputs) -> FeatureContext:
        extra = {
            X_N_EVENTS_SAME_DAY: self.n_same_day.get(value(event, E.event_id)),
            X_VIX_DAILY: einputs.vix_daily, X_SECTOR_DAILY: uinputs.sector_daily,
            X_PERP_DAILY: einputs.perp_daily, X_FUNDING: einputs.funding,
            X_MAX_LEVERAGE: einputs.max_leverage, X_LISTING_START: einputs.listing_start,
            "sector_etf": uinputs.sector_etf, **einputs.extra,
        }
        return FeatureContext(event=event, as_of=to_utc(as_of, assume_tz=UTC), decision_time=decision_time,
                              bars=bars, daily=uinputs.daily, market_bars=market_bars,
                              market_daily=einputs.market_daily, history=history,
                              perp_ctx=einputs.perp_ctx, horizon_hours=int(self.settings.horizon_hours),
                              p0_buffer_minutes_sec_8k=float(self.settings.p0_buffer_minutes_sec_8k),
                              extra=extra)

    def context_for(self, event: pd.Series, decision_time: str, *, events: pd.DataFrame,
                    targets: pd.DataFrame | None, as_of: pd.Timestamp | None = None) -> FeatureContext:
        """One FeatureContext for `event` (used by live prediction): as_of defaults to
        t0 + DECISION_TIMES[decision_time]; pass an explicit as_of for an expected t0."""
        from . import decision_as_of

        event = event.copy()
        event[E.t0] = to_utc(event[E.t0], assume_tz=UTC)
        as_of = to_utc(as_of, assume_tz=UTC) if as_of is not None else decision_as_of(event[E.t0], decision_time)
        underlying = str(value(event, E.underlying))
        uinputs = self.underlying_inputs(underlying)
        bars, market_bars = self.event_bars(event)
        einputs = self.event_inputs(event, uinputs, bars)
        hist = history_view(events, targets if targets is not None else pd.DataFrame(), underlying, as_of,
                            int(self.settings.horizon_hours))
        return self.context(event, decision_time, as_of, history=hist, bars=bars, market_bars=market_bars,
                            uinputs=uinputs, einputs=einputs)


def load_context(settings: Settings, event: pd.Series, decision_time: str, *, events: pd.DataFrame,
                 targets: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None,
                 loader: ContextLoader | None = None) -> FeatureContext:
    """Convenience for one event (freedom predict): build a loader for `events` and return the
    FeatureContext at `decision_time` (or the explicit `as_of`)."""
    loader = loader or ContextLoader(settings, events)
    return loader.context_for(event, decision_time, events=events, targets=targets, as_of=as_of)
