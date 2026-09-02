"""Universe: which Hyperliquid markets are equities with earnings events.

`build_universe` = live markets (HyperliquidClient.all_markets) + automatic SEC ticker match
+ configs/universe_overrides.yaml (authoritative) + listing_start and 30-day median notional.
Output frame uses schemas.U columns and is written to settings.universe_path.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yaml

from ..config import Settings
from ..schemas import EVENT_KINDS, Kind, U

log = logging.getLogger(__name__)

BENCHMARK_MARKETS = ("xyz:SP500", "xyz:VIX")  # context markets that also get listing/volume info
DEFAULT_DEX_PRIORITY = ["xyz", "para", "io", "mkts", "hyna"]


def load_overrides(settings: Settings) -> dict:
    """Parsed universe_overrides.yaml: {'defaults': {...}, 'markets': {market: {...}}}."""
    path = settings.universe_overrides_path
    if not path.exists():
        log.warning("no universe overrides at %s; every market will be marked verify=True", path)
        return {"defaults": {}, "markets": {}}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    markets = raw.get("markets") or {}
    for k, v in list(markets.items()):
        if v is None:
            markets[k] = {}
        elif not isinstance(v, dict):
            raise ValueError(f"override for {k!r} must be a mapping, got {type(v).__name__}")
        if "kind" in markets[k]:
            Kind(markets[k]["kind"])  # validate
    return {"defaults": raw.get("defaults") or {}, "markets": markets}


def classify(markets: pd.DataFrame, sec_tickers: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """Pure function: assign kind/underlying/cik/verify/exclude_reason to every market row.
    Override entries win; unmatched markets default to kind='other', verify=True."""
    sec = sec_tickers.drop_duplicates("ticker").set_index("ticker") if len(sec_tickers) else pd.DataFrame(
        columns=["cik", "title"]
    )
    ov_markets = overrides.get("markets", {})
    rows = []
    for _, m in markets.iterrows():
        market, symbol = m["market"], m["symbol"]
        ov = ov_markets.get(market)
        sec_hit = sec.loc[symbol] if symbol in sec.index else None
        if ov is not None and "kind" in ov:
            kind = Kind(ov["kind"])
            verify = bool(ov.get("verify", False))
        elif sec_hit is not None:
            kind = Kind.equity_us  # automatic guess; a human must confirm
            verify = True
        else:
            kind = Kind.other
            verify = True
        is_event_kind = kind in EVENT_KINDS
        underlying = (ov or {}).get("underlying") or (symbol if is_event_kind else None)
        cik = (ov or {}).get("cik")
        if cik is None and is_event_kind and sec_hit is not None:
            cik = int(sec_hit["cik"])
        name = (ov or {}).get("note") if ov else None
        if sec_hit is not None and is_event_kind:
            name = str(sec_hit["title"])
        exclude_reason = (ov or {}).get("exclude_reason")
        if exclude_reason is None and not is_event_kind:
            exclude_reason = f"kind={kind.value}"
        rows.append({
            U.market: market, U.dex: m["dex"], U.symbol: symbol, U.kind: kind.value,
            U.underlying: underlying, U.cik: cik, U.name: name, U.exclude_reason: exclude_reason,
            U.verify: verify,
            U.max_leverage: m.get("max_leverage"), U.growth_mode: m.get("growth_mode"),
            U.deployer_fee_scale: m.get("deployer_fee_scale"),
        })
    out = pd.DataFrame(rows)
    out[U.cik] = out[U.cik].astype("Int64")
    return out


def choose_primary(universe: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """Set is_primary: for each underlying keep the market with the highest
    median_notional_30d, tie-break by defaults.dex_priority."""
    u = universe.copy()
    priority = (overrides.get("defaults") or {}).get("dex_priority") or DEFAULT_DEX_PRIORITY
    rank = {d: i for i, d in enumerate(priority)}
    if U.median_notional_30d not in u.columns:
        u[U.median_notional_30d] = np.nan
    u["_dex_rank"] = u[U.dex].map(lambda d: rank.get(d, len(rank)))
    u["_notional"] = u[U.median_notional_30d].fillna(-1.0)
    u[U.is_primary] = False
    event_rows = u[u[U.kind].isin([k.value for k in EVENT_KINDS]) & u[U.underlying].notna()]
    winners = (
        event_rows.sort_values(["_notional", "_dex_rank"], ascending=[False, True])
        .drop_duplicates(U.underlying, keep="first")
        .index
    )
    u.loc[winners, U.is_primary] = True
    u[U.in_event_universe] = u[U.is_primary] & u[U.kind].isin([k.value for k in EVENT_KINDS])
    return u.drop(columns=["_dex_rank", "_notional"])


def _listing_and_volume(hl, market: str, now: pd.Timestamp) -> tuple[pd.Timestamp | None, float]:
    """First daily candle start and median daily notional (close*volume) over the last 30 days."""
    start = hl.listing_start(market)
    if start is None:
        return None, float("nan")
    bars = hl.candles(market, "1d", now - pd.Timedelta(days=31), now, cache_ttl=6 * 3600)
    if bars is None or len(bars) == 0:
        return start, float("nan")
    notional = (bars["close"] * bars["volume"]).astype(float)
    return start, float(np.nanmedian(notional)) if len(notional) else float("nan")


def build_universe(settings: Settings, *, write: bool = True) -> pd.DataFrame:
    from ..data.hyperliquid import HyperliquidClient
    from ..data.sec import SECClient

    settings.ensure_dirs()
    hl = HyperliquidClient(settings)
    markets = hl.all_markets()
    try:
        sec_tickers = SECClient(settings).ticker_map()
    except Exception as exc:  # network or parsing trouble must not block the universe
        log.warning("SEC ticker map unavailable (%s); relying on overrides only", exc)
        sec_tickers = pd.DataFrame(columns=["ticker", "cik", "title"])
    overrides = load_overrides(settings)
    u = classify(markets, sec_tickers, overrides)

    # floored so the 30-day candle request body (the cache key) is stable within the hour and
    # its 6 h cache_ttl in _listing_and_volume actually hits on reruns
    now = pd.Timestamp.now(tz="UTC").floor("h")
    need = u[U.kind].isin([k.value for k in EVENT_KINDS]) | u[U.market].isin(BENCHMARK_MARKETS)
    starts: dict[str, pd.Timestamp | None] = {}
    notional: dict[str, float] = {}
    for market in u.loc[need, U.market]:
        try:
            s, n = _listing_and_volume(hl, market, now)
        except Exception as exc:
            log.warning("listing/volume lookup failed for %s: %s", market, exc)
            s, n = None, float("nan")
        starts[market], notional[market] = s, n
    u[U.listing_start] = u[U.market].map(starts)
    u[U.listing_start] = pd.to_datetime(u[U.listing_start], utc=True)
    u[U.median_notional_30d] = u[U.market].map(notional).astype(float)
    u = choose_primary(u, overrides)
    u = u.sort_values([U.in_event_universe, U.kind, U.market], ascending=[False, True, True]).reset_index(drop=True)
    if write:
        from ..data.archive import write_parquet_atomic

        write_parquet_atomic(u, settings.universe_path)  # readers never see a partial file
    return u


def load_universe(settings: Settings) -> pd.DataFrame:
    if not settings.universe_path.exists():
        raise FileNotFoundError(f"{settings.universe_path} missing; run `freedom universe` first")
    return pd.read_parquet(settings.universe_path)


def event_universe(universe: pd.DataFrame) -> pd.DataFrame:
    """Rows with in_event_universe == True (equity kinds, primary market)."""
    return universe[universe[U.in_event_universe]].reset_index(drop=True)


def verification_report(universe: pd.DataFrame) -> pd.DataFrame:
    """Rows a human should confirm: verify flag set, or event kind without a CIK."""
    mask = universe[U.verify] | (
        universe[U.kind].isin([k.value for k in EVENT_KINDS]) & universe[U.cik].isna()
    )
    cols = [U.market, U.kind, U.underlying, U.cik, U.name, U.exclude_reason, U.verify]
    return universe.loc[mask, cols].reset_index(drop=True)
