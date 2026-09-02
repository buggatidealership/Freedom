"""Local parquet archive of Hyperliquid candles, funding and asset contexts.

Hyperliquid serves only the most recent 5000 candles per (market, interval) (1m ~ 3.5 days),
so fine-grained perp history has to be captured continuously. `archive_markets` is what the
scheduled `freedom archive` job runs; `load_archive` is how the targets module reads it back.

Layout under `settings.archive_dir`. Market names have ':' replaced by '_' (`xyz:NVDA` lives
in `candles/xyz_NVDA/`); build paths with `candle_path` / `funding_path` / `ctx_path` rather
than by hand:

    candles/<market>/<interval>.parquet   schemas.C columns, source='hl_archive', unique sorted t
    candles/<market>/funding.parquet      market, t, funding_rate, premium (unique sorted t)
    ctx/<dex>/<YYYY-MM-DD>.parquet        one row per (market, snapshot time) of metaAndAssetCtxs

Funding `t` is the settlement *hour*: the server reports the settlement instant (~50-120 ms
after the hour) and `HyperliquidClient.funding_history` floors it, so `funding.parquet` joins
with `1h.parquet` on `t` directly. Every datetime column is datetime64[ns, UTC].

Every file is written atomically (tmp file then `replace`) and appended with dedup, keeping the
most recently fetched version of a row, so a bar that was still forming when archived is
overwritten by its final values on the next run. If a run finds the archive older than the
server's horizon (a missed schedule), the bars in between are gone for good: the run still
appends what the server has and reports the hole in the summary's `error` column.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pyarrow.parquet as pq

from ..config import Settings
from ..data.base import ProviderUnavailable, utcnow
from ..data.hyperliquid import (
    CANDLE_COLUMNS,
    FUNDING_COLUMNS,
    MAX_CANDLES_PER_REQUEST,
    HyperliquidClient,
    empty_candles,
    from_ms,
    interval_ms,
    to_ms,
)
from ..schemas import UTC, C, PriceSource, U
from ..timeutil import to_utc

CANDLES_SUBDIR = "candles"
CTX_SUBDIR = "ctx"
FUNDING_FILE = "funding"
# First funding pull for a market whose listing date cannot be determined (no daily candles);
# markets with candles are bootstrapped from their listing date instead.
FUNDING_BOOTSTRAP_DAYS = 35

# Failures of one item that must not abort the scheduled run: network / HTTP status errors and
# malformed responses (json.JSONDecodeError and the meta/ctx length mismatch of `ctx_frame`
# are ValueErrors). They are recorded in the summary's `error` column.
RECOVERABLE_ERRORS: tuple[type[Exception], ...] = (httpx.HTTPError, ValueError)

SUMMARY_COLUMNS: list[str] = ["market", "interval", "rows_added", "first_t", "last_t",
                              "rows_total", "error"]
CTX_COLUMNS: list[str] = ["t", "dex", "market", "funding", "open_interest", "prev_day_px",
                          "day_ntl_vlm", "premium", "oracle_px", "mark_px", "mid_px",
                          "impact_bid", "impact_ask", "day_base_vlm"]


# ---- paths -----------------------------------------------------------------------------------
def market_dir_name(market: str) -> str:
    return market.replace(":", "_")


def candle_path(settings: Settings, market: str, interval: str) -> Path:
    return settings.archive_dir / CANDLES_SUBDIR / market_dir_name(market) / f"{interval}.parquet"


def funding_path(settings: Settings, market: str) -> Path:
    return settings.archive_dir / CANDLES_SUBDIR / market_dir_name(market) / f"{FUNDING_FILE}.parquet"


def ctx_path(settings: Settings, dex: str, day: date) -> Path:
    return settings.archive_dir / CTX_SUBDIR / dex / f"{day.isoformat()}.parquet"


# ---- parquet helpers ---------------------------------------------------------------------------
def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a sibling tmp file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_parquet_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def archived_rows(path: Path) -> int:
    """Row count of the parquet at `path` from its footer (no data read); 0 without a file."""
    return int(pq.read_metadata(path).num_rows) if path.exists() else 0


def _ensure_utc(s: pd.Series) -> pd.Series:
    """Datetime column as tz-aware UTC (parquet round-trips keep the zone; be defensive)."""
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize(UTC)
    return s.dt.tz_convert(UTC)


def append_dedup(path: Path, new: pd.DataFrame, *, key: list[str],
                 sort: list[str]) -> tuple[pd.DataFrame, int]:
    """Merge `new` into the parquet at `path`: rows sharing `key` keep the newest (from `new`),
    output sorted by `sort`. Returns (merged frame, rows added). Nothing is written when
    `new` is empty."""
    old = read_parquet_or_none(path)
    n_old = 0 if old is None else len(old)
    if new.empty:
        return (old if old is not None else new), 0
    merged = new if old is None or old.empty else pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=key, keep="last")
    merged = merged.sort_values(sort, kind="mergesort").reset_index(drop=True)
    write_parquet_atomic(merged, path)
    return merged, len(merged) - n_old


# ---- per-item archivers ---------------------------------------------------------------------------
def _summary_row(market: str, interval: str, added: int, fetched: pd.DataFrame, total: int,
                 tcol: str = C.t, error: str | None = None) -> dict:
    return {
        "market": market, "interval": interval, "rows_added": int(added),
        "first_t": fetched[tcol].iloc[0] if len(fetched) else pd.NaT,
        "last_t": fetched[tcol].iloc[-1] if len(fetched) else pd.NaT,
        "rows_total": int(total), "error": error,
    }


def _error_row(market: str, interval: str, path: Path, exc: Exception) -> dict:
    """Summary row for an item whose fetch failed: nothing added, rows_total is what the
    archive already holds, error names the exception."""
    return _summary_row(market, interval, 0, empty_candles(), archived_rows(path),
                        error=f"{type(exc).__name__}: {exc}")


def _gap_message(interval: str, gap_from_ms: int, gap_to_ms: int, step: int) -> str:
    n_missing = (gap_to_ms - gap_from_ms) // step
    first, last = from_ms([gap_from_ms, gap_to_ms - step])
    return (f"gap: {n_missing} {interval} bars from {first.isoformat()} to {last.isoformat()} "
            f"were never archived (archive fell behind the server's "
            f"{MAX_CANDLES_PER_REQUEST}-bar horizon)")


def archive_candles(client: HyperliquidClient, settings: Settings, market: str, interval: str,
                    now: pd.Timestamp) -> dict:
    """Pull the newest candles the API serves for (market, interval) and append them.

    The window is the server's 5000-bar horizon ending at the bar in progress; when the archive
    already reaches into that horizon only the part from the last archived bar (inclusive, so a
    previously partial bar is refreshed) is requested. When the archive ends *before* the
    horizon (a missed schedule) the bars in between can no longer be fetched: the horizon is
    archived as usual and the summary row's `error` reports the hole ("gap: ...")."""
    step = interval_ms(interval)
    path = candle_path(settings, market, interval)
    now_ms = to_ms(now)
    end_ms = (now_ms // step) * step + step  # include the bar in progress
    horizon_ms = end_ms - MAX_CANDLES_PER_REQUEST * step
    start_ms, gap_from_ms = horizon_ms, None
    old = read_parquet_or_none(path)
    if old is not None and len(old):
        last_ms = to_ms(_ensure_utc(old[C.t]).iloc[-1])
        start_ms = max(horizon_ms, last_ms)
        if last_ms + step < horizon_ms:
            gap_from_ms = last_ms + step  # first bar the archive lacks
    fetched = client.candles(market, interval, from_ms([start_ms]).iloc[0],
                             from_ms([end_ms]).iloc[0], cache_ttl=None)
    fetched = fetched.assign(**{C.source: PriceSource.hl_archive.value})
    merged, added = append_dedup(path, fetched, key=[C.t], sort=[C.t])
    error = None
    if gap_from_ms is not None:
        gap_to_ms = to_ms(fetched[C.t].iloc[0]) if len(fetched) else end_ms
        error = _gap_message(interval, gap_from_ms, gap_to_ms, step)
    return _summary_row(market, interval, added, fetched, len(merged), error=error)


def archive_funding(client: HyperliquidClient, settings: Settings, market: str,
                    now: pd.Timestamp) -> dict:
    """Append hourly funding from the hour after the last archived one. The first pull for a
    market starts at its listing date (`client.listing_start`, one cached 1d request) so the
    whole funding history is captured; FUNDING_BOOTSTRAP_DAYS back when it has no daily candles."""
    path = funding_path(settings, market)
    now_utc = to_utc(now, assume_tz=UTC)
    old = read_parquet_or_none(path)
    if old is not None and len(old):
        start = _ensure_utc(old["t"]).iloc[-1].floor("h") + pd.Timedelta(hours=1)
    else:
        listing = client.listing_start(market)
        start = listing if listing is not None else now_utc - pd.Timedelta(days=FUNDING_BOOTSTRAP_DAYS)
    fetched = client.funding_history(market, start, now_utc, cache_ttl=None)[FUNDING_COLUMNS]
    merged, added = append_dedup(path, fetched, key=["t"], sort=["t"])
    return _summary_row(market, FUNDING_FILE, added, fetched, len(merged), tcol="t")


def _f(ctx: dict, key: str) -> float:
    v = ctx.get(key)
    return float(v) if v is not None else float("nan")


def ctx_frame(dex: str, meta: dict, ctxs: list[dict], t: pd.Timestamp) -> pd.DataFrame:
    """One row per non-delisted market of a metaAndAssetCtxs response, stamped with `t`.
    Raises ValueError when `meta['universe']` and `ctxs` differ in length."""
    rows = []
    for asset, ctx in zip(meta.get("universe", []), ctxs, strict=True):
        if asset.get("isDelisted", False):
            continue
        impact = ctx.get("impactPxs") or [None, None]
        rows.append({
            "t": t, "dex": dex, "market": asset["name"],
            "funding": _f(ctx, "funding"), "open_interest": _f(ctx, "openInterest"),
            "prev_day_px": _f(ctx, "prevDayPx"), "day_ntl_vlm": _f(ctx, "dayNtlVlm"),
            "premium": _f(ctx, "premium"), "oracle_px": _f(ctx, "oraclePx"),
            "mark_px": _f(ctx, "markPx"), "mid_px": _f(ctx, "midPx"),
            "impact_bid": float(impact[0]) if impact[0] is not None else float("nan"),
            "impact_ask": float(impact[1]) if impact[1] is not None else float("nan"),
            "day_base_vlm": _f(ctx, "dayBaseVlm"),
        })
    df = pd.DataFrame(rows, columns=CTX_COLUMNS)
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.as_unit("ns")
    return df


def snapshot_ctx(client: HyperliquidClient, settings: Settings, dex: str,
                 now: pd.Timestamp) -> dict:
    """Append a snapshot of the dex's asset contexts to ctx/<dex>/<utc date>.parquet."""
    meta, ctxs = client.meta_and_asset_ctxs(dex)
    t = to_utc(now, assume_tz=UTC)
    frame = ctx_frame(dex, meta, ctxs, t)
    path = ctx_path(settings, dex, t.date())
    merged, added = append_dedup(path, frame, key=["market", "t"], sort=["t", "market"])
    return _summary_row(dex, CTX_SUBDIR, added, frame, len(merged), tcol="t")


def has_event_universe(settings: Settings) -> bool:
    """True when data/universe.parquet exists and marks at least one market as part of the
    event universe (the scheduled job runs `freedom universe` before `freedom archive`)."""
    path = settings.universe_path
    if not path.exists():
        return False
    try:
        if U.in_event_universe not in pq.read_schema(path).names:
            return False
        flags = pd.read_parquet(path, columns=[U.in_event_universe])[U.in_event_universe]
    except (OSError, ValueError, KeyError):
        return False
    return bool(flags.fillna(False).astype(bool).any())


def snapshot_consensus_row(settings: Settings, now: pd.Timestamp) -> dict:
    """Consensus snapshot for the event universe (events.snapshot_consensus: the FMP earnings
    calendar and the Nasdaq calendar for the coming days) appended to
    archive/consensus/<UTC date>.parquet, as a summary row with interval='consensus'.

    `archive_markets` calls this only when `has_event_universe` holds: without an event
    universe there is nothing to snapshot consensus for. A provider that cannot run
    (FMP_API_KEY unset, daily budget spent) or fails is recorded in the row's `error` column
    instead of aborting the candle archive."""
    from ..events import CONSENSUS_ITEM, consensus_path, snapshot_consensus

    path = consensus_path(settings, now.date())
    try:
        written = snapshot_consensus(settings, now=now)
    except (ProviderUnavailable, *RECOVERABLE_ERRORS) as exc:
        return _summary_row(CONSENSUS_ITEM, CONSENSUS_ITEM, 0, pd.DataFrame(), archived_rows(path),
                            error=f"{type(exc).__name__}: {exc}")
    return _summary_row(CONSENSUS_ITEM, CONSENSUS_ITEM, len(written), written, archived_rows(path),
                        tcol="snapshot_time")


# ---- public API ----------------------------------------------------------------------------------
def archive_markets(settings: Settings, markets: list[str], intervals: list[str], *,
                    client: HyperliquidClient | None = None,
                    now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Archive candles (every interval), funding and asset-context snapshots for `markets`.

    Returns one summary row per (market, interval) plus one per market for funding
    (interval='funding'), one per dex for the context snapshot (interval='ctx') and, when
    data/universe.parquet has an event universe, one for the consensus snapshot (market and
    interval='consensus', see `snapshot_consensus_row`):
    columns market, interval, rows_added, first_t, last_t, rows_total, error.

    `error` is None for a clean item. It reads "<ExceptionType>: ..." for an item whose fetch
    failed (RECOVERABLE_ERRORS: network / HTTP status errors and malformed responses are
    caught per item so the rest of the run continues; rows_total then reports what the archive
    already holds) and "gap: ..." for candles whose archive had fallen behind the server
    horizon (the newest bars were still appended). Anything else propagates."""
    for interval in intervals:
        interval_ms(interval)  # fail early on a typo rather than after an hour of pulls
    client = client or HyperliquidClient(settings)
    now_ts = to_utc(now, assume_tz=UTC) if now is not None else pd.Timestamp(utcnow())
    rows: list[dict] = []
    for market in markets:
        for interval in intervals:
            try:
                rows.append(archive_candles(client, settings, market, interval, now_ts))
            except RECOVERABLE_ERRORS as exc:
                rows.append(_error_row(market, interval, candle_path(settings, market, interval),
                                       exc))
        try:
            rows.append(archive_funding(client, settings, market, now_ts))
        except RECOVERABLE_ERRORS as exc:
            rows.append(_error_row(market, FUNDING_FILE, funding_path(settings, market), exc))
    for dex in sorted({m.split(":", 1)[0] for m in markets if ":" in m}):
        try:
            rows.append(snapshot_ctx(client, settings, dex, now_ts))
        except RECOVERABLE_ERRORS as exc:
            rows.append(_error_row(dex, CTX_SUBDIR, ctx_path(settings, dex, now_ts.date()), exc))
    if has_event_universe(settings):
        rows.append(snapshot_consensus_row(settings, now_ts))
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    for col in ("first_t", "last_t"):
        summary[col] = pd.to_datetime(summary[col], utc=True)
    return summary


def load_archive(settings: Settings, market: str, interval: str, start: pd.Timestamp,
                 end: pd.Timestamp) -> pd.DataFrame:
    """Archived candles with start <= t < end as a schemas.C frame, source='hl_archive'.
    Empty frame (right columns and dtypes) when nothing is archived for the window."""
    path = candle_path(settings, market, interval)
    df = read_parquet_or_none(path)
    if df is None or df.empty:
        return empty_candles()
    t = _ensure_utc(df[C.t])
    start_utc, end_utc = to_utc(start, assume_tz=UTC), to_utc(end, assume_tz=UTC)
    mask = (t >= start_utc) & (t < end_utc)
    if not mask.any():
        return empty_candles()
    out = df.loc[mask].copy()
    out[C.t] = t[mask]
    out[C.t_end] = _ensure_utc(out[C.t_end])
    out[C.market] = market
    out[C.interval] = interval
    out[C.source] = PriceSource.hl_archive.value
    out = out.drop_duplicates(subset=C.t, keep="last").sort_values(C.t, kind="mergesort")
    return out[CANDLE_COLUMNS].reset_index(drop=True)
