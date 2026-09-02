"""Local parquet archive of Hyperliquid candles, funding and asset contexts.

Hyperliquid serves only the most recent 5000 candles per (market, interval) (1m ~ 3.5 days),
so fine-grained perp history has to be captured continuously. `archive_markets` is what the
scheduled `freedom archive` job runs; `load_archive` is how the targets module reads it back.

Layout under `settings.archive_dir` (market names have ':' replaced by '_'):

    candles/<market>/<interval>.parquet   schemas.C columns, source='hl_archive', unique sorted t
    candles/<market>/funding.parquet      market, t, funding_rate, premium (unique sorted t)
    ctx/<dex>/<YYYY-MM-DD>.parquet        one row per (market, snapshot time) of metaAndAssetCtxs

Every file is written atomically (tmp file then `replace`) and appended with dedup, keeping the
most recently fetched version of a row, so a bar that was still forming when archived is
overwritten by its final values on the next run.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from ..config import Settings
from ..data.base import utcnow
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
from ..schemas import UTC, C, PriceSource
from ..timeutil import to_utc

CANDLES_SUBDIR = "candles"
CTX_SUBDIR = "ctx"
FUNDING_FILE = "funding"
FUNDING_BOOTSTRAP_DAYS = 35  # first pull of funding for a market when nothing is archived yet

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


def archive_candles(client: HyperliquidClient, settings: Settings, market: str, interval: str,
                    now: pd.Timestamp) -> dict:
    """Pull the newest candles the API serves for (market, interval) and append them.

    The window is the server's 5000-bar horizon ending at the bar in progress; when the archive
    already reaches into that horizon only the part from the last archived bar (inclusive, so a
    previously partial bar is refreshed) is requested."""
    step = interval_ms(interval)
    path = candle_path(settings, market, interval)
    now_ms = to_ms(now)
    end_ms = (now_ms // step) * step + step  # include the bar in progress
    start_ms = end_ms - MAX_CANDLES_PER_REQUEST * step
    old = read_parquet_or_none(path)
    if old is not None and len(old):
        last_ms = to_ms(_ensure_utc(old[C.t]).iloc[-1])
        start_ms = max(start_ms, last_ms)
    fetched = client.candles(market, interval, from_ms([start_ms]).iloc[0],
                             from_ms([end_ms]).iloc[0], cache_ttl=None)
    fetched = fetched.assign(**{C.source: PriceSource.hl_archive.value})
    merged, added = append_dedup(path, fetched, key=[C.t], sort=[C.t])
    return _summary_row(market, interval, added, fetched, len(merged))


def archive_funding(client: HyperliquidClient, settings: Settings, market: str,
                    now: pd.Timestamp) -> dict:
    """Append hourly funding since the last archived entry (or FUNDING_BOOTSTRAP_DAYS back)."""
    path = funding_path(settings, market)
    old = read_parquet_or_none(path)
    if old is not None and len(old):
        start = _ensure_utc(old["t"]).iloc[-1] + pd.Timedelta(milliseconds=1)
    else:
        start = to_utc(now, assume_tz=UTC) - pd.Timedelta(days=FUNDING_BOOTSTRAP_DAYS)
    fetched = client.funding_history(market, start, now, cache_ttl=None)[FUNDING_COLUMNS]
    merged, added = append_dedup(path, fetched, key=["t"], sort=["t"])
    return _summary_row(market, FUNDING_FILE, added, fetched, len(merged), tcol="t")


def _f(ctx: dict, key: str) -> float:
    v = ctx.get(key)
    return float(v) if v is not None else float("nan")


def ctx_frame(dex: str, meta: dict, ctxs: list[dict], t: pd.Timestamp) -> pd.DataFrame:
    """One row per non-delisted market of a metaAndAssetCtxs response, stamped with `t`."""
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
    df["t"] = pd.to_datetime(df["t"], utc=True)
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


# ---- public API ----------------------------------------------------------------------------------
def archive_markets(settings: Settings, markets: list[str], intervals: list[str], *,
                    client: HyperliquidClient | None = None,
                    now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Archive candles (every interval), funding and asset-context snapshots for `markets`.

    Returns one summary row per (market, interval) plus one per market for funding
    (interval='funding') and one per dex for the context snapshot (interval='ctx'):
    columns market, interval, rows_added, first_t, last_t, rows_total, error. HTTP failures
    are recorded in `error` and do not stop the run; anything else propagates."""
    for interval in intervals:
        interval_ms(interval)  # fail early on a typo rather than after an hour of pulls
    client = client or HyperliquidClient(settings)
    now_ts = to_utc(now, assume_tz=UTC) if now is not None else pd.Timestamp(utcnow())
    rows: list[dict] = []
    for market in markets:
        for interval in intervals:
            try:
                rows.append(archive_candles(client, settings, market, interval, now_ts))
            except httpx.HTTPError as exc:
                rows.append(_summary_row(market, interval, 0, empty_candles(), 0, error=str(exc)))
        try:
            rows.append(archive_funding(client, settings, market, now_ts))
        except httpx.HTTPError as exc:
            rows.append(_summary_row(market, FUNDING_FILE, 0, empty_candles(), 0, error=str(exc)))
    for dex in sorted({m.split(":", 1)[0] for m in markets if ":" in m}):
        try:
            rows.append(snapshot_ctx(client, settings, dex, now_ts))
        except httpx.HTTPError as exc:
            rows.append(_summary_row(dex, CTX_SUBDIR, 0, empty_candles(), 0, error=str(exc)))
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
