"""Timezone and exchange-calendar helpers. All public functions accept and return UTC."""

from __future__ import annotations

from datetime import datetime, time
from functools import lru_cache

import pandas as pd

from .schemas import NY, UTC


@lru_cache(maxsize=1)
def xnys():
    import exchange_calendars as xc

    return xc.get_calendar("XNYS")


def to_utc(ts: datetime | pd.Timestamp | str, assume_tz: str | None = None) -> pd.Timestamp:
    """Return a tz-aware UTC Timestamp. Naive inputs need `assume_tz` (e.g. NY for FMP)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        if assume_tz is None:
            raise ValueError(f"naive timestamp {ts!r} needs assume_tz")
        t = t.tz_localize(assume_tz, ambiguous="NaT", nonexistent="shift_forward")
        if pd.isna(t):
            raise ValueError(f"ambiguous DST timestamp {ts!r} in {assume_tz}")
    return t.tz_convert(UTC)


def to_ny(ts: pd.Timestamp) -> pd.Timestamp:
    return to_utc(ts).tz_convert(NY)


def next_open_after(ts: pd.Timestamp) -> pd.Timestamp:
    """First XNYS regular-session open strictly after `ts`."""
    cal = xnys()
    return pd.Timestamp(cal.next_open(to_utc(ts))).tz_convert(UTC)


def next_close_after(ts: pd.Timestamp) -> pd.Timestamp:
    """First XNYS regular-session close strictly after `ts`."""
    cal = xnys()
    return pd.Timestamp(cal.next_close(to_utc(ts))).tz_convert(UTC)


def is_rth(ts: pd.Timestamp) -> bool:
    return bool(xnys().is_open_on_minute(to_utc(ts)))


def is_session_day(ts: pd.Timestamp) -> bool:
    return bool(xnys().is_session(to_ny(ts).normalize().tz_localize(None)))


def ny_clock(ts: pd.Timestamp) -> time:
    return to_ny(ts).time()


def classify_timing(t0: pd.Timestamp) -> str:
    """AMC / BMO / RTH / CLOSED for a release instant, in New York terms."""
    from .schemas import Timing

    if is_rth(t0):
        return Timing.rth
    if not is_session_day(t0):
        return Timing.closed
    clock = ny_clock(t0)
    if clock < time(9, 30):
        return Timing.bmo
    return Timing.amc
