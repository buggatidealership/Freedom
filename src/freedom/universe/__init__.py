"""Universe: which Hyperliquid markets are equities with earnings events.

`build_universe` = live markets (HyperliquidClient.all_markets) + automatic SEC ticker match
+ configs/universe_overrides.yaml (authoritative) + listing_start and 30-day median notional.
Output frame uses schemas.U columns and is written to settings.universe_path.
"""

from __future__ import annotations

import pandas as pd

from ..config import Settings


def load_overrides(settings: Settings) -> dict:
    """Parsed universe_overrides.yaml: {'defaults': {...}, 'markets': {market: {...}}}."""
    raise NotImplementedError


def classify(markets: pd.DataFrame, sec_tickers: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """Pure function: assign kind/underlying/cik/verify/exclude_reason to every market row.
    Override entries win; unmatched markets default to kind='other', verify=True."""
    raise NotImplementedError


def choose_primary(universe: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """Set is_primary: for each underlying keep the market with the highest
    median_notional_30d, tie-break by defaults.dex_priority."""
    raise NotImplementedError


def build_universe(settings: Settings, *, write: bool = True) -> pd.DataFrame:
    raise NotImplementedError


def load_universe(settings: Settings) -> pd.DataFrame:
    raise NotImplementedError


def event_universe(universe: pd.DataFrame) -> pd.DataFrame:
    """Rows with in_event_universe == True (equity kinds, primary market)."""
    raise NotImplementedError
