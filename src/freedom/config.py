"""Runtime settings. Everything comes from environment variables or a .env file.

Keys use their conventional names (FMP_API_KEY, ALPHAVANTAGE_API_KEY, ANTHROPIC_API_KEY);
harness options use the FREEDOM_ prefix (FREEDOM_DATA_DIR, FREEDOM_TAKER_FEE_BPS, ...).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FREEDOM_", env_file=".env", extra="ignore")

    # --- paths -------------------------------------------------------------------------------
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    configs_dir: Path = Path("configs")

    # --- API keys (optional; features degrade with a clear message when missing) --------------
    fmp_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("FMP_API_KEY", "FREEDOM_FMP_API_KEY")
    )
    alphavantage_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ALPHAVANTAGE_API_KEY", "FREEDOM_ALPHAVANTAGE_API_KEY"),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "FREEDOM_ANTHROPIC_API_KEY"),
    )
    sec_user_agent: str = "freedom-harness research@example.com"

    # --- endpoints and budgets ---------------------------------------------------------------
    hyperliquid_api_url: str = "https://api.hyperliquid.xyz"
    hyperliquid_weight_per_minute: int = 1100  # documented limit is 1200; keep headroom
    fmp_base_url: str = "https://financialmodelingprep.com"
    fmp_daily_budget: int = 240  # free plan documented at 250/day
    alphavantage_daily_budget: int = 20  # free plan is 25/day
    nasdaq_requests_per_minute: int = 30
    sec_requests_per_second: int = 8  # SEC fair-access limit is 10/s
    http_timeout_seconds: float = 30.0
    cache_ttl_seconds: int = 7 * 24 * 3600
    live_cache_ttl_seconds: int = 60

    # --- target and evaluation defaults -------------------------------------------------------
    horizon_hours: int = 24
    # P0 back-off for 8-K times (targets.p0_buffer_for; the reaction and pre-release feature
    # groups read it through FeatureContext): acceptance trails the wire by 25-134 s (measured)
    p0_buffer_minutes_sec_8k: float = 3.0
    taker_fee_bps: float = 4.5  # conservative; HIP-3 growth-mode fees are lower, see docs
    slippage_floor_bps: float = 5.0  # per leg, floor of the execution-cost model
    slippage_range_coeff: float = 0.25  # per leg: + coeff * range_bps, range_bps = 1e4*(high-low)/open of the execution bar
    max_fill_lag_minutes: float = 5.0  # a fill later than this after the signal is not traded
    gross_exposure_cap: float = 1.0  # equal_split capital rule across overlapping positions
    trade_threshold: float = 0.0  # simulation: no trade unless |p_up - 0.5| >= this
    no_trade_band: float = 0.10  # prediction card: NO TRADE unless |p_up - 0.5| >= this
    target_vol: float = 0.03  # by_magnitude sizing: size = target_vol / predicted |r_24h|, capped at 1
    min_train_events: int = 120
    embargo_days: int = 2
    min_t0_confidence: float = 0.6  # dataset filter; fixed per run, never a search dimension
    holdout_season: str | None = "2026Q3"  # pinned; scored only by `freedom evaluate --final`
    random_seed: int = 7

    # --- derived paths -----------------------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def archive_dir(self) -> Path:
        return self.data_dir / "archive"

    @property
    def universe_path(self) -> Path:
        return self.data_dir / "universe.parquet"

    @property
    def events_path(self) -> Path:
        return self.data_dir / "events.parquet"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "dataset.parquet"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def optuna_db(self) -> Path:
        return self.data_dir / "optuna.db"

    @property
    def targets_path(self) -> Path:
        return self.data_dir / "targets.parquet"

    @property
    def consensus_archive_dir(self) -> Path:
        return self.archive_dir / "consensus"

    @property
    def holdout_log_path(self) -> Path:
        return self.reports_dir / "holdout_scorings.jsonl"

    @property
    def universe_overrides_path(self) -> Path:
        return self.configs_dir / "universe_overrides.yaml"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.cache_dir, self.archive_dir, self.models_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)


def get_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]
