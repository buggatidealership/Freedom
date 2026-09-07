from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from freedom.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


REPO_CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings on a scratch data dir with a copy of the repo's YAML configs. The committed SEC
    bundle parquets are deliberately left out: with them in reach, fixture events whose fake
    EDGAR answers nothing pick up real filings and facts."""
    configs = tmp_path / "configs"
    configs.mkdir()
    for f in REPO_CONFIGS.glob("*.yaml"):
        shutil.copy(f, configs / f.name)
    s = Settings(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports", configs_dir=configs,
                 fmp_api_key="test", alphavantage_api_key=None, _env_file=None)
    s.ensure_dirs()
    return s
