from __future__ import annotations

from pathlib import Path

import pytest

from freedom.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "data", reports_dir=tmp_path / "reports",
                 configs_dir=Path(__file__).parent.parent / "configs",
                 fmp_api_key="test", alphavantage_api_key=None, _env_file=None)
    s.ensure_dirs()
    return s
