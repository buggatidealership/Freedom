"""The committed override files parse through the loaders that read them at run time. A YAML
syntax error here would otherwise surface only on the runner, as a failed card."""

from __future__ import annotations

import pandas as pd
import yaml

from freedom.events import (
    _load_manual_overrides,
    _load_release_clock_overrides,
    _load_report_date_overrides,
)
from tests.conftest import REPO_CONFIGS


def test_override_files_are_valid_yaml_mappings():
    for name in ("t0_overrides.yaml", "report_date_overrides.yaml", "release_clock_overrides.yaml"):
        raw = yaml.safe_load((REPO_CONFIGS / name).read_text(encoding="utf-8")) or {}
        assert isinstance(raw, dict), name
        assert all(str(k).strip() for k in raw), name


def test_override_loaders_read_the_committed_files(settings):
    manual = _load_manual_overrides(settings)
    assert all(isinstance(v, pd.Timestamp) and v.tzinfo is not None for v in manual.values())
    assert all(":" in k and k == k.upper() for k in manual)
    dates = _load_report_date_overrides(settings)
    assert all(":" in k for k in dates)
    clocks = _load_release_clock_overrides(settings)
    assert set(clocks) >= {"ASML", "TSM"}
