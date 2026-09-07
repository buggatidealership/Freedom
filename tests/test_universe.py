import json
from pathlib import Path

import pandas as pd

from freedom import universe as universe_mod
from freedom.data import archive as archive_mod
from freedom.schemas import U
from freedom.universe import choose_primary, classify, load_overrides, verification_report
from tests.fakes import FakeHyperliquidInfo

FIX = Path(__file__).parent / "fixtures"


def _markets_from_fixtures() -> pd.DataFrame:
    rows = []
    for meta_file in sorted((FIX / "hyperliquid").glob("meta_*.json")):
        dex = meta_file.stem.split("_", 1)[1]
        for a in json.load(open(meta_file))["universe"]:
            if a.get("isDelisted"):
                continue
            rows.append({
                "market": a["name"], "dex": dex, "symbol": a["name"].split(":", 1)[1],
                "max_leverage": a.get("maxLeverage"), "growth_mode": a.get("growthMode") == "enabled",
                "deployer_fee_scale": float(a.get("deployerFeeScale", 1.0)),
                "only_isolated": bool(a.get("onlyIsolated", False)),
            })
    return pd.DataFrame(rows)


def _sec_head() -> pd.DataFrame:
    d = json.load(open(FIX / "sec" / "company_tickers_head.json"))
    return pd.DataFrame([{"ticker": v["ticker"], "cik": v["cik_str"], "title": v["title"]} for v in d.values()])


def test_classify_uses_overrides_over_sec_match(settings):
    u = classify(_markets_from_fixtures(), _sec_head(), load_overrides(settings))
    by = u.set_index(U.market)
    assert by.loc["xyz:NVDA", U.kind] == "equity_us"
    assert int(by.loc["xyz:NVDA", U.cik]) == 1045810
    assert by.loc["xyz:NVDA", U.underlying] == "NVDA"
    assert not by.loc["xyz:NVDA", U.verify]
    # SEC false positives are corrected by the override file
    assert by.loc["xyz:GOLD", U.kind] == "commodity"
    assert by.loc["xyz:CL", U.kind] == "commodity"
    assert by.loc["xyz:GOLD", U.exclude_reason] == "kind=commodity"
    assert by.loc["xyz:TSM", U.kind] == "equity_fpi"
    # crypto dex never enters the event universe
    assert by.loc["hyna:BTC", U.kind] == "crypto"


def test_unknown_market_is_flagged_for_verification(settings):
    markets = pd.DataFrame([{"market": "zzz:FOO", "dex": "zzz", "symbol": "FOO", "max_leverage": 5,
                             "growth_mode": False, "deployer_fee_scale": 1.0, "only_isolated": False}])
    u = classify(markets, _sec_head(), load_overrides(settings))
    assert u.loc[0, U.kind] == "other" and bool(u.loc[0, U.verify])
    # SEC-matched but not curated -> equity_us guess, still needs verification
    markets2 = markets.assign(market="zzz:AAPL", symbol="AAPL")
    u2 = classify(markets2, _sec_head(), load_overrides(settings))
    assert u2.loc[0, U.kind] == "equity_us" and bool(u2.loc[0, U.verify]) and int(u2.loc[0, U.cik]) == 320193


def test_choose_primary_prefers_volume_then_dex_priority(settings):
    ov = load_overrides(settings)
    u = classify(_markets_from_fixtures(), _sec_head(), ov)
    u[U.median_notional_30d] = float("nan")
    u.loc[u[U.market] == "para:AVGO", U.median_notional_30d] = 5e6
    u.loc[u[U.market] == "xyz:AVGO", U.median_notional_30d] = 1e6
    u = choose_primary(u, ov)
    by = u.set_index(U.market)
    assert bool(by.loc["para:AVGO", U.is_primary]) and not bool(by.loc["xyz:AVGO", U.is_primary])
    # equal (missing) volume -> dex priority: xyz before para
    assert bool(by.loc["xyz:NET", U.is_primary]) and not bool(by.loc["para:NET", U.is_primary])
    assert bool(by.loc["xyz:NVDA", U.in_event_universe])
    assert not bool(by.loc["xyz:GOLD", U.in_event_universe])
    assert u[U.in_event_universe].sum() == u.loc[u[U.in_event_universe], U.underlying].nunique()


def test_build_universe_writes_atomically_with_a_stable_clock(settings, monkeypatch):
    FakeHyperliquidInfo().install(monkeypatch)  # markets from the fixtures; the SEC ticker map is unavailable
    clocks = []

    def listing_and_volume(hl, market, now):
        clocks.append(now)
        return None, float("nan")

    monkeypatch.setattr(universe_mod, "_listing_and_volume", listing_and_volume)
    writes = []
    real_write = archive_mod.write_parquet_atomic

    def write(df, path):
        writes.append(path)
        real_write(df, path)

    monkeypatch.setattr(archive_mod, "write_parquet_atomic", write)
    u = universe_mod.build_universe(settings)
    # one clock for the whole pull, floored to the hour so the candle request (the cache key) repeats
    assert clocks and len(set(clocks)) == 1 and clocks[0] == clocks[0].floor("h")
    assert writes == [settings.universe_path] and settings.universe_path.exists()
    assert not settings.universe_path.with_name(settings.universe_path.name + ".tmp").exists()
    assert len(pd.read_parquet(settings.universe_path)) == len(u) and bool(u[U.in_event_universe].any())
    settings.universe_path.unlink()
    universe_mod.build_universe(settings, write=False)
    assert not settings.universe_path.exists() and len(writes) == 1


def test_verification_report_lists_uncertain_rows(settings):
    ov = load_overrides(settings)
    u = choose_primary(classify(_markets_from_fixtures(), _sec_head(), ov), ov)
    rep = verification_report(u)
    assert "xyz:SPCX" in set(rep[U.market])
    assert "xyz:NVDA" not in set(rep[U.market])


def _write_cik_yaml(path: Path, tickers: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"tickers": tickers}), encoding="utf-8")


def test_cik_map_loads_merges_and_round_trips(settings, tmp_path):
    settings.configs_dir = tmp_path / "cfg2"  # not the fixture's copy of the repo configs
    assert len(universe_mod.load_cik_map(settings)) == 0  # absent file: empty, not an error
    _write_cik_yaml(settings.cik_map_path, {"nvda": {"cik": 1045810, "title": "NVIDIA CORP"},
                                            "AAPL": 320193, "BAD": None, "ZM": {"title": "no cik"}})
    m = universe_mod.load_cik_map(settings)
    assert sorted(m["ticker"]) == ["AAPL", "NVDA"] and dict(zip(m["ticker"], m["cik"], strict=True)) == {"NVDA": 1045810, "AAPL": 320193}
    assert m.set_index("ticker").loc["AAPL", "title"] == "AAPL"  # bare integer: the ticker names it
    # live rows win, bundled rows fill the gaps
    live = pd.DataFrame({"ticker": ["NVDA"], "cik": [1], "title": ["live"]})
    merged = universe_mod.merge_ticker_maps(live, m).set_index("ticker")
    assert int(merged.loc["NVDA", "cik"]) == 1 and int(merged.loc["AAPL", "cik"]) == 320193
    assert len(universe_mod.merge_ticker_maps(None, None)) == 0
    # write from a universe frame and read back: symbol and a different underlying both map
    u = pd.DataFrame({U.market: ["xyz:NVDA", "para:NVDA", "xyz:FOO", "xyz:GOLD"], U.symbol: ["NVDA", "NVDA", "FOO", "GOLD"],
                      U.kind: ["equity_us", "equity_us", "equity_fpi", "commodity"],
                      U.underlying: ["NVDA", "NVDA", "FOOX", None], U.cik: [1045810, 1045810, 77, 5],
                      U.name: ["NVIDIA CORP", "NVIDIA CORP", None, "Gold.com"]})
    u[U.cik] = u[U.cik].astype("Int64")
    path, n = universe_mod.write_cik_map(settings, u)
    assert path == settings.cik_map_path and n == 3
    back = universe_mod.load_cik_map(settings).set_index("ticker")
    assert back.index.tolist() == ["FOO", "FOOX", "NVDA"] and int(back.loc["FOOX", "cik"]) == 77
    assert back.loc["FOO", "title"] == "FOO" and "GOLD" not in back.index  # non-event kinds stay out
    assert path.read_text(encoding="utf-8").startswith(universe_mod.CIK_MAP_HEADER)


def test_build_universe_takes_ciks_from_the_committed_map_when_sec_is_blocked(settings, monkeypatch):
    """The data job's universe had no CIK at all (2026-09-07: sec.gov 403 from the runner), so the
    events build never consulted the SEC bundle. With configs/cik_map.yaml every event-universe
    row carries its CIK even when the live ticker file is unreachable."""
    FakeHyperliquidInfo().install(monkeypatch)  # HttpClient.get_json raises: the SEC map is unavailable
    monkeypatch.setattr(universe_mod, "_listing_and_volume", lambda hl, market, now: (None, float("nan")))
    u = universe_mod.build_universe(settings, write=False)
    ev = u[u[U.in_event_universe]]
    assert len(ev) and ev[U.cik].notna().all()
    assert int(u.set_index(U.market).loc["xyz:NVDA", U.cik]) == 1045810
    assert u.set_index(U.market).loc["xyz:NVDA", U.name] == "NVIDIA CORP"
    # the map holds only event names: nothing outside the curated universe gets a CIK from it
    assert u.loc[~u[U.kind].isin(["equity_us", "equity_fpi"]), U.cik].isna().all()


def test_build_universe_without_a_cik_map_still_builds(settings, monkeypatch, tmp_path):
    import shutil

    configs = tmp_path / "cfg2"  # only the overrides: no cik_map.yaml
    configs.mkdir()
    shutil.copy(settings.universe_overrides_path, configs / "universe_overrides.yaml")
    settings.configs_dir = configs
    FakeHyperliquidInfo().install(monkeypatch)
    monkeypatch.setattr(universe_mod, "_listing_and_volume", lambda hl, market, now: (None, float("nan")))
    u = universe_mod.build_universe(settings, write=False)
    assert bool(u[U.in_event_universe].any()) and u[U.cik].isna().all()
