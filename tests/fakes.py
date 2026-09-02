"""Offline stand-ins for provider HTTP calls, built on the committed JSON fixtures.

`FakeHyperliquidInfo` replaces `HttpClient.post_json` with the server semantics measured on
2026-09-02 (see docs/data-sources.md and the module docstring of freedom.data.hyperliquid):
candleSnapshot returns bars overlapping [startTime, endTime] (so the bar *containing* an
unaligned startTime is included) and never more than 5000; fundingHistory returns at most
500 entries, oldest first. Every call is recorded with its weight and cache_ttl.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from freedom.data import hyperliquid as hl
from freedom.data.base import HttpClient

FIXTURES = Path(__file__).parent / "fixtures"
NVDA = "xyz:NVDA"
FUNDING_STEP_MS = 3_600_000


def load_fixture(provider: str, name: str):
    return json.loads((FIXTURES / provider / name).read_text())


def synth_candles(market: str, interval: str, start_ms: int, n: int) -> list[dict]:
    step = hl.interval_ms(interval)
    return [
        {"t": start_ms + i * step, "T": start_ms + (i + 1) * step - 1, "s": market, "i": interval,
         "o": str(100 + i), "c": str(101 + i), "h": str(102 + i), "l": str(99 + i),
         "v": str(1.5 * (i + 1)), "n": i + 1}
        for i in range(n)
    ]


def synth_funding(market: str, start_ms: int, n: int) -> list[dict]:
    return [
        {"coin": market, "fundingRate": "0.0000125", "premium": str(i * 1e-6),
         "time": start_ms + i * FUNDING_STEP_MS + 40}
        for i in range(n)
    ]


class FakeHyperliquidInfo:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.candles: dict[tuple[str, str], list[dict]] = {
            (NVDA, "1h"): load_fixture("hyperliquid", "candles_xyzNVDA_1h_20260824_29.json"),
            (NVDA, "5m"): load_fixture("hyperliquid", "candles_xyzNVDA_5m_20260826.json"),
        }
        self.funding: dict[str, list[dict]] = {
            NVDA: load_fixture("hyperliquid", "funding_xyzNVDA_20260826_28.json"),
        }
        self.fail_markets: set[str] = set()  # candle/funding requests for these raise

    def install(self, monkeypatch) -> FakeHyperliquidInfo:
        def _post(http: HttpClient, url: str, body, **kw):
            return self.post_json(url, body, **kw)

        def _get(http: HttpClient, *a, **kw):
            raise AssertionError("Hyperliquid client must only use POST /info")

        monkeypatch.setattr(HttpClient, "post_json", _post)
        monkeypatch.setattr(HttpClient, "get_json", _get)
        return self

    def calls_of(self, typ: str) -> list[dict]:
        return [c for c in self.calls if c["body"]["type"] == typ]

    def post_json(self, url: str, body: dict, *, cache_ttl, weight=1.0, headers=None):
        self.calls.append({"url": url, "body": body, "cache_ttl": cache_ttl, "weight": weight})
        typ = body["type"]
        if typ == "perpDexs":
            return load_fixture("hyperliquid", "perpDexs.json")
        if typ == "meta":
            return load_fixture("hyperliquid", f"meta_{body['dex']}.json")
        if typ == "metaAndAssetCtxs":
            return load_fixture("hyperliquid", f"metaAndAssetCtxs_{body['dex']}.json")
        if typ == "candleSnapshot":
            req = body["req"]
            if req["coin"] in self.fail_markets:
                raise httpx.ConnectError("simulated network failure")
            step = hl.interval_ms(req["interval"])
            span = req["endTime"] - req["startTime"] + 1
            assert span <= hl.MAX_CANDLES_PER_REQUEST * step, "request would exceed 5000 candles"
            rows = self.candles.get((req["coin"], req["interval"]), [])
            return [r for r in rows if r["T"] >= req["startTime"] and r["t"] <= req["endTime"]]
        if typ == "fundingHistory":
            if body["coin"] in self.fail_markets:
                raise httpx.ConnectError("simulated network failure")
            rows = self.funding.get(body["coin"], [])
            hit = [r for r in rows if body["startTime"] <= r["time"] <= body["endTime"]]
            return sorted(hit, key=lambda r: r["time"])[: hl.FUNDING_PAGE_SIZE]
        raise AssertionError(f"unexpected info request type {typ!r}")
