"""Shared HTTP plumbing: disk cache, token-bucket rate limiter, daily budget, retries.

Contract for provider clients:
* all public methods return tz-aware UTC timestamps;
* every network call goes through `HttpClient.get_json` / `post_json` so that caching,
  rate limiting and budgets are impossible to bypass;
* a provider that cannot run (missing key, budget exhausted) raises `ProviderUnavailable`
  with a message that says what to set or wait for.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC as _UTC
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)


class ProviderUnavailable(RuntimeError):
    """The provider cannot be used right now (missing key, exhausted budget)."""


class BudgetExhausted(ProviderUnavailable):
    pass


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(provider: str, method: str, params: Any) -> str:
    return hashlib.sha256(f"{provider}|{method}|{_canonical(params)}".encode()).hexdigest()


class DiskCache:
    """JSON-on-disk cache with per-entry TTL. Entries are gzip files named by sha256 key."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, provider: str, key: str) -> Path:
        d = self.root / provider / key[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json.gz"

    def get(self, provider: str, key: str, max_age_seconds: int | None) -> Any | None:
        p = self._path(provider, key)
        if not p.exists():
            return None
        with gzip.open(p, "rt", encoding="utf-8") as f:
            env = json.load(f)
        if max_age_seconds is not None and time.time() - env["fetched_at"] > max_age_seconds:
            return None
        return env["payload"]

    def set(self, provider: str, key: str, payload: Any) -> None:
        p = self._path(provider, key)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "payload": payload}, f)
        tmp.replace(p)


class TokenBucket:
    """Weighted token bucket: `capacity` units refill evenly over each minute."""

    def __init__(self, capacity_per_minute: float):
        self.capacity = float(capacity_per_minute)
        self.tokens = float(capacity_per_minute)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, weight: float = 1.0) -> None:
        weight = min(weight, self.capacity)
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.capacity / 60)
                self.updated = now
                if self.tokens >= weight:
                    self.tokens -= weight
                    return
                wait = (weight - self.tokens) * 60 / self.capacity
            time.sleep(min(wait, 5.0))


class DailyBudget:
    """Persistent per-UTC-day request counter. Raises BudgetExhausted when the limit is hit."""

    def __init__(self, name: str, limit: int, state_dir: Path):
        self.name, self.limit = name, int(limit)
        self.path = Path(state_dir) / f"budget_{name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _load(self) -> dict[str, int]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def used_today(self) -> int:
        return self._load().get(date.today().isoformat(), 0)

    def consume(self, n: int = 1) -> None:
        with self.lock:
            st = self._load()
            today = date.today().isoformat()
            used = st.get(today, 0)
            if used + n > self.limit:
                raise BudgetExhausted(
                    f"{self.name}: daily budget of {self.limit} requests exhausted ({used} used). "
                    "Cached responses are still served; wait until tomorrow (UTC) or raise the budget."
                )
            st = {today: used + n}
            self.path.write_text(json.dumps(st))


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 425, 429, 500, 502, 503, 504)
    return False


@dataclass
class HttpClient:
    """One instance per provider. `provider` names the cache namespace."""

    provider: str
    cache: DiskCache
    limiter: TokenBucket | None = None
    budget: DailyBudget | None = None
    timeout: float = 30.0
    default_headers: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(timeout=self.timeout, headers=self.default_headers or {})

    def get_json(self, url: str, params: dict | None = None, *, cache_ttl: int | None,
                 weight: float = 1.0, headers: dict | None = None,
                 cache_params: Any = None) -> Any:
        key = cache_key(self.provider, f"GET {url}", cache_params if cache_params is not None else params)
        if cache_ttl is not None:
            hit = self.cache.get(self.provider, key, cache_ttl)
            if hit is not None:
                return hit
        if self.limiter is not None:
            self.limiter.acquire(weight)
        if self.budget is not None:
            self.budget.consume(1)
        payload = self._request("GET", url, params=params, headers=headers)
        self.cache.set(self.provider, key, payload)
        return payload

    def post_json(self, url: str, body: Any, *, cache_ttl: int | None, weight: float = 1.0,
                  headers: dict | None = None) -> Any:
        key = cache_key(self.provider, f"POST {url}", body)
        if cache_ttl is not None:
            hit = self.cache.get(self.provider, key, cache_ttl)
            if hit is not None:
                return hit
        if self.limiter is not None:
            self.limiter.acquire(weight)
        if self.budget is not None:
            self.budget.consume(1)
        payload = self._request("POST", url, json_body=body, headers=headers)
        self.cache.set(self.provider, key, payload)
        return payload

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _request(self, method: str, url: str, *, params=None, json_body=None, headers=None) -> Any:
        r = self._client.request(method, url, params=params, json=json_body, headers=headers)
        r.raise_for_status()
        return r.json()


def utcnow() -> datetime:
    return datetime.now(tz=_UTC)
