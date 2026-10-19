# Weekly earnings-date verification, 2026-10-19 (frugal run)

Window Oct 19 to Nov 2. Sources: FMP and Finnhub calendars (one request each, payloads in the scratchpad, not committed), the Nasdaq calendar per day as the third source, and issuer checks only where the vendors disagree or a vendor is missing (cost control). The freedom-data artifact step was skipped: GitHub Actions has produced no run since Sep 28, so the newest artifact predates this window; the local events table could not be used either because `data/` was not part of the restored repository (rebuilt `data/universe.parquet` with `freedom universe`, 163 rows, 37 flagged for human verification). Backup and catch-up scripts (`scripts/`) are absent from the restored repository; pushed directly.

## Events in the window (Hyperliquid equity universe)
| Event | FMP | Finnhub | Nasdaq | Issuer | Harness action |
|---|---|---|---|---|---|
| NFLX Q3 | 10-20 | missing | 10-20 after hours | **Confirmed**: company release of Sep 14, results posted 1:01 pm PT = 16:01 ET Oct 20, live interview 1:45 pm PT (M, s22.q4cdn.com PDF via search) | none needed (after-close clock) |
| META, TSLA, IBM, LRCX Q3 | 10-21 | 10-21 amc | 10-21 after hours | not checked (three vendors agree on date and clock) | none |
| BX, BE, WDC Q3 | 10-22 | 10-22 (BX bmo; BE, WDC amc) | 10-22 (BX pre-market; BE, WDC after hours) | BX confirmed last week (company notice, 10-22 pre-market); BE, WDC not checked (vendors agree) | none |
| NOK Q3 | 10-22 | missing | 10-22 pre-market | not checked | none |
| NBIS Q3 | missing | 10-22 (no time) | missing | **Not announced**; aggregators estimate Nov 11; Nebius announced its Q3 2025 date on Oct 22 last year (S) | Finnhub's 10-22 is unsupported: left open, no override (no issuer date to write) |
| GLW, SOFI, HOOD, STX, TER Q3 | 10-27 | 10-27 (GLW, SOFI bmo; HOOD, TER amc; STX blank) | 10-27 (GLW, SOFI pre-market; HOOD, STX, TER after hours) | not checked | none |
| EBAY Q3 | 10-28 | **10-27** (no time) | 10-28 after hours | **Not announced** (no 8-K or IR notice found; aggregators range Oct 28 to Nov 4) (S) | vendor disagreement left open; no override |
| SKHY (SK hynix ADR) Q3 | 10-28 | missing | missing | **Confirmed date**: company 6-K of Oct 14 says conference-call materials posted Oct 28 Seoul time (= evening of Oct 27 ET); the hour is not stated (M via search summary of the 6-K) | the harness's t0 would need a UTC instant; not written because the hour is not announced |
| GEV, GOOGL, MSFT, NOW Q3 | 10-28 | 10-28 (GOOGL, NOW amc; GEV, MSFT blank) | 10-28 (GEV pre-market; GOOGL, MSFT, NOW after hours) | not checked | none |
| AMZN, COIN, INTC, LLY, MSTR, RDDT, RIVN, SNDK Q3 | 10-29 | 10-29 (AMZN, COIN, INTC, RIVN amc; others blank) | 10-29 (LLY pre-market; the rest after hours) | not checked | none |
| CL, CVX Q3 | 10-30 | 10-30 bmo | 10-30 pre-market | not checked | none |
| DKNG Q3 | missing | 10-30 amc | missing | **Not announced**; aggregators range Oct 30 to Nov 6 (S) | left open; no override |
| AAPL, PLTR, CIFR | 11-02 | 11-02 (AAPL amc; PLTR blank; CIFR missing) | 11-02 (AAPL, PLTR after hours; CIFR time not supplied) | not checked | none |
| SPCX, STRC (FMP only) | 10-20 / 10-29 | missing | missing | not reporting issuers (pre-IPO market; a preferred-share ticker) | ignore |

Corrections made: none (no issuer date contradicted the harness's vendor-consensus date). Events still unconfirmed by their issuer: EBAY, DKNG, NBIS (not announced); all "not checked" rows rest on three-vendor agreement, not on the issuer. Vendor disagreements left open: EBAY (Finnhub 10-27 vs 10-28), NBIS (Finnhub 10-22 vs nothing), DKNG (Finnhub 10-30 vs nothing).

## SEC bundle refresh
`freedom sec-bundle` after rebuilding the universe: **8,847 filing rows** (newest filed 2026-10-16, accepted 21:30:16 UTC), **2,888 acceptance instants corrected**, **11,464 EPS-fact rows**, CIK map 79 tickers. `ruff check` clean; `pytest` 153 passed after installing the dev extras (respx was missing from the environment).

## Hyperliquid data pipeline
GitHub Actions still shows no archive.yml run since Sep 28; the stop-gap 1-minute captures continue every two days (last: Oct 19, with the weekly 5m/15m/1h sets).
