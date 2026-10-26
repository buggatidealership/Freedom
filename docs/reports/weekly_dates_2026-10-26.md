# Weekly earnings-date verification, 2026-10-26 (frugal run)

Window Oct 26 to Nov 9. Sources: FMP and Finnhub calendars (one request each; payloads in the scratchpad, not committed; both responses were capped at 4,000 / 1,500 rows and carried nothing for Oct 26-30, so the Oct 27-30 names rest on last week's two-vendor agreement plus the Nasdaq calendar), the Nasdaq calendar per day, and issuer checks for the disagreements, the open items from last week and the three hyperscalers. The freedom-data artifact step was skipped: GitHub Actions still shows no archive.yml run since Sep 28. Backup and catch-up scripts remain absent from the restored repository; pushed directly.

## Events in the window (Hyperliquid equity universe, 79 event-universe underlyings)
| Event | FMP | Finnhub | Nasdaq | Issuer | Harness action |
|---|---|---|---|---|---|
| VZ Q3 | (not in the capped payload) | — | 10-26 pre-market | Nasdaq; the 8-K pattern is pre-open (M); call locked 09:51 UTC today | none |
| GLW, SOFI (pre), HOOD, STX, TER (after) Q3 | — | — | 10-27 | not checked (two vendors agreed last week) | none |
| GEV, UMC (pre), EBAY, GOOGL, MSFT, NOW (after) Q3 | — | — | 10-28 | **MSFT confirmed**: company investor page, "Wednesday, October 28, 2026" (M). **GOOGL confirmed**: company notice of Oct 6, call 1:30 pm PT = 4:30 pm ET Oct 28 (M via the IR news page). **EBAY not announced** (IR pages show nothing; aggregator Oct 28). GEV, UMC, NOW not checked | none |
| LLY (pre), AMZN, COIN, INTC, MSTR, RDDT, RIVN, SNDK (after) Q3 | — | — | 10-29 | **AMZN confirmed**: Business Wire notice of Oct 15, call 2:00 pm PT = 5:00 pm ET Oct 29 (M). **INTC confirmed** Oct 22: company notice of Oct 6, after the close Oct 29, call 2 pm PT (M). Others not checked | none |
| CVX Q3 | — | — | 10-30 pre-market | not checked | none |
| AAPL, PLTR (after), CIFR (pre) | 11-02 | AAPL missing; PLTR missing; CIFR missing | 11-02 | not checked | none |
| AMD, SMCI Q3 | 11-03 | 11-03 (AMD amc) | 11-03 after hours | not checked | none |
| ARM fiscal Q2 | **11-04** | **11-05** | 11-04 after hours | **Confirmed 11-04 after the close**, call 2 pm PT (company newsroom, Oct 5) (M via search summary) | harness consensus (FMP + Nasdaq) already 11-04; Finnhub wrong; no override needed |
| COHR, QCOM (after), CRCL (pre), MELI (time not supplied) | 11-04 | 11-04 | 11-04 | not checked | none |
| SHAZ (SharonAI) Q3 | **11-05** | **11-04** | 11-05 (time not supplied) | **Not announced**; aggregators split Nov 5 (amc) and Nov 15 (S) | disagreement left open; no override |
| AAOI, DKNG, GPRO, IONQ, IREN, LITE, NET, QNT, USAR, BIRD | 11-05 | 11-05 (DKNG, LITE, NET amc) | 11-05 | **DKNG not announced** (IR pages show no event; three vendors agree) ; others not checked | none |
| MRNA, VST Q3 | 11-06 | 11-06 (MRNA bmo) | 11-06 pre-market | not checked | none |
| CRWV, HIMS, RKLB, TTWO | 11-09 | 11-09 (CRWV missing) | 11-09 | not checked | none |

Corrections made: none (no issuer date contradicted the harness's vendor-consensus date). Events still unconfirmed by their issuer: EBAY, DKNG, SHAZ (not announced); all "not checked" rows rest on vendor agreement, not on the issuer. Vendor disagreements left open: SHAZ (FMP/Nasdaq 11-05 vs Finnhub 11-04); ARM resolved by the issuer (11-04). NBIS has dropped out of all three calendars (last week Finnhub carried 10-22; nothing was filed); it is outside the 14-day window.

## SEC bundle refresh
`freedom sec-bundle`: **8,864 filing rows** (17 new since Oct 19), **0 acceptance instants corrected**, **11,464 EPS-fact rows**, CIK map 79 tickers. `ruff check` clean; `pytest` result recorded in the commit message.

## Hyperliquid data pipeline
GitHub Actions still shows no archive.yml run since Sep 28; the stop-gap 1-minute captures continue every two days (last: Oct 25, 21:48Z Oct 21 to 09:48Z Oct 25).
