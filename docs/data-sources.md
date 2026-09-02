# Data sources: what was measured (2026-09-02)

Every statement here was verified by a live request from the build environment on
2026-09-02 unless marked *assumed*. Re-verify before relying on anything marked *assumed*.

## Hyperliquid (tradable instrument, no key)

* `POST https://api.hyperliquid.xyz/info` is reachable. `perpDexs` returns 10 HIP-3 dexes
  plus the validator-run main dex (`null`). Several dexes are fully delisted (`flx`, `vntl`,
  `km`, `abcd`, `cash`); `xyz` (103 active markets), `para` (23), `mkts` (4), `io` (4) and
  `hyna` (6, crypto only) are live.
* Equity-style markets live on `xyz`, `para`, `io`. Market names are `dex:SYMBOL`
  (e.g. `xyz:NVDA`). Asset metadata fields: `szDecimals, maxLeverage, marginTableId,
  isDelisted, growthMode, deployerFeeScale, lastFeeScaleChangeTime, onlyIsolated, marginMode`.
* Stock perps trade **24/7**: `xyz:NVDA` 1h candles carried volume in every hour of every
  weekday and weekend over the last 14 days. Funding settles **hourly** (168 entries / 7d).
* `candleSnapshot` intervals: 1m 3m 5m 15m 30m 1h 2h 4h 8h 12h 1d 3d 1w 1M. **Only the most
  recent 5000 candles per (market, interval) are served.** Measured: 1m ≈ 3.5 days, 5m ≈ 17
  days, 1h ≈ 208 days, 4h / 1d cover the full listing history. Listing starts measured:
  `xyz:NVDA` 2025-11-12, `xyz:AAPL` 2025-11-21, `xyz:SP500` 2026-03-18, `para:AVGO` 2026-05-21.
  Consequence: fine-grained perp history for past earnings events does not exist unless we
  archive it ourselves. The harness ships an archiver for that reason.
* Rate limit: 1200 weight per minute per IP. `candleSnapshot` costs 20 + 1 per 60 candles
  returned (a full 5000-candle page ≈ 103 weight, so ≈ 11 pages/minute). Other info requests
  cost 20 (`l2Book`, `allMids`, `clearinghouseState` cost 2).
* `xyz:VIX` is delisted (`isDelisted: true`, no candles), so the VIX context features are
  always missing until another volatility source is wired; `xyz:SP500` (listed 2026-03-18) is the
  benchmark for perp-era events and SPY via FMP before that.
* Fees (docs): validator perps taker 0.045 % / maker 0.015 % at tier 0. HIP-3 growth-mode
  markets have a 90 %-reduced protocol fee (docs quote a 0.0045–0.009 % baseline) plus a
  deployer share; `xyz` reports `deployerFeeScale = 1.0`, `growthMode = enabled`. The exact
  all-in taker rate for `xyz` was **not measured**; the backtester takes it as a parameter and
  defaults to the conservative 0.045 %.
* Oracle (docs.trade.xyz, summarised by search, page not fetched directly): during the cash
  session the spot index is the oracle; outside it, a futures-implied or LP (Pyth) price is
  used; during closures the oracle holds the last reference and drifts with a 30-minute EWMA of
  the market's impact-price difference. So post-market perp moves reflect real after-hours
  trading, and weekend moves are perp-internal. *Assumed until the docs page is read directly.*
* Worked example (`xyz:NVDA`, 8-K accepted 2026-08-26 20:21:19 Z, 5-minute candles): the last
  bar ending at or before t0 is [20:15, 20:20) with close 211.07; the [20:20, 20:25) bar contains
  the release and is never used for P0 (with the 3-minute 8-K buffer the harness backs off one
  more bar, to 210.63). Log returns versus 211.07 using only bars that end at or before each
  checkpoint: +5m −1.49 % (207.94, the release bar: an initial dump to 203.52), +15m −2.18 %,
  +30m −0.62 %, +60m +3.09 %, next open 13:30 Z +5.45 %, next close 20:00 Z +7.72 %, +24 h +7.16 %
  (226.74). The path dumped about 2 %, was back to flat after roughly 40 minutes and only then
  trended. An earlier draft of this note quoted +9.06 % by taking P0 from inside the release bar
  and checkpoint prices from bars that ended after the checkpoint; that is exactly the look-ahead
  the target code is tested against.

## SEC EDGAR (release timing and actuals, no key, needs a User-Agent)

* `https://www.sec.gov/files/company_tickers.json`: 10 391 tickers → CIK. Matching Hyperliquid
  symbols against it yields 87 hits with **known false positives** (`xyz:GOLD` → Gold.com,
  `xyz:CL` → Colgate, `hyna:BTC` → an ETF). The universe module therefore layers a curated
  override file on top of the automatic match.
* `https://data.sec.gov/submissions/CIK##########.json`: recent filings with
  `acceptanceDateTime`. For US filers the earnings 8-K carries item `2.02` and its acceptance
  time is within about a minute of the press release: NVDA 2026-08-26 20:21:19 Z, AAPL
  2026-07-30 20:30:28 Z, 2026-04-30 20:30:41 Z, 2026-01-29 21:30:33 Z (winter time).
* Foreign private issuers (TSM, ASML, BABA, ARM, NOK, NBIS, …) file 6-K with no item codes,
  and the filing time can lag the release by hours (TSM 2026-07-16 11:45 Z for a release made
  in Asian hours). 6-K acceptance time is **not** a release-time source; those names fall back
  to the calendar time-of-day flag plus event detection from intraday data.
* `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`: diluted EPS facts per
  period, usable as an actuals cross-check (values are GAAP, calendars quote non-GAAP).
* Full-text search `https://efts.sec.gov/LATEST/search-index?...` works and can locate the
  EX-99.1 press release for optional text features.

## Financial Modeling Prep (key present as `FMP_API_KEY`)

* `stable/earnings?symbol=` : per-quarter `date, epsActual, epsEstimated, revenueActual,
  revenueEstimated` (no time-of-day). `stable/earnings-calendar?from=&to=` for upcoming events.
* `stable/historical-chart/{1min,5min}?symbol=&from=&to=&extended=true` returns **pre-market
  and after-hours bars (04:00–19:55 ET)**; without `extended` only 09:30–15:55. Depth measured
  to at least 2024-08 for 5min and 2025-02 for 1min. Timestamps are exchange-local (America/New_York)
  with no zone suffix. **A 1-minute request covering five sessions returned only the latest
  three (2880 bars)**, silently dropping the first two days; a 5-minute request covering ten
  sessions returned all ten. The client therefore chunks 1-minute windows at three calendar
  days, and the resolver and the price-path loader both request `[report day − 1, report day + 1]`.
* `stable/historical-price-eod/full?symbol=&from=&to=`: daily bars back to at least 2020.
* `stable/profile`, `stable/quote`, `stable/aftermarket-trade` (live after-hours trade) work.
* `stable/splits?symbol=` lists splits (NFLX 10:1 on 2025-11-17). **Both EOD and intraday bars
  are split-adjusted**: NFLX closed 2025-11-14 at 111.22 and opened 2025-11-17 at 110.75 in the
  EOD series, and the 04:00 ET bar of 2025-11-14 prints 115.20, all on the post-split basis.
  `xyz:NFLX` had no candles in November 2025 (listed later), so no perp split discontinuity has
  been observed yet. The harness fetches the calendar once per underlying (`FMPClient.splits`,
  cached a week), flags events with an ex-date inside `[t0 − 60 d, t0 + 24 h]`
  (`corporate_action`, `corporate_action_ex_date`), NaNs the headline target when the ex-date
  lies in `[P0, t0 + 24 h]`, and on a perp path (not adjusted) NaNs every checkpoint from the
  ex-date on. Fixture: `tests/fixtures/fmp/splits_NFLX.json`.
* `stable/earning-call-transcript` is **restricted** on this plan. Rate-limit headers are not
  exposed; the free plan is documented at 250 requests/day (*assumed*). The client caches every
  response on disk and budgets requests per run.

## Nasdaq (no key, needs a browser-like User-Agent)

* `https://api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD` works for past and future
  dates: `symbol, eps, epsForecast, surprise, noOfEsts, fiscalQuarterEnding, time`. `time` is
  mostly `time-not-supplied` for past dates, so it is a consensus/surprise cross-check, not a
  timing source.
* `https://api.nasdaq.com/api/quote/{sym}/historical?assetclass=stocks&fromdate=&todate=&limit=`
  gives daily bars (669 rows for NVDA since 2024-01-01).

## Alpha Vantage (key present as `ALPHAVANTAGE_API_KEY`, free tier)

* `EARNINGS` returns 110 quarters for NVDA with `reportedDate, reportedEPS, estimatedEPS,
  surprisePercentage, reportTime (pre-market | post-market)`. Useful as a timing flag for
  foreign filers.
* Free tier: **25 requests/day**; intraday endpoints hit the limit immediately. Treated as an
  optional, quota-budgeted cross-check only.

## Blocked or unusable from this environment

* Yahoo Finance (`query1.finance.yahoo.com`, `fc.yahoo.com`): connection reset by the egress
  proxy; `yfinance` fails. Stooq: JavaScript challenge page. Neither is used.
