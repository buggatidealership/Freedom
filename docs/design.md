# Freedom: post-earnings price-action harness for Hyperliquid equities

## 1. What the harness is for

A research and prediction harness that, for every equity tradable as a perpetual on
Hyperliquid, predicts the price action from the earnings release through the following 24
hours, and that *optimizes itself* (models, features, decision time) against honest
out-of-sample evaluation rather than in-sample fit.

It is a harness, not a strategy: it produces a dataset, a leaderboard of models with
confidence intervals, a cost-aware trading simulation, and live predictions with the evidence
behind them. Whether anything is predictable is an output of the harness, not an input.

### Premise checks (what is wrong or incomplete in the naive framing)

1. "Stocks on Hyperliquid" are perpetual futures on a builder-deployed dex (`xyz`, `para`,
   `io`), not shares. They trade 24/7 with hourly funding, and their price outside the cash
   session comes from an oracle that follows after-hours liquidity providers and, during
   closures, drifts with the perp's own order flow. The 24-hour window therefore includes
   hours in which the perp *is* the price discovery venue. That is a feature (the window is
   tradable end to end) and a risk (weekend and overnight moves are thinner).
2. The universe is small: 87 SEC-matched names before curation, roughly 70–80 genuine
   equities with earnings after removing commodities, indices, pre-IPO tokens and duplicates.
   At ~4 events per name per year that is ~300 events/year. Any model must be built for small
   N: strong regularisation, few features, walk-forward evaluation with bootstrap intervals,
   and baselines that are hard to beat.
3. Fine-grained perp history is short (listings start Nov 2025; only the last 5000 candles per
   interval are served). Historical events must use the underlying's extended-hours equity
   bars (FMP) as the proxy for the perp path, with the perp used where it exists and archived
   from now on. The harness records which source each target came from.
4. Predicting the *direction* of a 24 h post-earnings move before the release from public data
   is close to a coin flip in the literature; what is more tractable is (a) magnitude (how big
   the move will be, which matters for sizing), and (b) conditional questions once the release
   is out: does the first 15–30 minutes of after-hours reaction extend or reverse by t0+24h.
   The harness therefore supports several *decision times* (before the release, and k minutes
   after it) and reports each separately. It must never mix them.

## 2. Definitions (exact)

* **Event** `e = (underlying, fiscal_period, t0)`. `t0` is the release instant in UTC.
* **Release-time resolver**, in priority order, each producing `(t0, confidence, source)`:
  1. SEC 8-K with item 2.02: `acceptanceDateTime` (US filers; measured within ~1 min of the
     press release).
  2. Event detection: first 1-minute extended-hours bar within the calendar day whose volume
     z-score against the same-clock-time baseline exceeds a threshold and whose absolute return
     exceeds a threshold; used to *refine* (1) when both exist and as the primary source for
     foreign private issuers.
  3. Calendar flag (`post-market` / `pre-market` from Alpha Vantage or Nasdaq) mapped to a
     default clock time (16:05 / 07:00 America/New_York), low confidence.
  4. Manual override file.
  Events whose resolver confidence is below a configurable floor are kept in the table but
  excluded from training by default (`--min-t0-confidence`).
* **Price series** `P(t)`: for `t ≥ listing` the Hyperliquid perp mid/close (archived 1m/5m
  candles, else 1h); before listing, or where archived candles are missing, the FMP 1-minute
  extended-hours bar of the underlying. Every target row carries `price_source`.
* **Reference price** `P0 = P(t0⁻)`: the close of the last bar that *ends* strictly before
  `t0`. Bars are half-open `[start, end)`; a bar containing `t0` is never used for `P0`.
* **Checkpoints** `h ∈ {+5m, +15m, +30m, +60m, +2h, next_open, next_open+30m, next_close,
  +24h}`; `next_open/close` are the next XNYS regular session boundaries after `t0`
  (exchange calendar aware). `P(t0+h)` is the close of the last bar ending at or before `t0+h`.
* **Targets** per checkpoint: `r_h = ln P(t0+h) − ln P0`; abnormal `ar_h = r_h − r_h(mkt)`
  where `mkt` is `xyz:SP500` if listed at the time, else SPY via FMP. Labels derived from
  `r_24h`: `direction = sign`, `magnitude = |r_24h|`, `continuation = sign(r_24h − r_30m)`.
* **Decision time** `d ∈ {pre_5m, post_1m, post_15m, post_30m, post_60m}` meaning
  `t0 − 5min`, `t0 + k min`. A feature is admissible at `d` only if its own timestamp `≤ d`.
  The feature builder enforces this with an explicit `as_of` argument on every provider call;
  there is no code path that builds features without one.
* **Horizon** is fixed at `t0 + 24h` for the headline prediction; intermediate checkpoints are
  auxiliary targets and diagnostics.

## 3. Universe

`freedom universe` → `data/universe.parquet`

1. Pull `perpDexs`, then `meta` for every dex; keep `isDelisted == false`.
2. Auto-classify each market by matching `SYMBOL` to SEC `company_tickers.json`.
3. Apply `configs/universe_overrides.yaml` (checked into git, reviewed by a human): kind ∈
   `{equity_us, equity_fpi, etf, index, commodity, fx, crypto, preipo, rate, other}`,
   `underlying_ticker`, `cik`, `exclude_reason`. Only `equity_us` and `equity_fpi` enter the
   event universe. Duplicated underlyings across dexes (e.g. `xyz:AVGO` and `para:AVGO`) keep one
   *primary* market (deepest volume) and record the others as alternates.
4. Record `listing_start` (first daily candle), `max_leverage`, `growth_mode`,
   `deployer_fee_scale`, and 30-day median notional volume.

## 4. Data layer

`src/freedom/data/` — one client per provider, all sharing:

* an on-disk cache (`data/cache/<provider>/<sha256(request)>.json|parquet`) with TTLs so reruns
  are free and tests can run offline from committed fixtures;
* a token-bucket rate limiter per provider (Hyperliquid weight/min; FMP and Alpha Vantage daily
  budgets that abort the run with a clear message instead of silently returning partial data);
* retries with exponential backoff on 429/5xx/network errors;
* strict timezone handling: every timestamp leaving a client is tz-aware UTC. FMP intraday is
  parsed as America/New_York then converted; Hyperliquid `t` is epoch-ms UTC.

Providers: `hyperliquid` (meta, candles with paging, funding, asset contexts),
`fmp` (earnings, calendar, intraday extended, EOD, profile, aftermarket trade),
`sec` (tickers, submissions, companyfacts, full-text search, EX-99.1 fetch),
`nasdaq` (calendar, daily), `alphavantage` (earnings with reportTime; optional).

**Archiver** (`freedom archive`): pulls 1m, 5m, 15m and 1h candles plus funding for every
universe market and appends to `data/archive/candles/<market>/<interval>.parquet` (dedup on
`t`). Designed to run from cron or a GitHub Actions schedule at least every 3 days (the 1m
window is ~3.5 days). Also snapshots `metaAndAssetCtxs` (open interest, premium, oracle vs mark).

## 5. Event table

`freedom events build` → `data/events.parquet`, one row per (underlying, fiscal period):
`t0, t0_confidence, t0_source, timing (AMC|BMO|RTH|closed), eps_actual, eps_estimate,
eps_surprise_pct, rev_actual, rev_estimate, rev_surprise_pct, n_estimates, sources_used,
market (primary perp), listing_start, has_perp_at_t0`.
Rules: an event is created from FMP earnings history and Nasdaq/Alpha Vantage cross-checks;
disagreements on the date by more than one day are flagged, not silently resolved.

## 6. Features (all built with `as_of = d`)

| group | examples | admissible at |
|---|---|---|
| calendar | AMC/BMO, weekday, days since last event, number of universe events same day, holiday adjacency | pre |
| pre_price | returns 1/5/20/60 d, realised vol 20 d, distance to 52 w high/low, drift in last 60/30 min before t0, extended-hours volume vs baseline, gap since last close | pre |
| history | this name's past `r_24h` mean/std/skew, hit rate of continuation, last four reactions, historical sensitivity of `r_24h` to EPS surprise | pre |
| market | `xyz:SP500` / SPY 1/5 d returns, `xyz:VIX` level and 5 d change, sector proxy (SMH, XLE, XBI) returns | pre |
| perp_state | funding rate, premium (mark−oracle), open-interest change 24 h, 30-day volume, leverage cap | pre (when perp exists) |
| surprise | EPS and revenue surprise %, standardised against the name's own surprise history, sign agreement | post |
| reaction | `r_1m…r_k`, path high/low range, volume z-score, perp premium after release | post_k |
| text (optional) | guidance change (raised/maintained/lowered), tone score, extracted from the 8-K EX-99.1 via Claude with a JSON schema; cached per filing | post |

Missing features are explicit (`NaN` + indicator), never imputed silently. Feature functions are
pure and unit-tested against fixtures that include a look-ahead trap (a bar starting before `d`
but ending after it must be excluded).

## 7. Models

* Baselines: `zero` (predict 0 / p=0.5), `historical_mean` (name's past mean `r_24h`),
  `sign_of_reaction` (for post decision times), `surprise_sign`.
* `ridge` / `logistic` with standardisation and strong L2 (grid over α).
* `lightgbm` with small-N settings (`num_leaves ≤ 7`, `min_data_in_leaf ≥ 20`, feature
  fraction, early stopping on the inner fold).
* `quantile` (LightGBM quantile objective, τ = 0.1/0.5/0.9) for magnitude bands.
* `ensemble`: mean of calibrated members.
Each model exposes `fit(X, y)`, `predict_proba_up`, `predict_return`, `predict_interval`.

## 8. Evaluation

* Walk-forward by event date: expanding window, minimum 120 training events, step = one
  earnings season (quarter), with a 2-day embargo around each test event so that same-day
  events of other names are never on both sides.
* Metrics per decision time: directional accuracy, balanced accuracy, Brier, log-loss,
  Spearman IC of predicted vs realised `r_24h`, MAE, pinball loss for quantiles, calibration
  table (deciles).
* Trading simulation: enter at `d` at the perp price (or proxy) with taker fee (param, default
  0.045 %) and slippage (param, default 5 bp), accrue hourly funding from the archive, exit at
  `t0 + 24h`; sizing = fixed or proportional to `|p − 0.5|`; report mean return per event,
  hit rate, Sharpe-like ratio on the event series, max drawdown, turnover, and the same for the
  baselines. Bootstrap (block by season) 95 % intervals on every headline number; paired
  comparison vs best baseline.
* Every evaluation writes `reports/<run_id>/` with a JSON summary and the per-event
  predictions so results are reproducible and auditable.

## 9. Optimisation ("optimizes for predicting")

`freedom optimize` runs an Optuna study over: model family and hyper-parameters, feature groups
on/off, decision time, target variant (raw vs abnormal), training-window length, and the
minimum `t0` confidence. Objective: the walk-forward metric chosen by `--objective` (default
`brier` for direction, `pnl_sharpe` available) computed **only on folds whose test seasons end
before a held-out final season**; the held-out season is scored once at the end and reported
separately as the honest number. Studies persist to `data/optuna.db`; a leaderboard is written.
Overfitting guard: the number of trials is reported alongside the improvement over the best
baseline, and the report states the probability that the improvement is noise (bootstrap).

## 10. Live prediction

`freedom upcoming` lists universe events in the next N days (FMP calendar, SEC resolver where
an 8-K already exists). `freedom predict --market xyz:NVDA --decision post_30m` builds features
`as_of now` from live Hyperliquid candles and FMP aftermarket trades, loads the best model for
that decision time, and prints: `p_up`, expected `r_24h`, 10/90 % band, the top feature
contributions, the data sources used, and the model's out-of-sample record.

## 11. Repository layout

```
pyproject.toml            # package `freedom`, CLI entry point `freedom`
src/freedom/
  config.py               # pydantic settings (keys from env, paths, fees)
  universe/               # hyperliquid pull + classification + overrides loader
  data/                   # provider clients, cache, rate limiter, archiver
  events/                 # earnings event table + release-time resolver
  targets/                # price path, checkpoints, returns, labels
  features/               # feature groups with as_of discipline
  models/                 # baselines, ridge/logistic, lightgbm, quantile, ensemble
  eval/                   # walk-forward, metrics, trading sim, bootstrap, reports
  optimize/               # optuna study + leaderboard
  cli.py                  # typer app
configs/                  # universe_overrides.yaml, default.yaml
tests/                    # offline, fixture based; look-ahead traps
docs/                     # this design, data-sources, results
.github/workflows/        # ci (lint+tests), archive (scheduled candle archiving)
```

## 12. What would show this design is wrong

* If the perp path around historical events (where both exist) diverges materially from the
  FMP extended-hours path, the proxy assumption fails and pre-listing events must be dropped.
  The harness measures this divergence on overlapping events and reports it.
* If the 8-K acceptance time is systematically late for some filers, `t0` is wrong and early
  reaction features leak. The event detector's disagreement with the 8-K time is logged per
  filer.
* If no model beats the baselines outside the bootstrap interval at any decision time, the
  honest conclusion is "not predictable with these inputs" and the harness must say so.
