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
  2. Event detection: first 1-minute extended-hours bar on the report date whose volume
     z-score against the same-clock-minute baseline exceeds a threshold and whose absolute return
     exceeds a threshold. Detection may only move an 8-K time **earlier**: `t0 = min(acceptance,
     first detected bar start)` when they are within 15 minutes; a later detection never changes
     `t0` and is logged as the reaction lag. Confidence of an 8-K event does not depend on
     whether detection fires. Detection is the primary source for foreign private issuers; a
     detection that fires on the very first bar of the extended session is flagged
     (`detection_first_bar`) and gets calendar-flag confidence, because the release probably
     happened while no bars existed.
  3. Calendar flag (`post-market` / `pre-market` from Alpha Vantage or Nasdaq) mapped to a
     default clock time (16:05 / 07:00 America/New_York), low confidence.
  4. Manual override file.
  Events whose resolver confidence is below `min_t0_confidence` (a fixed run parameter, default
  0.6, raisable from the CLI, **never** a search dimension) are kept in the table but excluded
  from training. Every metric is additionally stratified by `t0_source` and `kind`; the headline
  pools only `sec_8k`, `manual` and non-first-bar detections.
* **Price series** `P(t)`: one source per event, never mixed inside `[t0 − 1h, t0 + 24h]`.
  The Hyperliquid perp candle close (trade price) is used when archived or live 1m/5m candles
  cover that whole window (1m preferred; the archive keeps 1m from now on), otherwise the
  underlying's FMP 1-minute extended-hours bars. 1h or coarser candles are never used for `P0`,
  any checkpoint, or a fill; when neither fine source covers the window the event's targets are
  NaN. Every target row carries `price_source`, `price_interval` and `price_market`.
* **Reference price** `P0`: close of the last bar with `t_end ≤ t0 − δ(t0_source)`, where
  `δ = 3 min` for `sec_8k` (acceptance trails the wire by 25–134 s on the five NVDA/AAPL filings
  measured) and `0` for `manual` and `detected`. Bars are half-open `[t, t_end)`; a bar containing
  the instant is never used. `p0_time` and `p0_staleness_min` are recorded on every row.
* **Checkpoints** `h ∈ {+5m, +15m, +30m, +60m, +2h, next_open, next_open+30m, next_close,
  +24h}`; `next_open/close` are the next XNYS regular-session boundaries after `t0`.
  `P(t0+h)` is the close of the last bar with `t_end ≤ t0+h`. A checkpoint is **valid only if**
  (i) that bar ends after the `P0` bar and (ii) its staleness `(t0+h) − t_end ≤ max(2 × interval,
  5 min)`; otherwise it is NaN. Each row records `t_<cp>` (bar end used) and `s_<cp>` (staleness,
  minutes), plus `horizon_actual_h` and `h24_in_closure` (XNYS closed at `t0+24h`). Consequence on
  the FMP proxy (04:00–19:55 ET bars): `+24h` is unobservable for Friday, pre-holiday and
  outside-window releases; those events keep the intermediate and `next_close` checkpoints but
  have no headline label until a perp path exists.
* **Targets** per checkpoint: `r_h = ln P(t0+h) − ln P0`; abnormal `ar_h = r_h − r_h(mkt)`
  where `mkt` is `xyz:SP500` when the path is a perp, else SPY via FMP. Labels from `r_24h`:
  `direction = sign`, `magnitude = |r_24h|`, and `continuation_k = sign(r_k) × sign(r_24h − r_k)`
  for `k ∈ {15m, 30m}` (+1 the early reaction extended, −1 it reversed, NaN when `|r_k|` is
  inside a 25 bp dead band; the count of dead-band events is reported). `post_15m` reports
  `continuation_15m`, later decision times `continuation_30m`.
* **Corporate actions**: price inputs within one source are put on the as-of basis using the
  FMP splits calendar (measured 2026-09-02: FMP intraday and EOD bars are already
  split-adjusted for NFLX's 10:1 split; see data-sources.md). Events with a split or spin-off
  ex-date inside `[t0 − 60 d, t0 + 24 h]` get `flags += corporate_action`; those with the
  ex-date inside `[P0, t0 + 24 h]` have NaN headline targets. A perp path across a split is used
  only if measured continuous.
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
`t`), where `<market>` is the market name with `:` replaced by `_` (`xyz:NVDA` →
`candles/xyz_NVDA/1h.parquet`); build paths through `archive.candle_path` / `load_archive`,
never by hand. Funding goes to `candles/<market>/funding.parquet` with `t` floored to the
settlement hour, so it joins hourly bars on `t`; the first pull for a market starts at its
listing date. Designed to run from cron or a GitHub Actions schedule at least every 3 days
(the 1m window is ~3.5 days); a run that finds the archive older than the server horizon
appends what is still served and reports the lost span in its summary. Also snapshots
`metaAndAssetCtxs` (open interest, premium, oracle vs mark) to `ctx/<dex>/<date>.parquet`.

## 5. Event table

`freedom events` → `data/events.parquet`, one row per (underlying, fiscal period).
`fiscal_period` is the fiscal quarter-end month (`YYYY-MM`), derived deterministically per
name: SEC companyfacts quarterly EPS period end for US filers, Alpha Vantage `fiscalDateEnding`
for foreign private issuers, else the calendar quarter end preceding the FMP date with
`flags += fiscal_period_derived`. `event_id = "{underlying}:{fiscal_period}"`. Nasdaq and Alpha
Vantage rows are matched to the FMP row by nearest report date within ±10 days; a matched pair
whose dates differ by more than one day gets `flags += date_conflict` and confidence 0.
Columns: `t0, t0_confidence, t0_source, timing, eps_actual, eps_estimate, eps_surprise_pct,
rev_actual, rev_estimate, rev_surprise_pct, n_estimates, estimate_source,
estimate_snapshot_time, sources_used, market, listing_start, has_perp_at_t0, pending, flags`.
**Consensus provenance:** vendor estimates for past events are the vendor's final value, not
the consensus as of `t0`; they are stored with `estimate_source = fmp_final` and the surprise
feature group is marked non-point-in-time in reports. From now on the archiver's consensus
snapshots provide `estimate_source = consensus_snapshot` with the capture time, and live
prediction uses only those. `has_perp_at_t0 = t0 ≥ min(listing_start)` over all markets of the
underlying; when the primary market was unlisted at `t0` but an alternate existed, the
alternate's candles are used and recorded in `price_market`.

## 6. Features (all built with `as_of = d`)

| group | examples | admissible at |
|---|---|---|
| calendar | AMC/BMO, weekday, days since last event, number of universe events same day, holiday adjacency | pre |
| pre_price | returns 1/5/20/60 d, realised vol 20 d, distance to 52 w high/low, drift in last 60/30 min before t0, extended-hours volume vs baseline, gap since last close | pre |
| history | this name's past `r_24h` mean/std/skew, hit rate of continuation, last four reactions, historical sensitivity of `r_24h` to EPS surprise — computed only from `history_view(events, targets, underlying, as_of)` = rows with `t0 + 24h ≤ as_of` | pre |
| market | `xyz:SP500` / SPY 1/5 d returns, `xyz:VIX` level and 5 d change, sector proxy (SMH, XLE, XBI) returns | pre |
| perp_state | funding rate, premium (mark−oracle), open-interest change 24 h, 30-day volume, leverage cap | pre (when perp exists) |
| surprise | EPS and revenue surprise %, standardised against the name's own surprise history, sign agreement | post |
| reaction | `r_1m…r_k`, path high/low range, volume z-score, perp premium after release | post_k |
| text (deferred) | guidance change and tone from the 8-K EX-99.1 via an LLM; **not built in v1** because an LLM trained after the event knows the outcome, so the feature cannot be made point-in-time for historical rows | — |

Missing features are explicit (`NaN` + indicator), never imputed silently. `as_of` gates the
harness's own event/target store as well as provider calls: `build_features` asserts
`(history.t0 + 24h ≤ as_of).all()`. Feature functions are pure and unit-tested against fixtures
with two look-ahead traps: a bar starting before `d` but ending after it must be excluded, and
at `post_60m` setting the event's own `r_24h` to +5.0 must leave every feature unchanged.

## 7. Models

* Baselines: `zero` (r = 0, p = 0.5), `base_rate` (p = training-window up-rate, r = training
  mean `r_24h`), `historical_mean` (name's past mean `r_24h`, pooled fallback), `hist_abs_mean`
  (name's past mean `|r_24h|`, a magnitude baseline), `vol_scaled` (σ from 20-day realised
  vol scaled to the horizon; magnitude and quantiles from the training distribution of
  `r_24h/σ`), `sign_of_reaction` (post decision times: direction = sign(`r_k`)),
  `always_extends` (continuation = +1), `surprise_sign`. Every comparison in §8/§9 is against
  the **best baseline per metric**, and the report names which one it was.
* `ridge` / `logistic` with standardisation and strong L2 (grid over α).
* `lightgbm` with small-N settings (`num_leaves ≤ 7`, `min_data_in_leaf ≥ 20`, feature
  fraction, early stopping on the inner fold).
* `ensemble`: mean of members' `p_up` and `r_hat`.
Each model exposes `fit(X, y_return, y_direction)`, `predict_proba_up`, `predict_return`.
The 10/90 % band is not a model method: for each (model, decision time) it is the empirical
10th/90th percentile of walk-forward out-of-sample residuals `r_true − r_hat`, saved with the
trained model and added to `r_hat` at prediction time.

## 8. Evaluation

* Walk-forward by earnings season (calendar quarter of `t0`): expanding window, minimum 120
  training events, one season per test fold, with a 2-day embargo so same-day events of other
  names are never on both sides. The **holdout season is pinned in settings**
  (`holdout_season`, default `2026Q3`, advanced only by a human edit) and excluded from every
  fold by `freedom evaluate` and `freedom optimize`. It is scored only by
  `freedom evaluate --final`; every such scoring appends `{timestamp, git commit, dataset hash,
  models}` to `reports/holdout_scorings.jsonl`, and the report prints how many times the holdout
  has been scored so the reader can discount it.
* Metrics per decision time: directional accuracy, balanced accuracy, Brier, log-loss, Spearman
  IC of `r_hat` vs `r_24h`, MAE, magnitude MAE of `predict_magnitude` vs `|r_24h|` (so
  `hist_abs_mean` / `vol_scaled` are scored as magnitude forecasts), calibration table
  (deciles). Every metric is reported (a) for
  events with `has_perp_at_t0` (headline) and (b) for all events, and stratified by `t0_source`,
  `kind` and `timing`.
* Trading simulation. A fill at instant `x` (entry at `d`, exit at `t0+24h`) is the **open of
  the first bar whose start is ≥ x** plus execution cost; it is never the close of a bar ending
  at or before `x`, so a fill is always strictly later than the signal it acts on. `fill_lag`
  is recorded per trade; trades whose lag exceeds `max_fill_lag_minutes` (default 5) are not
  taken and their count is reported. Execution cost per leg is
  `slip_bps = slippage_floor_bps + slippage_range_coeff × (high − low)/open` of the execution
  bar (defaults 5 bp and 0.25), plus the taker fee (`taker_fee_bps`, default 4.5). Funding is
  accrued hourly from the archive only when the perp existed at `t0` and archived funding covers
  `[d, t0+24h]`; otherwise zero, with `funding_source ∈ {archive, none}` recorded and the share of
  events (and of PnL) with real funding reported. Side = sign(`p_up` − 0.5) when
  `|p_up − 0.5| ≥ trade_threshold` (a setting, default 0), else no trade. Sizing variants:
  `fixed`, `by_confidence` (∝ `|p_up − 0.5|`), `by_magnitude` (`target_vol` / predicted
  `|r_24h|`, capped) and a `magnitude_gate` (trade only when predicted `|r_24h|` exceeds the
  round-trip cost), so a magnitude forecast has its own PnL line. The predicted `|r_24h|` is
  the model's `predict_magnitude` (predictions column `magnitude_hat`); `|r_hat|` stands in
  only for a model without one. `trade_threshold` and `target_vol` are settings and part of
  the report's config hash.
* Sample size is part of every result: each cell (model × decision time × subset) reports `n`,
  the bootstrap interval, and the **minimum detectable improvement** over the best baseline at
  that `n` (Brier and accuracy), derived from the paired comparison's own standard error:
  MDE = (z₀.₉₇₅ + z₀.₈) × SE of the mean per-event score difference, i.e. the MDE of the test
  the report actually runs. The closed-form unpaired bound (Bhatia-Davis for Brier) is shown
  only for cells without a comparison and is labelled an upper bound. The conclusion "not
  predictable with these inputs" may be drawn only when the interval excludes the minimum
  detectable improvement; otherwise the report says "inconclusive at n = …". With listings from Nov 2025 the perp-era pre-holdout cohort is a few
  hundred events at best, so early reports will mostly be inconclusive, and the report says so.
* `evaluate --final` refuses to run until the holdout season is closed (now ≥ start of the
  next season + horizon: a dataset built mid-season passes every per-row test yet cannot hold
  the events still scheduled), while any universe event scheduled in the holdout season has
  `t0 + 24h` in the future or targets pending, or when `events.parquet` lists a holdout-season
  event the dataset lacks. The holdout advances at most once per closed
  season by a human edit of `holdout_season`; the previous holdout joins the folds, and after the
  first advance the live record (§10) is the primary honest number.
* Portfolio metrics: capital rule `equal_split` with gross exposure cap 1.0: each position
  holds `cap × min(size, 1) / (peak n_open over its [entry, exit] interval)` for its whole
  life — a constant weight per position, so the summed exposure never exceeds the cap; capital
  freed when an overlapping position closes early is not redeployed (a time-varying
  `1 / n_open(t)` share would need the price path at every concurrency change). The rule is
  printed in the summary and leaderboard. Net PnL is aggregated into a daily series keyed by
  UTC exit date and the Sharpe-like ratio, max drawdown (starting capital counts as the first
  peak) and turnover are computed on that series. Per-event mean net return and hit rate keep
  their bootstrap 95 % intervals and a paired comparison against the best baseline.
* Bootstrap resampling: blocks by season when at least 5 seasons are present, else by UTC day
  of `t0` (same-day dependence is kept), else iid rows; the scheme is recorded per cell. A
  single-season block — the holdout — can therefore never report a zero-width interval, which
  is what a season block bootstrap would degenerate to.
* Reproducibility: `run_id = <UTC yyyymmddTHHMMSSZ>-<sha256(dataset.parquet)[:8]>` (the
  file's bytes; a frame with no file on disk falls back to a content hash and the summary
  records `dataset_hash_source`);
  `reports/<run_id>/summary.json` records the dataset hash, git commit and dirty flag, non-secret
  settings, library versions, all metrics, and `predictions.parquet` holds every out-of-sample
  prediction. All stochastic steps derive from `settings.random_seed` (LightGBM
  `seed`/`deterministic`, Optuna `TPESampler(seed)`, numpy generators).

## 9. Optimisation ("optimizes for predicting")

`freedom optimize --decision-times <list> [--objective brier] [--n-trials N]` runs **one Optuna
study per decision time** (study name `freedom_<d>_<objective>`, persisted in
`data/optuna.db`). Decision time is not a search dimension and neither is the `t0` confidence
floor. Per study the search space is: model family and hyper-parameters, feature groups on/off
(only groups admissible at `d`), target variant (raw vs abnormal), and training-window length.
The objective is the walk-forward metric on folds that exclude the pinned holdout season. The
leaderboard is per decision time; `optimize` never scores the holdout. The report states the
number of trials next to the improvement over the best baseline and the bootstrap probability
that the improvement is noise.

## 10. Live prediction

`freedom upcoming` lists universe events in the next N days (FMP calendar with the newest
archived consensus snapshot). `freedom predict --event <event_id> --decision <d>` builds
features `as_of` a well-defined instant and loads the trained model for `d`:

* **pre_5m**: `as_of = expected_t0 − 5 min`, where `expected_t0` is the issuer's median
  acceptance clock time over its past `sec_8k` events (fallback: the calendar-flag default).
  The live row stores `as_of`, `expected_t0`, and, once the 8-K arrives, `t0_actual` and
  `t0_lag_s`.
* **post_k**: `t0_live` comes from the live detector on 1-minute perp or FMP bars (the same
  code as the historical detector); the 8-K acceptance is back-filled afterwards and the row is
  scored in the `detected` stratum. `predict` marks the row `off_schedule` (and does not trade)
  when `now − t0_live` is outside `[k − 1 min, k + max_fill_lag]`.
* Every live row records `model_id`, the data sources used, `input_lag_s` per source (FMP bar
  availability, Hyperliquid candle, SEC submissions), and is appended to
  `data/live_predictions.parquet`; `freedom evaluate --live` later scores those rows against
  realised targets and compares live and backtest decision instants. The one-off measurement of
  input lags during an earnings evening is recorded in data-sources.md when done.
* Output: `p_up`, expected `r_24h`, residual-based 10/90 % band, top feature contributions,
  consensus provenance, and the model's walk-forward record for `d`.

`freedom train` writes `model.json` next to the model with `decision_time`, `dataset_sha256`,
git sha, config hash, `trained_at`, `n_events`, the filters applied (`min_t0_confidence`,
`has_perp_at_t0`) and the holdout-score reference.

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

## 13. Review outcomes (2026-09-02)

Four independent reviewers (leakage, data engineering, trading realism, scope) produced 32
findings; 25 survived adversarial verification and are folded into the sections above. The
rejected ones, for the record: an "8 quarters of history" requirement the design never made; a
claim that the cache cannot support reruns (it is the checkpoint); a leverage/liquidation model
(out of scope for a research harness); funding materiality (measured immaterial on weekday
events); zero-key behaviour (FMP is required and fails fast); and two checkpoint-computability
claims that the bar rule already handles. Text features were deferred, the per-model interval
method was dropped in favour of residual bands, and `predict` reports only point-in-time
consensus.
