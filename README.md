# Freedom

A research and prediction harness for the price action of an equity in the 24 hours after its
earnings release, for every equity that trades as a perpetual future on Hyperliquid (HIP-3
markets such as `xyz:NVDA`, `xyz:AAPL`, `para:AVGO`). It builds the event dataset, evaluates
models honestly out of sample, simulates trading with fees, slippage and funding, optimises
itself against that evaluation, and produces live predictions with the evidence behind them.

What it does **not** do: promise that post-earnings moves are predictable. Whether they are, at
which decision time, and by how much, is an output of the harness and every result comes with a
sample size and a minimum detectable improvement.

## Read this first: what the harness assumes and what it measured

* "Stocks on Hyperliquid" are perpetual futures on builder-deployed dexes (`xyz`, `para`, `io`),
  not shares. They trade 24/7 with hourly funding; outside the cash session their price comes
  from an oracle that follows after-hours liquidity providers and, during closures, drifts with
  the perp's own order flow. The 24-hour window is therefore tradable end to end, but overnight
  and weekend segments are thin.
* The universe is small. On 2026-09-02 the live pull found 138 markets, of which 71 are
  equities with earnings (US filers and US-listed foreign issuers). Most were listed in 2026, so
  the perp-era event history is roughly a hundred events; older events use the underlying's
  extended-hours equity bars as the price proxy and are labelled as such.
* Hyperliquid serves only the most recent 5000 candles per interval (1-minute history ≈ 83
  hours). The harness ships an archiver, and a scheduled GitHub Actions job, so that fine-grained
  perp history accumulates from now on.
* Release times come from SEC 8-K item 2.02 acceptance timestamps, which trail the newswire by
  25–134 s on the filings measured, so the reference price backs off three minutes. Foreign
  issuers' 6-K timestamps are not usable and fall back to the issuer's documented release
  clock (`configs/release_clock_overrides.yaml`), else detection from 1-minute bars.
* Consensus estimates from vendors are their *final* values, not the consensus as of the
  release. Historical surprise features are therefore labelled non-point-in-time; from now on the
  archiver snapshots the calendar daily so future events have real point-in-time consensus.

`docs/data-sources.md` lists every measurement behind these statements; `docs/design.md` is the
design, including the 31 review findings that shaped it.

## Definitions

* **Event**: `(underlying, fiscal quarter, t0)` with `t0` the release instant in UTC.
* **Reference price** `P0`: close of the last bar ending at or before `t0 − δ` (`δ` = 3 min for
  8-K-timed events, 0 otherwise). Bars are half-open; a bar containing the instant is never used.
* **Checkpoints**: +5m, +15m, +30m, +60m, +2h, next regular open, open+30m, next regular close,
  +24h. Each is valid only if its bar ends after the `P0` bar and within max(2 × bar interval,
  5 min) of the checkpoint.
* **Targets**: log returns `r_h` from `P0`, abnormal returns against `xyz:SP500` (or SPY), and
  labels `direction`, `magnitude` and `continuation_k = sign(r_k) · sign(r_24h − r_k)` for
  k ∈ {15m, 30m} (+1 the early reaction extended, −1 it reversed).
* **Decision times**: `pre_10m`, `pre_5m`, `post_1m`, `post_15m`, `post_30m`, `post_60m`. Every feature is
  built `as_of` the decision instant and the models for different decision times are never mixed.

## Install

```bash
uv venv && . .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # add FMP_API_KEY (required for equity bars and earnings history)
freedom status
```

Keys: `FMP_API_KEY` is required (earnings history, extended-hours bars, daily bars).
`ALPHAVANTAGE_API_KEY` is optional (report-time flag for foreign issuers; 25 requests/day on the
free tier). Hyperliquid, SEC EDGAR and Nasdaq need no key. Yahoo Finance is not used.

## Run it end to end

```bash
freedom universe                 # Hyperliquid markets -> data/universe.parquet (+ rows to verify)
freedom archive                  # candles, funding, context and consensus snapshots (run every 12 h)
freedom events --since 2024-01-01 --underlyings NVDA,AAPL,MSFT,AMZN,META,GOOGL,TSLA,AMD,MU,INTC
freedom dataset --decision-times pre_10m,pre_5m,post_15m,post_30m
freedom evaluate --models zero,base_rate,historical_mean,hist_abs_mean,vol_scaled,sign_of_reaction,always_extends,surprise_sign,linear,lightgbm --decision-times pre_5m,post_30m
freedom optimize --decision-times post_30m --n-trials 50
freedom train --model lightgbm --decision-time pre_10m
freedom train --model lightgbm --decision-time post_30m
freedom upcoming --days 14
freedom predict --event NVDA:2026-10 --decision pre_10m   # card 1, ten minutes before
freedom predict --event NVDA:2026-10 --decision post_30m  # card 2, thirty minutes after
freedom cards --horizon-minutes 45                        # every card due in the next 45 min, unattended
```

When a vendor calendar has the wrong day (Oracle's Q1 FY2027 date was two days off in FMP and
four in Finnhub), put the correction in `configs/report_date_overrides.yaml` as
`ORCL:2026-09-08: 2026-09-10` (vendor date to issuer-confirmed date). It applies before the 8-K
search, the Nasdaq lookup and the `upcoming` schedule. `configs/t0_overrides.yaml` still pins an
exact release instant when the time, not the day, is wrong. An issuer that releases at the same
local clock every quarter (ASML at 07:00 Amsterdam, TSMC at 14:00 Taipei) goes in
`configs/release_clock_overrides.yaml` as `ASML: "07:00 Europe/Amsterdam"`: every past and
upcoming event of that issuer then gets `t0` at that clock (`t0_source = issuer_clock`,
confidence 0.7, below an 8-K acceptance and above a bar detection). Such releases fall outside
the FMP proxy's 04:00–20:00 ET bars, so their pre-listing rows keep `p0` but carry no labels
(`label_reason = p0_stale`) until a perp path exists.

Each `predict` prints a card first: `CALL: LONG | SHORT | NO TRADE` (NO TRADE unless `p_up` is at
least `no_trade_band` = 0.10 away from 0.5), the expected 24 h move with its 10/90 % band, and the
five features that pushed the call, signed, each with a plain-language description. When the
direction head is untrained (fewer than 30 usable rows) the reasons are importance-ranked and
unsigned, and the card says so. Off-schedule and replay rows print `NOT TRADEABLE`.

```
```

## Cards

`freedom cards` produces the cards without anyone typing. It lists the upcoming events, works
out every decision instant (`expected_t0` − 10 min for `pre_10m`, + 15 / + 30 min for
`post_15m` / `post_30m`) and, for each instant in the next `--horizon-minutes` (45), sleeps until
it, runs the same prediction as `freedom predict`, prints the card, appends the row to
`data/live_predictions.parquet` and writes `reports/cards/<event>__<decision>.md` (a compact
card), the same `.json`, and a line in `reports/cards/index.md`. A post-release card whose
release the detector has not seen yet is retried every 60 s for 15 minutes and then recorded
as a "no release detected" note; a release that came late is re-run at its real `as_of`; a pair
already predicted live is skipped, so overlapping runs never duplicate a card.

The `freedom-cards` GitHub Actions job (`.github/workflows/cards.yml`) runs it every 15 minutes
around the clock (Hyperliquid trades 24/7; ASML and TSMC release overnight for New York) on top
of the newest `freedom-data` artifact, whose job now also trains the `pre_10m`, `post_15m` and
`post_30m` LightGBM models, and keeps the rows and cards in a `freedom-live` artifact that the
data job folds back into `freedom-data`. One-time setup, after the data job's secrets:

1. Create an issue titled "Cards".
2. Add a repository variable `CARDS_ISSUE` (Settings → Secrets and variables → Actions →
   Variables) holding that issue's number.
3. Subscribe to the issue in the GitHub mobile app: every card the job produces is posted as a
   comment, and the app's notification delivers it to the phone.

Caveat: GitHub's cron can start minutes late, and a run that is waiting for an instant blocks
the next scheduled run until it finishes. A card computed after its window (more than 5 minutes
after the instant for a post-release decision, after the expected release for a pre-release one)
still comes out but prints `NOT TRADEABLE` with the reason, and an instant missed by more than
5 minutes produces no card at all. `freedom cards --now <UTC ISO>` replays a window on the
command line without sleeping; its rows are marked replay and its cards `NOT TRADEABLE`.

The ten-name/2024 example yields fewer than the default 120 trainable events per walk-forward
fold, so `evaluate` and `optimize` need `FREEDOM_MIN_TRAIN_EVENTS=12` (what the first-run
reports in `docs/results.md` used); without it they exit 2 and say so.

The FMP free plan allows about 250 requests a day. `events` and `dataset` treat an exhausted
budget as a checkpoint: they write what they have, mark the rest `pending`, exit non-zero, and
resume from the on-disk cache when rerun the next day. Building the full universe back to 2024
takes several days of quota; a subset of underlyings, as above, fits in one.

## Evaluation protocol

Walk-forward by earnings season with an embargo; a **pinned holdout season**
(`FREEDOM_HOLDOUT_SEASON`, default `2026Q3`) that `evaluate` and `optimize` never touch and that
only `evaluate --final` scores, with every scoring logged. Metrics per decision time, for events
with a live perp at `t0` (headline) and for all events, stratified by release-time source,
issuer kind and timing, each with `n`, bootstrap intervals and the minimum detectable improvement
over the best baseline per metric. The trading simulation fills at the open of the bar after the
signal, charges a floor plus range-based execution cost and taker fees, accrues archived funding
where it exists, and reports portfolio metrics under an equal-split capital rule.

## Results so far

First end-to-end run (2026-09-02, twelve early-listed names, 48 past events, two walk-forward
folds, 24 out-of-sample events per cell; details and how to read them in `docs/results.md`):

* Release times resolved from 8-K acceptance for all 48 events; detection moved 35 of them
  earlier by a median of 85 s (the wire-to-filing lag).
* At the pre-release decision time nothing beats the base rate, and the learners never trained
  (folds of 12 and 24 rows are below the 30-row floor at which they fall back to the base
  rate); the report labels them "untrained" rather than judging predictability.
* After the release, the sign of the first 15–30 minutes agrees with the 24-hour direction in
  63–67 % of events, but the 95 % interval includes 50 % and the paired comparison is
  inconclusive at n = 24. The `always_extends` baseline is the honest comparator for the
  continuation question and is not yet distinguishable from a coin flip.
* Sample-size arithmetic: the minimum detectable Brier improvement at n = 24 is about 0.25
  (unpaired bound); the full 71-name universe yields roughly 280 events a year, at which point
  it falls to the 0.01–0.02 range where a real edge would show.

## Layout

```
src/freedom/universe   Hyperliquid markets -> classified event universe (configs/universe_overrides.yaml)
src/freedom/data       provider clients (hyperliquid, fmp, sec, nasdaq, alphavantage), cache, budgets, archiver
src/freedom/events     earnings events and the release-time resolver
src/freedom/targets    price paths, checkpoints, returns, labels
src/freedom/features   feature groups with as_of discipline, dataset builder
src/freedom/models     baselines, linear, lightgbm, ensemble
src/freedom/eval       walk-forward folds, metrics, trading simulation, bootstrap, reports
src/freedom/optimize   Optuna study per decision time
src/freedom/live.py    one live prediction (freedom predict); src/freedom/card.py its card
src/freedom/cards.py   every card due in a window, unattended (freedom cards; the freedom-cards job)
src/freedom/cli.py     the `freedom` command
```

## Known limitations

* Small samples: expect most cells to be "inconclusive" until several more earnings seasons of
  perp-era data have been archived.
* Historical consensus is vendor-final, not point-in-time.
* The all-in taker fee for `xyz` growth-mode markets was not measured; the simulation defaults to
  the conservative 0.045 %.
* Text features (guidance tone from the press release) are deferred: an LLM trained after the
  event knows the outcome, so they cannot be made point-in-time for historical rows.
