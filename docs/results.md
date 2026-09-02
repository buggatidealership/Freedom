# First end-to-end run (2026-09-02)

Run `20260902T142936Z-154046cc` (`reports/20260902T142936Z-154046cc/`), twelve underlyings with the earliest Hyperliquid listings
(NVDA, TSLA, AAPL, MSFT, GOOGL, AMZN, META, MU, INTC, PLTR, AMD, NFLX), events since
2025-09-01, decision times `pre_5m`, `post_15m`, `post_30m`, minimum training events lowered to
12 for this smoke run (`FREEDOM_MIN_TRAIN_EVENTS=12`; the default is 120). Everything below is
reproducible from the commands in the README and about 190 FMP requests.

## What was measured

| quantity | value |
|---|---|
| Hyperliquid markets pulled | 138 (71 in the event universe, 15 flagged for human verification) |
| events resolved | 48 past + 12 upcoming; 48 from 8-K acceptance times, 0 pending |
| release times refined by detection | 35 of 48, median lag between wire and 8-K acceptance 85 s (IQR 32–257 s) |
| price paths | 47 events on FMP 1-minute extended-hours bars, 1 (NVDA 2026-08-26) on archived 1-minute perp candles |
| 24-hour label coverage | 48 of 48 past events |
| dataset | 180 rows (60 events × 3 decision times), 78 features, 8 never populated (open interest and VIX: no archive yet; surprise z-scores: no history; `r_60m` at earlier decision times by construction) |
| walk-forward | 2 folds (test 2026Q1 trained on 12 events; test 2026Q2 trained on 24), holdout 2026Q3 never scored |
| out-of-sample events per cell | 24 |

Post-release reactions in this sample are large: median `|r_24h|` 7.7 %, ten of 48 beyond
±10 %. Fourteen of the 24 test-fold events with a valid early reaction reversed sign between
30 minutes and 24 hours (`continuation_30m = −1`), ten extended.

## Out-of-sample results (headline cohort, n = 24)

| decision | model | accuracy [95 % CI] | Brier | Spearman IC | verdict |
|---|---|---|---|---|---|
| pre_5m | base_rate | 0.542 [0.36, 0.71] | 0.249 | −0.22 | baseline |
| pre_5m | vol_scaled | 0.542 | 0.249 | 0.04 | baseline (magnitude MAE 0.043 vs 0.077 for zero) |
| pre_5m | linear, lightgbm | 0.542 | 0.249 | −0.22 | untrained: every fold below 30 rows, so they reproduce the base rate |
| post_15m | sign_of_reaction | 0.667 [0.52, 0.80] | 0.229 | 0.70 | baseline |
| post_15m | always_extends | 0.667 | 0.229 | 0.71 | baseline |
| post_15m | surprise_sign | 0.417 | 0.354 | 0.18 | baseline |
| post_30m | sign_of_reaction | 0.625 [0.46, 0.79] | 0.250 | 0.78 | baseline |
| post_30m | linear, lightgbm | 0.542 | 0.249 | −0.22 | untrained |

Trading simulation (fixed size, taker 4.5 bp + floor 5 bp + range-based slippage per leg, no
funding because no event had archived funding): the reaction-sign baselines at `post_30m`
average +90 bp net per event with a Sharpe-like ratio of 1.8 on the daily series; the surprise
sign loses 169 bp per event; the base rate makes +63 bp because the sample skews up. None of
these intervals exclude zero.

## How to read this

* **Verified**: the pipeline runs end to end on live data; release times, reference prices and
  checkpoints obey the bar rules (unit-tested on the real NVDA reaction and checked by hand on
  MU, AAPL and NVDA filings); the holdout season was not touched; every number above carries
  its sample size.
* **Derived**: with 24 out-of-sample events the minimum detectable Brier improvement is about
  0.25 in the unpaired bound and 0.03–0.06 in the paired comparisons, so no verdict other than
  "inconclusive" or "untrained" is possible. The learners never trained: 12 and 24 rows are
  below the 30-row floor at which they fall back to the base rate, by design.
* **Hypothesised, not shown**: that the early-reaction sign carries information about the rest
  of the 24 hours (accuracy 0.63–0.67, IC 0.7–0.8 at post_15m/post_30m). The IC is inflated
  mechanically because `r_30m` is a component of `r_24h`; the accuracy interval includes 0.5.
  The `always_extends` baseline is the right comparator for the continuation question and it
  is not yet distinguishable from a coin flip here.
* **What would change the picture**: the full 71-name universe (≈280 events a year) and two
  more archived seasons of 1-minute perp candles. At n ≈ 150 the paired MDE for Brier falls to
  roughly 0.01–0.02, enough to see a real edge if one exists.

## Known issues surfaced by this run

* FMP and Nasdaq report different EPS actuals (GAAP vs non-GAAP) for 21 of 48 events; the
  harness flags `eps_actual_conflict` and uses FMP's value, so the `surprise` feature group
  is on a mixed basis.
* `xyz:VIX` is delisted; the VIX features are always missing.
* The closing-auction minute fired the release detector once (MU 2026-06-24); the resolver now
  refuses to move an after-close filing into the regular session.
* Consensus for all past events is vendor-final (`estimate_source = fmp_final`), as documented.
