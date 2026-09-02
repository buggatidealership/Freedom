# Freedom

A research and prediction harness for the price action of equities in the 24 hours after an
earnings release, for every equity that trades as a perpetual on Hyperliquid (HIP-3 markets such
as `xyz:NVDA`). The harness builds the event dataset, evaluates models honestly out of sample,
simulates trading with fees and funding, optimises itself against that evaluation, and produces
live predictions with the evidence behind them.

Status: scaffold. See `docs/design.md` for the design and `docs/data-sources.md` for what each
data source was measured to provide.
