# Ad hoc call method N3 (from 2026-10-03)

Decided by the user on 2026-10-03: research picks the side; a coin is kept only as the benchmark;
the goal is to get better at predicting direction and magnitude over time, not to beat the other
models on any one card. Judgement is built and trusted from graded outcomes: the mechanical
inputs tested so far (the stock's own up/down record, price trends, short volume, options skew,
release timing) showed no edge, but research-led judgement had not yet set a single direction, so
its record starts now. Replaces Rule N2 (coin side of record), which applied to no graded call.

## Before the release

1. Read [LESSONS.md](LESSONS.md) and apply what is relevant.
2. Research the print against the bar: the numbers the market will judge (guide, margins, the key
   segment), the consensus and any whisper, positioning, peers' recent prints.
3. Kill switches: 3 to 5 conditions observable in the release or on the call (exact metric and
   threshold) that would flip or confirm the call. Give each a probability that it fires, computed
   from data (consensus spread, the company's beat and guide history, peer prints, monthly data,
   channel checks), with the basis written down.
4. Build P(up) from scenarios: P(up) = sum over scenarios of P(scenario) x P(up | scenario), with
   P(up | scenario) taken from history where it can be measured. Direction = the side above 0.50.
   The stock's own up/down record is shown as context only: it scored 0.501 on 976 held-out prints.
5. Magnitude = the expected absolute move (options-implied where a liquid chain exists, otherwise
   the median of the stock's own event-day absolute moves); 10/90 band.
6. Benchmark: record the coin side, SHA-256("<TICKER> <date>") first hex digit 0-7 = UP, 8-f = DOWN,
   computed before research. It never sets the call.

## After the release (grading)

1. Direction hit for the call and for the coin; Brier score on P(up) against the coin's 0.25;
   magnitude error; band.
2. For each kill switch: fired or not, against its probability.
3. Classify the miss: a fundamentals miss (a switch's probability was wrong) or a reaction miss
   (the switches fired as expected but the price went the other way).
4. "What I missed and how to compute it next time": the data that would have given a better
   probability for the switch that decided the move. Append a dated entry to LESSONS.md.
5. Lessons are hypotheses with a track record, not rules. Each one says how it would change a call;
   at every grading, mark each lesson that applied as helped, hurt or neutral. A lesson with a
   clear positive record becomes part of the method; one that keeps hurting is retired. This is how
   the method changes from experience rather than from priors.
6. Running tallies: hit rate of research against the coin, and the share of DOWN calls (a one-way
   lean is a warning sign).
