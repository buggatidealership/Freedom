# Market questions (graded yes-or-no forecasts on indexes)

A separate class from the earnings calls. Each question names an index, a level, a deadline and a source, and carries one probability fixed at the lock. Scoring: Brier score, (probability minus outcome) squared, outcome 1 for yes and 0 for no, set beside the Brier score of the control rate (how often the same thing happened from any day in the index's history). A locked probability is not revised; a later view is a new question with a new id. The earnings gate has no mode for this class, so these documents are not gate-checked. That gap is stated to the user and is theirs to close.

## Ledger

| id | Locked | Question | Deadline | P(yes) | Fitting cell | Control | Outcome | Brier | Control Brier |
|---|---|---|---|---|---|---|---|---|---|
| MQ1 | 2026-11-21 | Nasdaq Composite closes at or under 22,079.91 on any day from Nov 23 through Dec 31 2026 | 2026-12-31 | 0.72 | 17 of 23; 8 of 9 fast; 8 of 17 on a first fear spike | 0.49 | pending | | |
| MQ2 | 2026-11-21 | Nasdaq Composite closes at or under 19,319.92 on any day from Nov 23 2026 through Mar 31 2027 | 2027-03-31 | 0.32 | 8 of 23; 3 of 9 fast; 6 of 17 | 0.15 | pending | | |
| MQ3 | 2026-11-21 | S&P 500 closes at or under 6,255.14 on any day from Nov 23 2026 through Mar 31 2027 | 2027-03-31 | 0.28 | 7 of 33; 3 of 13 fast; 5 of 17 | 0.16 | pending | | |
| MQ4 | 2026-11-21 | Nasdaq Composite closes above 22,617.73 on Feb 26 2027 | 2027-02-26 | 0.55 | 14 of 23; 6 of 9 fast; 13 of 17; 1 of 5 with calm credit | 0.66 | pending | | |
| MQ5 | 2026-11-25 | Nasdaq Composite closes under 22,617.73 on any day from Nov 25 2026 through Jan 29 2027 | 2027-01-29 | 0.55 | cells disagree: 1 of 7 after a close 5 % above came first; 6 of 6 on the Nasdaq 100 fund's fall-rise-gap pattern (three episodes); 6 of 7 fast cases went lower | not computed | pending | | |
| MQ6 | 2026-11-25 | October 2026 core PCE price index, year on year as first published and rounded to one decimal, is 3.3 % or higher (estimate 3.2 %) | 2026-11-25 | 0.38 | my fit on 18 months of core CPI: plus 0.29 % on the month, 3.23 to 3.25 % on the year | none | pending | | |
| MQ7 | 2026-11-25 | October 2026 headline PCE price index, year on year as first published and rounded to one decimal, is 3.9 % or higher (estimate 3.8 %) | 2026-11-25 | 0.31 | my fit on 18 months of CPI: plus 0.32 % on the month, 3.82 % on the year | none | pending | | |

Range forecast: Nasdaq Composite close on Dec 31 2026, 10th percentile 19,200, median 22,400, 90th percentile 24,650. Scored inside or outside; over many such ranges eight in ten should land inside.

## 2026-11-21 lock: MQ1 to MQ4

Locked 2026-11-21 21:51 UTC, a Saturday; no session has traded since the reference closes. Lessons receipt: 094d16ab04f5 read 2026-11-21T21:45Z. What was read: every part of LESSONS.md, 1 to 15, printed in full and uncut in this same context before this lock. The questions were written by me from a template the user sent with its blanks unfilled.

**Resolution.** Official daily closes. Nasdaq Composite from the FRED series NASDAQCOM, cross-checked with Yahoo's index series. S&P 500 from Yahoo's index series, cross-checked with FRED SP500. Read after 22:00 UTC on each deadline. A yes on MQ1, MQ2 or MQ3 is fixed by the first qualifying close and its date is recorded.

**Reference state (M, two sources agree).** Nasdaq Composite 22,617.73 on Nov 20 2026, 18.05 % under its record close of 27,599.89 (Oct 6), session 33. S&P 500 6,910.39, 11.62 % under 7,818.93. Volatility index 35.89, its first close at 35 or more. Distance still needed: MQ1 a further 2.38 %, MQ2 a further 14.58 %, MQ3 a further 9.48 %. Sessions from Nov 23: 27 to Dec 31, 65 to Feb 26, 87 to Mar 31. The Fed decision on Dec 9 comes in session 12.

**Decision it serves.** The user left this blank. The probabilities do not depend on it.

### 1. Counted cases and base rates (D)

Each past case is measured by the distance still needed today, from its own close, not by whether it crossed a named line. Cases are listed one by one in the session's count; the totals:

| Cell | n | MQ1: further 2.4 % in 27 sessions | MQ2: further 14.6 % in 87 | MQ4: higher 65 sessions on |
|---|---|---|---|---|
| Nasdaq, first close 18 % under its peak, 1971 on | 23 | 17 | 8 | 14 |
| Same, reached within 40 sessions (today's speed) | 9 | 8 | 3 | 6 |
| Same, fast and from a record close | 6 | 5 | 2 | 5 |
| First volatility-index close at 35 or more, 1990 on | 17 | 8 | 6 | 13 |
| Fed funds higher 63 sessions later (line-crossing count) | 7 | 6 | 2 | 4 |
| Fed funds lower 63 sessions later (line-crossing count) | 10 | 10 | 6 | 6 |
| Baa spread not wider over the prior 63 sessions | 5 | 5 | 3 | 1 |
| Baa spread wider by 0.05 or more | 13 | 9 | 5 | 10 |
| Control: every session since 1971 | 13,980 | 49 % | 15 % | 66 % |

The 23 first closes: 1973, 1978, 1980, 1981, 1983, 1987, 1990, 1998, Apr 2000, Oct 2000, Feb 2001, Aug 2001, 2002, 2004, 2008, 2009, 2011, 2016, 2018, 2020, Feb 2022, Sep 2022, 2025. The six that did not fall a further 2.4 %: 1978, 1983, 2004, 2011, 2016, 2025.

| Cell for MQ3 | n | S&P fell a further 9.5 % in 87 sessions |
|---|---|---|
| S&P, first close 11.5 % under its peak, 1950 on | 33 | 7 (1962, 1966, 1987, 2002, 2009, 2020, 2022) |
| Same, reached within 35 sessions | 13 | 3 |
| Same, Nasdaq already 15 % or more down that day | 17 | 3 |
| First volatility-index close at 35 or more | 17 | 5 |
| Control: every session since 1950 | | 16 % |

### 2. Adjustments
- MQ1. The drawdown cell says 0.74 and its fast cases 8 of 9. The fear-spike cell says 8 of 17, which is the control rate: a first panic close carried no information about the next leg. Where both conditions held in a first leg down (1990, 1998, 2011, 2018, 2022, 2025) the further fall came in 2 of 6. Taken at 0.72: two points under the drawdown cell. No adjustment for the Fed; the count does not support one.
- MQ2. Every cell gives 0.33 to 0.35. Taken at 0.32. A first draft took points off for calm credit; the count does not support that (3 of 5 with calm credit went on), so no credit adjustment is made.
- MQ3. Cells give 0.18 to 0.29. Taken at 0.28, at the top of the cells, because MQ2 and MQ3 are nearly one event: at the ratio the two indexes have fallen since Oct 6 (1.55), the S&P's 9.5 % is the Nasdaq's 14.7 %. A first draft added points because the Nasdaq was already deep; the wider count (3 of 17) does not support that.
- MQ4. Cells range from 1 of 5 to 13 of 17. The drawdown cell says 0.61 and the fear-spike cell 0.76. Against them: where credit had not yet widened at this stage the index was lower 65 sessions on in 4 of 5 (1987, 1990, 2002, 2022; 2009 the exception), and with the Fed still raising 4 of 7. Taken at 0.55, six points under the drawdown cell. This one is close to a coin and is logged as such.

### 3. Final probabilities
MQ1 0.72. MQ2 0.32. MQ3 0.28. MQ4 0.55.

### 4. Range for the Nasdaq Composite close on Dec 31 2026
10th percentile 19,200 (minus 15 %). Median 22,400 (minus 1 %). 90th percentile 24,650 (plus 9 %). From the 27-session changes after the 23 first closes (10th percentile minus 16.4 %, median plus 2.4 %, 90th plus 12.3 %) and the nine fast ones (minus 16.4, minus 1.6, plus 7.3).

### 5. Under each outcome of the Fed decision on Dec 9 2026 (H)
| Outcome | Market odds, Nov 17 to 19 readings (S) | MQ1 | MQ2 | MQ3 | MQ4 |
|---|---|---|---|---|---|
| No change | about 3 % | 0.68 | 0.22 | 0.18 | 0.65 |
| Raise of 0.25 | 90 to 97 % | 0.72 | 0.32 | 0.28 | 0.55 |
| Raise of 0.50 | 3 to 10 % | 0.82 | 0.48 | 0.42 | 0.42 |

These are judgment, not counts. The count runs against the simple story: the falls in which fed funds came down over the next 63 sessions were the deeper ones (6 of 10 reached 30 % in 87 sessions, against 2 of 7 where funds went up), because the Fed cuts into a crisis. Eleven of MQ1's 27 closes come before the decision. A second dated event: Nvidia's quarterly filing. None was on EDGAR at the lock (M); the due date near Dec 4 is my estimate (H).

### 6. The strongest case for the opposite outcomes
The low close was Nov 20. Fear closed at a level that has marked lows: after a first volatility close at 35 the Nasdaq was higher 65 sessions on in 13 of 17, and in nine of those 17 it never closed 2.4 % lower in the next 27 sessions. Bonds were bought on the day and the average stock has barely fallen, so this may be one crowded trade finishing its unwind. The Nov 5 chip crash was retraced in full within four sessions, and my own notes say I under-allow for that. On that reading MQ1, MQ2 and MQ3 are no and MQ4 is yes.
For MQ4 the opposite case is the calm-credit cell: 4 of 5 lower, in falls driven by rates and not by a credit event, which is today's kind.

### 7. The one measurement, and what was not retrieved
The measurement: the Baa credit spread. It is 0.16 narrower than three months ago (1.52 on Nov 18, 1.56 on Nov 19). In past falls it had already widened by this stage in 13 of 18 cases, and those were the ones that were higher three months on.
Not retrieved: index option prices, which would give the market's own odds for these exact levels and are the benchmark I would most want; Fed odds taken after the Nov 20 close; the Baa spread for Nov 20; earnings estimates and their revisions; valuations; margin debt and positioning; fund flows.

### Facts used, dated
- Oct 6 2026: record closes, Nasdaq Composite 27,599.89 and S&P 500 7,818.93 (M).
- Fed target upper bound 4.00 % (M). Next decision Dec 9; a raise of 0.25 priced at 90 to 97 %, of 0.50 at 3 to 10 %, after a governor's remarks; the New York Fed president ruled out a half point (S).
- Nov 20: Nasdaq Composite minus 5.28 %, S&P 500 minus 3.70 %, volatility index 35.89 (M). Nasdaq minus 10.0 % on the week (S).
- Nov 20: a second report says the SEC inquiry into Nvidia concerns its accounting; neither party has confirmed it (S). No quarterly report or late-filing notice from Nvidia on EDGAR by Nov 21 (M).
- Nov 21 news sweep: doubts about AI financing and a large new listing falling after its debut are the reasons the press gives for the week (S).

### Reconciliation with my unlogged view of Nov 20
Given then, with no deadline: Nasdaq Composite under 22,080 at some point, 90 % (the line-crossing count, 20 of 23); S&P under 6,255 before a new record, 45 %; S&P higher in about three months, 65 %. MQ1 to MQ3 are the same events with deadlines and are lower for that reason. MQ4 is lower than the Nov 20 figure by ten points and on a different index; the calm-credit count was made today.

### What would disprove the method
A Brier score on this class no better than the control's after twenty graded questions.

### Alternative reading and shared assumptions
Alternative: the cells disagree because none of them describes this market, and a single forecaster's index odds are no better than the control rates. Shared assumptions: closes are the right measure; falls since 1971 and 1950 are comparable with one led by a handful of AI companies; the legs of 2000 to 2002 count as separate cases (four of the 23).

## 2026-11-25 lock: MQ5 to MQ7

Locked 2026-11-25 11:06 UTC, before the 13:30 UTC release. Lessons receipt 69c3c601e72b: all 15 parts printed in full on Nov 24 ending 18:31 UTC in this same context, and the entries added since printed uncut today with `read --changes`. Not gate-checked; the gate has no mode for this class.

**MQ5 and my point view of the low.** Nasdaq Composite: record close 27,599.89 (Oct 6), first close 18 % under it 22,617.73 (Nov 20, session 33), then plus 6.51 % and minus 1.48 %; 23,733.54 on Nov 24. Twenty-one earlier first closes 18 % under a peak (one vendor's daily closes from 1971; a simpler episode rule than the Nov 21 count, which found 23): the final low was within 2.4 % of that close in 5, a further 2.4 to 10 % in 3, more than 10 % in 13; median further fall 13.7 %. In the seven reached within 40 sessions of the peak the low came 3 to 29 sessions later in all seven, with a median further fall of 13.5 % (1.4, 7.8, 19.0, 13.5, 22.0, 4.1, 13.7). No earlier case had a 6 % rise on the very next session. My point view, a hypothesis: a lower closing low near 20,400 between Dec 9 and Dec 23, and if no close under 22,617.73 comes by Jan 29 the low was Nov 20. Reconciliation with MQ1 (0.72, locked, unchanged): the bounce cell written on Nov 21 (1 of 7) now applies, and my present figure for MQ1's event would be near 0.45; MQ1 is not revised.

**MQ6 and MQ7.** Inputs (M): October CPI plus 0.350 % on the month, core CPI plus 0.296 %, producer prices plus 0.348 % and plus 0.486 % without food, energy and trade; September PCE index 132.257 and core 130.902; October 2025 bases 127.790 and 127.165. A least-squares fit of monthly PCE on monthly CPI over 18 months of 2025-2026 (residual standard deviation 0.06 to 0.07 points) gives headline plus 0.32 % and core plus 0.29 %. Year on year: 3.82 % and 3.23 to 3.25 %. Chance of each rounded figure: headline 3.7 about 0.12, 3.8 about 0.5, 3.9 or more about 0.31; core 3.1 about 0.10, 3.2 about 0.45, 3.3 or more about 0.38. Estimates (S): 3.8 % and 3.2 %, each plus 0.3 % on the month. Not modelled: revisions to earlier months, which move the year-on-year figure; the components the statistics office takes from producer prices; a longer fitting sample, which two data services refused today.
