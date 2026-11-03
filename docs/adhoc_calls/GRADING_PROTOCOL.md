# Grading protocol for ad hoc earnings calls

Written 2026-11-03 at the user's request ("follow a process that aims you at getting better at predicting directions and magnitude over time; take my prompt and write a better version of it"). This file is the process; the prompt at the bottom is what to paste. Both are hypotheses with a record, like everything else in `LESSONS.md`.

## What "getting better" means here, in numbers

A grade is only useful if it changes the next call. Four series are tracked, each against a benchmark, and each lesson is written so that one of them can move:

| Series | Statistic | Benchmark | Where it lives |
|---|---|---|---|
| Direction | hit rate, and Brier score of P(up) | SHA-256 coin (hit rate 0.50, Brier 0.25) | `CALIBRATION.csv` (side, p_up, direction), tallies in `LESSONS.md` |
| Magnitude | size error = called size − \|r\|, signed and absolute | the options event-only move; the no-change line (0); the own-median line | `CALIBRATION.csv` (size_pct, realized_pct) |
| Mechanism | share of hits that are "right for the right reason" vs "luck"; share of misses that are "reaction" vs "fundamentals" | none; the share itself is the asset | the rubric block of each call doc, marked at grading |
| Calibration | hit rate by P(up) band (≤ 0.42, 0.43-0.47, 0.48-0.52, ≥ 0.53) and by call type (regime cell) | each band should hit at the rate it claims | "Trust calibration" section of `LESSONS.md` |

A call that hits on direction but misses on mechanism is scored as a hit on the sheet and a miss in the lessons. A call that misses on direction with the mechanism intact (the named line fired, the crowd ignored it) is a reaction miss and teaches a tape rule, not a reading rule. The two kinds of miss get different lessons and must not be pooled.

## Pre-registration (at the lock, or the grade is not clean)

Every call doc carries, before the release: the grading window (which close vs which close, on the settled row); the scenario table with P(s), P(up|s) and sized legs; the kill switches with thresholds and P(fire); the sizing lines (own median, conditional, options, C leg, shrink); the governing sentence (which rule decides the side); the independence statement if another call on the same print exists in the same session; the rubric (right reason / luck / reaction miss / fundamentals miss, with the switch that decides each); the lessons-read stamp (UTC time, LESSONS.md commit hash). A switch threshold must be read off the company's last exhibit or a sourced consensus, with the source named; a bar written from memory is a premise error waiting to be found (AMD v1/v2, Nov 3: wrong drawdown, wrong Q4 bar).

## The grading run (one pass, no agents, under 30 minutes of tool time)

1. **Settled closes first.** Reference close and graded close from a settled daily row (stockanalysis, CBOE 15:59:59, the exchange), two sources agreeing where possible. A 4 pm quote-endpoint print or a press "closed down X %" is provisional; mark it so and re-stamp when the row posts. Never grade on a pre-market or after-hours price. If the sites serve stale pages (this container, Nov 3), take the press close as provisional and schedule the re-stamp.
2. **Release facts from the primary source.** The 8-K exhibit or the company release for every switch line; a transcript or press figure is S and is labeled. Mark each kill switch fired / not fired / unscorable against its stated P(fire). Name the scenario the print landed in by the table's own letter.
3. **Classify every call with its own rubric**, written before the print. Do not re-interpret the rubric after the fact; if the rubric was ambiguous, say so and score the ambiguity against the writer. Five outcomes: right reason; luck; **table right, mechanism wrong** (added Nov 3 after SHOP: the scenario letter was right, the named sign-setter did not fire; a hit on the sheet, a half-miss on the mechanism series); reaction miss; fundamentals miss.
4. **Compute**: r (D), hit/miss, Brier, coin side and hit, size error against the call and against each sizing line, the relative line where a basket was pre-registered, gap and gap-retention (open from the settled row). Write the arithmetic.
5. **Mark every standing lesson** that applied: helped / hurt / neutral, one line each, with the number that says so. A lesson that was not applied when its gate was met is marked "not followed" and counted as hurt.
6. **Write the new lessons**, each as: the rule, the configuration it applies to, the number to compute next time, the current record (n), and what disproves it. A lesson without a disproof is an anecdote.
7. **Alternative interpretation**, written as a skeptic: what else explains the move (tape, sector, election, the first scan vs the session), and what observation would separate the two readings.
8. **Non-independent pairs** (two models or two versions on one print from one session): grade each on the sheet, labeled, but count the pair as one entry in the tallies, using the mean P(up) for the side and the mean size, the way the Oct 13 bank cluster was counted. Record the within-pair comparison (which size, which P(up) was nearer) as a series of its own; it says something only after ten pairs.
9. **Update the files**: one `CALIBRATION.csv` row per call (format `call,date,side,size_pct,p_up,realized_pct,direction,note`, the note carrying the closes, the scenario letter, the switches that fired and the sizing line that won); the dated `LESSONS.md` entry (lessons first, marks, tallies, pending); the "Trust calibration" section (counts by type); the call doc's rubric block (outcome marked); a grade file for the day when several calls grade together.
10. **Commit, push to both branches, copy to the project-files folder.** Lock the pending items (AMD on the Nov 4 close) with a scheduled reminder.
11. **Reply with the scorecard**: one line per call (side, size, r, hit/miss, rubric outcome), the lesson of the day in one sentence, what is provisional, and what the next call will do differently.

## What the grading must not do

Grade on a pre-market or after-hours price; write "re-read" or "settled" when it is not; move a threshold after the print to rescue a rubric; pool the two miss types; write the lesson for a call another session made; spend an agent fan-out on what two data pulls answer; let a hit at P(up) 0.52 count as evidence of anything.

## Monthly review (first grading day of each month)

Recompute the four series on the full `CALIBRATION.csv`; list the lessons whose disproof condition has been met and retire them in the trust section; list the sizing rule with the lowest mean absolute error over the last 20 calls and make it the default; list the call types under 60 % with n ≥ 5 and stop making them until a rule change is pre-registered.

## The prompt (paste this at the close)

> Grade the calls on <TICKERS> against the <DATE> close. Settled closes from two sources, provisional if the rows are not posted. Release facts from the exhibit. Mark each kill switch against its stated probability, name the scenario letter the print landed in, and classify each call with its own pre-registered rubric: right reason, luck, reaction miss, or fundamentals miss. Compute r, hit, Brier, coin, and the size error against every sizing line. Mark each standing lesson helped / hurt / neutral / not followed. Write new lessons only with a configuration, a number to compute, and a disproof. Add an alternative interpretation. Count non-independent pairs as one entry in the tallies. Update CALIBRATION.csv, LESSONS.md, the trust section and the call docs; commit, push both branches, back up. Reply with a one-line-per-call scorecard, the day's lesson, what is provisional, and what the next lock will do differently.

The user's original wording was "check the market close and grade your predictions against the outcome; follow a process that aims you at getting better at predicting directions and magnitude over time." What the better version adds: the settled-row rule, the pre-registered rubric as the only classifier, the two kinds of miss kept apart, the disproof requirement on lessons, the pair-as-one tally rule, and the four tracked series with their benchmarks, so that "getting better" is a number that can go the wrong way.
