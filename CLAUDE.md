# Working rules for this repository

## Operating core

This block is the part that must survive. It reaches the conversation by three routes that do not
depend on any summary: this file is loaded from disk at every session start and after every
compaction; `.claude/hooks/lessons_guard.py prompt` re-sends the block with every user prompt; and
the same script sends it at session start. Change the rules here, between the markers, and nowhere else.

<!-- core:start -->
1. **Translate first.** Before any work, restate the request with every vague word replaced by a number, a named source or a pass/fail test, and show that translation at the top of the reply. The user's words are leads, not orders. Name a wrong premise before building on it.
2. **Verify, never assume.** A number goes in a reply only if it was opened from a source in this context. Tag it measured, derived, secondary or hypothesis. A recalled figure is a hypothesis. Work that was not done is reported as not done.
3. **A summary is a claim, not a record.** After a compaction or in a new session, state comes from files: `docs/adhoc_calls/LESSONS.md` (trust section first), the call document, `CALIBRATION.csv`, `git log`. "Read" means printed in full in this context; the lessons receipt is the proof.
4. **Lessons before any prediction.** The order of work below is enforced by the gate. Name the fitting evidence cell and its count before choosing a side. No side on a gut override; no weight on a recalled count.
5. **A locked call is not revised.** After the lock, record only. A first reaction (after-hours scan, open, pre-market) is never reported as a result.
6. **Every finding carries its disproof**, the assumptions it shares with its alternatives, and one alternative reading.
7. **Reply form.** Outcome first, then what is wrong or incomplete. Short sentences; numbers in tables. After a prediction, exactly three lines: TAKE, THESIS, KILL.
8. **Never:** print a secret value; use FMP, Finnhub or Alpha Vantage in ad hoc work; commit vendor-derived or Hyperliquid data; put a model name in a file or a commit; open a pull request unasked; force-push; run `freedom evaluate --final`; work around a hook.
9. **Commit form.** `git pull` before every write. Identity `Claude <noreply@anthropic.com>`. Trailers `Co-Authored-By: Claude <noreply@anthropic.com>` and the `Claude-Session` line.
10. **Write down what was learned** in `LESSONS.md` at every grade, as a rule with a count and a disproof.
<!-- core:end -->

## Ad hoc prediction calls (`docs/adhoc_calls/`)

Two harness hooks make the method a precondition of the output, not something to remember.

1. **The call gate** (`.claude/hooks/call_gate.py hook`, on every `git commit`). A new call
   document, or an existing one whose `CALL` lines changed, cannot be committed unless its
   `## Gate` section passes every binary check. One failure blocks the commit.
2. **The stop check** (`.claude/hooks/call_gate.py stop`, at the end of every turn). A reply
   that carries a `TAKE:` line is refused unless that exact line is in a committed call
   document. A prediction therefore reaches the user only after its document passed the gate.

### Order of work for every prediction

1. `git pull`; `python3 .claude/hooks/lessons_guard.py status`. If the receipt is not valid,
   print every part of `LESSONS.md` into the conversation, one part per command
   (`read --part N`), and read each one. Never redirect that output. `read --changes`
   refreshes a receipt when only the file changed.
2. `python3 .claude/hooks/call_gate.py prep TICKER CIK` (US filers). Open every filing it
   lists and label each raise, cut or neutral. Read the company's own news page as well.
3. Do the research in `SEARCH_CHECKLIST.md`, rows 1 to 12. Row 12 is the macro and AI-tape news sweep, run on the day of the lock; the gate refuses "none found" for it. "Not read" is not an accepted
   status for any row except social feeds (not reachable) and perps (not needed, with a reason).
4. `python3 .claude/hooks/call_gate.py template TICKER` prints the Gate block. Fill it from
   measured numbers. The gate recomputes the base label, the habit label, the coin, the
   scenario table and the chained figure, and compares them with what the document states.
5. Write the document, the `CALIBRATION.csv` row and the `STATEMENTS.md` entry;
   `python3 .claude/hooks/call_gate.py check <doc>` until every check passes; commit; push;
   only then give the prediction to the user.

### What the gate checks (39 binary checks)

Lessons receipt current and quoted. Release time sourced, lock before release and not ahead
of the clock. Grade window. Coin recomputed. Base label and habit label equal to what the
numbers give. Regime classified. Scenario weights sum to 1, each with a written gate, the
middle scenario at most 0.20 unless two of the last four prints were held tables, table
arithmetic exact. Chained figure exact, the up-line switch at the counted rate or with a
written reason. Side consistent with the probability and with the `CALL` line. Outside the
AI tape a DOWN call names the line that will be cut and its source. Inside the tape a
residual call against a named basket. A call against the fitting cell needs an override
cell with n of at least 3. All twelve checklist rows done with evidence, the macro and AI sweep dated the day of the lock; every filing since
the last results 8-K labelled. KILL line with a number, tied to the scenario that defeats
the call. TAKE at most 140 characters, THESIS at most three sentences, Statement under 60
words, no hashtags or emojis. No model names. Ledger rows written.

Do not work around a block. Do the work the failed check names.

**Post-release class.** When the user asks for a call after a release, the Gate block carries `"call_class": "post_release"` and a `reference` object (price, time, source). The gate then requires the release to be before the lock and the reference price to be stamped within 20 minutes of it. These calls are tallied apart from pre-release calls. Approved by the user on 2026-11-12.

Grading follows `GRADING_PROTOCOL.md` and is not gated.
