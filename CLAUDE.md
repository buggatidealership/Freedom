# Working rules for this repository

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

Grading follows `GRADING_PROTOCOL.md` and is not gated.
