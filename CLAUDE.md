# Working rules for this repository

## Ad hoc prediction calls (`docs/adhoc_calls/`)

A harness hook enforces one rule: **no new call document can be committed unless
`docs/adhoc_calls/LESSONS.md` was read in full in the current conversation.**

Before deciding any side:

1. `git pull`, then run `python3 .claude/hooks/lessons_guard.py status`.
2. If the receipt is not valid, print every part into the conversation, one part per
   command, and read each one: `python3 .claude/hooks/lessons_guard.py read --part 1`
   up to the last part the status command reports. Do not redirect the output.
3. If only the file changed since the receipt, run
   `python3 .claude/hooks/lessons_guard.py read --changes`.
4. Put the receipt line the script prints (`Lessons receipt: <hash> read <time>`) in the
   lock line of the call document. It states exactly what was read.

The receipt is dropped when the conversation is compacted, cleared or started fresh,
because the text is then no longer in context. It also expires after 24 hours and
whenever `LESSONS.md` changes. The hook blocks `git commit` while a new call document
lacks a valid receipt line. Do not work around it; read the file.

Then follow `SEARCH_CHECKLIST.md` and the "Trust calibration" section of `LESSONS.md`.
Grading follows `GRADING_PROTOCOL.md`.
