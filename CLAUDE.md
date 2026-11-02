# Working notes for Claude in this repository

## The record that must survive

- `docs/adhoc_calls/LESSONS.md` is the accumulated lessons from every graded call. Read it before
  every call and update it at every grading. Companions in the same folder: `CALIBRATION.csv`
  (one row per call) and `METHOD_N3.md`.
- `main` and the working branch `claude/system-instructions-documentation-1xq2zj` must both carry
  it. A session that starts from a branch that does not exist fails with "branch or commit not
  found" (this stopped the long-running session on 2026-11-01, when `main` did not exist).
- Push after every grading commit. If a push fails, follow the protocol below before doing anything
  else, so the work never lives only in one container.

## Offline backup protocol (used during the 2026-09-28 to ~10-22 GitHub suspension)

While pushes return 403:

- Keep committing locally as usual (author `Claude <noreply@anthropic.com>`, the usual trailers).
- After every local commit run `scripts/offline_backup.sh` and send the resulting bundle to
  the user with the file tool, so the work survives the container. The bundle is a complete
  copy of the repository; `git clone <bundle> Freedom` or `git fetch <bundle>` restores it.
- At every scheduled wake, attempt `scripts/catch_up_push.sh`; a 403 means still suspended.
- When the user says **"commit since GitHub block"**, run `scripts/catch_up_push.sh`
  (with the latest bundle as the argument if this checkout no longer holds the local-only
  commits) and report which commits went up. Never force-push and never rewrite history.

## Known gaps in the record

- Git history starts at the 2026-10-22 re-import; earlier history is only in the offline bundles.
- `docs/adhoc_calls/NKE_2026-10-01.md` (the Nike call document) was not re-imported and is lost
  from the repository; its resolution is in `NKE_resolution_pending.md`.
- `CCL_2026-09-29.md` and `MU_2026-09-30.md` were restored on 2026-11-02 from the 2026-09-30
  backup; the MU copy predates its grading (the grade is in LESSONS.md).
