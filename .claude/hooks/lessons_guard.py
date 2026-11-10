#!/usr/bin/env python3
"""Lessons guard for ad hoc prediction calls.

A new call document under docs/adhoc_calls cannot be committed unless
LESSONS.md was printed in full to the session (every part), the file has not
changed since, the conversation has not been compacted or restarted since, and
the document carries the receipt line this script prints.

Sub-commands
  read --part N     print part N of LESSONS.md; after the last part a receipt is written
  read --changes    print what changed in LESSONS.md since the receipt and refresh it
  status            show the receipt state and the line a call document must carry
  hook              PreToolUse(Bash) hook: block `git commit` when a new call document lacks a valid receipt
  prompt            UserPromptSubmit hook: remind when a prompt asks for a prediction
  invalidate        SessionStart / PostCompact hook: drop the receipt (the text left the context)

State lives in .git/lessons_guard/ so it is per clone and never committed.
"""
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALLS = os.path.join(ROOT, "docs", "adhoc_calls")
LESSONS = os.path.join(CALLS, "LESSONS.md")
STATE_DIR = os.path.join(ROOT, ".git", "lessons_guard")
STATE = os.path.join(STATE_DIR, "state.json")
SNAPSHOT = os.path.join(STATE_DIR, "snapshot.md")
PART_CHARS = 22000          # stays under the 30,000-character inline output limit
MAX_AGE_HOURS = 24          # backstop; compaction and session start invalidate sooner
CALL_DOC = re.compile(r"^(?!GRADE_)[A-Z0-9_]+_\d{4}-\d{2}-\d{2}\.md$")
PROMPT_RE = re.compile(r"\b(predict\w*|next call|earnings call|reports? (today|tonight|tomorrow)|lock (the|a) call)\b", re.I)


def text():
    with open(LESSONS, encoding="utf-8") as f:
        return f.read()


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parts(s):
    """Split on line boundaries into chunks of about PART_CHARS characters."""
    out, cur, n = [], [], 0
    for line in s.splitlines(keepends=True):
        while len(line) > PART_CHARS:          # a single very long line
            if cur:
                out.append("".join(cur)); cur, n = [], 0
            out.append(line[:PART_CHARS]); line = line[PART_CHARS:]
        if n + len(line) > PART_CHARS and cur:
            out.append("".join(cur)); cur, n = [], 0
        cur.append(line); n += len(line)
    if cur:
        out.append("".join(cur))
    return out


def load():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)


def receipt_line(st):
    return "Lessons receipt: %s read %s" % (st["sha"][:12], st["completed_utc"])


def receipt_problem(st, cur_sha):
    if not st.get("completed_utc"):
        return "LESSONS.md has not been read in full in this session"
    if st.get("sha") != cur_sha:
        return "LESSONS.md changed since it was read (run: read --changes)"
    if time.time() - st.get("completed_ts", 0) > MAX_AGE_HOURS * 3600:
        return "the full read is older than %d hours" % MAX_AGE_HOURS
    return None


def cmd_read(argv):
    s = text(); cur = sha(s); st = load()
    if "--changes" in argv:
        if not st.get("completed_utc") or not os.path.exists(SNAPSHOT):
            print("No receipt to refresh. Read every part first: read --part 1"); return 1
        if time.time() - st.get("completed_ts", 0) > MAX_AGE_HOURS * 3600:
            print("The full read is older than %d hours. Read every part again." % MAX_AGE_HOURS); return 1
        with open(SNAPSHOT, encoding="utf-8") as f:
            old = f.read()
        if sha(old) == cur:
            print("No change since the receipt."); print(receipt_line(st)); return 0
        diff = list(difflib.unified_diff(old.splitlines(), s.splitlines(), "LESSONS.md at receipt", "LESSONS.md now", lineterm="", n=0))
        body = "\n".join(diff)
        if len(body) > 28000:
            print("The change is too large to show as a diff (%d characters). Read every part again." % len(body))
            save({}); return 1
        print(body)
        st["sha"] = cur
        st["refreshed_utc"] = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
        save(st)
        with open(SNAPSHOT, "w", encoding="utf-8") as f:
            f.write(s)
        print("\n--- changes shown; receipt refreshed ---"); print(receipt_line(st)); return 0
    try:
        n = int(argv[argv.index("--part") + 1])
    except Exception:
        print("usage: lessons_guard.py read --part N | read --changes"); return 1
    ps = parts(s)
    if not 1 <= n <= len(ps):
        print("part must be 1..%d" % len(ps)); return 1
    if st.get("reading_sha") != cur:           # file changed mid-read, or a fresh read
        st = {"reading_sha": cur, "seen": []}
    seen = set(st.get("seen", [])); seen.add(n); st["seen"] = sorted(seen)
    print("=== LESSONS.md part %d of %d (sha %s) ===" % (n, len(ps), cur[:12]))
    sys.stdout.write(ps[n - 1])
    if not ps[n - 1].endswith("\n"):
        print()
    missing = [i for i in range(1, len(ps) + 1) if i not in seen]
    if missing:
        save(st)
        print("=== end of part %d; still to read: %s ===" % (n, ", ".join(map(str, missing))))
    else:
        st.update({"sha": cur, "completed_ts": time.time(),
                   "completed_utc": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "parts": len(ps)})
        save(st)
        with open(SNAPSHOT, "w", encoding="utf-8") as f:
            f.write(s)
        print("=== end of part %d; every part read. Put this line in the call document: ===" % n)
        print(receipt_line(st))
    return 0


def cmd_status(_argv):
    s = text(); cur = sha(s); st = load(); n = len(parts(s))
    prob = receipt_problem(st, cur)
    print("LESSONS.md: %d characters, %d parts, sha %s" % (len(s), n, cur[:12]))
    if prob:
        print("Receipt: NOT VALID - " + prob)
        print("Parts read so far: %s" % (st.get("seen") if st.get("reading_sha") == cur else []))
    else:
        print("Receipt: valid"); print(receipt_line(st))
    return 0


def new_call_docs():
    try:
        out = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=all", "--", "docs/adhoc_calls"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    docs = []
    for line in out.splitlines():
        code, path = line[:2], line[3:].strip().strip('"')
        if code == "??" or code[0] == "A":
            if CALL_DOC.match(os.path.basename(path)):
                docs.append(path)
    return docs


def cmd_hook(_argv):
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    command = (data.get("tool_input") or {}).get("command", "") or ""
    if not re.search(r"\bgit\b[^\n;&|]*\bcommit\b", command):
        return 0
    docs = new_call_docs()
    if not docs:
        return 0
    cur = sha(text()); st = load(); prob = receipt_problem(st, cur)
    here = "python3 .claude/hooks/lessons_guard.py"
    if prob:
        n = len(parts(text()))
        sys.stderr.write(
            "BLOCKED by the lessons guard: new call document(s) %s, but %s.\n"
            "Read the file into this conversation first, one part per command: %s read --part 1 ... --part %d "
            "(or `%s read --changes` if only the file changed), then put the receipt line it prints into each "
            "call document and commit again. Do not skip parts and do not bypass this check.\n"
            % (", ".join(docs), prob, here, n, here))
        return 2
    want = "Lessons receipt: %s" % st["sha"][:12]
    missing = []
    for d in docs:
        try:
            with open(os.path.join(ROOT, d), encoding="utf-8") as f:
                if want not in f.read():
                    missing.append(d)
        except Exception:
            missing.append(d)
    if missing:
        sys.stderr.write(
            "BLOCKED by the lessons guard: %s must carry the line `%s`. "
            "Add it to the lock line of the document (it states exactly what was read) and commit again.\n"
            % (", ".join(missing), receipt_line(st)))
        return 2
    return 0


def cmd_prompt(_argv):
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    if not PROMPT_RE.search(data.get("prompt", "") or ""):
        return 0
    try:
        s = text(); st = load(); prob = receipt_problem(st, sha(s)); n = len(parts(s))
    except Exception:
        return 0
    if prob:
        msg = ("Lessons guard: this prompt looks like a prediction request and %s. Before deciding any side, run "
               "`python3 .claude/hooks/lessons_guard.py read --part 1` through `--part %d`, one part per command, and read "
               "each part. The commit of a new call document is blocked until every part has been printed and the "
               "receipt line is in the document." % (prob, n))
    else:
        msg = ("Lessons guard: receipt valid (%s). If LESSONS.md changes before the lock, run "
               "`python3 .claude/hooks/lessons_guard.py read --changes`." % receipt_line(st))
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": msg}}))
    return 0


def cmd_invalidate(_argv):
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}
    # A resumed session still has the text in context; anything else does not.
    if data.get("hook_event_name") == "SessionStart" and data.get("source") == "resume":
        return 0
    try:
        if os.path.exists(STATE):
            os.remove(STATE)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    cmds = {"read": cmd_read, "status": cmd_status, "hook": cmd_hook, "prompt": cmd_prompt, "invalidate": cmd_invalidate}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__); sys.exit(1)
    sys.exit(cmds[sys.argv[1]](sys.argv[2:]))
