#!/usr/bin/env python3
"""Binary gate for ad hoc prediction calls.

Every call document must carry a `## Gate` section holding one fenced JSON block.
Each check below returns pass or fail; a single failure blocks the commit.

  prep TICKER CIK     fetch the EDGAR filing list since the last results 8-K (row 11 source of truth)
  template TICKER     print an empty Gate block
  check PATH          run every check on one document and print the result
  hook                PreToolUse(Bash): on `git commit`, check every new call document and every
                      existing one whose CALL lines changed
  stop                Stop: refuse to end a turn whose last message carries TAKE lines that are not
                      in a committed call document
"""
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lessons_guard as LG  # noqa: E402

ROOT = LG.ROOT
CALLS = LG.CALLS
GATES = os.path.join(LG.STATE_DIR, "gates")
STOP_STATE = os.path.join(LG.STATE_DIR, "stop.json")
ROWS = ["1_bar", "2_documents", "3_unnumbered_language", "4_sell_side", "5_people_regulators",
        "6_index_events", "7_sibling", "8_social", "9_perps", "10_counterparties", "11_inter_print"]
MODEL_WORDS = re.compile(r"\b(fable|opus|sonnet|haiku|claude|gemini|grok|gpt-?\d)\b", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
TOL = 0.006


def parse_utc(s):
    return dt.datetime.strptime(s.replace("Z", "+0000"), "%Y-%m-%dT%H:%M%z")


def gate_block(text):
    m = re.search(r"^## Gate\b.*?```json\s*\n(.*?)\n```", text, re.S | re.M)
    if not m:
        raise ValueError("no `## Gate` section with a fenced json block")
    return json.loads(m.group(1))


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_doc(path):
    """Return a list of (id, ok, message)."""
    out = []

    def add(i, ok, msg):
        out.append((i, bool(ok), msg))

    text = open(path, encoding="utf-8").read()
    name = os.path.basename(path)
    try:
        g = gate_block(text)
    except Exception as e:
        return [("G00 gate block parses", False, str(e))]
    add("G00 gate block parses", True, "")

    # G01 lessons receipt
    st = LG.load(); cur = LG.sha(LG.text()); prob = LG.receipt_problem(st, cur)
    ok = (not prob) and g.get("lessons_receipt") == st.get("sha", "")[:12] and ("Lessons receipt: " + st.get("sha", "x")[:12]) in text
    add("G01 lessons file read in full, receipt current and quoted", ok, prob or "gate.lessons_receipt and the receipt line must equal the current receipt")

    # G02 release clock, lock before release, lock not ahead of the clock
    try:
        rel = parse_utc(g["release"]["utc"]); lock = parse_utc(g["doc_lock_utc"]); now = dt.datetime.now(dt.timezone.utc)
        src = str(g["release"].get("source_url", ""))
        ok = src.startswith("http") and lock < rel and lock <= now + dt.timedelta(minutes=5) and lock >= now - dt.timedelta(hours=6)
        add("G02 release time sourced; lock before release; lock within 6 h of the clock and not ahead of it", ok,
            "release.source_url must be a URL; need doc_lock_utc < release.utc and now-6h <= doc_lock_utc <= now+5m")
    except Exception as e:
        add("G02 release time sourced; lock before release", False, "release.utc / doc_lock_utc as YYYY-MM-DDTHH:MMZ (%s)" % e)

    # G03 grade window
    try:
        a = dt.date.fromisoformat(g["grade"]["reference_close_date"]); b = dt.date.fromisoformat(g["grade"]["graded_close_date"])
        add("G03 grade window: graded close after reference close", b > a, "graded_close_date must be later than reference_close_date")
    except Exception as e:
        add("G03 grade window", False, str(e))

    # G04 coin recomputed
    try:
        c = g["coin"]; h = hashlib.sha256(c["string"].encode()).hexdigest()[0]
        side = "UP" if h in "01234567" else "DOWN"
        add("G04 coin recomputed from its string", c["hex"] == h and c["side"] == side, "coin should be hex %s side %s" % (h, side))
    except Exception as e:
        add("G04 coin recomputed", False, str(e))

    # G05 base label derived from numbers
    try:
        s = g["setup"]; pb = s["pct_below_252d_high"]; d21 = s["d21_pct"]
        assert num(pb) and num(d21) and pb <= 0 and isinstance(s["up_of_last8"], int) and 0 <= s["up_of_last8"] <= 8 and isinstance(s["n_prints"], int)
        base = "derated" if pb <= -25 else ("chased" if (pb >= -10 or d21 >= 15) else "neutral")
        add("G05 base label equals the label the numbers give", s["base"] == base, "numbers give base = %s (derated <= -25%% below high; chased within 10%% of high or d21 >= +15%%)" % base)
        habit = "down" if s["up_of_last8"] <= 1 and s["n_prints"] >= 8 else ("up" if s["up_of_last8"] >= 7 and s["n_prints"] >= 8 else "none")
        add("G05b own habit label equals the count (>= 7 of 8 one way)", s.get("habit") == habit, "counts give habit = %s" % habit)
    except Exception as e:
        add("G05 base and habit from numbers", False, "setup needs pct_below_252d_high, d21_pct, up_of_last8, n_prints, base, habit (%s)" % e)

    # G06 regime
    regime = g.get("regime")
    add("G06 regime classified", regime in ("inside_ai_tape", "outside_ai_tape"), "regime must be inside_ai_tape or outside_ai_tape")

    # G07 scenarios
    try:
        sc = g["scenarios"]; ps = [sc[k]["p"] for k in "ABC"]; pu = [sc[k]["p_up"] for k in "ABC"]
        assert all(num(x) and 0 <= x <= 1 for x in ps + pu) and all(len(str(sc[k].get("gate", ""))) >= 10 for k in "ABC")
        add("G07 scenario weights sum to 1 and every scenario has a written gate", abs(sum(ps) - 1) <= 0.005, "sum is %.3f" % sum(ps))
        add("G07b middle scenario capped at 0.20 unless two of the last four prints were held tables", sc["B"]["p"] <= 0.20 + 1e-9 or int(g.get("held_tables_last4", 0)) >= 2, "P(B) = %.2f" % sc["B"]["p"])
        table = sum(p * u for p, u in zip(ps, pu))
        add("G07c table P(up) equals its arithmetic", num(g.get("p_up_table")) and abs(g["p_up_table"] - table) <= TOL, "arithmetic gives %.3f" % table)
    except Exception as e:
        add("G07 scenarios", False, "scenarios.A/B/C need gate, p, p_up (%s)" % e)

    # G08 counted switches and chained figure
    try:
        sw = g["switches"]; k1 = sw["ks1"]; k2 = sw["ks2"]
        chained = k1["p"] * k1["p_up"] + k2["p"] * k2["p_up"] + max(0.0, 1 - k1["p"] - k2["p"]) * sw["rest_p_up"]
        add("G08 chained P(up) equals its arithmetic", num(g.get("p_up_chained")) and abs(g["p_up_chained"] - chained) <= TOL, "arithmetic gives %.3f" % chained)
        add("G08b up-line switch at the counted rate (0.60-0.95) or a written reason of 20+ characters", 0.60 <= k1["p"] <= 0.95 or len(str(k1.get("reason", ""))) >= 20, "ks1.p = %s" % k1["p"])
    except Exception as e:
        add("G08 switches", False, "switches.ks1{p,p_up,reason}, ks2{p,p_up}, rest_p_up (%s)" % e)

    # G09 call consistent with itself and with the CALL line
    side = None
    try:
        c = g["call"]; side = c["side"]; size = float(c["size_pct"]); pg = float(c["p_up_gut"])
        assert side in ("UP", "DOWN") and size > 0 and 0 < pg < 1
        add("G09 side agrees with the stated probability", (side == "UP" and pg > 0.5) or (side == "DOWN" and pg < 0.5), "side %s with P(up) %.2f" % (side, pg))
        calls = re.findall(r"^CALL[^\n:]*: (UP|DOWN) (\d+(?:\.\d+)?) % \(P\(up\) (0\.\d+)", text, re.M)
        last = calls[-1] if calls else None
        add("G09b the last CALL line states the same side, size and probability", bool(last) and last[0] == side and abs(float(last[1]) - size) < 1e-9 and abs(float(last[2]) - pg) < 1e-9, "last CALL line: %s" % (last,))
    except Exception as e:
        add("G09 call", False, "call.side, size_pct, p_up_gut (%s)" % e)

    # G10 outside the tape a DOWN call names the line that will be cut
    if regime == "outside_ai_tape" and side == "DOWN":
        add("G10 outside the AI tape a DOWN call names a cut line and its source", len(str(g.get("cut_line") or "")) >= 15 and len(str(g.get("cut_line_source") or "")) >= 8, "cut_line and cut_line_source are required")
    else:
        add("G10 cut line (not required for this regime and side)", True, "")

    # G11 inside the tape a residual line against a named basket
    if regime == "inside_ai_tape":
        r = g.get("residual") or {}
        add("G11 inside the AI tape a residual call against a named basket", isinstance(r.get("basket"), list) and len(r["basket"]) >= 2 and num(r.get("call_pct")), "residual.basket (2+ tickers) and residual.call_pct are required")
    else:
        add("G11 residual line (not required outside the tape)", True, "")

    # G12 fitting cell and override rule
    try:
        fc = g["fitting_cell"]
        assert len(str(fc["name"])) >= 10 and isinstance(fc["n"], int) and isinstance(fc["hits"], int) and fc["says"] in ("UP", "DOWN", "NONE")
        if side and fc["says"] not in ("NONE", side):
            oc = g.get("override_cell") or {}
            add("G12 a call against the fitting cell needs an override cell with n >= 3", g.get("overrides_fitting_cell") is True and isinstance(oc.get("n"), int) and oc["n"] >= 3 and len(str(oc.get("name", ""))) >= 10,
                "fitting cell says %s; an override needs override_cell{name,n>=3,hits}" % fc["says"])
        else:
            add("G12 call agrees with the fitting cell", True, "")
    except Exception as e:
        add("G12 fitting cell", False, "fitting_cell{name,n,hits,says} (%s)" % e)

    # G13 checklist rows
    cl = g.get("checklist") or {}
    for row in ROWS:
        r = cl.get(row) or {}
        stt = r.get("status"); ev = r.get("evidence") or []
        if row == "8_social":
            ok = stt in ("read", "none_found", "not_reachable")
        elif row == "9_perps":
            ok = stt in ("read", "none_found") and len(ev) >= 1 or (stt == "not_needed" and len(str(r.get("reason", ""))) >= 10)
        else:
            ok = stt in ("read", "none_found") and len(ev) >= 1 and all(len(str(x)) >= 8 for x in ev)
        add("G13 checklist row %s done with evidence" % row, ok, "status read or none_found with at least one source in evidence")
    # row 11: every filing since the last results 8-K is labelled
    try:
        r11 = cl.get("11_inter_print") or {}
        items = {str(i["id"]): i.get("tail") for i in (r11.get("items") or [])}
        tick = str(g.get("ticker", ""))
        if g.get("cik"):
            pf = os.path.join(GATES, tick + ".json")
            assert os.path.exists(pf), "run: call_gate.py prep %s %s" % (tick, g["cik"])
            prep = json.load(open(pf))
            assert time.time() - prep["fetched_ts"] <= 24 * 3600, "the prepared filing list is older than 24 h; run prep again"
            missing = [f["accession"] for f in prep["filings"] if items.get(f["accession"]) not in ("raise", "cut", "neutral")]
            add("G13b row 11: all %d filings since the last results 8-K are labelled raise, cut or neutral" % len(prep["filings"]), not missing, "unlabelled: %s" % ", ".join(missing[:6]))
        else:
            alt = r11.get("press_sources") or []
            add("G13b row 11 (no EDGAR filer): at least two company sources listed", len(alt) >= 2, "press_sources needs 2+ URLs")
        add("G13c row 11: the company news page is listed", len(r11.get("press_sources") or []) >= 1, "press_sources needs the company's news page URL")
    except Exception as e:
        add("G13b row 11 inter-print statements", False, str(e))

    # G14-15 the public lines
    def line(tag):
        m = re.findall(r"^%s: (.+)$" % tag, text, re.M)
        return m[-1].strip() if m else None
    take, thesis, kill = line("TAKE"), line("THESIS"), line("KILL")
    add("G14 KILL line present, carries a number, and names the scenario that defeats the call", bool(kill) and bool(re.search(r"\d", kill or "")) and (g.get("kill") or {}).get("scenario") == ("A" if side == "DOWN" else "C") and (g.get("kill") or {}).get("text") == kill,
        "kill.scenario must be %s and kill.text must equal the KILL line" % ("A" if side == "DOWN" else "C"))
    add("G15 TAKE at most 140 characters", bool(take) and len(take) <= 140, "TAKE is %s characters" % (len(take) if take else "missing"))
    add("G15b THESIS at most three sentences", bool(thesis) and len(re.findall(r"[.!?](?:\s|$)", thesis)) <= 3, "THESIS missing or longer than three sentences")
    add("G15c no hashtags or emojis in the three lines", all(x and "#" not in x and not EMOJI.search(x) for x in (take, thesis, kill)), "remove hashtags or emojis")
    m = re.search(r"^## Statement\s*\n(.+?)\n\s*\n", text, re.S | re.M)
    add("G15d Statement under 60 words", bool(m) and len(m.group(1).split()) < 60, "Statement missing or 60+ words")

    # G16 no model names
    hit = MODEL_WORDS.search(text)
    add("G16 no model names in the document", not hit, "found `%s`" % (hit.group(0) if hit else ""))

    # G17 ledger rows written
    try:
        cid = name[:-3].rsplit("_", 1)[0]
        cal = open(os.path.join(CALLS, "CALIBRATION.csv"), encoding="utf-8").read()
        stm = open(os.path.join(CALLS, "STATEMENTS.md"), encoding="utf-8").read()
        add("G17 pending row in CALIBRATION.csv and the TAKE line in STATEMENTS.md", re.search(r"^%s(_V\d+)?,%s," % (re.escape(cid), re.escape(g["release"]["utc"][:10])), cal, re.M) is not None and bool(take) and ("TAKE: " + take) in stm,
            "CALIBRATION row must start `%s,%s,` and STATEMENTS.md must contain the TAKE line" % (cid, g.get("release", {}).get("utc", "")[:10]))
    except Exception as e:
        add("G17 ledger rows", False, str(e))
    return out


def report(path, res):
    bad = [r for r in res if not r[1]]
    lines = ["%s: %d of %d checks passed" % (os.path.relpath(path, ROOT), len(res) - len(bad), len(res))]
    lines += ["  FAIL %s -> %s" % (i, m) for i, _, m in bad]
    return "\n".join(lines), not bad


def cmd_prep(argv):
    tick, cik = argv[0].upper(), int(argv[1])
    req = urllib.request.Request("https://data.sec.gov/submissions/CIK%010d.json" % cik, headers={"User-Agent": "Freedom research noreply@anthropic.com"})
    d = json.loads(urllib.request.urlopen(req, timeout=40).read())["filings"]["recent"]
    filings = []; last = "none found"
    for i in range(len(d["form"])):
        if d["form"][i] == "8-K" and "2.02" in d["items"][i]:
            last = d["filingDate"][i]; break
        if d["form"][i] in ("8-K", "8-K/A", "10-Q", "10-K", "424B5", "424B3", "424B2", "S-3", "S-3ASR", "S-1", "SC 13D", "SCHEDULE 13D", "425", "6-K"):
            filings.append({"accession": d["accessionNumber"][i], "form": d["form"][i], "date": d["filingDate"][i], "items": d["items"][i], "doc": d["primaryDocument"][i]})
    os.makedirs(GATES, exist_ok=True)
    json.dump({"ticker": tick, "cik": cik, "last_results_8k": last, "fetched_ts": time.time(), "filings": filings}, open(os.path.join(GATES, tick + ".json"), "w"), indent=1)
    print("%s: %d filings since the results 8-K of %s. Each must be opened and labelled raise, cut or neutral in checklist.11_inter_print.items:" % (tick, len(filings), last))
    for f in filings:
        print("  %s  %s  %-8s items %s  https://www.sec.gov/Archives/edgar/data/%d/%s/%s" % (f["accession"], f["date"], f["form"], f["items"] or "-", cik, f["accession"].replace("-", ""), f["doc"]))
    return 0


def cmd_template(argv):
    t = argv[0].upper() if argv else "TICKER"
    tpl = {"ticker": t, "cik": None, "doc_lock_utc": "YYYY-MM-DDTHH:MMZ", "lessons_receipt": "", "release": {"utc": "YYYY-MM-DDTHH:MMZ", "source_url": ""},
           "grade": {"reference_close_date": "", "graded_close_date": ""}, "coin": {"string": "", "hex": "", "side": ""}, "regime": "",
           "setup": {"pct_below_252d_high": 0, "d21_pct": 0, "base": "", "up_of_last8": 0, "n_prints": 0, "habit": ""},
           "fitting_cell": {"name": "", "hits": 0, "n": 0, "says": ""}, "overrides_fitting_cell": False, "override_cell": None,
           "scenarios": {k: {"gate": "", "p": 0, "p_up": 0} for k in "ABC"}, "held_tables_last4": 0, "p_up_table": 0,
           "switches": {"ks1": {"p": 0.8, "p_up": 0, "reason": ""}, "ks2": {"p": 0.45, "p_up": 0}, "rest_p_up": 0}, "p_up_chained": 0,
           "call": {"side": "", "size_pct": 0, "p_up_gut": 0}, "cut_line": None, "cut_line_source": None, "residual": None,
           "kill": {"scenario": "", "text": ""},
           "checklist": {r: ({"status": "", "items": [], "press_sources": []} if r == "11_inter_print" else {"status": "", "evidence": []}) for r in ROWS}}
    tpl["checklist"]["11_inter_print"]["evidence"] = []
    print("## Gate\n\n```json\n" + json.dumps(tpl, indent=1) + "\n```")
    return 0


def cmd_check(argv):
    msg, ok = report(argv[0], check_doc(argv[0]))
    print(msg)
    return 0 if ok else 1


def docs_to_gate():
    out = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=all", "--", "docs/adhoc_calls"], capture_output=True, text=True, timeout=20).stdout
    docs = []
    for ln in out.splitlines():
        code, path = ln[:2], ln[3:].strip().strip('"')
        if not LG.CALL_DOC.match(os.path.basename(path)):
            continue
        if code == "??" or code[0] == "A":
            docs.append(path)
        elif "M" in code:                       # an existing document: gated when its CALL lines changed
            old = subprocess.run(["git", "-C", ROOT, "show", "HEAD:" + path], capture_output=True, text=True, timeout=20).stdout
            new = open(os.path.join(ROOT, path), encoding="utf-8").read()
            if re.findall(r"^CALL.*$", old, re.M) != re.findall(r"^CALL.*$", new, re.M):
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
    try:
        docs = docs_to_gate()
    except Exception:
        return 0
    fails = []
    for d in docs:
        try:
            msg, ok = report(os.path.join(ROOT, d), check_doc(os.path.join(ROOT, d)))
        except Exception as e:                  # fail closed when a call document is in play
            msg, ok = "%s: gate crashed: %s" % (d, e), False
        if not ok:
            fails.append(msg)
    if fails:
        sys.stderr.write("BLOCKED by the call gate. Every check is binary and all must pass before a prediction can be locked.\n" + "\n".join(fails) +
                         "\nFix each item by doing the work it names, then commit again. `python3 .claude/hooks/call_gate.py check <doc>` re-runs the checks.\n")
        return 2
    return 0


def last_assistant_text(path):
    try:
        lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()[-400:]
    except Exception:
        return ""
    for ln in reversed(lines):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        m = d.get("message") or {}
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            txt = "\n".join(b.get("text", "") for b in m["content"] if isinstance(b, dict) and b.get("type") == "text")
            if txt.strip():
                return txt
    return ""


def cmd_stop(_argv):
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    txt = data.get("last_assistant_message") or last_assistant_text(data.get("transcript_path", ""))
    takes = [t.strip() for t in re.findall(r"^TAKE: (.+)$", txt or "", re.M)]
    missing = []
    for t in takes:
        r = subprocess.run(["git", "-C", ROOT, "grep", "-q", "-F", "-e", "TAKE: " + t, "HEAD", "--", "docs/adhoc_calls"], capture_output=True, timeout=20)
        if r.returncode != 0:
            missing.append(t)
    try:
        stc = json.load(open(STOP_STATE))
    except Exception:
        stc = {"blocks": 0}
    if not missing:
        if stc.get("blocks"):
            json.dump({"blocks": 0}, open(STOP_STATE, "w"))
        return 0
    stc["blocks"] = stc.get("blocks", 0) + 1
    os.makedirs(LG.STATE_DIR, exist_ok=True)
    json.dump(stc, open(STOP_STATE, "w"))
    if stc["blocks"] > 3:                       # deadlock breaker, made visible to the user
        print(json.dumps({"systemMessage": "CALL GATE BYPASSED: a prediction was delivered whose TAKE line is not in any committed, gate-checked call document."}))
        return 0
    print(json.dumps({"decision": "block", "reason": "The message carries a prediction (TAKE line) that is not in any committed call document: %s. A prediction may only be delivered after its document passes the call gate and is committed. Run the gate, commit, and then answer; or remove the prediction from the message." % "; ".join(x[:80] for x in missing)}))
    return 0


if __name__ == "__main__":
    cmds = {"prep": cmd_prep, "template": cmd_template, "check": cmd_check, "hook": cmd_hook, "stop": cmd_stop}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__); sys.exit(1)
    sys.exit(cmds[sys.argv[1]](sys.argv[2:]))
