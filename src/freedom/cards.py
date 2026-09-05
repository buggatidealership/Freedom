"""`freedom cards`: every prediction card due in the next N minutes, without anyone typing.

The scheduled job (.github/workflows/cards.yml) runs this as a chain of lingering runs. One scan:

1. lists the universe's upcoming events (events.upcoming_events, two days ahead, with the ids
   `freedom upcoming` prints) and, for each event and each requested decision time, the
   decision instant `expected_t0 + DECISION_TIMES[decision]` minutes;
2. keeps the instants inside [now - DUE_LOOKBACK, now + horizon] and drops the (event_id,
   decision) pairs data/live_predictions.parquet already holds as a live (non-replay) row, so
   overlapping runs never predict a card twice;
3. in chronological order sleeps until each instant (small increments; never with --no-wait or
   a --now replay) and calls live.predict_event with append=True. A post-release decision
   whose release the detector has not seen yet (live.ReleaseNotDetected) is retried every
   RETRY_EVERY_S until instant + RETRY_FOR, then recorded as a "no release detected" note. A
   post-release row that came back off schedule because the release was late (as_of still in
   the future) is re-run at its as_of within the same deadline — the first row stays recorded,
   off schedule, as predict would have recorded it. Any other exception is logged, recorded as
   a failure note and the run continues with the next instant.
4. prints every card with the renderer `freedom predict` uses (print_card) and writes
   <out>/<event slug>__<decision>.md (a compact markdown card the job posts to the Cards
   issue), the same stem .json (the card, the recorded row, the schedule and the model record)
   and one line per card or note in <out>/index.md. The slug replaces the ':' of an event id
   (a GitHub artifact path may not contain one).

A `--now` override replays: nothing sleeps, the rows carry replay=True (so they never count as
live predictions and never block a live card later) and the cards print NOT TRADEABLE.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from . import events as events_mod
from . import live
from .config import Settings
from .data.archive import read_parquet_or_none
from .schemas import DECISION_TIMES, UTC, D, E
from .timeutil import to_utc

log = logging.getLogger(__name__)

DEFAULT_DECISIONS = "pre_10m,post_15m,post_30m"
DEFAULT_HORIZON_MINUTES = 45
LOOKAHEAD_DAYS = 2  # calendar days of upcoming events to consider
# An instant this recently past is still due: GitHub cron starts late and a run that slept until its
# last instant blocks the next one. A late pre card comes out off schedule (NOT TRADEABLE), which is
# a better record than no card at all.
DUE_LOOKBACK = pd.Timedelta(minutes=20)
RETRY_EVERY_S = 60.0  # post decisions: poll for the release this often ...
RETRY_FOR = pd.Timedelta(minutes=15)  # ... until this long after the instant
RESCAN_EVERY_S = 600.0  # linger mode: look for newly due instants this often
POSTED_SUFFIX = ".posted"  # marker next to a card/note the run itself posted on the Cards issue
SLEEP_STEP_S = 30.0  # waiting for an instant sleeps in increments of at most this
CARDS_SUBDIR = "cards"
INDEX_FILE = "index.md"
NOTE_NO_RELEASE = "no release detected"
NOTE_FAILED = "prediction failed"


def utcnow() -> pd.Timestamp:
    """The wall clock (module-level so tests can drive it)."""
    return pd.Timestamp.now(tz=UTC)


# ---- formatting shared with `freedom predict` ------------------------------------------------------
def blank(v: object) -> bool:
    return v is None or v is pd.NaT or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def fmt_value(v: object) -> str:
    """A cell: timestamps to the minute, floats to 4 significant-ish digits, missing blank."""
    if blank(v):
        return ""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-3 or abs(v) >= 1e6 else f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def fmt_pct(v: object) -> str:
    """A return as a signed percentage ('' when missing)."""
    return "" if blank(v) else f"{float(v) * 100:+.2f} %"  # type: ignore[arg-type]


def print_card(card: dict, console: Console | None = None) -> None:
    """The operator's card on the console: the call, its size, and the reasons."""
    console = console or Console()
    style = {"LONG": "bold green", "SHORT": "bold red"}.get(card["call"], "bold yellow")
    console.print(f"\nCARD {card['decision']}  {card['event_id']}  ({card['market'] or 'no perp market'})",
                  style="bold", markup=False)
    console.print(f"CALL: {card['call']}", style=style, markup=False)
    if card.get("forced_call"):
        console.print(f"  forced pick (graded on every event): {card['forced_call']}", markup=False)
    console.print(f"  p_up {fmt_value(card['p_up'])}   edge {card['edge']:+.3f} vs band ±{card['band']:.2f}   "
                  f"expected 24h move {fmt_pct(card['expected_r_24h'])}   typical size "
                  f"{fmt_pct(card['magnitude_hat']).lstrip('+-')}"
                  f"   10/90 % band {fmt_pct(card['r_lo'])} .. {fmt_pct(card['r_hi'])}", markup=False)
    for why in card["not_tradeable_because"]:
        console.print(f"  NOT TRADEABLE: {why}", style="yellow", markup=False)
    if card["reasons"]:
        table = Table(title=f"why: {card['reason_basis']}")
        for col in ("push", "feature", "value", "what it measures"):
            table.add_column(col)
        for r in card["reasons"]:
            push = "" if blank(r["push"]) else f"{r['push']:+.3f} ({r['direction']})"
            table.add_row(push, r["feature"], fmt_value(r["value"]), r["what"])
        console.print(table)
    else:
        console.print("  no reasons: the direction head is untrained (base rate) and the model exposes no "
                      "feature importance; the call above is the base rate, not a read of this event",
                      style="yellow", markup=False)


# ---- what is due --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Due:
    event_id: str
    underlying: str
    market: str | None
    decision: str
    expected_t0: pd.Timestamp
    instant: pd.Timestamp  # expected_t0 + DECISION_TIMES[decision] minutes

    @property
    def label(self) -> str:
        return f"{self.event_id} {self.decision}"


def slug(event_id: str) -> str:
    """File stem for an event id: 'NVDA:2026-07' -> 'NVDA_2026-07' (':' is not a valid
    artifact path character)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(event_id))


def live_pairs(settings: Settings) -> set[tuple[str, str]]:
    """(event_id, decision_time) of every live (replay == False) row already recorded."""
    df = read_parquet_or_none(live.live_predictions_path(settings))
    if df is None or len(df) == 0 or E.event_id not in df.columns or D.decision_time not in df.columns:
        return set()
    if "replay" in df.columns:
        df = df[~df["replay"].fillna(False).astype(bool)]
    return {(str(e), str(d)) for e, d in zip(df[E.event_id], df[D.decision_time], strict=True)}


def due_instants(settings: Settings, *, now: pd.Timestamp, horizon: pd.Timedelta, decisions: list[str],
                 days: int = LOOKAHEAD_DAYS, lookback: pd.Timedelta = DUE_LOOKBACK) -> tuple[list[Due], list[Due]]:
    """(due, already predicted): the (event, decision) instants inside [now - lookback,
    now + horizon] in chronological order, split by whether a live row exists for the pair."""
    upcoming = events_mod.upcoming_events(settings, days=days, source="table")  # never spends provider quota
    if upcoming is None or len(upcoming) == 0:
        return [], []
    upcoming = live.with_event_ids(upcoming)
    done = live_pairs(settings)
    lo, hi = now - lookback, now + horizon
    due: list[Due] = []
    skipped: list[Due] = []
    for r in upcoming.to_dict("records"):
        t0 = r.get("expected_t0")
        event_id = r.get(E.event_id)
        if blank(t0) or blank(event_id):
            continue
        t0 = to_utc(pd.Timestamp(t0), assume_tz=UTC)
        market = r.get(E.market)
        for decision in decisions:
            instant = t0 + pd.Timedelta(minutes=DECISION_TIMES[decision])
            if not lo <= instant <= hi:
                continue
            d = Due(str(event_id), str(r.get(E.underlying)), market if isinstance(market, str) else None,
                    decision, t0, instant)
            (skipped if (d.event_id, d.decision) in done else due).append(d)
    key = lambda d: (d.instant, d.event_id, d.decision)  # noqa: E731
    return sorted(due, key=key), sorted(skipped, key=key)


# ---- the run ------------------------------------------------------------------------------------------
@dataclass
class CardsRun:
    now: pd.Timestamp
    horizon: pd.Timedelta
    out_dir: Path
    replay: bool
    due: list[Due] = field(default_factory=list)
    skipped: list[Due] = field(default_factory=list)
    cards: list[dict] = field(default_factory=list)  # {"due", "card", "md", "json"}
    notes: list[dict] = field(default_factory=list)  # {"due", "kind", "message", "md"}


def _jsonable(v: object) -> object:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, str | bool | int) or v is None:
        return v
    if isinstance(v, float):
        return None if math.isnan(v) else v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(v, "item", None)  # numpy scalars
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    return str(v)


def card_markdown(res: dict) -> str:
    """The compact markdown card the job posts: the CALL first (a phone notification shows
    the first line), then the numbers, the tradeability, the reasons, the model and as_of."""
    card, row, sched = res["card"], res.get("row") or {}, res.get("schedule") or {}
    market = card.get("market") or "no perp market"
    lines = [f"## CALL: {card['call']} — {card['event_id']} {card['decision']} ({market})", ""]
    lines.append(f"**p_up {fmt_value(card['p_up'])}** · edge {card['edge']:+.3f} vs band ±{card['band']:.2f}")
    lines.append(f"expected 24 h move **{fmt_pct(card['expected_r_24h']) or 'n/a'}** · typical size "
                 f"{fmt_pct(card['magnitude_hat']).lstrip('+-') or 'n/a'} · 10/90 % band "
                 f"{fmt_pct(card['r_lo']) or 'n/a'} .. {fmt_pct(card['r_hi']) or 'n/a'}")
    lines.append("")
    if card.get("tradeable"):
        lines.append("tradeable: **yes**")
    else:
        lines.append("**NOT TRADEABLE**: " + ", ".join(card.get("not_tradeable_because") or ["unknown"]))
    lines.append("")
    if card.get("reasons"):
        lines.append(f"why ({card.get('reason_basis', '')}):")
        lines.append("")
        lines.append("| push | feature | value | what it measures |")
        lines.append("|---|---|---|---|")
        for r in card["reasons"]:
            push = "" if blank(r.get("push")) else f"{r['push']:+.3f} ({r['direction']})"
            lines.append(f"| {push} | `{r['feature']}` | {fmt_value(r.get('value'))} | {r.get('what', '')} |")
    else:
        lines.append("no reasons: the direction head is untrained (base rate) and the model exposes no "
                     "feature importance; the call above is the base rate, not a read of this event")
    lines.append("")
    as_of = card.get("as_of") if not blank(card.get("as_of")) else row.get(D.as_of)
    lines.append(f"as_of {fmt_value(as_of)} UTC · t0 used {fmt_value(row.get('t0_used'))} UTC "
                 f"({row.get('t0_source_live') or 'unknown source'})")
    if sched.get("note"):
        lines.append(f"schedule: {sched['note']}")
    if row.get("replay"):
        lines.append(f"REPLAY: `now` was overridden to {fmt_value(row.get('now_override'))} UTC; "
                     "this row does not count as a live prediction")
    lines.append(f"model `{row.get('model_id') or 'unknown'}` · run at {fmt_value(row.get('run_at'))} UTC")
    return "\n".join(lines) + "\n"


def note_markdown(due: Due, kind: str, message: str, when: pd.Timestamp) -> str:
    market = due.market or "no perp market"
    return (f"## NOTE: {kind} — {due.event_id} {due.decision} ({market})\n\n"
            f"No card was produced for the decision instant {fmt_value(due.instant)} UTC "
            f"(expected release {fmt_value(due.expected_t0)} UTC).\n\n{message}\n\n"
            f"recorded at {fmt_value(when)} UTC\n")


def index_line(due: Due, *, summary: str, stem: str, when: pd.Timestamp) -> str:
    return f"- {fmt_value(when)} UTC · {due.event_id} {due.decision} · {summary} · [{stem}.md]({stem}.md)\n"


def _append_index(out_dir: Path, line: str) -> None:
    path = out_dir / INDEX_FILE
    if not path.exists():
        path.write_text("# Cards\n\nOne line per card or note, oldest first (the cards job appends).\n\n",
                        encoding="utf-8")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def write_card(out_dir: Path, due: Due, res: dict, *, when: pd.Timestamp) -> dict:
    """<out>/<slug>__<decision>.md and .json plus the index line; returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{slug(due.event_id)}__{due.decision}"
    md, js = out_dir / f"{stem}.md", out_dir / f"{stem}.json"
    md.write_text(card_markdown(res), encoding="utf-8")
    payload = {"card": res["card"], "row": res.get("row"), "schedule": res.get("schedule"),
               "model_meta": res.get("model_meta"), "consensus": res.get("consensus"),
               "written_at": when, "instant": due.instant, "expected_t0": due.expected_t0}
    js.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    card = res["card"]
    trade = "tradeable" if card.get("tradeable") else f"NOT TRADEABLE ({', '.join(card.get('not_tradeable_because') or [])})"
    summary = (f"CALL: {card['call']} · p_up {fmt_value(card['p_up'])} · expected "
               f"{fmt_pct(card['expected_r_24h']) or 'n/a'} · {trade}")
    _append_index(out_dir, index_line(due, summary=summary, stem=stem, when=when))
    return {"md": md, "json": js}


def write_note(out_dir: Path, due: Due, *, kind: str, message: str, when: pd.Timestamp) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{slug(due.event_id)}__{due.decision}"
    md = out_dir / f"{stem}.md"
    md.write_text(note_markdown(due, kind, message, when), encoding="utf-8")
    _append_index(out_dir, index_line(due, summary=f"NOTE: {kind}: {message}", stem=stem, when=when))
    return md


def _sleep_until(instant: pd.Timestamp, *, step_s: float = SLEEP_STEP_S) -> None:
    while True:
        remaining = (instant - utcnow()).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(step_s, remaining))


def _off_schedule_early(res: dict, now: pd.Timestamp) -> pd.Timestamp | None:
    """The as_of to re-run at when a post-release row is off schedule only because the release
    came late (the detector saw it, its as_of is still ahead of now); else None."""
    row = res.get("row") or {}
    as_of = row.get(D.as_of)
    if not row.get("off_schedule") or blank(as_of):
        return None
    as_of = to_utc(pd.Timestamp(as_of), assume_tz=UTC)
    return as_of if as_of > now else None


def _predict_due(settings: Settings, due: Due, *, now_override: pd.Timestamp | str | None,
                 wait: bool, console: Console, model_name: str | None = None) -> tuple[dict | None, str | None]:
    """(result, message) — the prediction, or None with the reason no card was produced.
    Post decisions retry an undetected release every RETRY_EVERY_S until instant + RETRY_FOR
    (only when waiting is allowed) and re-run a too-early row at its as_of."""
    deadline = due.instant + RETRY_FOR
    post = DECISION_TIMES[due.decision] >= 0
    message = None
    while True:
        try:
            res = live.predict_event(settings, event_id=due.event_id, decision=due.decision,
                                     model_name=model_name, now=now_override, append=True)
        except live.ReleaseNotDetected as exc:
            message = str(exc)
            now = utcnow()
            if not (post and wait) or now >= deadline:
                return None, message
            console.print(f"  {due.label}: {message}; retrying in {RETRY_EVERY_S:.0f} s until "
                          f"{fmt_value(deadline)} UTC", style="yellow", markup=False)
            time.sleep(RETRY_EVERY_S)
            continue
        now = utcnow()
        rerun_at = _off_schedule_early(res, now) if post and wait else None
        if rerun_at is not None and rerun_at <= deadline:
            console.print(f"  {due.label}: the release came late; the row at {fmt_value(now)} UTC is off "
                          f"schedule (recorded), re-running at as_of {fmt_value(rerun_at)} UTC",
                          style="yellow", markup=False)
            _sleep_until(rerun_at)
            continue
        return res, None


def post_to_issue(md: Path, console: Console) -> bool:
    """Post a card or note as a comment on the Cards issue when the job environment names one
    (CARDS_ISSUE and GITHUB_REPOSITORY; the gh CLI is authenticated by GH_TOKEN), right after it
    is written rather than at the end of a run that may linger for hours. A marker file
    <name>.posted records success so the workflow's catch-up step never posts it twice."""
    issue = os.environ.get("CARDS_ISSUE", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not issue or not repo:
        return False
    marker = md.with_name(md.name + POSTED_SUFFIX)
    if marker.exists():
        return True
    try:
        proc = subprocess.run(["gh", "issue", "comment", issue, "-R", repo, "--body-file", str(md)],
                              capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        console.print(f"  could not post {md.name}: {exc}", style="yellow", markup=False)
        return False
    if proc.returncode != 0:
        console.print(f"  could not post {md.name}: {proc.stderr.strip()[:200]}", style="yellow", markup=False)
        return False
    marker.write_text(proc.stdout.strip() + "\n", encoding="utf-8")
    console.print(f"  posted {md.name} on issue #{issue}", markup=False)
    return True


def _handle_due(settings: Settings, due: Due, run: CardsRun, *, now: pd.Timestamp | str | None, wait: bool,
                console: Console, model_name: str | None) -> None:
    """Wait for one instant, predict it, write the card or the note, post it."""
    if wait and due.instant > utcnow():
        console.print(f"waiting {(due.instant - utcnow()).total_seconds() / 60:.1f} min for {due.label} "
                      f"at {fmt_value(due.instant)} UTC", markup=False)
        _sleep_until(due.instant)
    try:
        res, message = _predict_due(settings, due, now_override=now, wait=wait, console=console,
                                    model_name=model_name)
    except Exception as exc:  # one failed card must not stop the others
        log.exception("%s failed", due.label)
        res, message = None, f"{type(exc).__name__}: {exc}"
        kind = NOTE_FAILED
    else:
        kind = NOTE_NO_RELEASE
    when = utcnow()
    if res is None:
        md = write_note(run.out_dir, due, kind=kind, message=message or "", when=when)
        run.notes.append({"due": due, "kind": kind, "message": message, "md": md})
        console.print(f"NOTE {due.label}: {kind}: {message}", style="yellow", markup=False)
        post_to_issue(md, console)
        return
    print_card(res["card"], console)
    paths = write_card(run.out_dir, due, res, when=when)
    run.cards.append({"due": due, "card": res["card"], **paths})
    console.print(f"wrote {paths['md']}", markup=False)
    post_to_issue(Path(paths["md"]), console)


def _print_due(due: list[Due], *, horizon_minutes: int, now: pd.Timestamp, replay: bool, console: Console) -> None:
    table = Table(title=f"freedom cards: {len(due)} due in the next {horizon_minutes} minutes "
                        f"(now {fmt_value(now)} UTC{', REPLAY' if replay else ''})")
    for col in ("instant (UTC)", "event", "decision", "market", "expected t0 (UTC)"):
        table.add_column(col)
    for d in due:
        table.add_row(fmt_value(d.instant), d.event_id, d.decision, d.market or "", fmt_value(d.expected_t0))
    console.print(table)


def run_cards(settings: Settings, *, horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
              decisions: list[str] | None = None, now: pd.Timestamp | str | None = None,
              wait: bool = True, out_dir: Path | None = None, console: Console | None = None,
              model_name: str | None = None, linger_minutes: int = 0) -> CardsRun:
    """Predict, print and write every card due in the next `horizon_minutes` minutes. `model_name`
    picks the trained model under data/models/<decision>/ (None: the only one there).
    `linger_minutes` > 0 keeps the run alive that long, rescanning every RESCAN_EVERY_S for
    instants that enter the horizon, so a chain of long runs covers the clock even though
    GitHub's cron fires hours apart (measured 2026-09-05: two to five hours between firings of a
    15-minute schedule). Replays and --no-wait never linger."""
    console = console or Console()
    decisions = list(decisions) if decisions else DEFAULT_DECISIONS.split(",")
    for d in decisions:
        if d not in DECISION_TIMES:
            raise ValueError(f"unknown decision time {d!r}; choose from {', '.join(DECISION_TIMES)}")
    replay = now is not None
    now_ts = to_utc(now, assume_tz=UTC) if replay else utcnow()
    wait = wait and not replay
    horizon = pd.Timedelta(minutes=int(horizon_minutes))
    out = Path(out_dir) if out_dir is not None else settings.reports_dir / CARDS_SUBDIR
    run = CardsRun(now=now_ts, horizon=horizon, out_dir=out, replay=replay)
    linger = pd.Timedelta(minutes=int(linger_minutes)) if wait and linger_minutes > 0 else pd.Timedelta(0)
    watch_until = now_ts + linger
    done: set[tuple[str, str]] = set()
    first = True
    while True:
        scan_now = now_ts if first else utcnow()
        due, skipped = due_instants(settings, now=scan_now, horizon=horizon, decisions=decisions)
        due = [d for d in due if (d.event_id, d.decision) not in done]
        if first:
            run.due, run.skipped = list(due), skipped
            if skipped:
                console.print(f"already predicted live (skipped): {', '.join(d.label for d in skipped)}",
                              markup=False)
            if not due:
                console.print(f"no cards due in the next {horizon_minutes} minutes (now {fmt_value(scan_now)} UTC)",
                              markup=False)
        else:
            run.due.extend(due)
        if due:
            _print_due(due, horizon_minutes=horizon_minutes, now=scan_now, replay=replay, console=console)
        for d in due:
            _handle_due(settings, d, run, now=now, wait=wait, console=console, model_name=model_name)
            done.add((d.event_id, d.decision))
        first = False
        if linger == pd.Timedelta(0):
            return run
        remaining = (watch_until - utcnow()).total_seconds()
        if remaining <= 0:
            console.print(f"linger window over (until {fmt_value(watch_until)} UTC): {len(run.cards)} card(s), "
                          f"{len(run.notes)} note(s)", markup=False)
            return run
        nap = min(RESCAN_EVERY_S, remaining)
        console.print(f"lingering until {fmt_value(watch_until)} UTC; next scan in {nap / 60:.0f} min", markup=False)
        time.sleep(nap)


__all__ = ["CardsRun", "Due", "card_markdown", "due_instants", "fmt_pct", "fmt_value", "live_pairs",
           "print_card", "run_cards", "slug", "utcnow", "write_card", "write_note"]
