"""Forward scorecard: every live card graded against the realised 24 h outcome.

A card is a falsifiable call locked before the outcome exists. The live row records, at as_of,
p_up, the banded call (LONG / SHORT / NO TRADE) and the forced pick (LONG when p_up >= 0.5,
else SHORT); the daily data build later measures r_24h for the same event. `build_scorecard`
joins the two.

* Counted rows are live (replay False) and on schedule. Replays and off-schedule rows are
  listed as excluded and never graded.
* Pending rows are counted calls whose 24 h window has not closed yet; unlabelled rows are
  counted calls whose window closed but whose outcome the build could not measure.
* The forced pick is graded on every scored row (a coin flip scores 50 %). The banded call is
  graded only where it was not NO TRADE: that is the money rule.
* Hit rates carry a Wilson 90 % interval so a handful of calls is never read as evidence.
* Disclosure order is enforced once the release minute is measured (t0 from an 8-K, a detection,
  an issuer clock or a manual override): a pre-release card whose decision instant is not
  strictly before that minute is "contaminated", a post-release card whose instant precedes it
  is "premature"; both are excluded from grading and listed. Every scored row records the margin
  in minutes between its instant and the measured release.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .card import CALL_LONG, CALL_SHORT, call_for, forced_call_for
from .config import Settings
from .data.archive import read_parquet_or_none
from .live import live_predictions_path
from .schemas import DECISION_TIMES, UTC, D, E, T, T0Source
from .timeutil import to_utc

Z90 = 1.6448536269514722
WINDOW_SLACK = pd.Timedelta(hours=2)  # after t0 + horizon the build needs a little time to label
MEASURED_T0 = frozenset({T0Source.sec_8k.value, T0Source.detected.value, T0Source.issuer_clock.value,
                         T0Source.manual.value})  # release minutes that are measurements, not schedules


def wilson(k: int, n: int, z: float = Z90) -> tuple[float, float]:
    """Wilson score interval for k hits in n trials; (nan, nan) when n == 0."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _bool(v) -> bool:
    return not (v is None or (isinstance(v, float) and math.isnan(v))) and bool(v)


def dedupe_live_rows(live: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """One live row per (event, decision) -> (kept, n_duplicates, n_superseded).

    An on-schedule row outranks an off-schedule one. When the release is detected after the
    scheduled instant, `freedom cards` records the first attempt off schedule and re-runs the
    card at its true as_of (t0_live + k); that re-run is the card (ORCL 2026-09-10: both post
    cards, the release detected five minutes after the pinned wire time). Keeping the earliest
    run there would grade nothing: the attempt is excluded as off schedule and the card dropped
    as its duplicate. A dropped attempt is `superseded` and counts as off schedule.
    Among rows of the same standing the earliest run wins (run_at, then posted_at, then file
    order): two overlapping card runs once produced the same post_30m card twice (ORCL
    2026-09-10); the later one is a duplicate of the record, not a second call."""
    if len(live) == 0 or E.event_id not in live.columns or D.decision_time not in live.columns:
        return live, 0, 0
    order = live.copy()
    order["_i"] = range(len(order))
    order["_off"] = (order["off_schedule"].map(_bool).astype(bool) if "off_schedule" in order.columns
                     else pd.Series(False, index=order.index))
    for col in ("run_at", "posted_at"):
        order["_" + col] = pd.to_datetime(order[col], utc=True, errors="coerce") if col in order.columns else pd.NaT
    order = order.sort_values(["_off", "_run_at", "_posted_at", "_i"], na_position="last", kind="mergesort")
    key = [E.event_id, D.decision_time]
    keep = ~order.duplicated(key, keep="first")
    kept = order[keep]
    kept_off = {(str(e), str(d)): bool(o) for e, d, o in zip(kept[E.event_id], kept[D.decision_time], kept["_off"],
                                                            strict=True)}
    dropped = order[~keep]
    n_superseded = sum(bool(o) and not kept_off[(str(e), str(d))]
                       for e, d, o in zip(dropped[E.event_id], dropped[D.decision_time], dropped["_off"], strict=True))
    kept = kept.sort_values("_i").drop(columns=["_i", "_off", "_run_at", "_posted_at"])
    return kept, int(len(dropped) - n_superseded), int(n_superseded)


def build_scorecard(settings: Settings, *, now: pd.Timestamp | None = None) -> dict:
    """Grade data/live_predictions.parquet against data/targets.parquet."""
    now_ts = to_utc(now) if now is not None else pd.Timestamp.now(tz=UTC)
    live = read_parquet_or_none(live_predictions_path(settings))
    targets = read_parquet_or_none(settings.targets_path)
    events = read_parquet_or_none(settings.events_path)
    band = float(settings.no_trade_band)
    out: dict = {"generated_at": now_ts.isoformat(), "n_live_rows": 0 if live is None else int(len(live)),
                 "excluded": {"replay": 0, "off_schedule": 0, "no_probability": 0, "contaminated": 0, "premature": 0,
                              "duplicate": 0},
                 "by_decision": {}, "rows": [], "scored_total": 0, "pending_total": 0, "unlabelled_total": 0}
    if live is None or len(live) == 0:
        return out
    live, n_dup, n_superseded = dedupe_live_rows(live)
    out["excluded"]["duplicate"] = n_dup
    out["excluded"]["off_schedule"] = n_superseded  # attempts replaced by an on-schedule re-run
    truth = {}
    if targets is not None and len(targets):
        for _, t in targets.iterrows():
            truth[str(t[E.event_id])] = _num(t.get(T.r("24h")))
    t0s, measured = {}, set()
    if events is not None and len(events) and E.t0 in events.columns:
        for _, e in events.iterrows():
            if not pd.isna(e[E.t0]):
                t0s[str(e[E.event_id])] = to_utc(pd.Timestamp(e[E.t0]))
                if E.t0_source in events.columns and str(e.get(E.t0_source)) in MEASURED_T0:
                    measured.add(str(e[E.event_id]))
    horizon = pd.Timedelta(hours=float(settings.horizon_hours))
    per: dict[str, dict] = {}
    for _, r in live.iterrows():
        if _bool(r.get("replay")):
            out["excluded"]["replay"] += 1
            continue
        if _bool(r.get("off_schedule")):
            out["excluded"]["off_schedule"] += 1
            continue
        p_up = _num(r.get("p_up"))
        if math.isnan(p_up):
            out["excluded"]["no_probability"] += 1
            continue
        event_id, decision = str(r[E.event_id]), str(r[D.decision_time])
        call = str(r.get("call")) if isinstance(r.get("call"), str) and r.get("call") else call_for(p_up, band)
        forced = str(r.get("forced_call")) if isinstance(r.get("forced_call"), str) and r.get("forced_call") \
            else forced_call_for(p_up)
        t0 = t0s.get(event_id)
        as_of = r.get(D.as_of)
        as_of_ts = to_utc(pd.Timestamp(as_of)) if as_of is not None and not pd.isna(as_of) else None
        margin_min = None
        if t0 is not None and event_id in measured and as_of_ts is not None:
            margin_min = round((as_of_ts - t0).total_seconds() / 60, 1)
            if DECISION_TIMES.get(decision, 0) < 0 and as_of_ts >= t0:
                out["excluded"]["contaminated"] += 1
                out["rows"].append({"event_id": event_id, "decision": decision, "as_of": str(as_of), "p_up": round(p_up, 4),
                                    "call": call, "forced_call": forced, "status": "contaminated",
                                    "margin_min": margin_min, "r_24h": None, "forced_hit": None, "banded_hit": None})
                continue
            if DECISION_TIMES.get(decision, 0) >= 0 and as_of_ts < t0:
                out["excluded"]["premature"] += 1
                out["rows"].append({"event_id": event_id, "decision": decision, "as_of": str(as_of), "p_up": round(p_up, 4),
                                    "call": call, "forced_call": forced, "status": "premature",
                                    "margin_min": margin_min, "r_24h": None, "forced_hit": None, "banded_hit": None})
                continue
        cell = per.setdefault(decision, {"counted": 0, "scored": 0, "pending": 0, "unlabelled": 0,
                                         "forced_hits": 0, "banded_calls": 0, "banded_hits": 0,
                                         "banded_signed_r": [], "brier": [], "ups": 0})
        cell["counted"] += 1
        r24 = truth.get(event_id, float("nan"))
        window_closed = t0 is not None and now_ts >= t0 + horizon + WINDOW_SLACK
        row = {"event_id": event_id, "decision": decision, "as_of": str(r.get(D.as_of)), "model_id": r.get("model_id"),
               "p_up": round(p_up, 4), "call": call, "forced_call": forced, "r_24h": None, "status": "pending",
               "forced_hit": None, "banded_hit": None, "margin_min": margin_min}
        if math.isnan(r24):
            if window_closed:
                cell["unlabelled"] += 1
                row["status"] = "unlabelled"
            else:
                cell["pending"] += 1
            out["rows"].append(row)
            continue
        up = r24 > 0
        cell["scored"] += 1
        cell["ups"] += int(up)
        forced_hit = (forced == CALL_LONG) == up
        cell["forced_hits"] += int(forced_hit)
        cell["brier"].append((p_up - float(up)) ** 2)
        row.update({"r_24h": round(r24, 5), "status": "scored", "forced_hit": bool(forced_hit)})
        if call in (CALL_LONG, CALL_SHORT):
            hit = (call == CALL_LONG) == up
            cell["banded_calls"] += 1
            cell["banded_hits"] += int(hit)
            cell["banded_signed_r"].append(r24 if call == CALL_LONG else -r24)
            row["banded_hit"] = bool(hit)
        out["rows"].append(row)
    for decision in sorted(per, key=lambda d: DECISION_TIMES.get(d, 0)):
        c = per[decision]
        lo, hi = wilson(c["forced_hits"], c["scored"])
        blo, bhi = wilson(c["banded_hits"], c["banded_calls"])
        out["by_decision"][decision] = {
            "counted": c["counted"], "scored": c["scored"], "pending": c["pending"], "unlabelled": c["unlabelled"],
            "base_rate_up": (c["ups"] / c["scored"]) if c["scored"] else None,
            "forced_hit_rate": (c["forced_hits"] / c["scored"]) if c["scored"] else None,
            "forced_hit_90": [lo, hi] if c["scored"] else None,
            "brier": (sum(c["brier"]) / len(c["brier"])) if c["brier"] else None,
            "banded_calls": c["banded_calls"],
            "banded_hit_rate": (c["banded_hits"] / c["banded_calls"]) if c["banded_calls"] else None,
            "banded_hit_90": [blo, bhi] if c["banded_calls"] else None,
            "banded_mean_signed_r_24h": (sum(c["banded_signed_r"]) / len(c["banded_signed_r"]))
            if c["banded_signed_r"] else None,
        }
        out["scored_total"] += c["scored"]
        out["pending_total"] += c["pending"]
        out["unlabelled_total"] += c["unlabelled"]
    return out


def _pct(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.0f} %"


def scorecard_markdown(sc: dict) -> str:
    lines = ["## Forward scorecard", "",
             f"Generated {sc['generated_at'][:16]} UTC. Live cards on record: {sc['n_live_rows']} "
             f"(excluded: {sc['excluded']['replay']} replays, {sc['excluded']['off_schedule']} off schedule, "
             f"{sc['excluded']['no_probability']} without a probability, "
             f"{sc['excluded'].get('contaminated', 0)} pre cards made after the measured release, "
             f"{sc['excluded'].get('premature', 0)} post cards made before it, "
             f"{sc['excluded'].get('duplicate', 0)} duplicates of an earlier run).",
             "", "Forced pick = LONG when p_up >= 0.5 else SHORT, graded on every scored call (a coin flip scores "
             "50 %). Banded call = the money rule (NO TRADE inside the band). Intervals are Wilson 90 %.", "",
             "| decision | counted | scored | pending | forced hit rate | 90 % interval | banded calls | banded hit rate "
             "| mean signed 24 h return | Brier | base rate up |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, c in sc["by_decision"].items():
        fi = f"{_pct(c['forced_hit_90'][0])} to {_pct(c['forced_hit_90'][1])}" if c["forced_hit_90"] else ""
        msr = "" if c["banded_mean_signed_r_24h"] is None else f"{100 * c['banded_mean_signed_r_24h']:+.2f} %"
        brier = "" if c["brier"] is None else f"{c['brier']:.3f}"
        lines.append(f"| {d} | {c['counted']} | {c['scored']} | {c['pending']} | {_pct(c['forced_hit_rate'])} | {fi} | "
                     f"{c['banded_calls']} | {_pct(c['banded_hit_rate'])} | {msr} | {brier} | {_pct(c['base_rate_up'])} |")
    if not sc["by_decision"]:
        lines.append("| (no counted live cards yet) | | | | | | | | | | |")
    scored = [r for r in sc["rows"] if r["status"] == "scored"]
    if scored:
        lines += ["", "| event | decision | as of (UTC) | minutes vs release | call | forced | p_up | realised 24 h | forced hit |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in scored:
            m = "" if r.get("margin_min") is None else f"{r['margin_min']:+.0f}"
            lines.append(f"| {r['event_id']} | {r['decision']} | {r['as_of'][:16]} | {m} | {r['call']} | {r['forced_call']} | "
                         f"{r['p_up']:.2f} | {100 * r['r_24h']:+.2f} % | {'yes' if r['forced_hit'] else 'no'} |")
    bad = [r for r in sc["rows"] if r["status"] in ("contaminated", "premature")]
    if bad:
        lines += ["", "Excluded for disclosure order: " + ", ".join(
            f"{r['event_id']} {r['decision']} ({r['status']}, {r['margin_min']:+.0f} min vs release)" for r in bad)]
    pending = [r for r in sc["rows"] if r["status"] not in ("scored", "contaminated", "premature")]
    if pending:
        lines += ["", "Awaiting an outcome: " + ", ".join(f"{r['event_id']} {r['decision']} ({r['status']})" for r in pending)]
    return "\n".join(lines) + "\n"


def write_scorecard(settings: Settings, sc: dict) -> tuple[Path, Path]:
    """reports/scorecard.md (readable) and data/scorecard.json (travels in the data artifact)."""
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    md = settings.reports_dir / "scorecard.md"
    js = settings.data_dir / "scorecard.json"
    md.write_text(scorecard_markdown(sc), encoding="utf-8")
    js.write_text(json.dumps(sc, indent=1, default=str), encoding="utf-8")
    return md, js


__all__ = ["build_scorecard", "scorecard_markdown", "wilson", "write_scorecard"]
