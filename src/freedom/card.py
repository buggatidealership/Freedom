"""The two-card prediction output: the operator-facing summary of one live prediction.

Card 1 is the pre-release call (decision time pre_10m: at least ten minutes before the release),
card 2 the post-release call (post_15m / post_30m: the first reaction is in). Both carry:

* call: LONG when p_up - 0.5 >= band, SHORT when 0.5 - p_up >= band, else NO TRADE; a NaN
  probability is NO TRADE. The band is settings.no_trade_band.
* reasons: the features that pushed the direction head, signed (a positive push raises p_up),
  each with the plain-language description from features.groups.DESCRIPTIONS. When the
  direction head fell back to the base rate (fewer than models.MIN_TRAIN_ROWS usable rows) or
  the model has no `contributions` method, the reasons are the importance-ranked features
  instead and carry no sign; reason_basis says which.
* tradeable: False on an off-schedule or replay row. The call is still shown, but must not be
  traded (the row is recorded with the same flags).

Nothing here is a claim of skill: the card reports what the model says and why. Whether that
model beats the base rate is the evaluation report's job (docs/results.md).
"""

from __future__ import annotations

import math

import pandas as pd

from .features.groups import DESCRIPTIONS
from .schemas import D, E

CALL_LONG = "LONG"
CALL_SHORT = "SHORT"
CALL_NO_TRADE = "NO TRADE"
N_REASONS = 5
EDGE_EPS = 1e-9  # a p_up exactly `band` from 0.5 clears the band (0.6 - 0.5 is 0.0999.. in floats)
BASIS_SIGNED = "signed contributions of the direction head (a positive push raises p_up)"
BASIS_IMPORTANCE = ("importance-ranked features; the direction head is untrained (base rate), "
                    "so the pushes carry no sign")


def _float(v: object) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return f


def call_for(p_up: float | None, band: float) -> str:
    """LONG / SHORT / NO TRADE from the up-probability and the abstain band."""
    p = _float(p_up)
    if math.isnan(p):
        return CALL_NO_TRADE
    edge = p - 0.5
    if edge >= band - EDGE_EPS:
        return CALL_LONG
    if -edge >= band - EDGE_EPS:
        return CALL_SHORT
    return CALL_NO_TRADE


def forced_call_for(p_up: float | None) -> str:
    """The forced pick, graded on every event: LONG when p_up >= 0.5, else SHORT ('' when the
    probability is missing). A coin flip scores 50 % here; the banded call is the money rule."""
    p = _float(p_up)
    if math.isnan(p):
        return ""
    return CALL_LONG if p >= 0.5 else CALL_SHORT


def describe(column: str) -> str:
    """Plain-language description of a feature column (f_<key> or f_<key>__missing); an unknown
    key describes itself."""
    key = column[len(D.feature_prefix):] if column.startswith(D.feature_prefix) else column
    if key.endswith(D.missing_suffix):
        base = key[: -len(D.missing_suffix)]
        return f"whether this was unavailable: {DESCRIPTIONS.get(base, base)}"
    return DESCRIPTIONS.get(key, key)


def signed_reasons(model, X: pd.DataFrame, n: int = N_REASONS) -> list[dict] | None:
    """The n largest signed pushes of the direction head for the single row of X, or None when
    the model cannot give them (no `contributions` method, or the head fell back)."""
    contributions = getattr(model, "contributions", None)
    if contributions is None:
        return None
    try:
        c = contributions(X, "direction")
    except RuntimeError:
        return None
    pushes = c.iloc[0].drop(labels=["bias"], errors="ignore").astype(float)
    pushes = pushes[pushes.notna() & (pushes != 0.0)]
    pushes = pushes.reindex(pushes.abs().sort_values(ascending=False).index)
    return [_reason(str(name), X, push=float(push)) for name, push in pushes.head(n).items()]


def _reason(name: str, X: pd.DataFrame, *, push: float) -> dict:
    value = X[name].iloc[0] if name in X.columns else float("nan")
    direction = "" if math.isnan(push) else ("up" if push > 0 else "down")
    return {"feature": name, "what": describe(name), "value": _float(value), "push": push,
            "direction": direction}


def build_card(row: dict, *, model, X: pd.DataFrame, band: float, fallback: list[dict]) -> dict:
    """The card for one scored row. `fallback` is the importance-ranked list
    (live.top_contributions) used when signed contributions are unavailable."""
    p_up = _float(row.get("p_up"))
    reasons = signed_reasons(model, X)
    basis = BASIS_SIGNED
    if reasons is None:
        basis = BASIS_IMPORTANCE
        reasons = [_reason(str(r["feature"]), X, push=float("nan")) for r in fallback[:N_REASONS]]
    blockers = [name for name, flag in (("off schedule", row.get("off_schedule")),
                                        ("replay", row.get("replay"))) if bool(flag)]
    return {
        "event_id": row.get(E.event_id), "market": row.get(E.market),
        "decision": row.get(D.decision_time), "as_of": row.get(D.as_of),
        "call": call_for(p_up, band), "forced_call": forced_call_for(p_up),
        "p_up": p_up, "edge": p_up - 0.5, "band": float(band),
        "expected_r_24h": _float(row.get("r_hat")), "r_lo": _float(row.get("r_lo")),
        "r_hi": _float(row.get("r_hi")), "magnitude_hat": _float(row.get("magnitude_hat")),
        "reasons": reasons, "reason_basis": basis,
        "tradeable": not blockers, "not_tradeable_because": blockers,
    }


__all__ = ["CALL_LONG", "CALL_NO_TRADE", "CALL_SHORT", "N_REASONS", "build_card", "call_for",
           "describe", "forced_call_for", "signed_reasons"]
