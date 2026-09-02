"""The prediction card: the call, the abstain band, and the plain-language reasons."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from freedom import card
from freedom.features.groups import DESCRIPTIONS, GROUP_KEYS
from freedom.models import BaseModel, make_model

BAND = 0.10


@pytest.mark.parametrize(("p_up", "expected"), [
    (0.62, card.CALL_LONG), (0.60, card.CALL_LONG), (0.59, card.CALL_NO_TRADE), (0.5, card.CALL_NO_TRADE),
    (0.41, card.CALL_NO_TRADE), (0.40, card.CALL_SHORT), (0.1, card.CALL_SHORT),
    (float("nan"), card.CALL_NO_TRADE), (None, card.CALL_NO_TRADE),
])
def test_call_needs_the_edge_to_clear_the_band(p_up, expected):
    assert card.call_for(p_up, BAND) == expected


def test_every_feature_key_has_a_plain_language_description():
    for group, keys in GROUP_KEYS.items():
        for key in keys:
            assert key in DESCRIPTIONS, f"{group}.{key} has no description"
            assert DESCRIPTIONS[key] != key and len(DESCRIPTIONS[key]) > 12
    assert card.describe("f_ret_5d") == DESCRIPTIONS["ret_5d"]
    assert card.describe("f_ret_5d__missing").startswith("whether this was unavailable: ")
    assert card.describe("f_made_up") == "made_up"  # an unknown key describes itself


def _fit(n: int, seed: int = 3):
    """A linear model on a direction that follows f_a - f_b; n < MIN_TRAIN_ROWS leaves the
    heads at the base rate."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    r = 0.03 * a - 0.02 * b + 0.005 * rng.normal(size=n)
    X = pd.DataFrame({"f_a": a, "f_b": b, "f_a__missing": 0.0, "f_b__missing": 0.0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_model("linear", seed=seed).fit(X, pd.Series(r), pd.Series(np.sign(r)))
    return model, X


def _row(p_up: float, **extra) -> dict:
    return {"event_id": "NVDA:2026-07", "market": "xyz:NVDA", "decision_time": "pre_10m",
            "as_of": pd.Timestamp("2026-08-26 20:00", tz="UTC"), "p_up": p_up, "r_hat": 0.012,
            "r_lo": -0.03, "r_hi": 0.05, "magnitude_hat": 0.02, "off_schedule": False, "replay": False,
            **extra}


def test_signed_reasons_follow_the_direction_head():
    model, X = _fit(120)
    probe = pd.DataFrame({"f_a": [2.0], "f_b": [-1.5], "f_a__missing": [0.0], "f_b__missing": [0.0]})
    reasons = card.signed_reasons(model, probe)
    assert reasons is not None and [r["feature"] for r in reasons[:2]] == ["f_a", "f_b"] or \
        [r["feature"] for r in reasons[:2]] == ["f_b", "f_a"]
    by_name = {r["feature"]: r for r in reasons}
    assert by_name["f_a"]["push"] > 0 and by_name["f_a"]["direction"] == "up"  # a high f_a says up
    assert by_name["f_b"]["push"] > 0 and by_name["f_b"]["direction"] == "up"  # a low f_b also says up
    assert by_name["f_a"]["value"] == 2.0 and by_name["f_a"]["what"] == "a"
    pushes = [abs(r["push"]) for r in reasons]
    assert pushes == sorted(pushes, reverse=True)  # largest push first
    assert all(r["feature"] not in ("f_a__missing", "f_b__missing") for r in reasons)  # zero pushes dropped
    p_up = float(model.predict_proba_up(probe)[0])
    c = card.build_card(_row(p_up), model=model, X=probe, band=BAND, fallback=[])
    assert c["call"] == card.CALL_LONG and c["reason_basis"] == card.BASIS_SIGNED and c["tradeable"]
    assert c["edge"] == pytest.approx(p_up - 0.5) and c["expected_r_24h"] == 0.012
    assert c["decision"] == "pre_10m" and c["event_id"] == "NVDA:2026-07"


def test_card_falls_back_to_importance_when_the_direction_head_is_untrained():
    model, X = _fit(12)  # below MIN_TRAIN_ROWS: both heads at the base rate
    assert "direction" in model.fallback_heads_
    assert card.signed_reasons(model, X.head(1)) is None
    fallback = [{"feature": "f_a", "importance": 0.6, "value": 0.4}, {"feature": "f_b", "importance": 0.4, "value": -0.2}]
    c = card.build_card(_row(0.5), model=model, X=X.head(1), band=BAND, fallback=fallback)
    assert c["call"] == card.CALL_NO_TRADE and c["reason_basis"] == card.BASIS_IMPORTANCE
    assert [r["feature"] for r in c["reasons"]] == ["f_a", "f_b"]
    assert all(np.isnan(r["push"]) and r["direction"] == "" for r in c["reasons"])


def test_a_model_without_contributions_uses_the_fallback_too():
    class Plain(BaseModel):
        def fit(self, X, y_return, y_direction):
            return self

        def predict_proba_up(self, X):
            return np.full(len(X), 0.3)

        def predict_return(self, X):
            return np.full(len(X), -0.01)

    X = pd.DataFrame({"f_a": [1.0]})
    c = card.build_card(_row(0.3), model=Plain(), X=X, band=BAND, fallback=[{"feature": "f_a", "importance": 1.0, "value": 1.0}])
    assert c["call"] == card.CALL_SHORT and c["reason_basis"] == card.BASIS_IMPORTANCE
    assert c["reasons"][0]["feature"] == "f_a" and c["reasons"][0]["value"] == 1.0
    assert card.build_card(_row(0.3), model=Plain(), X=X, band=BAND, fallback=[])["reasons"] == []


def test_off_schedule_and_replay_rows_are_not_tradeable():
    model, X = _fit(120)
    c = card.build_card(_row(0.7, off_schedule=True), model=model, X=X.head(1), band=BAND, fallback=[])
    assert c["call"] == card.CALL_LONG and not c["tradeable"] and c["not_tradeable_because"] == ["off schedule"]
    c = card.build_card(_row(0.7, replay=True), model=model, X=X.head(1), band=BAND, fallback=[])
    assert c["not_tradeable_because"] == ["replay"]
    c = card.build_card(_row(0.7), model=model, X=X.head(1), band=BAND, fallback=[])
    assert c["tradeable"] and c["not_tradeable_because"] == []
