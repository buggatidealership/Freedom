import hashlib
import json
import logging
import math

import numpy as np
import pandas as pd
import pytest

from freedom import eval as ev
from freedom import models as models_mod
from freedom.data.base import BudgetExhausted
from freedom.eval import runner
from freedom.eval.folds import season_start
from freedom.schemas import C, D, E, P, T
from tests.synth_eval import (
    PERP_LISTING,
    make_bars,
    make_dataset,
    make_funding,
    make_paths,
    register_fakes,
)

register_fakes()

UTC = "UTC"
T0 = pd.Timestamp("2026-06-10 20:21", tz=UTC)  # a Wednesday, after the close
MODELS = ["zero", "base_rate", "linear"]


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return make_dataset()


def _minute_bars(start: pd.Timestamp, n: int, *, step_min: int = 1, shift: pd.Timedelta | None = None) -> pd.DataFrame:
    t = pd.date_range(start, periods=n, freq=f"{step_min}min") + (shift or pd.Timedelta(0))
    open_ = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame({C.market: "xyz:TEST", C.interval: f"{step_min}m", C.t: t,
                         C.t_end: t + pd.Timedelta(minutes=step_min), C.open: open_, C.high: open_ + 1.0,
                         C.low: open_ - 1.0, C.close: open_ + 0.5, C.volume: 1.0, C.n_trades: 1,
                         C.source: "hl_archive"})


def _prediction(p_up: float, r_hat: float = 0.02, *, has_perp: bool = False, market: str | None = None,
                decision_time: str = "post_30m", event_id: str = "TEST:2026-06",
                magnitude: float | None = None) -> pd.DataFrame:
    row = {P.event_id: event_id, P.decision_time: decision_time, P.model: "m", P.fold: 0,
           P.test_season: "2026Q2", P.p_up: p_up, P.r_hat: r_hat, P.r_true: 0.01,
           P.direction_true: 1.0, E.t0: T0, E.has_perp_at_t0: has_perp, E.market: market}
    if magnitude is not None:  # the column is absent otherwise, so |r_hat| stands in
        row[P.magnitude_hat] = magnitude
    return pd.DataFrame([row])


# ---- folds ---------------------------------------------------------------------------------------
def test_folds_exclude_holdout_respect_embargo_and_min_train(dataset):
    sub = dataset[dataset[D.decision_time] == "post_30m"].reset_index(drop=True)
    folds, holdout = ev.walk_forward_folds(sub, min_train=120, embargo_days=2, holdout_season="2026Q3")
    assert holdout is not None and holdout.fold == ev.HOLDOUT_FOLD and holdout.test_season == "2026Q3"
    seasons = [f.test_season for f in folds]
    assert seasons == sorted(seasons) and "2026Q3" not in seasons
    assert [f.fold for f in folds] == list(range(len(folds)))
    # 40 events per season: the first season with >= 120 prior events is the fourth one
    assert seasons[0] == "2025Q2"
    embargo = pd.Timedelta(days=2)
    for f in folds:
        train, test = sub.loc[f.train_idx], sub.loc[f.test_idx]
        assert (train[E.t0] < season_start(f.test_season) - embargo).all()
        assert (test["season"] == f.test_season).all() and len(test) == 40
        assert not set(train[D.event_id]) & set(test[D.event_id])
        assert (train["season"] != "2026Q3").all()
        assert len(train) >= 120
    assert (sub.loc[holdout.test_idx, "season"] == "2026Q3").all()
    assert (sub.loc[holdout.train_idx, E.t0] < season_start("2026Q3") - embargo).all()
    # no holdout pinned: every season is a candidate test fold
    folds2, holdout2 = ev.walk_forward_folds(sub, min_train=120, embargo_days=2, holdout_season=None)
    assert holdout2 is None and [f.test_season for f in folds2][-1] == "2026Q3"


def test_embargo_removes_events_just_before_the_test_season():
    start = season_start("2026Q2")
    t0 = [start - pd.Timedelta(days=100 + i) for i in range(130)]  # far before the season
    t0 += [start - pd.Timedelta(hours=30), start - pd.Timedelta(days=5)]  # one inside, one outside the embargo
    t0 += [start + pd.Timedelta(days=3)]  # the test event
    events = pd.DataFrame({E.event_id: [f"E{i}" for i in range(len(t0))], E.t0: t0})
    folds, _ = ev.walk_forward_folds(events, min_train=120, embargo_days=2, holdout_season=None)
    fold = [f for f in folds if f.test_season == "2026Q2"][0]
    train_ids = set(events.loc[fold.train_idx, E.event_id])
    assert "E130" not in train_ids and "E131" in train_ids and list(events.loc[fold.test_idx, E.event_id]) == ["E132"]
    folds0, _ = ev.walk_forward_folds(events, min_train=120, embargo_days=0, holdout_season=None)
    fold0 = [f for f in folds0 if f.test_season == "2026Q2"][0]
    assert "E130" in set(events.loc[fold0.train_idx, E.event_id])
    # min_train: nothing qualifies when the requirement exceeds the prior events
    assert ev.walk_forward_folds(events, min_train=500, embargo_days=2, holdout_season=None)[0] == []


# ---- fills, costs, funding, sizing ---------------------------------------------------------------
def test_fill_uses_next_bar_open_never_the_signal_bars_close():
    bars = _minute_bars(pd.Timestamp("2026-08-26 20:00", tz=UTC), 10)
    when = pd.Timestamp("2026-08-26 20:03:30", tz=UTC)  # inside the [20:03, 20:04) bar
    fill = ev.fill_price(bars, when, max_lag=pd.Timedelta(minutes=5))
    assert fill is not None
    price, start, range_bps = fill
    assert start == pd.Timestamp("2026-08-26 20:04", tz=UTC) and start > when
    assert price == 104.0  # open of the 20:04 bar
    signal_bar = bars[bars[C.t] == pd.Timestamp("2026-08-26 20:03", tz=UTC)].iloc[0]
    assert price != signal_bar[C.close] and price != signal_bar[C.open]
    assert range_bps == pytest.approx(2.0 / 104.0 * 1e4)
    # exactly on a bar boundary: that bar's open (its start is >= the signal)
    on_boundary = ev.fill_price(bars, pd.Timestamp("2026-08-26 20:05", tz=UTC), max_lag=pd.Timedelta(minutes=5))
    assert on_boundary[0] == 105.0 and on_boundary[1] == pd.Timestamp("2026-08-26 20:05", tz=UTC)
    # lag limit: with the next five bars missing the first available bar is 5.5 minutes later
    gapped = bars[~bars[C.t].isin(pd.date_range("2026-08-26 20:04", periods=5, freq="1min", tz=UTC))]
    assert ev.fill_price(gapped, when, max_lag=pd.Timedelta(minutes=5)) is None
    late = ev.fill_price(gapped, when, max_lag=pd.Timedelta(minutes=6))
    assert late[1] == pd.Timestamp("2026-08-26 20:09", tz=UTC)
    # nothing after the signal, or no bars at all
    assert ev.fill_price(bars, pd.Timestamp("2026-08-26 21:00", tz=UTC), max_lag=pd.Timedelta(minutes=5)) is None
    assert ev.fill_price(pd.DataFrame(), when, max_lag=pd.Timedelta(minutes=5)) is None


def test_simulate_cost_arithmetic_fills_and_lag(settings):
    bars = _minute_bars(T0 - pd.Timedelta(minutes=10), 24 * 60 + 30)
    trades = ev.simulate(_prediction(0.8), lambda _eid: bars, settings=settings)
    assert len(trades) == 1
    row = trades.iloc[0]
    entry_signal, exit_signal = T0 + pd.Timedelta(minutes=30), T0 + pd.Timedelta(hours=24)
    entry_bar = bars[bars[C.t] >= entry_signal].iloc[0]
    exit_bar = bars[bars[C.t] >= exit_signal].iloc[0]
    assert bool(row["traded"]) and row["side"] == 1 and row["size"] == 1.0
    assert row["entry_fill"] == entry_bar[C.open] and row["entry_fill_time"] == entry_bar[C.t]
    assert row["exit_fill"] == exit_bar[C.open] and row["exit_fill_time"] == exit_bar[C.t]
    assert row["fill_lag_min"] == 0.0 and row["exit_fill_lag_min"] == 0.0

    def leg(bar):
        return settings.slippage_floor_bps + settings.slippage_range_coeff * (bar[C.high] - bar[C.low]) / bar[C.open] * 1e4 + settings.taker_fee_bps

    expected_cost = leg(entry_bar) + leg(exit_bar)
    assert row["cost_bps"] == pytest.approx(expected_cost)
    gross = exit_bar[C.open] / entry_bar[C.open] - 1
    assert row["gross_return"] == pytest.approx(gross)
    assert row["net_return"] == pytest.approx(gross - expected_cost / 1e4)
    assert row["funding_bps"] == 0.0 and row["funding_source"] == ev.FUNDING_NONE
    # a short is the mirror image on the same fills
    short = ev.simulate(_prediction(0.2), lambda _eid: bars, settings=settings).iloc[0]
    assert short["side"] == -1 and short["gross_return"] == pytest.approx(-gross)
    assert short["cost_bps"] == pytest.approx(expected_cost)
    # bars shifted by 30 s: the fill lags the signal by half a minute and is still the next bar's open
    shifted = _minute_bars(T0 - pd.Timedelta(minutes=10), 24 * 60 + 30, shift=pd.Timedelta(seconds=30))
    lagged = ev.simulate(_prediction(0.8), lambda _eid: shifted, settings=settings).iloc[0]
    assert lagged["fill_lag_min"] == pytest.approx(0.5) and lagged["entry_fill_time"] > entry_signal
    # beyond max_fill_lag_minutes the trade is not taken and the reason is recorded: 10-minute bars
    # starting at 20:18 put the first bar after the 20:51 signal at 20:58 (lag 7 min > 5)
    sparse = _minute_bars(T0 - pd.Timedelta(minutes=3), 24 * 60 + 30, step_min=10)
    dropped = ev.simulate(_prediction(0.8), lambda _eid: sparse, settings=settings).iloc[0]
    assert not dropped["traded"] and dropped["untraded_reason"] == "entry_fill_lag"
    assert math.isnan(dropped["cost_bps"]) and dropped["side"] == 1
    # no signal at exactly 0.5 and no bars
    assert ev.simulate(_prediction(0.5), lambda _eid: bars, settings=settings).iloc[0]["untraded_reason"] == "no_signal"
    assert ev.simulate(_prediction(0.8), lambda _eid: None, settings=settings).iloc[0]["untraded_reason"] == "no_bars"


def test_funding_sign_and_coverage(settings):
    bars = _minute_bars(T0 - pd.Timedelta(minutes=10), 24 * 60 + 30)
    rate = 1e-5
    funding = make_funding(rate)
    long = ev.simulate(_prediction(0.8, has_perp=True, market="xyz:TEST"), lambda _e: bars, settings=settings,
                       funding=funding).iloc[0]
    # settlement hours strictly after the 20:51 entry up to the 20:21 exit next day: 21:00 ... 20:00 = 24 hours
    assert long["funding_source"] == ev.FUNDING_ARCHIVE
    assert long["funding_bps"] == pytest.approx(24 * rate * 1e4)
    assert long["net_return"] == pytest.approx(long["gross_return"] - (long["cost_bps"] + long["funding_bps"]) / 1e4)
    short = ev.simulate(_prediction(0.2, has_perp=True, market="xyz:TEST"), lambda _e: bars, settings=settings,
                        funding=funding).iloc[0]
    assert short["funding_bps"] == pytest.approx(-24 * rate * 1e4)  # the short receives positive funding
    assert short["net_return"] == pytest.approx(short["gross_return"] - short["cost_bps"] / 1e4 + 24 * rate)
    # no perp at t0 -> no funding even though the archive has it
    no_perp = ev.simulate(_prediction(0.8, has_perp=False, market="xyz:TEST"), lambda _e: bars, settings=settings,
                          funding=funding).iloc[0]
    assert no_perp["funding_source"] == ev.FUNDING_NONE and no_perp["funding_bps"] == 0.0
    # one settlement hour missing from the archive -> not covered -> zero with source 'none'
    gap = make_funding(rate, drop_hours=[T0.floor("h") + pd.Timedelta(hours=5)])
    partial = ev.simulate(_prediction(0.8, has_perp=True, market="xyz:TEST"), lambda _e: bars, settings=settings,
                          funding=gap).iloc[0]
    assert partial["funding_source"] == ev.FUNDING_NONE and partial["funding_bps"] == 0.0
    assert ev.simulate(_prediction(0.8, has_perp=True, market="xyz:TEST"), lambda _e: bars, settings=settings,
                       funding=lambda _m: None).iloc[0]["funding_source"] == ev.FUNDING_NONE


def test_sizing_variants(settings):
    bars = _minute_bars(T0 - pd.Timedelta(minutes=10), 24 * 60 + 30)
    conf = ev.simulate(_prediction(0.75), lambda _e: bars, settings=settings, sizing="by_confidence").iloc[0]
    assert conf["size"] == pytest.approx(0.5) and conf["pnl"] == pytest.approx(0.5 * conf["net_return"])
    mag = ev.simulate(_prediction(0.75, r_hat=0.06), lambda _e: bars, settings=settings, sizing="by_magnitude").iloc[0]
    assert mag["size"] == pytest.approx(0.03 / 0.06)
    capped = ev.simulate(_prediction(0.75, r_hat=0.01), lambda _e: bars, settings=settings, sizing="by_magnitude").iloc[0]
    assert capped["size"] == 1.0
    gated = ev.simulate(_prediction(0.75, r_hat=0.001), lambda _e: bars, settings=settings, sizing="magnitude_gate").iloc[0]
    assert not gated["traded"] and gated["untraded_reason"] == "magnitude_gate"
    assert gated["cost_bps"] / 1e4 > 0.001  # the round-trip cost exceeds the predicted move
    passed = ev.simulate(_prediction(0.75, r_hat=0.05), lambda _e: bars, settings=settings, sizing="magnitude_gate").iloc[0]
    assert passed["traded"] and passed["size"] == 1.0
    assert passed[P.magnitude_hat] == pytest.approx(0.05)  # |r_hat| stood in for the missing forecast
    # the model's magnitude forecast (predict_magnitude) drives by_magnitude / magnitude_gate even
    # when its direction forecast r_hat is 0, as for hist_abs_mean and vol_scaled
    mag_only = ev.simulate(_prediction(0.75, r_hat=0.0, magnitude=0.06), lambda _e: bars, settings=settings,
                           sizing="by_magnitude").iloc[0]
    assert mag_only["traded"] and mag_only["size"] == pytest.approx(0.5) and mag_only[P.magnitude_hat] == 0.06
    gate_ok = ev.simulate(_prediction(0.75, r_hat=0.0, magnitude=0.05), lambda _e: bars, settings=settings,
                          sizing="magnitude_gate").iloc[0]
    assert gate_ok["traded"] and gate_ok["size"] == 1.0
    gate_no = ev.simulate(_prediction(0.75, r_hat=0.05, magnitude=0.0005), lambda _e: bars, settings=settings,
                          sizing="magnitude_gate").iloc[0]
    assert not gate_no["traded"] and gate_no["untraded_reason"] == "magnitude_gate"
    assert ev.position_size("by_magnitude", 0.25, 0.0, target_vol=0.03, round_trip_cost=0.002, magnitude=0.06) == (0.5, None)
    assert ev.position_size("by_magnitude", 0.25, 0.06, target_vol=0.03, round_trip_cost=0.002) == (0.5, None)
    assert ev.position_size("magnitude_gate", 0.25, 0.0, target_vol=0.03, round_trip_cost=0.002, magnitude=0.001) == (0.0, "magnitude_gate")
    assert ev.position_size("magnitude_gate", 0.25, 0.0, target_vol=0.03, round_trip_cost=0.002, magnitude=math.nan) == (0.0, "magnitude_gate")
    # threshold on |p_up - 0.5|
    below = ev.simulate(_prediction(0.55), lambda _e: bars, settings=settings, threshold=0.1).iloc[0]
    assert not below["traded"] and below["untraded_reason"] == "below_threshold"
    with pytest.raises(ValueError):
        ev.simulate(_prediction(0.75), lambda _e: bars, settings=settings, sizing="kelly")


# ---- portfolio -----------------------------------------------------------------------------------
def _trades_frame(entries, exits, sizes, rets, traded=None) -> pd.DataFrame:
    n = len(entries)
    return pd.DataFrame({P.event_id: [f"E{i}" for i in range(n)], "traded": traded if traded is not None else [True] * n,
                         "entry_fill_time": pd.to_datetime(entries, utc=True),
                         "exit_fill_time": pd.to_datetime(exits, utc=True), "size": sizes, "net_return": rets,
                         "pnl": np.asarray(sizes) * np.asarray(rets)})


def test_equal_split_exposure_never_exceeds_the_cap():
    rng = np.random.default_rng(3)
    base = pd.Timestamp("2026-05-04", tz=UTC)
    entries = [base + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 10 * 24 * 60, size=60)]
    exits = [e + pd.Timedelta(hours=24) for e in entries]
    sizes = rng.uniform(0.2, 1.0, size=60)
    rets = rng.normal(0, 0.02, size=60)
    trades = _trades_frame(entries, exits, sizes, rets)
    for cap in (1.0, 0.5):
        pm = ev.portfolio_metrics(trades, gross_exposure_cap=cap)
        assert pm["n_trades"] == 60 and pm["max_gross_exposure"] <= cap + 1e-12
        # independent check on a one-minute grid
        entry_ns = trades["entry_fill_time"].astype("int64").to_numpy()
        exit_ns = trades["exit_fill_time"].astype("int64").to_numpy()
        w, _ = ev.equal_split_weights(entry_ns, exit_ns, sizes, cap=cap)
        grid = pd.date_range(base, base + pd.Timedelta(days=12), freq="1min").astype("int64").to_numpy()
        exposure = np.zeros(len(grid))
        for i in range(60):
            exposure += w[i] * ((grid >= entry_ns[i]) & (grid <= exit_ns[i]))
        assert exposure.max() <= cap + 1e-12
        assert (w > 0).all()
    # non-overlapping positions each hold the whole capital; overlapping ones split it
    solo = _trades_frame([base, base + pd.Timedelta(days=3)], [base + pd.Timedelta(days=1), base + pd.Timedelta(days=4)],
                         [1.0, 1.0], [0.01, -0.02])
    pm = ev.portfolio_metrics(solo)
    assert pm["max_gross_exposure"] == pytest.approx(1.0) and pm["total_return"] == pytest.approx(-0.01)
    assert pm["n_days"] == 2 and pm["max_drawdown"] == pytest.approx(0.02) and pm["turnover"] == pytest.approx(4.0)
    assert pm["hit_rate"] == 0.5
    daily = pd.Series([0.01, -0.02])
    assert pm["sharpe_like"] == pytest.approx(daily.mean() / daily.std(ddof=1) * math.sqrt(252))
    pair = _trades_frame([base, base], [base + pd.Timedelta(days=1)] * 2, [1.0, 1.0], [0.02, 0.02])
    pm2 = ev.portfolio_metrics(pair)
    assert pm2["max_gross_exposure"] == pytest.approx(1.0) and pm2["total_return"] == pytest.approx(0.02)
    empty = ev.portfolio_metrics(_trades_frame([base], [base + pd.Timedelta(days=1)], [1.0], [0.01], traded=[False]))
    assert empty["n_trades"] == 0 and empty["n_untraded"] == 1 and math.isnan(empty["sharpe_like"])
    # a loss from the starting capital is a drawdown: the initial equity is the first peak
    early_loss = _trades_frame([base, base + pd.Timedelta(days=3)], [base + pd.Timedelta(days=1), base + pd.Timedelta(days=4)],
                               [1.0, 1.0], [-0.02, 0.01])
    assert ev.portfolio_metrics(early_loss)["max_drawdown"] == pytest.approx(0.02)
    single_loss = _trades_frame([base], [base + pd.Timedelta(days=1)], [1.0], [-0.05])
    assert ev.portfolio_metrics(single_loss)["max_drawdown"] == pytest.approx(0.05)
    assert ev.portfolio_metrics(_trades_frame([base], [base + pd.Timedelta(days=1)], [1.0], [0.05]))["max_drawdown"] == 0.0


# ---- metrics -------------------------------------------------------------------------------------
def test_classification_metrics_on_known_inputs():
    p = [0.9, 0.8, 0.3, 0.6, 0.5, 0.7]
    y = [1, -1, -1, 1, 1, 0]  # the zero label is dropped
    m = ev.classification_metrics(p, y)
    assert m["n"] == 5
    assert m["accuracy"] == pytest.approx((1 + 0 + 1 + 1 + 0.5) / 5)
    assert m["balanced_accuracy"] == pytest.approx(((1 + 1 + 0.5) / 3 + (0 + 1) / 2) / 2)
    assert m["brier"] == pytest.approx((0.01 + 0.64 + 0.09 + 0.16 + 0.25) / 5)
    expected_ll = -np.mean([math.log(0.9), math.log(0.2), math.log(0.7), math.log(0.6), math.log(0.5)])
    assert m["log_loss"] == pytest.approx(expected_ll)
    assert ev.classification_metrics([], [])["n"] == 0 and math.isnan(ev.classification_metrics([], [])["brier"])
    assert ev.classification_metrics([0.5, 0.5], [1, -1])["accuracy"] == 0.5


def test_regression_metrics_calibration_and_residual_band():
    r_hat, r_true = [0.1, 0.2, 0.3, np.nan], [0.1, 0.25, 0.2, 0.5]
    m = ev.regression_metrics(r_hat, r_true)
    assert m["n"] == 3 and m["mae"] == pytest.approx(0.05)
    assert m["rmse"] == pytest.approx(math.sqrt((0 + 0.0025 + 0.01) / 3))
    assert m["spearman_ic"] == pytest.approx(0.5)
    assert math.isnan(ev.regression_metrics([1, 1, 1], [1, 2, 3])["spearman_ic"])
    table = ev.calibration_table([0.05, 0.15, 0.95, 0.9, 0.92], [-1, 1, 1, 1, -1], bins=5)
    assert list(table["bin"]) == [0, 1, 2, 3, 4] and table["n"].sum() == 5
    last = table.iloc[4]
    assert last["n"] == 3 and last["frac_up"] == pytest.approx(2 / 3)
    assert last["mean_p"] == pytest.approx((0.95 + 0.9 + 0.92) / 3) and last["gap"] == pytest.approx(2 / 3 - last["mean_p"])
    assert table.iloc[2]["n"] == 0 and math.isnan(table.iloc[2]["mean_p"])
    res = np.linspace(-1.0, 1.0, 201)
    lo, hi = ev.residual_band(np.zeros(201), res)
    assert lo == pytest.approx(-0.8) and hi == pytest.approx(0.8)
    assert all(math.isnan(v) for v in ev.residual_band([], []))


def test_mde_is_monotone_in_n_and_matches_the_formula():
    z = 1.959963984540054 + 0.8416212335729143
    assert ev.min_detectable_improvement(100, "accuracy", 0.5) == pytest.approx(z * math.sqrt(0.25 / 100))
    assert ev.min_detectable_improvement(300, "brier", 0.25) == pytest.approx(z * math.sqrt(0.1875 / 300))
    values = [ev.min_detectable_improvement(n, "brier", 0.25) for n in (50, 100, 200, 400, 1000, 5000)]
    assert all(a > b for a, b in zip(values[:-1], values[1:], strict=True))
    assert ev.min_detectable_improvement(100, "accuracy", 0.5, alpha=0.01) > ev.min_detectable_improvement(100, "accuracy", 0.5)
    assert math.isnan(ev.min_detectable_improvement(0, "accuracy", 0.5))
    with pytest.raises(ValueError):
        ev.min_detectable_improvement(100, "mae", 0.1)
    # the paired MDE comes from the comparison's own standard error and sits well below the
    # closed-form (unpaired, Bhatia-Davis) bound at the same n
    rng = np.random.default_rng(5)
    diff = pd.Series(rng.normal(0.01, 0.214, size=158))
    se = ev.paired_se(diff)
    assert se == pytest.approx(diff.std(ddof=1) / math.sqrt(158))
    assert ev.paired_mde(se) == pytest.approx(z * se)
    assert ev.paired_mde(se) < 0.6 * ev.min_detectable_improvement(158, "brier", 0.25)
    assert ev.paired_mde(se, alpha=0.01) > ev.paired_mde(se)
    assert ev.paired_se(diff.to_numpy()) == se and math.isnan(ev.paired_se([0.1])) and math.isnan(ev.paired_mde(math.nan))
    assert ev.paired_se([0.1, np.nan, 0.3]) == pytest.approx(np.std([0.1, 0.3], ddof=1) / math.sqrt(2))


def test_bootstrap_ci_iid_and_block():
    rng = np.random.default_rng(1)
    values = pd.Series(rng.normal(0.1, 1.0, size=400))

    def mean(v: pd.Series) -> float:
        return float(v.mean())

    point, lo, hi = ev.bootstrap_ci(values, mean, n=400, seed=7)
    assert point == pytest.approx(values.mean()) and lo < point < hi
    assert ev.bootstrap_ci(values, mean, n=400, seed=7) == (point, lo, hi)  # deterministic
    assert ev.MIN_BLOCKS == 5
    seasons = pd.Series(np.repeat(["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1"], 80))
    bp, blo, bhi = ev.bootstrap_ci(values, mean, n=400, block=seasons, seed=7)
    assert bp == pytest.approx(point) and blo < bp < bhi and (blo, bhi) != (lo, hi)
    # a single block (the holdout season) must not give a degenerate [point, point] interval:
    # with fewer than MIN_BLOCKS distinct blocks the rows are resampled iid instead
    one_block = ev.bootstrap_ci(values, mean, n=400, block=pd.Series(["s"] * 400), seed=7)
    assert one_block[1] < point < one_block[2] and one_block == (point, lo, hi)
    four = pd.Series(np.repeat(list("abcd"), 100))
    assert ev.bootstrap_ci(values, mean, n=400, block=four, seed=7) == (point, lo, hi)
    four_block = ev.bootstrap_ci(values, mean, n=400, block=four, seed=7, min_blocks=4)
    assert four_block != (point, lo, hi) and four_block[1] < point < four_block[2]
    dist = ev.bootstrap_distribution(values, mean, n=200, block=pd.Series(["s"] * 400), seed=7)
    assert dist.std() > 0.02  # not collapsed onto the point estimate
    # choose_blocks takes the coarsest structure that has enough blocks and reports it
    days = pd.Series([f"d{i % 40}" for i in range(400)])
    name, labels = ev.choose_blocks([("block:season", pd.Series(["s"] * 400)), ("block:day", days)])
    assert name == "block:day" and labels is days
    assert ev.choose_blocks([("block:season", seasons), ("block:day", days)])[0] == "block:season"
    assert ev.choose_blocks([("block:season", pd.Series(["s"] * 400))]) == ("iid", None)
    assert ev.choose_blocks([("block:season", None)]) == ("iid", None)
    assert all(math.isnan(v) for v in ev.bootstrap_ci(pd.Series([], dtype=float), mean)[1:])


def test_verdicts():
    assert ev.verdict("brier", 0.01, 0.02, 0.005, 100) == "improves"
    assert ev.verdict("brier", -0.01, 0.001, 0.005, 100) == "not_predictable"
    assert ev.verdict("brier", -0.01, 0.02, 0.005, 100) == "inconclusive at n = 100"
    assert ev.verdict("brier", -0.02, -0.01, 0.005, 100) == "worse"
    assert ev.verdict("spearman_ic", -0.1, 0.2, None, 40) == "inconclusive at n = 40"


# ---- evaluate end to end ---------------------------------------------------------------------------
def test_evaluate_end_to_end_writes_reports(settings, dataset):
    paths, funding = make_paths(dataset), make_funding()
    summary = ev.evaluate(settings, dataset, model_names=MODELS, decision_times=["pre_5m", "post_30m"],
                          paths=paths, funding=funding, n_boot=60)
    out = settings.reports_dir / summary["run_id"]
    for name in ("summary.json", "predictions.parquet", "trades.parquet", "leaderboard.md"):
        assert (out / name).exists()
    loaded = json.loads((out / "summary.json").read_text())
    assert loaded["run_id"] == summary["run_id"] and len(loaded["dataset_sha256"]) == 64
    assert summary["run_id"].endswith(summary["dataset_sha256"][:8]) and "T" in summary["run_id"]
    assert summary["git"]["sha"] and "pandas" in summary["versions"] and "fmp_api_key" not in summary["settings"]
    assert summary["settings"]["trade_threshold"] == 0.0 and summary["settings"]["target_vol"] == 0.03
    assert summary["dataset_hash_source"] == "content" and summary["capital_rule"] == ev.CAPITAL_RULE
    assert summary["final"] is False and summary["holdout_results"] is None
    assert summary["holdout"] == {"season": "2026Q3", "scorings_before": 0, "scorings_after": 0,
                                  "scored_now": False, "n_events": 40, "n_unobservable_24h": 0,
                                  "unobservable_24h": []}
    assert not settings.holdout_log_path.exists()
    # no surprise / max_leverage column in this dataset: nothing is non-point-in-time
    assert summary["non_point_in_time_groups"] == {} and summary["trading_subsets"] == list(ev.TRADING_SUBSETS)
    assert summary["cohorts"]["post_30m"]["non_point_in_time_groups"] == []
    assert not any("non-point-in-time" in n for n in summary["notes"])

    preds = pd.read_parquet(out / "predictions.parquet")
    assert set(preds[P.model]) == set(MODELS) and set(preds[P.decision_time]) == {"pre_5m", "post_30m"}
    assert (preds[P.test_season] != "2026Q3").all() and (preds[P.fold] >= 0).all()
    assert not preds.duplicated([P.event_id, P.decision_time, P.model]).any()
    # 2025Q2 has 120 prior events but fewer than 120 *trainable* ones (low-confidence and
    # target-missing events are never trained on), so it is skipped and reported as such
    assert set(preds[P.test_season]) == {"2025Q3", "2025Q4", "2026Q1", "2026Q2"}
    assert [f["test_season"] for f in summary["folds"]["post_30m"]] == sorted(set(preds[P.test_season]))
    assert all(f["n_train"] >= settings.min_train_events for f in summary["folds"]["post_30m"])
    skipped = summary["skipped_seasons"]["post_30m"]
    # every season before the first usable fold is listed with its (too small) training count
    assert [s["test_season"] for s in skipped] == ["2024Q3", "2024Q4", "2025Q1", "2025Q2"]
    assert all(s["n_train_trainable"] < 120 for s in skipped)
    # the first fold has no earlier residuals, later folds carry a band around r_hat
    lin = preds[(preds[P.model] == "linear") & (preds[P.decision_time] == "post_30m")]
    assert lin.loc[lin[P.fold] == 0, P.r_lo].isna().all() and lin.loc[lin[P.fold] > 0, P.r_lo].notna().all()
    assert (lin.loc[lin[P.fold] > 0, P.r_lo] < lin.loc[lin[P.fold] > 0, P.r_hat]).all()
    # low-confidence events are predicted (kept in the table) but never trained on: they are in the test rows
    assert (preds[E.t0_confidence] < settings.min_t0_confidence).any()
    # the magnitude forecast is a column of its own; these stand-ins use the default |r_hat|
    assert (preds[P.magnitude_hat] == preds[P.r_hat].abs()).all()

    res = summary["results"]["post_30m"]
    cell = res["linear"]["subsets"]["all"]
    assert cell["n"] > 100 and cell["n_direction"] <= cell["n"]
    for key in ("accuracy", "balanced_accuracy", "brier", "log_loss", "spearman_ic", "mae", "rmse", "magnitude_mae"):
        assert key in cell and cell[key] is not None
    assert cell["n_magnitude"] == cell["n_return"] and cell["magnitude_mae"] > 0
    assert cell["spearman_ic"] > 0.3  # the synthetic signal is recoverable
    assert set(cell["mde"]) == {"accuracy", "brier"} and cell["mde"]["brier"] > 0
    assert cell["ci"]["brier"][0] < cell["brier"] < cell["ci"]["brier"][1]
    # four test seasons are too few for season blocks: the report says which scheme it used
    assert cell["resampling"] == "block:day"
    cmp = cell["comparison"]["brier"]
    assert cmp["baseline"] in ("zero", "base_rate") and cmp["mde"] == cell["mde"]["brier"]
    assert cmp["ci"][0] <= cmp["improvement"] <= cmp["ci"][1] and 0 <= cmp["p_noise"] <= 1
    assert cmp["verdict"] in ("improves", "not_predictable", "worse") or cmp["verdict"].startswith("inconclusive at n = ")
    assert summary["best_baseline"]["post_30m"]["all"]["brier"]["model"] == cmp["baseline"]
    # the MDE is the paired comparison's own, below the closed-form upper bound at the same n
    assert cell["mde_source"] == {"accuracy": "paired_se", "brier": "paired_se"} and cmp["mde_source"] == "paired_se"
    assert cmp["se"] > 0 and cmp["mde"] == pytest.approx(ev.paired_mde(cmp["se"]))
    assert cmp["mde"] < ev.min_detectable_improvement(cmp["n"], "brier", cmp["baseline_value"])
    assert cmp["resampling"] == cell["resampling"] and cmp["se_bootstrap"] > 0
    assert cell["comparison"]["magnitude_mae"]["baseline"] in ("zero", "base_rate")
    assert res["zero"]["subsets"]["all"]["mde_source"]["brier"] == "closed_form_upper_bound"
    best_brier = summary["best_baseline"]["post_30m"]["all"]["brier"]["value"]  # the bound is taken at the best baseline
    assert res["zero"]["subsets"]["all"]["mde"]["brier"] == pytest.approx(
        ev.min_detectable_improvement(res["zero"]["subsets"]["all"]["n_direction"], "brier", best_brier))
    assert res["zero"]["is_baseline"] and res["zero"]["subsets"]["all"]["comparison"] is None
    assert res["zero"]["subsets"]["all"]["accuracy"] == pytest.approx(0.5)
    assert res["zero"]["subsets"]["all"]["brier"] == pytest.approx(0.25)
    for subset in ("has_perp_at_t0", "headline", "t0_source=sec_8k", "t0_source=calendar_flag", "kind=equity_us",
                   "kind=equity_fpi", "timing=AMC", "timing=BMO"):
        assert subset in res["linear"]["subsets"], subset
    assert res["linear"]["subsets"]["has_perp_at_t0"]["n"] == int(
        (lin[E.has_perp_at_t0] & lin[P.r_true].notna()).sum())
    assert res["linear"]["subsets"]["headline"]["n"] <= res["linear"]["subsets"]["has_perp_at_t0"]["n"]
    assert res["linear"]["subsets"]["all"]["calibration"] is not None and len(res["linear"]["subsets"]["all"]["calibration"]) == 10
    assert res["linear"]["subsets"]["kind=equity_us"]["calibration"] is None
    band = res["linear"]["residual_band"]
    assert band["q10"] < 0 < band["q90"] and 0.6 < band["coverage"] < 0.95

    trades = pd.read_parquet(out / "trades.parquet")
    assert set(trades["sizing"]) == set(ev.SIZINGS) and set(trades[P.model]) == set(MODELS)
    zero_trades = trades[trades[P.model] == "zero"]
    assert not zero_trades["traded"].any() and (zero_trades["untraded_reason"] == "no_signal").all()
    fixed = trades[(trades[P.model] == "linear") & (trades["sizing"] == "fixed") & (trades[P.decision_time] == "post_30m")]
    taken = fixed[fixed["traded"]]
    # a fill is the open of the first bar starting at or after the signal: never before it, and
    # never later than max_fill_lag_minutes
    assert len(taken) > 100 and (taken["entry_fill_time"] >= taken["signal_time"]).all()
    assert (taken["entry_fill_time"] - taken["signal_time"] <= pd.Timedelta(minutes=settings.max_fill_lag_minutes)).all()
    assert (taken["fill_lag_min"] >= 0).all() and (taken["fill_lag_min"] > 0).any()
    assert (taken["exit_fill_time"] >= taken["exit_signal_time"]).all()
    assert (taken["cost_bps"] > 2 * (settings.slippage_floor_bps + settings.taker_fee_bps)).all()
    # funding only for perp-era events with archive coverage, and only for them
    perp_era = taken[E.t0] >= PERP_LISTING
    assert (taken.loc[perp_era, "funding_source"] == ev.FUNDING_ARCHIVE).all()
    assert (taken.loc[~perp_era, "funding_source"] == ev.FUNDING_NONE).all()
    tr = res["linear"]["trading"]["fixed"]
    assert tr["n_trades"] == len(taken) and tr["max_gross_exposure"] <= settings.gross_exposure_cap + 1e-12
    assert 0 < tr["funding_share_events"] < 1 and "sharpe_like" in tr and "max_drawdown" in tr
    assert tr["mean_pnl"]["n"] == len(taken) and tr["comparison"]["baseline"] in ("zero", "base_rate")
    assert tr["resampling"] == "block:day" and tr["mean_pnl"]["lo"] < tr["mean_pnl"]["point"] < tr["mean_pnl"]["hi"]
    assert tr["comparison"]["resampling"] == "block:day" and tr["comparison"]["ci"][0] < tr["comparison"]["ci"][1]
    assert P.magnitude_hat in trades.columns and (taken[P.magnitude_hat] == taken[P.r_hat].abs()).all()
    assert tr["untraded_reasons"].get("no_bars", 0) == int(lin[P.r_true].isna().sum())  # no path without targets
    assert res["linear"]["trading"]["magnitude_gate"]["n_trades"] < tr["n_trades"]
    assert res["base_rate"]["trading"]["fixed"]["comparison"] is None
    # trading statistics per subset: `trading` is the every-row simulation, `trading_subsets`
    # holds it next to the headline slice (perp era, confident, non-calendar t0), and the
    # paired PnL comparison is against the best baseline on the same rows
    assert tr["subset"] == "all" and tr["n_events"] == lin[P.event_id].nunique()
    ts = res["linear"]["trading_subsets"]
    assert set(ts) == set(ev.TRADING_SUBSETS) and ts["all"]["fixed"] == tr
    n_head = res["linear"]["subsets"]["headline"]["n"]
    head = ts["headline"]["fixed"]
    assert head["subset"] == "headline" and head["n_events"] == n_head
    assert 0 < head["n_trades"] <= n_head < tr["n_trades"] and head["mean_pnl"]["n"] == head["n_trades"]
    assert head["comparison"]["baseline"] in ("zero", "base_rate") and head["comparison"]["ci"][0] < head["comparison"]["ci"][1]
    assert "headline" in trades.columns and int(fixed["headline"].sum()) == n_head
    assert head["n_trades"] == int(taken["headline"].sum())
    assert (fixed.loc[fixed["headline"], E.has_perp_at_t0]).all()
    assert (fixed.loc[fixed["headline"], E.t0] >= PERP_LISTING).all()
    # the headline slice's PnL is recomputed on its own trades, not inherited from the all-rows line
    head_pnl = taken.loc[taken["headline"], "pnl"]
    assert head["mean_pnl"]["point"] == pytest.approx(head_pnl.mean()) and head["mean_pnl"]["point"] != tr["mean_pnl"]["point"]

    md = (out / "leaderboard.md").read_text()
    assert "## post_30m" in md and "| linear |" in md and "MDE" in md and summary["run_id"] in md
    assert "magnitude MAE" in md and "block:day" in md and "‡" in md and "equal_split" in md
    # the leaderboard's trading columns describe the rows of the row's subset (headline here)
    post = md.split("## post_30m", 1)[1]
    line = next(ln for ln in post.splitlines() if ln.startswith("| linear | headline |"))
    cells = [c.strip() for c in line.split("|")[1:-1]]
    assert cells[2] == str(n_head) and cells[-2] == f"{head['sharpe_like']:.2f}"
    assert cells[-1] == f"{head['mean_pnl']['point'] * 1e4:.1f}"
    assert cells[-1] != f"{tr['mean_pnl']['point'] * 1e4:.1f}"
    assert any("inconclusive" in n or "Brier comparisons" in n for n in summary["notes"])
    assert "continuation_dead_band_n" in summary["cohorts"]["post_30m"]


def test_evaluate_rejects_bad_inputs(settings, dataset):
    with pytest.raises(KeyError):
        ev.evaluate(settings, dataset, model_names=["no_such_model"], decision_times=["post_30m"], paths=lambda _e: None)
    with pytest.raises(ValueError):
        ev.evaluate(settings, dataset, model_names=["zero"], decision_times=["post_7m"], paths=lambda _e: None)
    tiny = dataset[dataset["season"].isin(["2026Q1", "2026Q2"])]
    with pytest.raises(ValueError, match="trainable events"):
        ev.evaluate(settings, tiny, model_names=["zero"], decision_times=["post_30m"], paths=lambda _e: None)


def test_trade_threshold_and_target_vol_are_settings(settings, dataset):
    strict = settings.model_copy(update={"trade_threshold": 0.45, "target_vol": 0.01})
    assert ev.report.config_hash(strict) != ev.report.config_hash(settings)
    summary = ev.evaluate(strict, dataset, model_names=["linear"], decision_times=["post_30m"],
                          paths=make_paths(dataset), n_boot=10)
    assert summary["settings"]["trade_threshold"] == 0.45 and summary["settings"]["target_vol"] == 0.01
    tr = summary["results"]["post_30m"]["linear"]["trading"]
    assert tr["fixed"]["untraded_reasons"].get("below_threshold", 0) > 0
    assert summary["results"]["post_30m"]["linear"]["trading_subsets"]["headline"]["fixed"]["untraded_reasons"].get("below_threshold", 0) > 0
    trades = pd.read_parquet(strict.reports_dir / summary["run_id"] / "trades.parquet")
    taken = trades[trades["traded"] & (trades["sizing"] == "by_magnitude")]
    assert (taken["size"] == np.minimum(1.0, 0.01 / taken[P.magnitude_hat])).all()


def test_magnitude_forecasts_reach_the_simulation_and_metrics(settings, dataset):
    """hist_abs_mean forecasts r_hat = 0 and a magnitude of its own: it must trade under
    by_magnitude / magnitude_gate and be scored as a magnitude forecast."""
    summary = ev.evaluate(settings, dataset, model_names=["zero", "hist_abs_mean", "linear"],
                          decision_times=["post_30m"], paths=make_paths(dataset), funding=make_funding(), n_boot=20)
    out = settings.reports_dir / summary["run_id"]
    preds = pd.read_parquet(out / "predictions.parquet")
    ham = preds[preds[P.model] == "hist_abs_mean"]
    assert (ham[P.r_hat] == 0).all() and (ham[P.magnitude_hat] > 0).all()
    res = summary["results"]["post_30m"]
    assert res["hist_abs_mean"]["is_baseline"]
    tr = res["hist_abs_mean"]["trading"]
    assert tr["magnitude_gate"]["n_trades"] > 0 and tr["by_magnitude"]["n_trades"] > 0
    assert tr["magnitude_gate"]["untraded_reasons"].get("magnitude_gate", 0) == 0
    trades = pd.read_parquet(out / "trades.parquet")
    bym = trades[(trades[P.model] == "hist_abs_mean") & (trades["sizing"] == "by_magnitude") & trades["traded"]]
    assert len(bym) and np.allclose(bym["size"], np.minimum(1.0, settings.target_vol / bym[P.magnitude_hat]))
    cell = res["hist_abs_mean"]["subsets"]["all"]
    assert np.isfinite(cell["magnitude_mae"]) and cell["magnitude_mae"] < res["zero"]["subsets"]["all"]["magnitude_mae"]
    assert summary["best_baseline"]["post_30m"]["all"]["magnitude_mae"]["model"] == "hist_abs_mean"
    cmp = res["linear"]["subsets"]["all"]["comparison"]["magnitude_mae"]
    assert cmp["baseline"] == "hist_abs_mean" and cmp["ci"][0] <= cmp["improvement"] <= cmp["ci"][1]
    assert cmp["mde"] is None  # MDE is reported for brier / accuracy only


def test_holdout_band_pools_only_pre_holdout_residuals(settings):
    """A dataset that holds seasons after the pinned holdout (the normal state between a season
    closing and the human edit that advances holdout_season) must not lend the holdout's
    r_lo/r_hi the out-of-sample errors observed after the holdout period."""
    from tests.synth_eval import SEASONS

    ds = make_dataset(seasons=[*SEASONS, "2026Q4"])
    s = settings.model_copy(update={"holdout_season": "2026Q2"})
    df = runner.prepare_dataset(ds, "r_24h", s)
    sub = df[df[D.decision_time] == "post_30m"].drop_duplicates(D.event_id).reset_index(drop=True)
    sub = sub[sub[runner.Y].notna()].reset_index(drop=True)
    folds, holdout, _, _ = runner._fold_plan(sub, s)
    assert holdout is not None and {"2026Q3", "2026Q4"} <= {f.test_season for f in folds}
    preds = runner._walk_forward(sub, runner.feature_columns(df), folds, "linear", s, holdout=holdout)
    wf = preds[preds[P.fold] != ev.HOLDOUT_FOLD]
    res = wf[P.r_true] - wf[P.r_hat]
    before = res[wf[P.test_season] < "2026Q2"]
    assert set(wf.loc[before.index, P.test_season]) == {"2025Q3", "2025Q4", "2026Q1"}
    q10, q90 = np.quantile(before, [0.1, 0.9])
    hp = preds[preds[P.fold] == ev.HOLDOUT_FOLD]
    assert len(hp) and (hp[P.test_season] == "2026Q2").all()
    assert np.allclose(hp[P.r_lo] - hp[P.r_hat], q10) and np.allclose(hp[P.r_hi] - hp[P.r_hat], q90)
    all_q10, all_q90 = np.quantile(res, [0.1, 0.9])
    assert not (np.isclose(q10, all_q10) and np.isclose(q90, all_q90))
    # walk-forward folds still use the residuals of the folds before them, the first has none
    first = wf[wf[P.fold] == 0]
    assert first[P.r_lo].isna().all()
    for fold in sorted(set(wf[P.fold]) - {0}):
        part = wf[wf[P.fold] == fold]
        season = part[P.test_season].iloc[0]
        earlier = res[wf[P.test_season] < season]
        assert np.allclose(part[P.r_lo] - part[P.r_hat], np.quantile(earlier, 0.1))


def test_non_point_in_time_inputs_are_marked(settings, dataset):
    """Design §5: the surprise group (vendor-final consensus) and perp_state.max_leverage (the
    current cap) are not point-in-time; every run that has them in scope says so, with the
    estimate_source breakdown of the trainable events."""
    from freedom.features.groups import NON_POINT_IN_TIME

    rng = np.random.default_rng(2)
    ds = dataset.assign(**{"f_eps_surprise": rng.normal(size=len(dataset)), "f_eps_surprise__missing": 0.0,
                           "f_max_leverage": 10.0, "f_max_leverage__missing": 0.0})
    assert E.estimate_source not in ds.columns
    ids = ds[D.event_id].astype(str)
    calendar = pd.DataFrame({E.event_id: ids.unique(), E.t0: ds.drop_duplicates(D.event_id)[E.t0].to_numpy()})
    calendar[E.estimate_source] = np.where(np.arange(len(calendar)) % 3 == 0, "consensus_snapshot", "fmp_final")
    calendar.loc[calendar.index[:2], E.estimate_source] = None
    summary = ev.evaluate(settings, ds, model_names=["zero", "linear"], decision_times=["pre_5m", "post_30m"],
                          paths=make_paths(ds), n_boot=10, events=calendar)
    # both groups are admissible at post_30m, only perp_state at pre_5m
    assert summary["non_point_in_time_groups"] == {g: NON_POINT_IN_TIME[g] for g in ("perp_state", "surprise")}
    assert summary["cohorts"]["post_30m"]["non_point_in_time_groups"] == ["perp_state", "surprise"]
    assert summary["cohorts"]["pre_5m"]["non_point_in_time_groups"] == ["perp_state"]
    # estimate_source joined from the events calendar when the dataset predates the column
    counts = summary["cohorts"]["post_30m"]["estimate_source"]
    n_trainable = summary["cohorts"]["post_30m"]["n_trainable"]
    assert set(counts) <= {"consensus_snapshot", "fmp_final", "missing"} and sum(counts.values()) == n_trainable
    assert counts["fmp_final"] > counts["consensus_snapshot"] > 0
    preds = pd.read_parquet(settings.reports_dir / summary["run_id"] / "predictions.parquet")
    assert E.estimate_source in preds.columns and set(preds[E.estimate_source].dropna()) == {"consensus_snapshot", "fmp_final"}
    notes = [n for n in summary["notes"] if "non-point-in-time" in n]
    assert len(notes) == 2 and any(n.startswith("post_30m:") and "surprise" in n and "linear" in n for n in notes)
    md = (settings.reports_dir / summary["run_id"] / "leaderboard.md").read_text()
    assert "Non-point-in-time inputs: **perp_state**" in md and "**surprise**" in md
    assert "Non-point-in-time inputs in scope at post_30m: perp_state, surprise" in md and "fmp_final:" in md
    # a dataset that carries the column itself is used as is; without either source the count is 'unavailable'
    with_col = ds.assign(**{E.estimate_source: "nasdaq_final"})
    summary2 = ev.evaluate(settings, with_col, model_names=["zero"], decision_times=["post_30m"],
                           paths=make_paths(ds), n_boot=10, events=calendar)
    assert summary2["cohorts"]["post_30m"]["estimate_source"] == {"nasdaq_final": n_trainable}
    assert not any("non-point-in-time" in n for n in summary2["notes"])  # no learner consumed them
    summary3 = ev.evaluate(settings, ds, model_names=["zero"], decision_times=["post_30m"], paths=make_paths(ds), n_boot=10)
    assert summary3["cohorts"]["post_30m"]["estimate_source"] == {"unavailable": n_trainable}
    assert ev.non_point_in_time_in_scope(["f_a", "f_eps_surprise"], "pre_5m") == {}
    assert list(ev.non_point_in_time_in_scope(["f_a", "f_eps_surprise"], "post_60m")) == ["surprise"]
    assert ev.non_point_in_time_in_scope(["f_funding_rate"], "pre_5m") == {}  # perp_state without max_leverage
    assert ev.estimate_source_counts(pd.DataFrame({E.estimate_source: ["fmp_final", None, np.nan, "fmp_final"]})) == {"fmp_final": 2, "missing": 2}
    nullable = pd.DataFrame({E.estimate_source: pd.array(["fmp_final", pd.NA], dtype="string")})
    assert ev.estimate_source_counts(nullable) == {"fmp_final": 1, "missing": 1}
    assert ev.estimate_source_counts(pd.DataFrame({"x": [1, 2]})) == {"unavailable": 2}


class _Recorder(models_mod.BaseModel):
    """Stand-in that records the direction target it was fitted on and returns preset outputs."""

    fitted: list[np.ndarray] = []
    p_out: np.ndarray | None = None
    r_out: np.ndarray | None = None
    m_out: np.ndarray | None = None

    def fit(self, X, y_return, y_direction):
        type(self).fitted.append(np.asarray(y_direction, dtype=float))
        return self

    def predict_proba_up(self, X):
        return self.p_out if self.p_out is not None else np.full(len(X), 0.6)

    def predict_return(self, X):
        return self.r_out if self.r_out is not None else np.full(len(X), 0.01)

    def predict_magnitude(self, X):
        return self.m_out if self.m_out is not None else np.abs(self.predict_return(X))


def test_fit_predict_passes_zero_direction_through_and_validates_outputs(request, caplog):
    if "recorder" not in models_mod.REGISTRY:
        models_mod.register("recorder")(_Recorder)
    request.addfinalizer(lambda: models_mod.REGISTRY.pop("recorder", None))
    train = pd.DataFrame({"f_a": [0.1, 0.2, 0.3], runner.Y: [0.0, 0.02, -0.01], runner.DIR: [0.0, 1.0, -1.0]})
    test = pd.DataFrame({"f_a": [0.5, 0.6, 0.7]})
    _Recorder.fitted.clear()
    _Recorder.p_out = _Recorder.r_out = _Recorder.m_out = None
    p, r, mag, model = runner._fit_predict("recorder", 7, train, test, ["f_a"])
    assert list(_Recorder.fitted[-1]) == [0.0, 1.0, -1.0]  # a zero move is not 'up'
    assert list(p) == [0.6] * 3 and list(mag) == [0.01] * 3 and isinstance(model, _Recorder)
    # out-of-range probabilities are clipped with a warning; NaN magnitude falls back to |r_hat|
    _Recorder.p_out = np.array([1.2, -0.1, 0.7])
    _Recorder.r_out = np.array([0.02, -0.03, np.nan])
    _Recorder.m_out = np.array([np.nan, -0.03, 0.04])
    with caplog.at_level(logging.WARNING, logger="freedom.eval.runner"):
        p, r, mag, _ = runner._fit_predict("recorder", 7, train, test, ["f_a"])
    assert list(p) == [1.0, 0.0, 0.7] and "outside [0, 1]" in caplog.text and "non-finite r_hat" in caplog.text
    assert mag[0] == pytest.approx(0.02) and mag[1] == pytest.approx(0.03) and mag[2] == pytest.approx(0.04)
    assert math.isnan(r[2])
    # too many non-finite probabilities reject the model outright
    _Recorder.p_out = np.array([np.nan, 0.5, 0.5])
    with pytest.raises(ValueError, match="non-finite p_up"):
        runner._fit_predict("recorder", 7, train, test, ["f_a"])
    _Recorder.p_out = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="predictions for 3 rows"):
        runner._fit_predict("recorder", 7, train, test, ["f_a"])
    _Recorder.p_out = _Recorder.r_out = _Recorder.m_out = None


def test_loader_paths_propagates_provider_aborts(settings, dataset, monkeypatch):
    from freedom.targets import loaders

    events = pd.DataFrame([{E.event_id: "TEST:2026-06", E.t0: T0, E.market: "xyz:TEST", E.underlying: "TEST"}])

    def exhausted(*_a, **_k):
        raise BudgetExhausted("FMP daily budget exhausted; wait for the next UTC day")

    monkeypatch.setattr(loaders, "load_event_bars", exhausted)
    with pytest.raises(BudgetExhausted, match="budget"):
        ev.loader_paths(settings, events)("TEST:2026-06")

    def missing(*_a, **_k):
        raise FileNotFoundError("no archive")

    monkeypatch.setattr(loaders, "load_event_bars", missing)
    assert ev.loader_paths(settings, events)("TEST:2026-06") is None
    monkeypatch.setattr(loaders, "load_event_bars", lambda *_a, **_k: (pd.DataFrame(), None))
    assert ev.loader_paths(settings, events)("TEST:2026-06") is None
    assert ev.loader_paths(settings, events)("UNKNOWN:2026-06") is None
    # and evaluate aborts instead of reporting trades on a partially fetched set of events
    with pytest.raises(BudgetExhausted):
        ev.evaluate(settings, dataset, model_names=["linear"], decision_times=["post_30m"], paths=exhausted, n_boot=10)
    assert not any(settings.reports_dir.iterdir())


def test_dataset_sha256_of_the_parquet_file_is_used_when_it_exists(settings, dataset):
    path = settings.dataset_path
    dataset.to_parquet(path, index=False)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert ev.dataset_sha256(path) == expected == ev.dataset_sha256(str(path))
    assert ev.dataset_sha256(dataset) != expected and len(ev.dataset_sha256(dataset)) == 64
    summary = ev.evaluate(settings, dataset, model_names=["zero"], decision_times=["post_30m"],
                          paths=lambda _e: None, n_boot=10)
    assert summary["dataset_sha256"] == expected and summary["dataset_hash_source"] == f"file:{path}"
    assert summary["run_id"].endswith(expected[:8])
    ev.train_final(settings, dataset, model_name="zero", decision_time="post_30m")
    meta = json.loads((settings.models_dir / "post_30m" / "zero" / "model.json").read_text())
    assert meta["dataset_sha256"] == expected and meta["dataset_hash_source"] == f"file:{path}"
    # an explicit path wins over the default
    other = settings.data_dir / "other.parquet"
    dataset.head(200).to_parquet(other, index=False)
    explicit = ev.evaluate(settings, dataset, model_names=["zero"], decision_times=["post_30m"],
                           paths=lambda _e: None, n_boot=10, dataset_path=other)
    assert explicit["dataset_sha256"] == hashlib.sha256(other.read_bytes()).hexdigest()


# ---- final: holdout scoring ----------------------------------------------------------------------------
def _unobservable_24h(dataset: pd.DataFrame, event_id: str) -> pd.DataFrame:
    """`dataset` with the event's +24h label blanked the way targets.compute_targets does when
    the checkpoint bar is missing: price_source (and p0) resolved, r_24h / direction NaN,
    target_missing True."""
    out = dataset.copy()
    rows = out[D.event_id] == event_id
    out.loc[rows, [T.r("24h"), T.ar("24h"), T.direction, T.magnitude]] = np.nan
    out.loc[rows, "target_missing"] = True
    assert out.loc[rows, T.price_source].notna().all()
    return out


def test_final_refuses_an_open_holdout_season(settings, dataset):
    clean = dataset  # the synthetic holdout season has complete targets for every event
    paths = make_paths(clean)
    with pytest.raises(ev.HoldoutNotReady, match="future"):
        ev.evaluate(settings, clean, model_names=["zero"], decision_times=["post_30m"], final=True, paths=paths,
                    now=pd.Timestamp("2026-08-01", tz=UTC), n_boot=20)
    assert not settings.holdout_log_path.exists() and not any(settings.reports_dir.iterdir())
    # a dataset built mid-season: every holdout event it holds is closed, but the season is not,
    # so events still scheduled in it cannot be in the dataset
    cutoff = pd.Timestamp("2026-08-15", tz=UTC)
    mid = clean[~((clean["season"] == "2026Q3") & (clean[E.t0] >= cutoff))].reset_index(drop=True)
    assert (mid.loc[mid["season"] == "2026Q3", E.t0] + pd.Timedelta(hours=24) < pd.Timestamp("2026-09-01", tz=UTC)).all()
    with pytest.raises(ev.HoldoutNotReady, match="not closed"):
        ev.evaluate(settings, mid, model_names=["zero"], decision_times=["post_30m"], final=True, paths=make_paths(mid),
                    now=pd.Timestamp("2026-09-01", tz=UTC), n_boot=20)
    assert ev.season_end("2026Q3") == pd.Timestamp("2026-10-01", tz=UTC)
    with pytest.raises(ev.HoldoutNotReady, match="not closed"):
        ev.check_holdout_ready(runner.prepare_dataset(clean, "r_24h", settings), settings,
                               pd.Timestamp("2026-10-01 23:59", tz=UTC))
    # the earnings calendar lists a holdout-season event the dataset lacks
    calendar = pd.DataFrame({E.event_id: ["NEW:2026-09", str(clean[D.event_id].iloc[0])],
                             E.t0: [pd.Timestamp("2026-09-15 20:30", tz=UTC), clean[E.t0].iloc[0]]})
    with pytest.raises(ev.HoldoutNotReady, match=r"not in the dataset \(NEW:2026-09\)"):
        ev.evaluate(settings, clean, model_names=["zero"], decision_times=["post_30m"], final=True, paths=paths,
                    now=pd.Timestamp("2026-12-01", tz=UTC), n_boot=20, events=calendar)
    calendar.to_parquet(settings.events_path, index=False)  # read by default when it exists
    with pytest.raises(ev.HoldoutNotReady, match="not in the dataset"):
        ev.evaluate(settings, clean, model_names=["zero"], decision_times=["post_30m"], final=True, paths=paths,
                    now=pd.Timestamp("2026-12-01", tz=UTC), n_boot=20)
    settings.events_path.unlink()
    assert not settings.holdout_log_path.exists() and not any(settings.reports_dir.iterdir())
    # every holdout event closed, but one has missing targets and no resolved price path (the
    # fetch never completed): complete the dataset first
    first_id = str(clean.loc[clean["season"] == "2026Q3", D.event_id].iloc[0])
    missing = _unobservable_24h(clean, first_id)
    missing.loc[missing[D.event_id] == first_id, T.price_source] = None
    with pytest.raises(ev.HoldoutNotReady, match="missing or pending"):
        ev.evaluate(settings, missing, model_names=["zero"], decision_times=["post_30m"], final=True,
                    paths=make_paths(missing), now=pd.Timestamp("2026-12-01", tz=UTC), n_boot=20)
    # a resolved path whose p0 could not be found is a gap too, not an unobservable label
    with_p0 = _unobservable_24h(clean, first_id).assign(**{T.p0: 100.0})
    with_p0.loc[with_p0[D.event_id] == first_id, T.p0] = np.nan
    with pytest.raises(ev.HoldoutNotReady, match="missing or pending"):
        ev.check_holdout_ready(runner.prepare_dataset(with_p0, "r_24h", settings), settings,
                               pd.Timestamp("2026-12-01", tz=UTC))
    pending = clean.copy()
    pending[E.pending] = pending["season"] == "2026Q3"
    with pytest.raises(ev.HoldoutNotReady, match="missing or pending"):
        ev.evaluate(settings, pending, model_names=["zero"], decision_times=["post_30m"], final=True, paths=paths,
                    now=pd.Timestamp("2026-12-01", tz=UTC), n_boot=20)
    no_holdout = settings.model_copy(update={"holdout_season": None})
    with pytest.raises(ev.HoldoutNotReady, match="no holdout_season"):
        ev.evaluate(no_holdout, clean, model_names=["zero"], decision_times=["post_30m"], final=True, paths=paths,
                    now=pd.Timestamp("2026-12-01", tz=UTC), n_boot=20)
    assert not settings.holdout_log_path.exists()


def test_final_scores_the_holdout_once_and_logs_it(settings, dataset):
    # one holdout event has a resolved FMP-proxy path but no +24h label by construction (a
    # Friday AMC release: t0 + 24h falls on Saturday, design §2); it must not block the run
    holdout_ids = dataset.loc[dataset["season"] == "2026Q3", D.event_id].astype(str).unique()
    gone = str(holdout_ids[3])
    clean = _unobservable_24h(dataset, gone)
    paths, funding = make_paths(clean), make_funding()
    now = pd.Timestamp("2026-12-01", tz=UTC)
    summary = ev.evaluate(settings, clean, model_names=MODELS, decision_times=["post_30m"], final=True, paths=paths,
                          funding=funding, n_boot=30, now=now)
    hold_rows = clean[(clean["season"] == "2026Q3") & (clean[D.decision_time] == "post_30m")]
    n_holdout = hold_rows.loc[~hold_rows["target_missing"], D.event_id].nunique()
    assert n_holdout == len(holdout_ids) - 1
    assert summary["final"] is True and summary["holdout"]["scorings_before"] == 0
    assert summary["holdout"]["scorings_after"] == 1 and summary["holdout"]["scored_now"] is True
    assert summary["holdout"]["n_unobservable_24h"] == 1 and summary["holdout"]["unobservable_24h"] == [gone]
    assert any("no +24h label by construction" in n and gone in n for n in summary["notes"])
    hold = summary["holdout_results"]["post_30m"]["models"]
    cell = hold["linear"]["subsets"]["all"]
    assert cell["n"] == n_holdout
    cmp = cell["comparison"]["brier"]
    assert cmp["baseline"] in ("zero", "base_rate")
    # one season can never be season-blocked: the holdout intervals must not collapse to a point
    assert cell["resampling"] == "block:day" and cmp["resampling"] == "block:day"
    assert cell["ci"]["brier"][0] < cell["ci"]["brier"][1] and cell["ci"]["accuracy"][0] < cell["ci"]["accuracy"][1]
    assert cmp["ci"][0] < cmp["ci"][1] and cmp["mde_source"] == "paired_se"
    for sizing, stats in hold["linear"]["trading"].items():
        if stats["n_trades"] > 1:
            assert stats["mean_pnl"]["lo"] < stats["mean_pnl"]["hi"], sizing
            if stats["comparison"] is not None:  # absent when no baseline traded under this sizing
                assert stats["comparison"]["ci"][0] < stats["comparison"]["ci"][1], sizing
    assert hold["linear"]["trading"]["fixed"]["comparison"]["resampling"] == "block:day"
    # the walk-forward cells never contain the holdout season
    assert "2026Q3" not in [f["test_season"] for f in summary["folds"]["post_30m"]]
    preds = pd.read_parquet(settings.reports_dir / summary["run_id"] / "predictions.parquet")
    hp = preds[preds[P.fold] == ev.HOLDOUT_FOLD]
    assert set(hp[P.test_season]) == {"2026Q3"} and len(hp) == n_holdout * len(MODELS)
    assert gone not in set(hp[P.event_id].astype(str))
    assert (preds.loc[preds[P.fold] >= 0, P.test_season] != "2026Q3").all()
    # the holdout band comes from the walk-forward residuals, all of which precede the holdout here
    wf = preds[(preds[P.model] == "linear") & (preds[P.fold] >= 0)]
    lin_h = hp[hp[P.model] == "linear"]
    q10 = np.quantile(wf[P.r_true] - wf[P.r_hat], 0.1)
    assert np.allclose(lin_h[P.r_lo] - lin_h[P.r_hat], q10)
    lines = [json.loads(ln) for ln in settings.holdout_log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["run_id"] == summary["run_id"] and rec["dataset_sha256"] == summary["dataset_sha256"]
    assert rec["models"] == MODELS and rec["git_commit"] == summary["git"]["sha"] and rec["holdout_season"] == "2026Q3"
    assert rec["timestamp"].startswith("2026-12-01")
    trades = pd.read_parquet(settings.reports_dir / summary["run_id"] / "trades.parquet")
    assert set(trades["block"]) == {"walk_forward", "holdout"}
    # a second scoring is counted so the reader can discount it
    again = ev.evaluate(settings, clean, model_names=["zero"], decision_times=["post_30m"], final=True, paths=paths,
                        funding=funding, n_boot=20, now=now)
    assert again["holdout"]["scorings_before"] == 1 and again["holdout"]["scorings_after"] == 2
    assert "scored 1 time(s) before" in again["notes"][0]


# ---- train_final -------------------------------------------------------------------------------------
def test_train_final_saves_model_with_provenance(settings, dataset):
    model = ev.train_final(settings, dataset, model_name="linear", decision_time="post_30m")
    out = settings.models_dir / "post_30m" / "linear"
    meta = json.loads((out / "model.json").read_text())
    assert (out / "fake_model.json").exists() or (out / "model.joblib").exists()
    assert meta["decision_time"] == "post_30m" and meta["model"] == "linear" and meta["target"] == "r_24h"
    assert len(meta["dataset_sha256"]) == 64 and meta["git_sha"] and len(meta["config_hash"]) == 64
    assert meta["dataset_hash_source"] == "content"  # no dataset.parquet in this data_dir
    assert meta["trained_at"] and meta["schema_version"] == 2
    sub = dataset[(dataset[D.decision_time] == "post_30m") & (dataset["season"] != "2026Q3")]
    trainable = sub[(sub[E.t0_confidence] >= settings.min_t0_confidence) & ~sub["target_missing"]]
    perp = trainable[trainable[E.has_perp_at_t0]]
    use_perp = len(perp) >= settings.min_train_events
    assert meta["filters"] == {"min_t0_confidence": 0.6, "has_perp_at_t0": use_perp,
                               "holdout_season_excluded": "2026Q3", "target_present": True}
    assert meta["n_events"] == (len(perp) if use_perp else len(trainable))
    assert meta["residual_band"]["source"].startswith("walk_forward") and meta["residual_band"]["n"] > 0
    assert model.residual_q_[0] == meta["residual_band"]["q10"] and model.residual_q_[0] < 0 < model.residual_q_[1]
    assert meta["holdout"] == {"season": "2026Q3", "scorings": 0, "last_scoring": None}
    assert meta["feature_names"] == [c for c in dataset.columns if c.startswith("f_")]
    with pytest.raises(KeyError):
        ev.train_final(settings, dataset, model_name="nope", decision_time="post_30m")


# ---- helpers ----------------------------------------------------------------------------------------
def test_synthetic_bars_are_consistent_with_targets(dataset):
    ev_row = dataset.iloc[0]
    bars = make_bars(pd.Timestamp(ev_row[E.t0]), float(ev_row[T.r("24h")]))
    assert (bars[C.t_end] > bars[C.t]).all() and (bars[C.low] <= bars[C.open]).all() and (bars[C.high] >= bars[C.close]).all()
    assert ev.to_jsonable({"a": np.float64(1.5), "b": np.nan, "c": pd.Timestamp("2026-01-01", tz=UTC),
                           "d": np.int64(3), "e": [np.bool_(True)]}) == {"a": 1.5, "b": None, "c": "2026-01-01T00:00:00+00:00",
                                                                        "d": 3, "e": [True]}
