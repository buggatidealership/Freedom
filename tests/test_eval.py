import json
import math

import numpy as np
import pandas as pd
import pytest

from freedom import eval as ev
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
                decision_time: str = "post_30m", event_id: str = "TEST:2026-06") -> pd.DataFrame:
    return pd.DataFrame([{P.event_id: event_id, P.decision_time: decision_time, P.model: "m", P.fold: 0,
                          P.test_season: "2026Q2", P.p_up: p_up, P.r_hat: r_hat, P.r_true: 0.01,
                          P.direction_true: 1.0, E.t0: T0, E.has_perp_at_t0: has_perp, E.market: market}])


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


def test_bootstrap_ci_iid_and_block():
    rng = np.random.default_rng(1)
    values = pd.Series(rng.normal(0.1, 1.0, size=400))
    point, lo, hi = ev.bootstrap_ci(values, lambda v: float(v.mean()), n=400, seed=7)
    assert point == pytest.approx(values.mean()) and lo < point < hi
    assert ev.bootstrap_ci(values, lambda v: float(v.mean()), n=400, seed=7) == (point, lo, hi)  # deterministic
    seasons = pd.Series(np.repeat(["2025Q1", "2025Q2", "2025Q3", "2025Q4"], 100))
    bp, blo, bhi = ev.bootstrap_ci(values, lambda v: float(v.mean()), n=400, block=seasons, seed=7)
    assert bp == pytest.approx(point) and blo < bp < bhi
    one_block = ev.bootstrap_ci(values, lambda v: float(v.mean()), n=50, block=pd.Series(["s"] * 400), seed=7)
    assert one_block[1] == pytest.approx(point) and one_block[2] == pytest.approx(point)
    assert all(math.isnan(v) for v in ev.bootstrap_ci(pd.Series([], dtype=float), lambda v: float(v.mean()))[1:])


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
    assert summary["final"] is False and summary["holdout_results"] is None
    assert summary["holdout"] == {"season": "2026Q3", "scorings_before": 0, "scorings_after": 0,
                                  "scored_now": False, "n_events": 40}
    assert not settings.holdout_log_path.exists()

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
    assert [s["test_season"] for s in skipped] == ["2025Q2"] and skipped[0]["n_train_trainable"] < 120
    # the first fold has no earlier residuals, later folds carry a band around r_hat
    lin = preds[(preds[P.model] == "linear") & (preds[P.decision_time] == "post_30m")]
    assert lin.loc[lin[P.fold] == 0, P.r_lo].isna().all() and lin.loc[lin[P.fold] > 0, P.r_lo].notna().all()
    assert (lin.loc[lin[P.fold] > 0, P.r_lo] < lin.loc[lin[P.fold] > 0, P.r_hat]).all()
    # low-confidence events are predicted (kept in the table) but never trained on: they are in the test rows
    assert (preds[E.t0_confidence] < settings.min_t0_confidence).any()

    res = summary["results"]["post_30m"]
    cell = res["linear"]["subsets"]["all"]
    assert cell["n"] > 100 and cell["n_direction"] <= cell["n"]
    for key in ("accuracy", "balanced_accuracy", "brier", "log_loss", "spearman_ic", "mae", "rmse"):
        assert key in cell and cell[key] is not None
    assert cell["spearman_ic"] > 0.3  # the synthetic signal is recoverable
    assert set(cell["mde"]) == {"accuracy", "brier"} and cell["mde"]["brier"] > 0
    assert cell["ci"]["brier"][0] <= cell["brier"] <= cell["ci"]["brier"][1]
    cmp = cell["comparison"]["brier"]
    assert cmp["baseline"] in ("zero", "base_rate") and cmp["mde"] == cell["mde"]["brier"]
    assert cmp["ci"][0] <= cmp["improvement"] <= cmp["ci"][1] and 0 <= cmp["p_noise"] <= 1
    assert cmp["verdict"] in ("improves", "not_predictable", "worse") or cmp["verdict"].startswith("inconclusive at n = ")
    assert summary["best_baseline"]["post_30m"]["all"]["brier"]["model"] == cmp["baseline"]
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
    assert tr["untraded_reasons"].get("no_bars", 0) == int(lin[P.r_true].isna().sum())  # no path without targets
    assert res["linear"]["trading"]["magnitude_gate"]["n_trades"] < tr["n_trades"]
    assert res["base_rate"]["trading"]["fixed"]["comparison"] is None

    md = (out / "leaderboard.md").read_text()
    assert "## post_30m" in md and "| linear |" in md and "MDE" in md and summary["run_id"] in md
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


# ---- final: holdout scoring ----------------------------------------------------------------------------
def _clean_holdout(dataset: pd.DataFrame) -> pd.DataFrame:
    """Holdout rows with complete targets (drop the target_missing events of that season)."""
    drop = dataset[(dataset["season"] == "2026Q3") & dataset["target_missing"]][D.event_id].unique()
    return dataset[~dataset[D.event_id].isin(drop)].reset_index(drop=True)


def test_final_refuses_an_open_holdout_season(settings, dataset):
    clean = _clean_holdout(dataset)
    paths = make_paths(clean)
    with pytest.raises(ev.HoldoutNotReady, match="future"):
        ev.evaluate(settings, clean, model_names=["zero"], decision_times=["post_30m"], final=True, paths=paths,
                    now=pd.Timestamp("2026-08-01", tz=UTC), n_boot=20)
    assert not settings.holdout_log_path.exists() and not any(settings.reports_dir.iterdir())
    # every holdout event closed, but one has missing targets
    missing = clean.copy()
    first_holdout = missing[missing["season"] == "2026Q3"].index[:2]
    missing.loc[first_holdout, "target_missing"] = True
    missing.loc[first_holdout, [T.r("24h"), T.direction]] = np.nan
    with pytest.raises(ev.HoldoutNotReady, match="missing or pending"):
        ev.evaluate(settings, missing, model_names=["zero"], decision_times=["post_30m"], final=True,
                    paths=make_paths(missing), now=pd.Timestamp("2026-12-01", tz=UTC), n_boot=20)
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
    clean = _clean_holdout(dataset)
    paths, funding = make_paths(clean), make_funding()
    now = pd.Timestamp("2026-12-01", tz=UTC)
    summary = ev.evaluate(settings, clean, model_names=MODELS, decision_times=["post_30m"], final=True, paths=paths,
                          funding=funding, n_boot=30, now=now)
    n_holdout = clean[(clean["season"] == "2026Q3") & (clean[D.decision_time] == "post_30m")][D.event_id].nunique()
    assert summary["final"] is True and summary["holdout"]["scorings_before"] == 0
    assert summary["holdout"]["scorings_after"] == 1 and summary["holdout"]["scored_now"] is True
    hold = summary["holdout_results"]["post_30m"]["models"]
    assert hold["linear"]["subsets"]["all"]["n"] == n_holdout
    assert hold["linear"]["subsets"]["all"]["comparison"]["brier"]["baseline"] in ("zero", "base_rate")
    # the walk-forward cells never contain the holdout season
    assert "2026Q3" not in [f["test_season"] for f in summary["folds"]["post_30m"]]
    preds = pd.read_parquet(settings.reports_dir / summary["run_id"] / "predictions.parquet")
    hp = preds[preds[P.fold] == ev.HOLDOUT_FOLD]
    assert set(hp[P.test_season]) == {"2026Q3"} and len(hp) == n_holdout * len(MODELS)
    assert (preds.loc[preds[P.fold] >= 0, P.test_season] != "2026Q3").all()
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
    assert (out / "fake_model.json").exists()
    assert meta["decision_time"] == "post_30m" and meta["model"] == "linear" and meta["target"] == "r_24h"
    assert len(meta["dataset_sha256"]) == 64 and meta["git_sha"] and len(meta["config_hash"]) == 64
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
