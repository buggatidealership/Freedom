"""Models: registry, baseline semantics, NaN handling, determinism, small samples, persistence.

The synthetic dataset has a fixed seed and a known generating process:
r_24h = 0.4 * r_30m + 0.03 * x1 - 0.02 * x2 + noise, so the learners have a real signal to find
while the baselines' outputs can be recomputed by hand from the feature columns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from freedom.models import (
    MIN_TRAIN_ROWS,
    PROBA_EPS,
    BaseModel,
    SmallSampleWarning,
    available_models,
    clip_proba,
    feature_columns,
    make_model,
)
from freedom.models.baselines import VolScaled
from freedom.models.ensemble import Ensemble
from freedom.models.lgbm import LightGBMModel
from freedom.models.linear import LinearModel

EXPECTED_MODELS = {
    "zero", "base_rate", "historical_mean", "hist_abs_mean", "vol_scaled", "sign_of_reaction",
    "always_extends", "surprise_sign", "linear", "lightgbm", "ensemble",
}
LEARNERS = ["linear", "lightgbm", "ensemble"]
SEED = 11
N = 600
N_TRAIN = 450


def make_dataset(n: int = N, seed: int = SEED, *, post: bool = True):
    """(X, y_return, y_direction). X carries non-feature columns, NaNs with __missing
    companions, a constant column and an all-NaN column; `post=False` drops the reaction
    features (a pre-release decision time)."""
    rng = np.random.default_rng(seed)
    x1, x2, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    r15 = rng.normal(scale=0.02, size=n)
    r30 = r15 + rng.normal(scale=0.01, size=n)
    r24 = 0.4 * r30 + 0.03 * x1 - 0.02 * x2 + 0.015 * noise

    def holes(values, frac):
        return np.where(rng.random(n) < frac, np.nan, values)

    X = pd.DataFrame({
        "event_id": [f"E{i}" for i in range(n)],
        "t0": pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC"),
        "f_x1": holes(x1, 0.1),
        "f_x2": x2,
        "f_noise": rng.normal(size=n),
        "f_const": 1.0,
        "f_allnan": np.nan,
        "f_history_mean_r24h": holes(rng.normal(0.005, 0.02, n), 0.3),
        "f_history_mean_abs_r24h": holes(np.abs(rng.normal(0.05, 0.02, n)), 0.3),
        "f_pre_price_rvol_20d": holes(rng.lognormal(np.log(0.03), 0.3, n), 0.1),
        "f_surprise_eps_surprise_pct": holes(rng.normal(0, 10, n), 0.2),
    })
    if post:
        X["f_reaction_r_15m"] = holes(r15, 0.05)
        X["f_reaction_r_30m"] = holes(r30, 0.05)
    for c in feature_columns(X):
        X[c + "__missing"] = X[c].isna().astype(float)
    y_return = pd.Series(holes(r24, 0.05), name="r_24h")
    y_direction = np.sign(y_return).rename("direction_24h")
    return X, y_return, y_direction


@pytest.fixture(scope="module")
def data():
    return make_dataset()


@pytest.fixture(scope="module")
def split(data):
    X, y, d = data
    tr, te = slice(0, N_TRAIN), slice(N_TRAIN, N)
    return X.iloc[tr], y.iloc[tr], d.iloc[tr], X.iloc[te], y.iloc[te], d.iloc[te]


def _fitted(name, split, **params):
    Xtr, ytr, dtr, *_ = split
    return make_model(name, seed=SEED, **params).fit(Xtr, ytr, dtr)


def _base_rates(ytr, dtr):
    lab = dtr[dtr.notna() & (dtr != 0)]
    return float((lab > 0).mean()), float(ytr.mean()), float(ytr.abs().mean())


# ---- registry ---------------------------------------------------------------------------------
def test_registry_is_complete():
    assert set(available_models()) == EXPECTED_MODELS
    for name in EXPECTED_MODELS:
        m = make_model(name, seed=3)
        assert isinstance(m, BaseModel) and m.name == name and m.seed == 3
        assert m.feature_names_ == [] and m.residual_q_ is None


def test_make_model_aliases_and_unknown():
    assert isinstance(make_model("ridge"), LinearModel)
    assert isinstance(make_model("logistic"), LinearModel)
    assert "ridge" not in available_models()
    with pytest.raises(KeyError, match="unknown model"):
        make_model("nope")


def test_only_feature_columns_are_features(data):
    X, y, d = data
    m = make_model("zero").fit(X, y, d)
    assert m.feature_names_ == feature_columns(X)
    assert all(c.startswith("f_") for c in m.feature_names_)
    assert "event_id" not in m.feature_names_ and "t0" not in m.feature_names_
    assert "f_x1__missing" in m.feature_names_


def test_unfitted_model_raises(data):
    X, *_ = data
    with pytest.raises(RuntimeError, match="not fitted"):
        make_model("base_rate").predict_proba_up(X)


# ---- baseline semantics -------------------------------------------------------------------------
def test_zero_and_base_rate(split):
    Xtr, ytr, dtr, Xte, yte, dte = split
    z = _fitted("zero", split)
    assert np.all(z.predict_proba_up(Xte) == 0.5) and np.all(z.predict_return(Xte) == 0.0)
    assert np.all(z.predict_magnitude(Xte) == 0.0)
    up, mean_r, _ = _base_rates(ytr, dtr)
    b = _fitted("base_rate", split)
    assert b.predict_proba_up(Xte) == pytest.approx(np.full(len(Xte), up))
    assert b.predict_return(Xte) == pytest.approx(np.full(len(Xte), mean_r))
    assert b.n_train_ == int(ytr.notna().sum())


def test_historical_mean_reads_history_feature_with_pooled_fallback(split):
    Xtr, ytr, dtr, Xte, *_ = split
    m = _fitted("historical_mean", split)
    hist = Xte["f_history_mean_r24h"].to_numpy()
    has = np.isfinite(hist)
    assert has.any() and (~has).any()
    r = m.predict_return(Xte)
    p = m.predict_proba_up(Xte)
    up, mean_r, _ = _base_rates(ytr, dtr)
    assert r[has] == pytest.approx(hist[has])
    assert r[~has] == pytest.approx(np.full((~has).sum(), mean_r))
    assert p[has] == pytest.approx(0.5 + 0.25 * np.sign(hist[has]))
    assert p[~has] == pytest.approx(np.full((~has).sum(), up))


def test_hist_abs_mean_is_a_magnitude_baseline(split):
    Xtr, ytr, dtr, Xte, *_ = split
    m = _fitted("hist_abs_mean", split)
    feat = Xte["f_history_mean_abs_r24h"].to_numpy()
    has = np.isfinite(feat)
    mag = m.predict_magnitude(Xte)
    up, _, mean_abs = _base_rates(ytr, dtr)
    assert mag[has] == pytest.approx(feat[has])
    assert mag[~has] == pytest.approx(np.full((~has).sum(), mean_abs))
    assert np.all(m.predict_return(Xte) == 0.0)
    assert m.predict_proba_up(Xte) == pytest.approx(np.full(len(Xte), up))


def test_vol_scaled_scales_with_rvol_and_calibrates_on_training_z(split):
    Xtr, ytr, dtr, Xte, *_ = split
    m: VolScaled = _fitted("vol_scaled", split)
    rvol_tr = Xtr["f_pre_price_rvol_20d"].to_numpy()
    both = np.isfinite(rvol_tr) & ytr.notna().to_numpy()
    z = ytr.to_numpy()[both] / rvol_tr[both]  # horizon 24h -> sigma_h = rvol
    assert m.z_abs_mean_ == pytest.approx(np.abs(z).mean())
    assert m.pooled_sigma_ == pytest.approx(rvol_tr[np.isfinite(rvol_tr)].mean())
    rvol = Xte["f_pre_price_rvol_20d"].to_numpy()
    has = np.isfinite(rvol)
    mag = m.predict_magnitude(Xte)
    assert mag[has] == pytest.approx(rvol[has] * np.abs(z).mean())
    assert mag[~has] == pytest.approx(np.full((~has).sum(), m.pooled_sigma_ * np.abs(z).mean()))
    assert m.predict_return(Xte)[has] == pytest.approx(rvol[has] * z.mean())
    q10, q90 = m.predict_quantile(Xte, 0.1), m.predict_quantile(Xte, 0.9)
    assert np.all(q10 < q90)
    with pytest.raises(KeyError):
        m.predict_quantile(Xte, 0.33)
    # the horizon-scaling constant cancels: magnitudes are calibrated on the training z
    m48 = make_model("vol_scaled", horizon_hours=48).fit(Xtr, ytr, dtr)
    assert m48.predict_magnitude(Xte) == pytest.approx(mag)


def test_sign_of_reaction_uses_longest_available_reaction(split):
    Xtr, ytr, dtr, Xte, *_ = split
    m = _fitted("sign_of_reaction", split)
    r30 = Xte["f_reaction_r_30m"].to_numpy()
    r15 = Xte["f_reaction_r_15m"].to_numpy()
    has30, only15 = np.isfinite(r30), ~np.isfinite(r30) & np.isfinite(r15)
    none = ~np.isfinite(r30) & ~np.isfinite(r15)
    assert has30.any() and only15.any()
    p, r = m.predict_proba_up(Xte), m.predict_return(Xte)
    assert p[has30] == pytest.approx(0.5 + 0.25 * np.sign(r30[has30]))
    assert r[has30] == pytest.approx(r30[has30])
    assert p[only15] == pytest.approx(0.5 + 0.25 * np.sign(r15[only15]))
    assert r[only15] == pytest.approx(r15[only15])
    up, mean_r, _ = _base_rates(ytr, dtr)
    if none.any():
        assert p[none] == pytest.approx(np.full(none.sum(), up))
        assert r[none] == pytest.approx(np.full(none.sum(), mean_r))
    # a pre-release frame has no reaction at all: pure base rate
    Xpre, ypre, dpre = make_dataset(post=False)
    pre = make_model("sign_of_reaction").fit(Xpre, ypre, dpre)
    assert np.all(pre.predict_proba_up(Xpre) == pre.up_rate_)
    assert np.all(pre.predict_return(Xpre) == pre.mean_return_)


def test_always_extends_pushes_the_reaction_further(split):
    Xtr, ytr, dtr, Xte, *_ = split
    m = _fitted("always_extends", split)
    rk_tr = Xtr["f_reaction_r_30m"].fillna(Xtr["f_reaction_r_15m"]).to_numpy()
    both = np.isfinite(rk_tr) & ytr.notna().to_numpy()
    assert m.extension_ == pytest.approx(np.abs(ytr.to_numpy()[both] - rk_tr[both]).mean())
    assert m.extension_ > 0
    rk = Xte["f_reaction_r_30m"].fillna(Xte["f_reaction_r_15m"]).to_numpy()
    has = np.isfinite(rk) & (rk != 0)
    r = m.predict_return(Xte)
    assert r[has] == pytest.approx(rk[has] + np.sign(rk[has]) * m.extension_)
    assert np.all(np.sign(r[has]) == np.sign(rk[has])) and np.all(np.abs(r[has]) > np.abs(rk[has]))
    assert m.predict_proba_up(Xte)[has] == pytest.approx(0.5 + 0.25 * np.sign(rk[has]))


def test_surprise_sign(split):
    Xtr, ytr, dtr, Xte, *_ = split
    m = _fitted("surprise_sign", split)
    s = Xte["f_surprise_eps_surprise_pct"].to_numpy()
    has = np.isfinite(s)
    up, mean_r, mean_abs = _base_rates(ytr, dtr)
    p, r = m.predict_proba_up(Xte), m.predict_return(Xte)
    assert p[has] == pytest.approx(0.5 + 0.25 * np.sign(s[has]))
    assert r[has] == pytest.approx(np.sign(s[has]) * mean_abs)
    assert p[~has] == pytest.approx(np.full((~has).sum(), up))
    assert r[~has] == pytest.approx(np.full((~has).sum(), mean_r))


ALTERNATE_NAMES = {  # the flat f_<key> names that features.build_features writes
    "f_history_mean_r24h": "f_hist_r24_mean",
    "f_history_mean_abs_r24h": "f_hist_abs_r24_mean",
    "f_pre_price_rvol_20d": "f_rvol_20d",
    "f_reaction_r_30m": "f_r_now",
    "f_reaction_r_15m": "f_r_15m",
    "f_surprise_eps_surprise_pct": "f_eps_surprise",
}


@pytest.mark.parametrize("name", ["historical_mean", "hist_abs_mean", "vol_scaled",
                                  "sign_of_reaction", "always_extends", "surprise_sign"])
def test_feature_baselines_accept_either_naming_convention(name, split):
    Xtr, ytr, dtr, Xte, *_ = split
    rename = {**ALTERNATE_NAMES, **{k + "__missing": v + "__missing" for k, v in ALTERNATE_NAMES.items()}}
    a = make_model(name).fit(Xtr, ytr, dtr)
    b = make_model(name).fit(Xtr.rename(columns=rename), ytr, dtr)
    Xalt = Xte.rename(columns=rename)
    assert np.array_equal(a.predict_proba_up(Xte), b.predict_proba_up(Xalt))
    assert np.array_equal(a.predict_return(Xte), b.predict_return(Xalt))
    assert np.array_equal(a.predict_magnitude(Xte), b.predict_magnitude(Xalt))
    # an explicit single column name is accepted too
    c = make_model(name, features=list(ALTERNATE_NAMES.values())[0]).fit(Xtr, ytr, dtr)
    assert c.features == ("f_hist_r24_mean",)


def test_sign_baseline_with_zero_signal_is_neutral():
    X = pd.DataFrame({"f_surprise_eps_surprise_pct": [0.0, 2.0, -3.0, np.nan]})
    y = pd.Series([0.01, 0.02, -0.03, 0.04])
    m = make_model("surprise_sign").fit(X, y, np.sign(y))
    assert m.predict_proba_up(X)[0] == 0.5 and m.predict_return(X)[0] == 0.0


# ---- learners ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name", LEARNERS)
def test_learners_tolerate_nan_constant_and_column_changes(name, split):
    Xtr, ytr, dtr, Xte, *_ = split
    assert Xtr["f_allnan"].isna().all() and Xtr["f_const"].nunique() == 1 and Xtr["f_x1"].isna().any()
    m = _fitted(name, split)
    p, r, mag = m.predict_proba_up(Xte), m.predict_return(Xte), m.predict_magnitude(Xte)
    assert np.isfinite(p).all() and np.isfinite(r).all() and np.isfinite(mag).all()
    assert np.all(p >= PROBA_EPS) and np.all(p <= 1 - PROBA_EPS)
    if name == "ensemble":  # mean of the members' magnitudes >= |mean r_hat| (triangle inequality)
        assert np.all(mag >= np.abs(r) - 1e-12)
    else:
        assert mag == pytest.approx(np.abs(r))
    # column order and extra columns do not matter
    shuffled = Xte.iloc[:, ::-1].copy()
    shuffled["f_new_feature"] = 1.0
    shuffled["not_a_feature"] = "x"
    assert np.array_equal(m.predict_proba_up(shuffled), p)
    assert np.array_equal(m.predict_return(shuffled), r)
    # a missing feature column becomes NaN and is handled like any other missing value
    dropped = Xte.drop(columns=["f_x1", "f_x1__missing"])
    assert np.isfinite(m.predict_return(dropped)).all()
    assert np.isfinite(m.predict_proba_up(dropped)).all()
    imp = m.feature_importance()
    assert imp is not None and list(imp.index) == m.feature_names_
    assert imp["f_allnan"] == 0.0 and imp["f_const"] == 0.0
    assert imp.idxmax() in {"f_x1", "f_x2"}


@pytest.mark.parametrize("name", LEARNERS)
def test_learners_beat_zero_on_a_real_signal(name, split):
    Xtr, ytr, dtr, Xte, yte, dte = split
    ok = yte.notna().to_numpy()
    y_true, up_true = yte.to_numpy()[ok], (dte.to_numpy()[ok] > 0)
    m = _fitted(name, split)
    r, p = m.predict_return(Xte)[ok], m.predict_proba_up(Xte)[ok]
    zero = _fitted("zero", split)
    mae_zero = np.abs(zero.predict_return(Xte)[ok] - y_true).mean()
    brier_zero = ((zero.predict_proba_up(Xte)[ok] - up_true) ** 2).mean()
    assert brier_zero == pytest.approx(0.25)
    assert np.abs(r - y_true).mean() < 0.6 * mae_zero
    assert ((p - up_true) ** 2).mean() < 0.6 * brier_zero
    assert ((p > 0.5) == up_true).mean() > 0.75


@pytest.mark.parametrize("name", LEARNERS)
def test_learners_are_deterministic_given_seed(name, split):
    Xtr, ytr, dtr, Xte, *_ = split
    a = make_model(name, seed=5).fit(Xtr, ytr, dtr)
    b = make_model(name, seed=5).fit(Xtr, ytr, dtr)
    assert np.array_equal(a.predict_proba_up(Xte), b.predict_proba_up(Xte))
    assert np.array_equal(a.predict_return(Xte), b.predict_return(Xte))


@pytest.mark.parametrize("name", ["linear", "lightgbm"])
def test_tiny_sample_falls_back_to_base_rate_with_warning(name, data):
    X, y, d = data
    n = MIN_TRAIN_ROWS - 10
    with pytest.warns(SmallSampleWarning):
        m = make_model(name).fit(X.iloc[:n], y.iloc[:n], d.iloc[:n])
    ref = make_model("base_rate").fit(X.iloc[:n], y.iloc[:n], d.iloc[:n])
    assert np.array_equal(m.predict_proba_up(X), ref.predict_proba_up(X))
    assert np.array_equal(m.predict_return(X), ref.predict_return(X))
    assert m.feature_importance() is None


@pytest.mark.parametrize("name", ["linear", "lightgbm"])
def test_single_class_direction_falls_back_for_that_head_only(name, split):
    Xtr, ytr, dtr, Xte, *_ = split
    with pytest.warns(SmallSampleWarning):
        m = make_model(name).fit(Xtr, ytr, pd.Series(np.ones(len(Xtr))))
    assert np.all(m.predict_proba_up(Xte) == 1 - PROBA_EPS)  # clipped up rate of 1.0
    assert np.std(m.predict_return(Xte)) > 0  # the return head still learned


@pytest.mark.parametrize("name", ["linear", "lightgbm"])
def test_direction_is_derived_from_the_return_when_absent(name, split):
    Xtr, ytr, dtr, Xte, *_ = split
    with warnings.catch_warnings():
        warnings.simplefilter("error", SmallSampleWarning)
        a = make_model(name).fit(Xtr, ytr, None)
        b = make_model(name).fit(Xtr, ytr, dtr)
    assert np.array_equal(a.predict_proba_up(Xte), b.predict_proba_up(Xte))


def test_no_feature_columns_falls_back(split):
    Xtr, ytr, dtr, Xte, *_ = split
    with pytest.warns(SmallSampleWarning):
        m = make_model("linear").fit(Xtr[["event_id"]], ytr, dtr)
    assert np.all(m.predict_return(Xte) == m.mean_return_)


def test_contributions_sum_to_the_prediction(split):
    Xtr, ytr, dtr, Xte, *_ = split
    for name in ("linear", "lightgbm"):
        m = _fitted(name, split)
        c = m.contributions(Xte.head(20))
        assert list(c.columns) == [*m.feature_names_, "bias"]
        assert c.sum(axis=1).to_numpy() == pytest.approx(m.predict_return(Xte.head(20)), abs=1e-6)
        logit = m.contributions(Xte.head(20), head="direction").sum(axis=1).to_numpy()
        p = m.predict_proba_up(Xte.head(20))
        assert clip_proba(1 / (1 + np.exp(-logit))) == pytest.approx(p, abs=1e-6)


def test_lightgbm_enforces_small_n_caps():
    m = make_model("lightgbm", num_leaves=31, min_data_in_leaf=5)
    assert m.lgb_params["num_leaves"] == 7 and m.lgb_params["min_data_in_leaf"] == 20
    assert isinstance(m, LightGBMModel)


def test_ensemble_is_the_mean_of_its_members(split):
    Xtr, ytr, dtr, Xte, *_ = split
    members = [make_model("base_rate"), make_model("sign_of_reaction"), make_model("hist_abs_mean")]
    e = Ensemble(members=members).fit(Xtr, ytr, dtr)
    assert e.member_names == ["base_rate", "sign_of_reaction", "hist_abs_mean"]
    p = np.mean([m.predict_proba_up(Xte) for m in members], axis=0)
    r = np.mean([m.predict_return(Xte) for m in members], axis=0)
    mag = np.mean([m.predict_magnitude(Xte) for m in members], axis=0)
    assert e.predict_proba_up(Xte) == pytest.approx(p)
    assert e.predict_return(Xte) == pytest.approx(r)
    assert e.predict_magnitude(Xte) == pytest.approx(mag)  # not |mean r_hat|
    w = Ensemble(members=["base_rate", "zero"], weights=(3, 1)).fit(Xtr, ytr, dtr)
    assert w.predict_proba_up(Xte) == pytest.approx(0.75 * w.members_[0].up_rate_ + 0.25 * 0.5)
    default = make_model("ensemble", member_params={"linear": {"cv_folds": 3}})
    assert default.member_names == ["linear", "lightgbm"] and default.members_[0].cv_folds == 3
    with pytest.raises(ValueError):
        Ensemble(members=[])
    with pytest.raises(ValueError):
        Ensemble(members=["zero"], weights=(1, 1))


# ---- persistence --------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(EXPECTED_MODELS))
def test_save_load_round_trip(name, split, tmp_path):
    Xtr, ytr, dtr, Xte, *_ = split
    m = _fitted(name, split)
    m.residual_q_ = (-0.0412, 0.0377)
    p, r, mag = m.predict_proba_up(Xte), m.predict_return(Xte), m.predict_magnitude(Xte)
    # a directory: the model lands in <dir>/model.joblib, leaving room for model.json
    d = tmp_path / "post_30m" / name
    m.save(d)
    assert (d / "model.joblib").exists()
    loaded = BaseModel.load(d)
    assert type(loaded) is type(m) and loaded.name == name
    assert loaded.feature_names_ == m.feature_names_ and loaded.residual_q_ == (-0.0412, 0.0377)
    assert loaded.seed == m.seed and loaded.params == m.params
    assert np.array_equal(loaded.predict_proba_up(Xte), p)
    assert np.array_equal(loaded.predict_return(Xte), r)
    assert np.array_equal(loaded.predict_magnitude(Xte), mag)
    lo, hi = loaded.predict_band(Xte)
    assert lo == pytest.approx(r - 0.0412) and hi == pytest.approx(r + 0.0377)
    # an explicit file path works too, and the subclass loader type-checks
    f = tmp_path / f"{name}.joblib"
    m.save(f)
    assert type(m).load(f).feature_names_ == m.feature_names_


def test_load_with_the_wrong_class_raises(split, tmp_path):
    m = _fitted("zero", split)
    m.save(tmp_path / "zero.joblib")
    with pytest.raises(TypeError):
        LinearModel.load(tmp_path / "zero.joblib")


def test_predict_band_is_nan_until_eval_sets_residuals(split):
    Xtr, ytr, dtr, Xte, *_ = split
    m = _fitted("base_rate", split)
    lo, hi = m.predict_band(Xte)
    assert np.isnan(lo).all() and np.isnan(hi).all()
