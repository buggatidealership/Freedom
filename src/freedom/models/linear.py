"""`linear`: ridge for the return and L2 logistic regression for the direction.

Pipeline (learned in fit, applied identically at prediction time):
1. median imputation per feature from the training rows (an all-NaN column imputes to 0);
   the f_*__missing companions already in X carry the missingness itself;
2. standardisation (a constant column gets scale 1 and therefore a zero coefficient);
3. RidgeCV over `alphas` (efficient leave-one-out) for the return head and an explicit grid
   over `Cs` for the L2 logistic direction head (seeded stratified inner folds scored by
   log-loss, ties broken towards the strongest regularisation, refit on all rows). The grids
   are deliberately on the strong-regularisation side because N is a few hundred events at
   best. A scalar `alpha` / `C` (the names the optimizer tunes) is shorthand for a
   one-element grid, i.e. no inner selection; passing both the scalar and the grid, or any
   other keyword, raises TypeError so a misspelt hyper-parameter cannot be a silent no-op.

A head with fewer than MIN_TRAIN_ROWS usable rows, or a direction head with a single class,
falls back to the training base rate for that head (SmallSampleWarning).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold

from . import MIN_TRAIN_ROWS, BaseModel, clip_proba, register

DEFAULT_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)  # ridge alpha on standardised X
DEFAULT_CS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)  # logistic C = 1 / lambda


def _grid(name: str, scalar, grid, default) -> tuple[float, ...]:
    """The search grid for one head: `scalar` (-> a one-element grid) or `grid`, not both."""
    if scalar is not None:
        if grid is not default:
            raise TypeError(f"linear: pass either {name} or {name}s, not both")
        grid = (scalar,)
    out = tuple(float(v) for v in grid)
    if not out:
        raise ValueError(f"linear: {name}s must not be empty")
    return out


@register("linear")
class LinearModel(BaseModel):
    def __init__(self, *, seed: int = 7, alphas=DEFAULT_ALPHAS, Cs=DEFAULT_CS, cv_folds: int = 5,
                 **params):
        alpha, C = params.pop("alpha", None), params.pop("C", None)
        if params:
            raise TypeError(f"linear: unknown parameter(s) {sorted(params)}; "
                            "accepted: alpha or alphas, C or Cs, cv_folds")
        alphas = _grid("alpha", alpha, alphas, DEFAULT_ALPHAS)
        Cs = _grid("C", C, Cs, DEFAULT_CS)
        super().__init__(seed=seed, alphas=alphas, Cs=Cs, cv_folds=cv_folds)
        self.alphas = alphas
        self.Cs = Cs
        self.cv_folds = int(cv_folds)
        self.medians_: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.active_: np.ndarray | None = None
        self.ridge_: RidgeCV | None = None
        self.logit_: LogisticRegression | None = None
        self.alpha_: float | None = None
        self.C_: float | None = None

    # ---- preprocessing ----------------------------------------------------------------------
    def _transform(self, M: np.ndarray) -> np.ndarray:
        Z = np.where(np.isnan(M), self.medians_, M)
        return (Z - self.mu_) / self.sd_

    def _features(self, X: pd.DataFrame) -> np.ndarray:
        return self._transform(self._matrix(X))

    # ---- fit / predict ----------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series) -> BaseModel:
        M, r, d = self._fit_base_rates(X, y_return, y_direction)
        self.ridge_ = self.logit_ = None
        self.alpha_ = self.C_ = None
        if M.shape[1] == 0:
            self.medians_ = self.mu_ = self.sd_ = np.empty(0)
            self.active_ = np.empty(0, dtype=bool)
            self._small_sample("return", 0)
            self._small_sample("direction", 0)
            return self
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            med = np.nanmedian(M, axis=0)
        self.medians_ = np.where(np.isfinite(med), med, 0.0)
        Z0 = np.where(np.isnan(M), self.medians_, M)
        self.mu_ = Z0.mean(axis=0)
        sd = Z0.std(axis=0)
        self.active_ = sd > 0  # constant (or all-NaN) columns carry no information
        self.sd_ = np.where(self.active_, sd, 1.0)
        Z = (Z0 - self.mu_) / self.sd_

        ok = np.isfinite(r)
        if ok.sum() >= MIN_TRAIN_ROWS:
            self.ridge_ = RidgeCV(alphas=list(self.alphas)).fit(Z[ok], r[ok])  # a list: sklearn mutates a 1-grid
            self.alpha_ = float(self.ridge_.alpha_)
        else:
            self._small_sample("return", int(ok.sum()))

        lab = np.isfinite(d) & (d != 0)
        y = (d[lab] > 0).astype(int)
        n_minority = int(min(y.sum(), len(y) - y.sum())) if len(y) else 0
        if lab.sum() >= MIN_TRAIN_ROWS and n_minority >= 2:
            self.C_ = self._select_C(Z[lab], y, n_splits=int(min(self.cv_folds, n_minority)))
            self.logit_ = self._logistic(self.C_).fit(Z[lab], y)
        else:
            self._small_sample("direction", int(lab.sum()))
        return self

    @staticmethod
    def _logistic(C: float) -> LogisticRegression:
        return LogisticRegression(C=C, solver="lbfgs", max_iter=1000)

    def _select_C(self, Z: np.ndarray, y: np.ndarray, *, n_splits: int) -> float:
        """Inner-fold log-loss over the C grid; the smallest C (strongest L2) wins ties."""
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed).split(Z, y))
        best_C, best_loss = None, np.inf
        for C in sorted(self.Cs):
            losses = []
            for tr, va in folds:
                p = clip_proba(self._logistic(C).fit(Z[tr], y[tr]).predict_proba(Z[va])[:, 1])
                losses.append(-np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)))
            loss = float(np.mean(losses))
            if loss < best_loss - 1e-12:
                best_C, best_loss = C, loss
        return float(best_C)

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self.ridge_ is None or len(X) == 0:  # sklearn rejects 0-row input
            return self._base_return(len(X))
        return np.asarray(self.ridge_.predict(self._features(X)), dtype=float)

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self.logit_ is None or len(X) == 0:
            return self._base_proba(len(X))
        return clip_proba(self.logit_.predict_proba(self._features(X))[:, 1])

    # ---- introspection ------------------------------------------------------------------------
    def feature_importance(self) -> pd.Series | None:
        """Mean of the normalised |standardised coefficients| over the fitted heads."""
        parts = []
        for est in (self.ridge_, self.logit_):
            if est is None:
                continue
            w = np.abs(np.ravel(est.coef_)) * self.active_  # solver noise on constant columns -> 0
            parts.append(w / w.sum() if w.sum() > 0 else w)
        if not parts:
            return None
        return pd.Series(np.mean(parts, axis=0), index=self.feature_names_, name="importance")

    def contributions(self, X: pd.DataFrame, head: str = "return") -> pd.DataFrame:
        """Per-row additive contributions coef * z (plus a `bias` column) of the return head
        (sums to r_hat) or the direction head (sums to the logit of p_up)."""
        self._check_fitted()
        est = self.ridge_ if head == "return" else self.logit_
        if est is None:
            raise RuntimeError(f"{head} head fell back to the base rate; no contributions")
        Z = self._features(X)
        out = pd.DataFrame(Z * np.ravel(est.coef_), columns=self.feature_names_, index=X.index)
        out["bias"] = float(np.ravel(est.intercept_)[0])
        return out
