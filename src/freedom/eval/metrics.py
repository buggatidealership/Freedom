"""Classification / regression metrics, calibration, residual bands, MDE and bootstrap.

Conventions: direction labels are in {-1, +1}; zero labels are dropped. A probability of
exactly 0.5 carries no direction, so it is scored as a coin flip (0.5 credit) in accuracy and
balanced accuracy; that makes the `zero` baseline sit at exactly 0.5 instead of at the up-rate.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

EPS = 1e-6
MDE_METRICS = ("brier", "accuracy")


def _floats(a) -> np.ndarray:
    if isinstance(a, np.ndarray) and a.dtype.kind == "f":
        return a
    return pd.Series(a, dtype="float64").to_numpy(dtype=float)


def _pairs(a, b, *, drop_zero_b: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x, y = _floats(a), _floats(b)
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    m = np.isfinite(x) & np.isfinite(y)
    if drop_zero_b:
        m &= y != 0
    return x[m], y[m]


def hit_scores(p_up, y_dir) -> np.ndarray:
    """Per-event accuracy credit: 1 right, 0 wrong, 0.5 when p_up == 0.5 (no direction).
    NaN where either input is missing or the label is 0."""
    p = np.asarray(pd.Series(p_up, dtype="float64"), dtype=float)
    y = np.asarray(pd.Series(y_dir, dtype="float64"), dtype=float)
    out = np.full(len(p), np.nan)
    m = np.isfinite(p) & np.isfinite(y) & (y != 0)
    pred = np.sign(p[m] - 0.5)
    out[m] = np.where(pred == 0, 0.5, (pred == np.sign(y[m])).astype(float))
    return out


def brier_scores(p_up, y_dir) -> np.ndarray:
    """Per-event squared error (p_up - 1[y > 0])^2; NaN where missing or the label is 0."""
    p = np.asarray(pd.Series(p_up, dtype="float64"), dtype=float)
    y = np.asarray(pd.Series(y_dir, dtype="float64"), dtype=float)
    out = np.full(len(p), np.nan)
    m = np.isfinite(p) & np.isfinite(y) & (y != 0)
    out[m] = (p[m] - (y[m] > 0).astype(float)) ** 2
    return out


def classification_metrics(p_up, y_dir) -> dict[str, float]:
    """accuracy, balanced_accuracy, brier, log_loss, n; y_dir in {-1, +1}; zero labels dropped."""
    p, y = _pairs(p_up, y_dir, drop_zero_b=True)
    n = int(len(p))
    if n == 0:
        return {"accuracy": math.nan, "balanced_accuracy": math.nan, "brier": math.nan,
                "log_loss": math.nan, "n": 0}
    yb = (y > 0).astype(float)
    credit = hit_scores(p, y)
    recalls = [float(np.mean(credit[yb == cls])) for cls in (0.0, 1.0) if np.any(yb == cls)]
    pc = np.clip(p, EPS, 1 - EPS)
    ll = -np.mean(yb * np.log(pc) + (1 - yb) * np.log(1 - pc))
    return {"accuracy": float(np.mean(credit)), "balanced_accuracy": float(np.mean(recalls)),
            "brier": float(np.mean((p - yb) ** 2)), "log_loss": float(ll), "n": n}


def spearman(a, b) -> float:
    """Spearman rank correlation; NaN with fewer than 3 pairs or a constant input."""
    x, y = _pairs(a, b)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    from scipy.stats import rankdata

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = np.corrcoef(rankdata(x), rankdata(y))[0, 1]  # average ranks for ties, as spearmanr
    return float(rho) if np.isfinite(rho) else math.nan


def regression_metrics(r_hat, r_true) -> dict[str, float]:
    """mae, rmse, spearman_ic, n."""
    x, y = _pairs(r_hat, r_true)
    n = int(len(x))
    if n == 0:
        return {"mae": math.nan, "rmse": math.nan, "spearman_ic": math.nan, "n": 0}
    err = y - x
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(math.sqrt(np.mean(err ** 2))),
            "spearman_ic": spearman(x, y), "n": n}


def calibration_table(p_up, y_dir, bins: int = 10) -> pd.DataFrame:
    """Reliability table over equal-width probability bins (deciles by default): one row per
    bin with p_lo, p_hi, n, mean_p, frac_up and gap = frac_up - mean_p. Empty bins keep n = 0."""
    p, y = _pairs(p_up, y_dir, drop_zero_b=True)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.floor(p * bins).astype(int), 0, bins - 1)
    up = (y > 0).astype(float)
    rows = []
    for b in range(bins):
        m = which == b
        n = int(m.sum())
        mean_p = float(p[m].mean()) if n else math.nan
        frac = float(up[m].mean()) if n else math.nan
        rows.append({"bin": b, "p_lo": float(edges[b]), "p_hi": float(edges[b + 1]), "n": n,
                     "mean_p": mean_p, "frac_up": frac, "gap": frac - mean_p if n else math.nan})
    return pd.DataFrame(rows, columns=["bin", "p_lo", "p_hi", "n", "mean_p", "frac_up", "gap"])


def residual_band(r_hat, r_true, lo: float = 0.1, hi: float = 0.9) -> tuple[float, float]:
    """Percentiles of r_true - r_hat over out-of-sample predictions."""
    x, y = _pairs(r_hat, r_true)
    if len(x) == 0:
        return (math.nan, math.nan)
    res = y - x
    return (float(np.quantile(res, lo)), float(np.quantile(res, hi)))


def min_detectable_improvement(n: int, metric: str, base_value: float, *, alpha: float = 0.05,
                               power: float = 0.8) -> float:
    """Smallest improvement in `metric` ('brier' or 'accuracy') over `base_value` detectable at
    sample size n with a two-sided test at `alpha` and the given power (normal approximation).

    Formula: MDE = (z_{1-alpha/2} + z_{power}) * sqrt(v * (1 - v) / n) with v = base_value
    clipped to [1e-3, 1 - 1e-3]. For accuracy this is the exact standard error of a Bernoulli
    hit indicator at rate v; for Brier it uses the Bhatia-Davis bound Var(s) <= E[s](1 - E[s])
    for a per-event score s = (p - y)^2 in [0, 1], so the Brier MDE is conservative (larger
    than the true detectable difference). Monotone decreasing in n; NaN when n <= 0."""
    if metric not in MDE_METRICS:
        raise ValueError(f"metric must be one of {MDE_METRICS}, got {metric!r}")
    if n is None or n <= 0 or base_value is None or not np.isfinite(base_value):
        return math.nan
    from scipy.stats import norm

    z = float(norm.ppf(1 - alpha / 2) + norm.ppf(power))
    v = float(np.clip(base_value, 1e-3, 1 - 1e-3))
    return z * math.sqrt(v * (1 - v) / n)


def _block_array(values: pd.Series, block) -> np.ndarray:
    if isinstance(block, pd.Series) and block.index.equals(values.index):
        arr = block.to_numpy()
    else:
        arr = np.asarray(block)
    if len(arr) != len(values):
        raise ValueError(f"block has {len(arr)} entries for {len(values)} values")
    return arr


def bootstrap_distribution(values: pd.Series, stat: Callable[[pd.Series], float], *, n: int = 2000,
                           block: pd.Series | None = None, seed: int = 7) -> np.ndarray:
    """Bootstrap replicates of `stat`: iid resampling of rows, or block resampling (whole
    blocks drawn with replacement, as many as there are blocks) when `block` labels are given.
    Replicates are passed to `stat` as plain RangeIndex Series (values only)."""
    v = values if isinstance(values, pd.Series) else pd.Series(values)
    arr = v.to_numpy()
    if len(arr) == 0 or n <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=float)
    if block is None:
        draws = rng.integers(0, len(arr), size=(n, len(arr)))
        for i in range(n):
            out[i] = stat(pd.Series(arr[draws[i]]))
        return out
    labels = _block_array(v, block)
    keys, inverse = np.unique(labels.astype(str), return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(len(keys))]
    draws = rng.integers(0, len(groups), size=(n, len(groups)))
    for i in range(n):
        pos = np.concatenate([groups[g] for g in draws[i]])
        out[i] = stat(pd.Series(arr[pos]))
    return out


def bootstrap_ci(values: pd.Series, stat: Callable[[pd.Series], float], *, n: int = 2000,
                 block: pd.Series | None = None, seed: int = 7) -> tuple[float, float, float]:
    """(point, lo95, hi95); block bootstrap by season when `block` is given."""
    v = values if isinstance(values, pd.Series) else pd.Series(values)
    point = float(stat(v)) if len(v) else math.nan
    dist = bootstrap_distribution(v, stat, n=n, block=block, seed=seed)
    if len(dist) == 0 or not np.isfinite(dist).any():
        return (point, math.nan, math.nan)
    lo, hi = np.nanpercentile(dist, [2.5, 97.5])
    return (point, float(lo), float(hi))
