"""Classification / regression metrics, calibration, residual bands, MDE and bootstrap.

Conventions: direction labels are in {-1, +1}; zero labels are dropped. A probability of
exactly 0.5 carries no direction, so it is scored as a coin flip (0.5 credit) in accuracy and
balanced accuracy; that makes the `zero` baseline sit at exactly 0.5 instead of at the up-rate.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

EPS = 1e-6
MDE_METRICS = ("brier", "accuracy")
# A block bootstrap needs at least this many distinct blocks: with fewer, every replicate
# redraws (almost) the same blocks and the interval collapses onto the point estimate (one
# season, as in the holdout block, gives CI == [point, point] and p_noise in {0, 1}).
MIN_BLOCKS = 5


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


def _z_total(alpha: float, power: float) -> float:
    """z_{1-alpha/2} + z_{power}: the multiplier of a standard error that gives the smallest
    effect a two-sided test at `alpha` detects with probability `power` (normal approximation)."""
    from scipy.stats import norm

    return float(norm.ppf(1 - alpha / 2) + norm.ppf(power))


def paired_se(diff) -> float:
    """Standard error of the mean of per-event score differences (model minus baseline on the
    same events): sd(diff, ddof=1) / sqrt(n) over the finite entries; NaN with fewer than 2."""
    d = _floats(diff)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return math.nan
    return float(np.std(d, ddof=1) / math.sqrt(len(d)))


def paired_mde(se: float, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable improvement of a paired comparison from its own standard error:
    (z_{1-alpha/2} + z_{power}) * se. This is the MDE of the test the report actually runs
    (the bootstrap interval of the mean paired difference), so it is the number a
    'not_predictable' verdict must be measured against. NaN when `se` is not finite."""
    if se is None or not np.isfinite(se):
        return math.nan
    return _z_total(alpha, power) * float(se)


def min_detectable_improvement(n: int, metric: str, base_value: float, *, alpha: float = 0.05,
                               power: float = 0.8) -> float:
    """Closed-form UPPER BOUND on the smallest improvement in `metric` ('brier' or 'accuracy')
    over `base_value` detectable at sample size n with a two-sided test at `alpha` and the given
    power (normal approximation). It is the fallback for cells that have no paired comparison
    (baselines, or no baseline present); wherever a comparison exists `paired_mde` of its own
    standard error is the honest number and is typically about half of this bound for Brier.

    Formula: MDE = (z_{1-alpha/2} + z_{power}) * sqrt(v * (1 - v) / n) with v = base_value
    clipped to [1e-3, 1 - 1e-3]. For accuracy this is the exact standard error of an unpaired
    Bernoulli hit indicator at rate v; for Brier it uses the Bhatia-Davis bound
    Var(s) <= E[s](1 - E[s]) for a per-event score s = (p - y)^2 in [0, 1]. Both ignore the
    pairing, which removes the between-event variance shared by model and baseline, so the
    bound is conservative. Monotone decreasing in n; NaN when n <= 0."""
    if metric not in MDE_METRICS:
        raise ValueError(f"metric must be one of {MDE_METRICS}, got {metric!r}")
    if n is None or n <= 0 or base_value is None or not np.isfinite(base_value):
        return math.nan
    v = float(np.clip(base_value, 1e-3, 1 - 1e-3))
    return _z_total(alpha, power) * math.sqrt(v * (1 - v) / n)


def _block_array(values: pd.Series, block) -> np.ndarray:
    if isinstance(block, pd.Series) and block.index.equals(values.index):
        arr = block.to_numpy()
    else:
        arr = np.asarray(block)
    if len(arr) != len(values):
        raise ValueError(f"block has {len(arr)} entries for {len(values)} values")
    return arr


def n_distinct_blocks(block: pd.Series | np.ndarray | None) -> int:
    """Number of distinct block labels (0 for None / empty)."""
    if block is None:
        return 0
    arr = block.to_numpy() if isinstance(block, pd.Series) else np.asarray(block)
    return int(len(np.unique(arr.astype(str)))) if len(arr) else 0


def choose_blocks(candidates: Sequence[tuple[str, pd.Series | None]], *,
                  min_blocks: int = MIN_BLOCKS) -> tuple[str, pd.Series | None]:
    """The resampling scheme for a bootstrap: the first `(name, labels)` candidate with at
    least `min_blocks` distinct labels, else `('iid', None)`. Callers list candidates from the
    coarsest dependence structure to the finest, e.g. season blocks first and UTC day of t0
    second, so a block that holds a single season (the holdout) still gets same-day
    dependence respected instead of a degenerate interval."""
    for name, labels in candidates:
        if n_distinct_blocks(labels) >= min_blocks:
            return name, labels
    return "iid", None


def bootstrap_distribution(values: pd.Series, stat: Callable[[pd.Series], float], *, n: int = 2000,
                           block: pd.Series | None = None, seed: int = 7,
                           min_blocks: int = MIN_BLOCKS) -> np.ndarray:
    """Bootstrap replicates of `stat`: iid resampling of rows, or block resampling (whole
    blocks drawn with replacement, as many as there are blocks) when `block` labels are given
    AND at least `min_blocks` distinct labels exist. With fewer blocks the block bootstrap is
    degenerate (a single block reproduces the sample in every replicate, so the interval has
    zero width), so rows are resampled iid instead; use `choose_blocks` to pick a finer block
    structure first and to record which scheme was used. Replicates are passed to `stat` as
    plain RangeIndex Series (values only)."""
    v = values if isinstance(values, pd.Series) else pd.Series(values)
    arr = v.to_numpy()
    if len(arr) == 0 or n <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=float)
    if block is not None:
        labels = _block_array(v, block)
        keys, inverse = np.unique(labels.astype(str), return_inverse=True)
        if len(keys) >= min_blocks:
            groups = [np.flatnonzero(inverse == k) for k in range(len(keys))]
            draws = rng.integers(0, len(groups), size=(n, len(groups)))
            for i in range(n):
                pos = np.concatenate([groups[g] for g in draws[i]])
                out[i] = stat(pd.Series(arr[pos]))
            return out
    draws = rng.integers(0, len(arr), size=(n, len(arr)))
    for i in range(n):
        out[i] = stat(pd.Series(arr[draws[i]]))
    return out


def bootstrap_ci(values: pd.Series, stat: Callable[[pd.Series], float], *, n: int = 2000,
                 block: pd.Series | None = None, seed: int = 7,
                 min_blocks: int = MIN_BLOCKS) -> tuple[float, float, float]:
    """(point, lo95, hi95); block bootstrap when `block` is given and has at least
    `min_blocks` distinct labels, iid rows otherwise (see bootstrap_distribution)."""
    v = values if isinstance(values, pd.Series) else pd.Series(values)
    point = float(stat(v)) if len(v) else math.nan
    dist = bootstrap_distribution(v, stat, n=n, block=block, seed=seed, min_blocks=min_blocks)
    if len(dist) == 0 or not np.isfinite(dist).any():
        return (point, math.nan, math.nan)
    lo, hi = np.nanpercentile(dist, [2.5, 97.5])
    return (point, float(lo), float(hi))
