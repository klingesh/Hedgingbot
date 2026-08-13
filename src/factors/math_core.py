"""
The actual factor mathematics — weighted moments, Gram-Schmidt
orthogonalization, and beta estimation — in PURE STANDARD LIBRARY Python.

Why stdlib and not numpy
------------------------
This is the load-bearing calculation of the whole overlay: these numbers decide
whether the bot hedges 0.4 lots or 4 lots. It therefore needs to be verifiable
in any environment, including a bare container with no scientific stack, and it
needs to be readable line by line by a human checking the algebra.

`estimate.py` is a thin pandas adapter over this module — it extracts columns to
lists, calls in here, and packages the results back into DataFrames. There is
exactly ONE implementation of the maths; pandas is a convenience skin, never a
second code path.

Sample sizes here are small (a few thousand daily bars, a handful of series) and
this runs monthly, offline. Pure Python is comfortably fast enough.
"""

from __future__ import annotations

import math

Vector = list[float]

_EPS = 1e-18


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def normalized_weights(n: int, halflife: float | None = None) -> Vector:
    """Weights summing to 1 over n observations, OLDEST FIRST.

    halflife None or <= 0 gives equal weights. Otherwise weight decays
    exponentially into the past with the given half-life in observations, so the
    most recent point has weight 1 before normalization and a point `halflife`
    observations back has weight 0.5.
    """
    if n <= 0:
        return []
    if halflife is None or halflife <= 0:
        return [1.0 / n] * n

    raw = [0.5 ** ((n - 1 - i) / halflife) for i in range(n)]
    total = math.fsum(raw)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / n] * n
    return [w / total for w in raw]


def _eff_divisor(w: Vector) -> float:
    """1 - sum(w^2), the reliability correction.

    For uniform weights w=1/n this equals (n-1)/n, which makes the weighted
    variance reduce exactly to the unbiased sample variance sum((x-m)^2)/(n-1).
    """
    d = 1.0 - math.fsum(wi * wi for wi in w)
    return d if d > 1e-12 else 1e-12


# ---------------------------------------------------------------------------
# Weighted moments
# ---------------------------------------------------------------------------


def wmean(x: Vector, w: Vector) -> float:
    return math.fsum(wi * xi for wi, xi in zip(w, x))


def wcov(x: Vector, y: Vector, w: Vector) -> float:
    mx = wmean(x, w)
    my = wmean(y, w)
    return math.fsum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, x, y)) / _eff_divisor(w)


def wvar(x: Vector, w: Vector) -> float:
    return wcov(x, x, w)


def wstd(x: Vector, w: Vector) -> float:
    return math.sqrt(max(wvar(x, w), 0.0))


def wcorr(x: Vector, y: Vector, w: Vector) -> float:
    sx = wstd(x, w)
    sy = wstd(y, w)
    if sx <= _EPS or sy <= _EPS:
        return 0.0
    return wcov(x, y, w) / (sx * sy)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def log_return_series(prices: Vector) -> list[float | None]:
    """Log returns from prices. Element 0 is None (no prior price).

    A non-positive or non-finite price yields None for the affected returns
    rather than raising — bad ticks happen, and one of them must not destroy an
    entire study. Callers align on the non-None entries.
    """
    out: list[float | None] = [None]
    for prev, cur in zip(prices, prices[1:]):
        if (prev is None or cur is None
                or not math.isfinite(prev) or not math.isfinite(cur)
                or prev <= 0.0 or cur <= 0.0):
            out.append(None)
        else:
            out.append(math.log(cur / prev))
    return out


def winsorize(values: Vector, fraction: float) -> Vector:
    """Clip `fraction` from EACH tail, by quantile.

    Natural gas routinely prints 15% daily moves. Un-clipped, a handful of those
    days set the ENERGY betas for the whole sample. We want a typical
    relationship to size a hedge from, not a tail-fitted one.
    """
    if fraction <= 0.0 or not values:
        return list(values)
    n = len(values)
    ordered = sorted(values)
    lo_i = int(math.floor(fraction * (n - 1)))
    hi_i = int(math.ceil((1.0 - fraction) * (n - 1)))
    lo, hi = ordered[lo_i], ordered[hi_i]
    if lo > hi:
        lo, hi = hi, lo
    return [min(max(v, lo), hi) for v in values]


# ---------------------------------------------------------------------------
# Gram-Schmidt orthogonalization
# ---------------------------------------------------------------------------


class OrthogonalFactors:
    """Mutually orthogonal factor series built from correlated proxies.

    Attributes
    ----------
    names            : factor names, in orthogonalization order
    series           : {name: orthogonalized return series}
    sigma            : {name: weighted std of the orthogonalized series}
    variance_retained: {name: fraction of the proxy's variance surviving
                       residualization against the earlier factors}
    raw_correlation  : {(a, b): correlation of the RAW proxies}
    n_obs            : observation count
    """

    __slots__ = ("names", "series", "sigma", "variance_retained",
                 "raw_correlation", "n_obs", "weights", "halflife")

    def __init__(self, names, series, sigma, variance_retained,
                 raw_correlation, n_obs, weights, halflife):
        self.names = tuple(names)
        self.series = series
        self.sigma = sigma
        self.variance_retained = variance_retained
        self.raw_correlation = raw_correlation
        self.n_obs = n_obs
        self.weights = weights
        self.halflife = halflife

    def orthogonality_error(self) -> float:
        """Largest |correlation| between two distinct factors.

        Zero by construction. A non-trivial value means the algebra broke, so
        this is asserted in tests and reported in the CLI output.
        """
        worst = 0.0
        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                c = abs(wcorr(self.series[a], self.series[b], self.weights))
                worst = max(worst, c)
        return worst


def build_orthogonal_factors(
    proxies: dict[str, Vector],
    order: tuple[str, ...],
    halflife: float | None = None,
    rescale_to_proxy: bool = True,
) -> OrthogonalFactors:
    """Sequentially residualize proxy return series into orthogonal factors.

    order[0] is kept as-is. order[k] is residualized against order[0..k-1]:

        f_k = p_k - sum_{j<k} (cov(p_k, f_j) / var(f_j)) * f_j

    Because each f_j is already orthogonal to the others, the coefficients can
    be computed independently rather than by solving a system.

    rescale_to_proxy multiplies each residual back up to the standard deviation
    of its own raw proxy. Direction, and therefore orthogonality, is unchanged;
    it exists purely so "a 1% METALS move" stays comparable in magnitude to "a
    1% gold move" and the exposure report reads sensibly.

    All proxy series must be the same length with no None values — align and drop
    before calling. Raises ValueError otherwise.
    """
    missing = [f for f in order if f not in proxies]
    if missing:
        raise ValueError(f"proxies missing factor(s) {missing}; got {sorted(proxies)}")

    names = list(order)
    lengths = {len(proxies[f]) for f in names}
    if len(lengths) != 1:
        raise ValueError(f"proxy series have differing lengths: {lengths}")
    n = lengths.pop()
    if n < 30:
        raise ValueError(f"need at least 30 aligned observations, got {n}")
    for f in names:
        if any(v is None or not math.isfinite(v) for v in proxies[f]):
            raise ValueError(f"proxy {f!r} contains None/non-finite values; align first")

    w = normalized_weights(n, halflife)

    raw_corr: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = wcorr(proxies[a], proxies[b], w)
            raw_corr[(a, b)] = c
            raw_corr[(b, a)] = c

    series: dict[str, Vector] = {}
    sigma: dict[str, float] = {}
    retained: dict[str, float] = {}

    for k, f in enumerate(names):
        p = list(proxies[f])
        raw_v = wvar(p, w)

        v = p
        for prev in names[:k]:
            fj = series[prev]
            vj = wvar(fj, w)
            if vj <= _EPS:
                continue
            coef = wcov(v, fj, w) / vj
            v = [vi - coef * fji for vi, fji in zip(v, fj)]

        res_v = wvar(v, w)
        retained[f] = (res_v / raw_v) if raw_v > _EPS else 0.0

        if rescale_to_proxy and res_v > _EPS and raw_v > _EPS:
            scale = math.sqrt(raw_v / res_v)
            v = [vi * scale for vi in v]

        series[f] = v
        sigma[f] = wstd(v, w)

    return OrthogonalFactors(
        names=names, series=series, sigma=sigma, variance_retained=retained,
        raw_correlation=raw_corr, n_obs=n, weights=w, halflife=halflife,
    )


# ---------------------------------------------------------------------------
# Beta estimation
# ---------------------------------------------------------------------------


class InstrumentFit:
    """Regression result for one instrument against the orthogonal factors."""

    __slots__ = ("betas", "r_squared", "sigma", "resid_sigma", "n_obs")

    def __init__(self, betas, r_squared, sigma, resid_sigma, n_obs):
        self.betas = betas
        self.r_squared = r_squared
        self.sigma = sigma
        self.resid_sigma = resid_sigma
        self.n_obs = n_obs


def fit_instrument(
    y: Vector,
    factors: OrthogonalFactors,
    halflife: float | None = None,
    factor_subset: dict[str, Vector] | None = None,
) -> InstrumentFit:
    """Regress one aligned return series on the orthogonal factors.

    Because the factors are mutually orthogonal, the multivariate OLS
    coefficient for factor f collapses to the univariate cov(y, f) / var(f).
    That is not an approximation — it is the payoff from orthogonalizing, and it
    also makes R^2 a clean sum of per-factor contributions:

        R^2 = sum_f  beta_f^2 * var(f) / var(y)

    `y` and the factor series must already be aligned and the same length.
    """
    ser = factor_subset if factor_subset is not None else factors.series
    n = len(y)
    for f in factors.names:
        if len(ser[f]) != n:
            raise ValueError(
                f"length mismatch: y has {n}, factor {f!r} has {len(ser[f])}"
            )

    w = normalized_weights(n, halflife)
    total_var = wvar(y, w)

    betas: dict[str, float] = {}
    explained = 0.0
    for f in factors.names:
        x = ser[f]
        vx = wvar(x, w)
        b = (wcov(y, x, w) / vx) if vx > _EPS else 0.0
        betas[f] = b
        explained += b * b * vx

    r2 = (explained / total_var) if total_var > _EPS else 0.0
    # Sampling noise can push explained marginally above total; clamp so
    # resid_sigma never goes imaginary.
    resid_var = max(total_var - explained, 0.0)

    return InstrumentFit(
        betas=betas,
        r_squared=r2,
        sigma=math.sqrt(max(total_var, 0.0)),
        resid_sigma=math.sqrt(resid_var),
        n_obs=n,
    )


def purity(
    betas: dict[str, float],
    factor: str,
    factor_sigma: dict[str, float] | None = None,
) -> float:
    """Share of an instrument's EXPLAINED variance coming from one factor.

        purity_f = (beta_f * sigma_f)^2 / sum_g (beta_g * sigma_g)^2

    Range 0..1. This is what makes an instrument a good single-factor hedge:
    1.0 means neutralizing that factor with it creates no other exposure.

    Variance weighting matters. EURUSD might have beta 1.0 to USD and 0.1 to
    RISK, but if RISK is three times as volatile then that "small" 0.1 carries
    real risk. Without factor_sigma this degrades to a beta-only approximation.
    """
    if factor_sigma is None:
        contrib = {f: b * b for f, b in betas.items()}
    else:
        contrib = {
            f: (b * factor_sigma.get(f, 0.0)) ** 2 for f, b in betas.items()
        }
    denom = math.fsum(contrib.values())
    if denom <= _EPS:
        return 0.0
    return contrib.get(factor, 0.0) / denom


# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------


def align(series: dict[str, list[float | None]]) -> tuple[list[int], dict[str, Vector]]:
    """Drop every index at which ANY series is None or non-finite.

    Returns (kept_indices, aligned_series). Estimating different factors on
    different samples is a classic source of "impossible" betas, so the factor
    model always uses one common sample.
    """
    if not series:
        return [], {}
    names = list(series)
    n = len(series[names[0]])
    for k in names:
        if len(series[k]) != n:
            raise ValueError(f"series {k!r} has length {len(series[k])}, expected {n}")

    keep: list[int] = []
    for i in range(n):
        ok = True
        for k in names:
            v = series[k][i]
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                ok = False
                break
        if ok:
            keep.append(i)

    return keep, {k: [float(series[k][i]) for i in keep] for k in names}
