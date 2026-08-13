"""
Orchestration: price history -> orthogonal factors -> BetaBook.

Pure stdlib. The algebra lives in math_core.py; this module handles alignment,
per-instrument sample selection, and packaging into a BetaBook.

The one subtlety worth understanding
------------------------------------
FACTORS are built on a single common sample: every factor proxy must have data
on a given day or that day is dropped for all of them. Estimating different
factors on different samples is a classic source of "impossible" betas, and it
also breaks the orthogonality guarantee that everything downstream relies on.

INSTRUMENTS are aligned independently against those factors. So an instrument
with a short history (a newly listed CFD, say) is estimated on its own shorter
overlap rather than truncating the sample for everything else. Its n_obs records
how much data it actually got, and instruments below `min_obs` are DROPPED rather
than estimated on noise — a silently unreliable beta is worse than a missing one,
because a missing one is caught by the unmapped-symbol guard while a bad one just
sizes a wrong hedge.
"""

from __future__ import annotations

import math

from .book import BetaBook
from .math_core import (
    OrthogonalFactors,
    align,
    build_orthogonal_factors,
    fit_instrument,
    log_return_series,
    winsorize,
)

Series = list[float | None]


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def returns_from_prices(
    prices: dict[str, list[float | None]],
    winsorize_fraction: float = 0.005,
) -> dict[str, Series]:
    """Log returns per series, each winsorized on its own distribution.

    Per-column clipping matters: NATGAS and EURUSD do not share a scale, so a
    single global quantile would clip one and ignore the other.
    """
    out: dict[str, Series] = {}
    for name, px in prices.items():
        rets = log_return_series(list(px))
        if winsorize_fraction > 0.0:
            finite_idx = [i for i, v in enumerate(rets) if v is not None]
            if finite_idx:
                clipped = winsorize([rets[i] for i in finite_idx], winsorize_fraction)
                for i, v in zip(finite_idx, clipped):
                    rets[i] = v
        out[name] = rets
    return out


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------


def build_factors(
    proxy_returns: dict[str, Series],
    order: tuple[str, ...],
    halflife: float | None = None,
) -> tuple[OrthogonalFactors, list[int]]:
    """Align the proxy return series and orthogonalize them.

    Returns (factors, kept_indices). The kept indices are needed so instrument
    series can be lined up against the same rows.
    """
    subset = {}
    for f in order:
        if f not in proxy_returns:
            raise ValueError(
                f"proxy_returns is missing factor {f!r}; got {sorted(proxy_returns)}"
            )
        subset[f] = proxy_returns[f]

    kept, aligned = align(subset)
    factors = build_orthogonal_factors(aligned, order, halflife=halflife)
    return factors, kept


# ---------------------------------------------------------------------------
# Betas
# ---------------------------------------------------------------------------


def estimate_betas(
    instrument_returns: dict[str, Series],
    factors: OrthogonalFactors,
    factor_row_indices: list[int],
    halflife: float | None = None,
    min_obs: int = 250,
) -> tuple[BetaBook, dict[str, int]]:
    """Fit every instrument against the orthogonal factors.

    instrument_returns : {name: full-length return series, None where missing}
    factor_row_indices : the original row indices the factors were built on
                         (returned by build_factors)

    Returns (book, skipped) where `skipped` maps dropped instrument names to the
    number of overlapping observations they had, so the caller can report why.
    """
    if not instrument_returns:
        raise ValueError("instrument_returns is empty")

    betas: dict[str, dict[str, float]] = {}
    r2: dict[str, float] = {}
    sig: dict[str, float] = {}
    resid: dict[str, float] = {}
    nobs: dict[str, int] = {}
    skipped: dict[str, int] = {}

    n_factor_rows = len(factor_row_indices)

    for name, series in instrument_returns.items():
        # Which of the factor rows does this instrument have data for?
        usable_positions = []
        for pos, row in enumerate(factor_row_indices):
            if row >= len(series):
                continue
            v = series[row]
            if v is None or not math.isfinite(v):
                continue
            usable_positions.append(pos)

        if len(usable_positions) < min_obs:
            skipped[name] = len(usable_positions)
            continue

        y = [float(series[factor_row_indices[p]]) for p in usable_positions]

        if len(usable_positions) == n_factor_rows:
            subset = None                      # full overlap, use factors as-is
        else:
            subset = {
                f: [factors.series[f][p] for p in usable_positions]
                for f in factors.names
            }

        fit = fit_instrument(y, factors, halflife=halflife, factor_subset=subset)

        betas[name] = dict(fit.betas)
        r2[name] = fit.r_squared
        sig[name] = fit.sigma
        resid[name] = fit.resid_sigma
        nobs[name] = fit.n_obs

    if not betas:
        detail = ", ".join(f"{k}({v} obs)" for k, v in sorted(skipped.items()))
        raise ValueError(
            f"no instrument had >= {min_obs} observations overlapping the factor "
            f"sample. Skipped: {detail or 'none'}"
        )

    book = BetaBook(
        factors=factors.names,
        betas=betas,
        r_squared=r2,
        sigma=sig,
        resid_sigma=resid,
        n_obs=nobs,
        factor_sigma=dict(factors.sigma),
    )
    return book, skipped


def estimate_from_prices(
    factor_prices: dict[str, list[float | None]],
    instrument_prices: dict[str, list[float | None]],
    order: tuple[str, ...],
    halflife: float | None = None,
    winsorize_fraction: float = 0.005,
    min_obs: int = 250,
) -> tuple[BetaBook, OrthogonalFactors, dict[str, int]]:
    """Convenience end-to-end: aligned price columns in, BetaBook out.

    All price lists must be indexed on the SAME date axis (use
    data.align_price_frame first). Missing observations are None.
    """
    factor_rets = returns_from_prices(factor_prices, winsorize_fraction)
    inst_rets = returns_from_prices(instrument_prices, winsorize_fraction)
    factors, kept = build_factors(factor_rets, order, halflife)
    book, skipped = estimate_betas(inst_rets, factors, kept, halflife, min_obs)
    return book, factors, skipped
