"""
Build factor proxies from BASKETS of instruments instead of single series.

The problem this solves
-----------------------
The METALS factor was proxied by GC=F — gold. GOLD is also a traded slot, using
the same GC=F series. So regressing gold on the factors put gold on both sides of
the equation, and the live run reported exactly what you would expect from that:

    GOLD   USD -1.03   METALS 0.93   R2 1.00   vol 2.67%   idio 0.00%

R2 of 1.00 and idiosyncratic vol of 0.00 are not measurements. They are the
regression discovering that gold explains gold. The consequences are concrete:

  * The risk model believes gold has NO unexplained risk, so it believes gold is
    perfectly hedgeable by trading the factors. Nothing is perfectly hedgeable.
  * portfolio_daily_risk loses the idiosyncratic term entirely for that position.
    On a book that is one gold trade, that is the majority of the position.
  * SILVER and PLATINUM betas are distorted too, because "METALS" literally means
    "gold", so their METALS betas (1.51, 1.10) are really their betas to GOLD,
    not to a metals complex.

The fix, and its honest limit
----------------------------
Build METALS from gold + silver + platinum at EQUAL RISK. No single traded
instrument is then the factor, gold's weight in it drops to about a third, and its
R2 and idio become real numbers.

The limit worth stating plainly: gold is still ~1/3 of the basket, so a residual
self-reference remains and gold's R2 will still be high — the metals are 0.58-0.79
correlated with each other, so any honest metals factor is largely a gold factor.
`self_weight()` quantifies exactly how much of each factor is an instrument's own
series, and the estimate script prints it. Fully eliminating it needs
leave-one-out factors (rebuild METALS from silver+platinum when regressing gold),
which costs the single-common-factor-set property that makes portfolio variance a
clean sum of squares. Basket first, measure the residual, then decide.

Why EQUAL RISK and not equal weight
-----------------------------------
Component volatilities differ enormously — on weekly data NATGAS runs 9.4% while
GOLD runs 2.7%. Averaging raw returns would let the noisiest member dominate the
factor and call it "energy". Weighting by 1/sigma gives every component the same
risk contribution, which is what "a metals complex factor" should mean.

Pure stdlib.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .math_core import normalized_weights, wstd

Series = List[Optional[float]]


def _common_rows(components: Dict[str, Series]) -> List[int]:
    """Row indices where EVERY component has a finite value.

    A factor must be one consistent series, so a day on which any component is
    missing is dropped for the whole basket rather than silently changing the
    basket's composition mid-sample.
    """
    names = list(components)
    if not names:
        return []
    n = len(components[names[0]])
    rows: List[int] = []
    for i in range(n):
        ok = True
        for k in names:
            col = components[k]
            if i >= len(col):
                ok = False
                break
            v = col[i]
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                ok = False
                break
        if ok:
            rows.append(i)
    return rows


def inverse_vol_weights(
    components: Dict[str, Series],
    rows: Sequence[int],
    halflife: Optional[float] = None,
) -> Dict[str, float]:
    """Weights proportional to 1/sigma, normalized to sum to 1.

    Equal risk contribution per component. A zero-variance component (a broken or
    constant series) is given zero weight rather than an infinite one.
    """
    w = normalized_weights(len(rows), halflife)
    sigmas: Dict[str, float] = {}
    for name, col in components.items():
        values = [float(col[i]) for i in rows]
        sigmas[name] = wstd(values, w)

    inv = {
        name: (1.0 / s if s > 1e-12 else 0.0) for name, s in sigmas.items()
    }
    total = math.fsum(inv.values())
    if total <= 1e-18:
        # Every component is degenerate; fall back to equal weights so the caller
        # gets a usable (if useless) series instead of a division by zero.
        n = len(components) or 1
        return {name: 1.0 / n for name in components}
    return {name: v / total for name, v in inv.items()}


def build_basket(
    components: Dict[str, Series],
    weights: Optional[Dict[str, float]] = None,
    halflife: Optional[float] = None,
    rescale_to_mean_component_vol: bool = True,
) -> Tuple[Series, Dict[str, float]]:
    """Combine component RETURN series into one factor return series.

    components : {yahoo symbol: return series}, all the same length, None for gaps
    weights    : explicit weights, or None for equal-risk (inverse vol)

    Returns (basket_series, weights_used). The basket series has the same length
    as its inputs, with None wherever any component was missing.

    rescale_to_mean_component_vol keeps the factor's magnitude comparable to a
    typical member. Inverse-vol weighting produces a basket whose volatility is
    lower than any single member (diversification), which would make "a 1% METALS
    move" mean something much larger in gold terms than in factor terms and make
    the exposure report hard to read. Rescaling is a pure change of units: it does
    not alter correlations, betas relative to each other, or orthogonality.
    """
    if not components:
        raise ValueError("no basket components supplied")

    rows = _common_rows(components)
    if len(rows) < 30:
        raise ValueError(
            f"basket has only {len(rows)} rows where all "
            f"{len(components)} components are present; need at least 30"
        )

    if weights is None:
        weights = inverse_vol_weights(components, rows, halflife)
    else:
        total = math.fsum(weights.values())
        if total <= 0:
            raise ValueError("basket weights must sum to a positive number")
        weights = {k: v / total for k, v in weights.items()}
        missing = [k for k in components if k not in weights]
        if missing:
            raise ValueError(f"no weight supplied for component(s) {missing}")

    length = len(next(iter(components.values())))
    out: Series = [None] * length
    row_set = set(rows)
    for i in range(length):
        if i not in row_set:
            continue
        out[i] = math.fsum(
            weights[name] * float(components[name][i]) for name in components
        )

    if rescale_to_mean_component_vol and len(components) > 1:
        w = normalized_weights(len(rows), halflife)
        basket_vals = [float(out[i]) for i in rows]
        basket_sigma = wstd(basket_vals, w)
        mean_component_sigma = math.fsum(
            wstd([float(components[name][i]) for i in rows], w)
            for name in components
        ) / len(components)
        if basket_sigma > 1e-12 and mean_component_sigma > 1e-12:
            scale = mean_component_sigma / basket_sigma
            for i in rows:
                out[i] = out[i] * scale

    return out, weights


# ---------------------------------------------------------------------------
# Circularity measurement
# ---------------------------------------------------------------------------


def self_weight(
    instrument_symbol: str,
    basket_weights: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """How much of each factor is this instrument's OWN price series.

    Returns {factor: weight}, only for factors where the weight is non-zero.

    Read it as a circularity score. 1.00 means the instrument IS the factor and
    its R2 is meaningless. 0.33 means a third of the factor is itself, so its R2
    is inflated but the remaining relationship is real. 0.00 is clean.
    """
    return {
        factor: weights[instrument_symbol]
        for factor, weights in basket_weights.items()
        if instrument_symbol in weights and weights[instrument_symbol] > 1e-9
    }


def circularity_report(
    instrument_symbols: Dict[str, str],
    basket_weights: Dict[str, Dict[str, float]],
    traded: Optional[Sequence[str]] = None,
) -> str:
    """Text table of self-weights, worst first.

    `traded` marks which instruments are actually held, because circularity
    matters very differently by role:

      * A TRADED instrument that is its own factor gets a fake R2 of 1.00 and a
        fake idio of 0.00, so the risk model understates its unhedgeable risk.
        That is a real problem.
      * A HEDGE CANDIDATE that is its own factor is a PURE lever — beta 1.0 by
        construction, so hedge sizing is exact. That is desirable, not a bug.

    Reporting both without that distinction would send someone "fixing" the
    beneficial case.
    """
    traded_set = set(traded or ())
    rows: List[Tuple[float, str, str, float, bool]] = []
    for name, symbol in instrument_symbols.items():
        for factor, weight in self_weight(symbol, basket_weights).items():
            rows.append((weight, name, factor, weight, name in traded_set))
    rows.sort(reverse=True)

    if not rows:
        return "   No instrument contributes to any factor basket. No circularity."

    lines = [
        f"   {'instrument':<12}{'factor':<9}{'self-weight':>12}{'role':>10}"
        f"{'  assessment':<40}",
    ]
    lines.append("   " + "-" * (len(lines[0]) - 3))
    for _sort, name, factor, weight, is_traded in rows:
        role = "traded" if is_traded else "hedge"
        if weight > 0.90:
            note = "IS the factor — R2/idio meaningless" if is_traded else \
                   "pure lever — exact hedge sizing, fine"
        elif weight > 0.20:
            note = "R2 inflated, idio understated" if is_traded else \
                   "good lever, mild self-reference"
        else:
            note = "negligible"
        lines.append(f"   {name:<12}{factor:<9}{weight:>11.2f} {role:>9}  {note}")

    bad = [r for r in rows if r[4] and r[0] > 0.90]
    if bad:
        lines += [
            "",
            "   *** A TRADED instrument IS a factor ***",
            "   Its R2 will read 1.00 and its idiosyncratic vol 0.00, which tells",
            "   the risk model it has no unhedgeable risk. Make that factor a",
            "   basket in FACTOR_BASKETS.",
        ]
    return "\n".join(lines)
