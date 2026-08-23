"""
One attempt to hedge ONE factor.

Extracted from hedge_decide() to fix "breach starvation", found in a read-only
architecture audit:

    9. Breach starvation — An unhedgeable worst breach can prevent a smaller,
       hedgeable breach from being addressed.

The original code picked the single worst breach and then `return`ed on every
failure path — no usable instrument, below minimum lot, no headroom, collateral
damage, futility. So if the largest breach happened to be unhedgeable, the overlay
did nothing at all, even when a second breach was sitting there perfectly
hedgeable. On the live book that was the actual situation: USD was unhedgeable
(43x, futile), while METALS and RISK were also in breach.

Returning an outcome instead of a plan lets the caller iterate factors by severity
and stop at the first one it can actually act on.

Pure stdlib, no I/O — same contract as the rest of the overlay package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..exposure.model import ExposureReport, InstrumentSpec, Position


@dataclass
class Attempt:
    """Result of trying to hedge one factor.

    `action` is None when this factor could not be hedged; `note` always explains
    what happened, in a form suitable for a journal line.
    """

    factor: str
    action: object | None            # HedgeAction | None (avoids a circular import)
    note: str
    category: str = ""               # short machine-readable failure reason


def total_risk_after(
    report: ExposureReport, betas, added_notional: float, hedge_key: str,
    factor_sigma: dict,
) -> float:
    """Portfolio 1-sigma daily risk if `added_notional` of `hedge_key` were added.

    Why this exists, from the audit:

        6. The system checks cap compliance but does not assert that total
           portfolio daily risk actually decreases after a hedge.

    Cap compliance is a PROXY for lower risk, not a guarantee of it. Caps constrain
    factors one at a time; risk aggregates them in quadrature. A hedge can satisfy
    every individual cap and still raise total risk — most obviously by trading a
    quiet factor down while pushing a loud one up, since a factor's contribution
    scales with its own volatility, not with its leverage number.

    The idiosyncratic term is carried over unchanged. That is deliberate: a hedge
    instrument does add its own idio risk, so holding it constant makes this
    estimate mildly OPTIMISTIC about the hedge. Since the check is used to REJECT
    hedges that fail to reduce risk, an optimistic estimate is the conservative
    direction — it cannot wave through a hedge that this model already says is bad.
    """
    factor_var = 0.0
    for f in report.factors:
        exposure = report.factor_exposure.get(f, 0.0) + added_notional * betas.beta(
            hedge_key, f
        )
        sigma = float(factor_sigma.get(f, 0.0) or 0.0)
        contribution = abs(exposure) * sigma
        factor_var += contribution * contribution
    return math.sqrt(factor_var + report.idio_daily_risk ** 2)


def hedge_gross_notional(
    existing_hedges: list[Position], hedge_specs: dict[str, InstrumentSpec]
) -> float:
    """Total absolute notional of hedges currently open."""
    total = 0.0
    for h in existing_hedges:
        spec = hedge_specs.get(h.symbol)
        if spec is None:
            continue
        total += abs(h.volume * spec.money_per_price_unit_per_lot * h.price)
    return total
