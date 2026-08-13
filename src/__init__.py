"""Hedgingbot — a factor-exposure overlay for the Tradingbot portfolio.

This is NOT a second strategy bot. It generates no alpha signals. Its only job
is to measure the *aggregate factor risk* of whatever positions are open and,
when a factor exposure breaches its cap, place the smallest offsetting trade
that brings it back inside.

Layering (mirrors Tradingbot deliberately):

    factors/    pure math  — return series -> orthogonal factors -> betas
    exposure/   pure math  — positions + betas -> factor exposures + risk decomp
    overlay/    pure logic — exposures + caps -> HedgePlan   (zero I/O)
    connectors/ I/O only   — the only place that talks to a broker
    state/      I/O        — shared, atomic, crash-safe kill-switch state
"""

__version__ = "0.1.0"
