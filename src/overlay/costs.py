"""
What a hedge actually costs to open and to hold.

Why this module is load-bearing for validation
----------------------------------------------
An overlay reduces drawdown and it charges you for the privilege. Phase 3 asks
"does it improve risk-adjusted return?", and that question is meaningless without
the bill. The bill has two parts and the second one is the one people forget:

  SPREAD  paid once per leg, on open and again on close.
  SWAP    paid EVERY NIGHT the hedge is held.

Measured on the live JustMarkets ECN account (login 1100219238):

    symbol        spread        swap_long   swap_short   per lot per night
    XAUUSD.ecn    0.0018%        -71.04       -84.12     -$71 / -$84
    EURUSD.ecn    0.0035%        -13.32        -3.72     -$13 / -$4
    XAGUSD.ecn    0.0371%         -8.52        -1.08
    US500.ecn     0.0026%         -5.00        -7.50
    XPTUSD.ecn    0.2918%        -16.25       -17.10

Two things fall straight out of those numbers:

1. A 0.90-lot EURUSD hedge costs $3.60 of spread to open and $12/night to hold —
   $360 a month, 3.4% of a 10.5k account. The SPREAD IS ROUNDING ERROR AND THE
   SWAP IS THE WHOLE COST. Any analysis that models spread and ignores swap will
   conclude hedging is nearly free, which is wrong by two orders of magnitude.

2. Same-symbol hedging on gold costs swap_long + swap_short = -$155 per lot per
   night, about 1.5% of that account daily, to hold a position pair whose net P&L
   is frozen. That is the arithmetic behind refusing same-symbol locking even on
   a hedging-enabled account.

Swap units
----------
JustMarkets reports swap in POINTS (SYMBOL_SWAP_MODE_POINTS, mode=1). One point
equals one tick for every symbol checked, so

    usd_per_lot_per_night = swap_points * tick_value

Gold: -84.12 points * $1.00 = -$84.12 per lot per night. Cross-checked against the
broker's own numbers rather than assumed.

Triple swap
-----------
Brokers charge three nights of swap on one weekday to cover the weekend, because
spot settlement is T+2. Wednesday is the usual day for FX and metals. Ignoring it
understates carry by ~40% (5 charges per week, not 3... precisely: 7 nights billed
over 5 trading days). `weekly_carry` handles this so a validation run does not
quietly under-count 2 nights in every 7.

Pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: MT5 SYMBOL_SWAP_MODE values we handle explicitly.
SWAP_MODE_POINTS = 1

#: Which weekday carries the triple charge (0=Monday). Wednesday for FX/metals.
TRIPLE_SWAP_WEEKDAY = 2


@dataclass(frozen=True)
class InstrumentCost:
    """Trading costs for one instrument, as reported by the broker.

    All fields come straight out of scripts/check_account_mode.py so there is no
    guessing and no hand-tuned "typical" values.
    """

    symbol: str
    tick_size: float
    tick_value: float          # account currency per tick per 1.0 lot
    bid: float
    ask: float
    swap_long: float           # points per lot per night
    swap_short: float          # points per lot per night
    swap_mode: int = SWAP_MODE_POINTS

    def __post_init__(self) -> None:
        for name in ("tick_size", "tick_value", "bid", "ask"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0:
                raise ValueError(
                    f"{self.symbol}: {name} must be finite and > 0, got {v!r}"
                )
        if self.ask < self.bid:
            raise ValueError(f"{self.symbol}: ask {self.ask} < bid {self.bid}")
        if self.swap_mode != SWAP_MODE_POINTS:
            raise ValueError(
                f"{self.symbol}: swap_mode {self.swap_mode} is not handled. Only "
                f"POINTS ({SWAP_MODE_POINTS}) is implemented, which is what "
                "JustMarkets reports. Percentage or interest modes would need "
                "their own conversion, and guessing would silently mis-cost every "
                "hedge."
            )

    # -- helpers -----------------------------------------------------------

    @property
    def money_per_price_unit_per_lot(self) -> float:
        return self.tick_value / self.tick_size

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_fraction(self) -> float:
        """Spread as a fraction of price."""
        return (self.ask - self.bid) / self.mid

    def notional(self, lots: float, price: float | None = None) -> float:
        return lots * self.money_per_price_unit_per_lot * (
            self.mid if price is None else price
        )

    # -- the two costs -----------------------------------------------------

    def spread_cost(self, lots: float) -> float:
        """Account-currency cost of crossing the spread once, always positive.

        Charged on open and again on close, so a full round trip is 2x this.
        """
        return abs(lots) * self.money_per_price_unit_per_lot * (self.ask - self.bid)

    def nightly_carry(self, lots: float, side: int) -> float:
        """Swap for one night. NEGATIVE means it costs you.

        side is +1 long, -1 short.
        """
        if side not in (1, -1):
            raise ValueError(f"side must be +1 or -1, got {side!r}")
        points = self.swap_long if side > 0 else self.swap_short
        return points * self.tick_value * abs(lots)

    def weekly_carry(self, lots: float, side: int) -> float:
        """Swap for a full week: 7 nights billed across 5 trading days."""
        return self.nightly_carry(lots, side) * 7.0

    def carry_for_weekday(self, lots: float, side: int, weekday: int) -> float:
        """Swap charged on a given weekday, accounting for the triple charge.

        weekday is 0=Monday .. 6=Sunday, matching datetime.date.weekday().
        Saturday and Sunday carry nothing; the weekend is billed on Wednesday.
        """
        if weekday >= 5:
            return 0.0
        nights = 3.0 if weekday == TRIPLE_SWAP_WEEKDAY else 1.0
        return self.nightly_carry(lots, side) * nights

    def both_directions_carry(self, lots: float) -> float:
        """Cost per night of holding BOTH a long and a short of the same size.

        This is the price of same-symbol locking. On gold it comes to about -$155
        per lot per night, roughly 1.5% of a 10.5k account daily, to hold a pair
        whose net P&L cannot move. Kept as a first-class method so the number is
        easy to quote rather than easy to forget.
        """
        return self.nightly_carry(lots, 1) + self.nightly_carry(lots, -1)

    # -- summary -----------------------------------------------------------

    def describe(self, lots: float, side: int, equity: float,
                 nights: int = 30) -> str:
        spread = self.spread_cost(lots)
        nightly = self.nightly_carry(lots, side)
        total = nightly * nights
        pct = 100.0 * total / equity if equity > 0 else 0.0
        direction = "long" if side > 0 else "short"
        return (
            f"{lots:.2f} lots {direction} {self.symbol}: "
            f"notional {self.notional(lots):,.0f}, "
            f"spread {spread:,.2f} to open, "
            f"swap {nightly:+,.2f}/night "
            f"({total:+,.2f} over {nights} nights = {pct:+.2f}% of equity)"
        )


def round_trip_cost(
    cost: InstrumentCost, lots: float, side: int, nights_held: float
) -> float:
    """Total cost of a hedge over its whole life. NEGATIVE is a cost.

    Two spread crossings (open and close) plus carry for every night held. This is
    the number the validation harness charges against the hedged equity curve.
    """
    spread = -2.0 * cost.spread_cost(lots)
    carry = cost.nightly_carry(lots, side) * nights_held
    return spread + carry


def breakeven_nights(
    cost: InstrumentCost, lots: float, side: int, risk_reduction_value: float
) -> float:
    """How many nights a hedge can be held before its cost exceeds its benefit.

    `risk_reduction_value` is what you judge the reduction in expected drawdown to
    be worth, in account currency. Returns inf when carry is positive (the hedge
    pays you), and 0 when the spread alone already exceeds the benefit.

    This exists to make an uncomfortable question explicit: a hedge that costs
    3.4% of equity a month has to be preventing a LOT of drawdown to be worth
    holding. Phase 3 should report this alongside any drawdown improvement.
    """
    spread = 2.0 * cost.spread_cost(lots)
    if risk_reduction_value <= spread:
        return 0.0
    nightly = cost.nightly_carry(lots, side)
    if nightly >= 0:
        return float("inf")
    return (risk_reduction_value - spread) / abs(nightly)
