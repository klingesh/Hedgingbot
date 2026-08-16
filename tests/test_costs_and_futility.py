"""
Tests for the cost model and the futility guard.

Both came directly out of running against the live account, so both are tested
against the ACTUAL broker numbers rather than invented ones. If a future change
makes the cost model disagree with what JustMarkets reported, these fail.
"""

from __future__ import annotations

import math

import pytest

from src.exposure.model import InstrumentSpec, Position, compute_exposure
from src.factors.book import BetaBook
from src.overlay.costs import (
    InstrumentCost,
    breakeven_nights,
    round_trip_cost,
)
from src.overlay.decision import HedgeCaps, HedgeInstrument, hedge_decide

EQUITY = 10_502.26          # the live account
FACTORS = ("USD", "RISK", "ENERGY", "METALS")


# Live values from scripts/check_account_mode.py on account 1100219238.
GOLD_COST = InstrumentCost(
    symbol="XAUUSD.ecn", tick_size=0.01, tick_value=1.0,
    bid=4375.88, ask=4375.96, swap_long=-71.04, swap_short=-84.12,
)
EURUSD_COST = InstrumentCost(
    symbol="EURUSD.ecn", tick_size=0.00001, tick_value=1.0,
    bid=1.15697, ask=1.15701, swap_long=-13.32, swap_short=-3.72,
)


# ---------------------------------------------------------------------------
# Cost model against the broker's own numbers
# ---------------------------------------------------------------------------


def test_notional_matches_the_broker_report():
    """check_account_mode printed NOTIONAL PER 1.0 LOT = 437,592 for gold."""
    assert GOLD_COST.money_per_price_unit_per_lot == pytest.approx(100.0)
    assert GOLD_COST.notional(1.0) == pytest.approx(437_592.0, rel=1e-4)
    # And 115,699 for EURUSD.
    assert EURUSD_COST.notional(1.0) == pytest.approx(115_699.0, rel=1e-4)


def test_gold_nightly_carry_matches_the_reported_swap():
    """swap_short -84.12 points, tick_value 1.0 => -$84.12 per lot per night."""
    assert GOLD_COST.nightly_carry(1.0, -1) == pytest.approx(-84.12)
    assert GOLD_COST.nightly_carry(1.0, +1) == pytest.approx(-71.04)
    assert GOLD_COST.nightly_carry(0.25, -1) == pytest.approx(-21.03)


def test_same_symbol_locking_on_gold_costs_about_155_a_night():
    """The arithmetic behind refusing same-symbol hedging even on a hedging
    account: the pair's net P&L is frozen and you pay both swaps."""
    both = GOLD_COST.both_directions_carry(1.0)
    assert both == pytest.approx(-155.16)
    # ~1.5% of this account, every night, for a position that cannot move.
    assert abs(both) / EQUITY == pytest.approx(0.0148, abs=0.002)


def test_spread_is_negligible_next_to_swap():
    """The finding that matters for validation: modelling spread and ignoring swap
    understates hedge cost by two orders of magnitude."""
    lots = 0.90
    spread = EURUSD_COST.spread_cost(lots)
    monthly_swap = abs(EURUSD_COST.nightly_carry(lots, +1) * 30)

    assert spread == pytest.approx(3.60, abs=0.10)
    assert monthly_swap == pytest.approx(359.64, abs=1.0)
    assert monthly_swap / spread > 50, (
        "swap must dominate spread; any cost model that omits it is wrong"
    )
    # And that monthly bill is a material share of the account.
    assert monthly_swap / EQUITY > 0.03


def test_triple_swap_on_wednesday():
    """Seven nights are billed across five trading days. Missing this
    under-counts carry by 2 nights in every 7."""
    nightly = EURUSD_COST.nightly_carry(1.0, +1)

    assert EURUSD_COST.carry_for_weekday(1.0, +1, 0) == pytest.approx(nightly)
    assert EURUSD_COST.carry_for_weekday(1.0, +1, 2) == pytest.approx(3 * nightly)
    assert EURUSD_COST.carry_for_weekday(1.0, +1, 5) == 0.0
    assert EURUSD_COST.carry_for_weekday(1.0, +1, 6) == 0.0

    week = sum(EURUSD_COST.carry_for_weekday(1.0, +1, d) for d in range(7))
    assert week == pytest.approx(EURUSD_COST.weekly_carry(1.0, +1))
    assert week == pytest.approx(7 * nightly)


def test_round_trip_cost_charges_two_spreads_plus_carry():
    lots, nights = 0.90, 30
    total = round_trip_cost(EURUSD_COST, lots, +1, nights)
    expected = (-2 * EURUSD_COST.spread_cost(lots)
                + EURUSD_COST.nightly_carry(lots, +1) * nights)
    assert total == pytest.approx(expected)
    assert total < 0, "this hedge costs money to hold"


def test_breakeven_nights():
    lots = 0.90
    # A benefit far larger than the costs gives a long runway.
    long_runway = breakeven_nights(EURUSD_COST, lots, +1, 1000.0)
    assert 70 < long_runway < 90

    # A benefit smaller than the spread alone: not worth opening at all.
    assert breakeven_nights(EURUSD_COST, lots, +1, 1.0) == 0.0

    # A positive-carry hedge can be held forever.
    positive = InstrumentCost(
        symbol="X", tick_size=0.00001, tick_value=1.0, bid=1.0, ask=1.00001,
        swap_long=5.0, swap_short=-1.0,
    )
    assert breakeven_nights(positive, 1.0, +1, 100.0) == float("inf")


def test_cost_rejects_unhandled_swap_mode():
    """Guessing a conversion for an unknown swap mode would silently mis-cost
    every hedge, so it must refuse instead."""
    with pytest.raises(ValueError, match="swap_mode"):
        InstrumentCost(
            symbol="X", tick_size=0.01, tick_value=1.0, bid=1.0, ask=1.01,
            swap_long=-1.0, swap_short=-1.0, swap_mode=3,
        )


def test_cost_rejects_inverted_quotes_and_bad_specs():
    with pytest.raises(ValueError, match="ask"):
        InstrumentCost(symbol="X", tick_size=0.01, tick_value=1.0,
                       bid=2.0, ask=1.0, swap_long=0.0, swap_short=0.0)
    with pytest.raises(ValueError, match="tick_value"):
        InstrumentCost(symbol="X", tick_size=0.01, tick_value=0.0,
                       bid=1.0, ask=1.01, swap_long=0.0, swap_short=0.0)


def test_describe_is_readable():
    text = EURUSD_COST.describe(0.90, +1, EQUITY, nights=30)
    assert "0.90 lots long" in text
    assert "swap" in text and "night" in text
    assert "% of equity" in text


# ---------------------------------------------------------------------------
# Futility guard — reproducing the live situation
# ---------------------------------------------------------------------------


def live_book() -> BetaBook:
    """Betas as measured on the weekly run with basket METALS."""
    return BetaBook(
        factors=FACTORS,
        betas={
            "GOLD":   {"USD": -1.03, "RISK": 0.21, "ENERGY": 0.00, "METALS": 0.48},
            "EURUSD": {"USD": -0.94, "RISK": 0.00, "ENERGY": 0.00, "METALS": -0.01},
        },
        sigma={"GOLD": 0.0267 / math.sqrt(5), "EURUSD": 0.0090 / math.sqrt(5)},
        resid_sigma={"GOLD": 0.0116 / math.sqrt(5),
                     "EURUSD": 0.0037 / math.sqrt(5)},
        r_squared={"GOLD": 0.81, "EURUSD": 0.83},
        factor_sigma={
            "USD": 0.00866 / math.sqrt(5), "RISK": 0.02044 / math.sqrt(5),
            "ENERGY": 0.06375 / math.sqrt(5), "METALS": 0.04572 / math.sqrt(5),
        },
    )


GOLD_SPEC = InstrumentSpec("XAUUSD.ecn", 0.01, 1.0, 0.01, 0.01, 100.0,
                           logical="GOLD")
EURUSD_SPEC = InstrumentSpec("EURUSD.ecn", 0.00001, 1.0, 0.01, 0.01, 100.0,
                             logical="EURUSD")


def live_report():
    """Four 0.25-lot gold shorts, exactly as on the account."""
    positions = [
        Position("XAUUSD.ecn", -1, 0.25, 4375.96, ticket=t, magic=0, logical="GOLD")
        for t in (2278989980, 2278989987, 2278989993, 2278989997)
    ]
    return compute_exposure(positions, {"XAUUSD.ecn": GOLD_SPEC},
                            live_book(), EQUITY)


def caps(**kw) -> HedgeCaps:
    base = dict(
        factor_caps={"USD": 1.50, "RISK": 1.50, "ENERGY": 1.00, "METALS": 1.25},
        max_hedge_gross_pct=1000.0,
    )
    base.update(kw)
    return HedgeCaps(**base)


def test_live_exposure_reproduces_the_reported_numbers():
    """Sanity-check the fixture against the real report before relying on it."""
    rep = live_report()
    assert rep.gross_notional == pytest.approx(437_596.0, rel=1e-3)
    # Tolerances are loose because the fixture uses betas ROUNDED to 2dp as they
    # were printed in the report, not the full-precision values from the beta
    # file. The point is to reproduce the reported figures closely enough to trust
    # the fixture, not to re-derive them exactly.
    assert rep.factor_leverage["USD"] == pytest.approx(43.07, abs=0.3)
    assert rep.factor_leverage["METALS"] == pytest.approx(-19.90, abs=0.3)
    assert rep.portfolio_daily_risk_pct == pytest.approx(46.0, abs=2.0)
    # The basket fix means gold now carries real idiosyncratic risk.
    assert rep.idio_daily_risk > 0.0


def test_futile_hedge_is_refused_with_advice_to_reduce_the_position():
    """THE regression. The overlay previously proposed 0.90 lots of EURUSD, which
    moved USD leverage 43.07x -> 33.72x: 22% of the excess removed, still 22x over
    cap, at ~3.4% of equity per month in swap."""
    rep = live_report()
    inst = HedgeInstrument("EURUSD", "EURUSD.ecn", EURUSD_SPEC, 1.15697, 1.15701)

    plan = hedge_decide(rep, live_book(), caps(), [inst])

    assert not plan.has_trades, f"should refuse, got: {plan.summary()}"
    action = plan.actions[0]
    assert action.action == "skip"
    assert "REDUCE THE POSITION" in action.reason
    assert "cosmetic" in action.reason
    assert "futile" in plan.reason


def test_the_futility_threshold_is_what_blocks_it():
    """Lowering the threshold must let the same hedge through, proving the guard
    is the thing making the decision rather than some other gate."""
    rep = live_report()
    inst = HedgeInstrument("EURUSD", "EURUSD.ecn", EURUSD_SPEC, 1.15697, 1.15701)

    permissive = hedge_decide(rep, live_book(), caps(min_excess_removed=0.0), [inst])

    assert permissive.has_trades
    a = permissive.trades()[0]
    assert a.logical == "EURUSD"
    assert a.side == 1, "book is long USD (short gold), so BUY EURUSD to offset"
    assert a.lots == pytest.approx(0.90, abs=0.05)


def test_an_effective_hedge_still_passes_the_guard():
    """The guard must not block hedges that actually work.

    0.10 lots of gold gives USD leverage ~4.3x. The excess is ~2.8x and the gross
    ceiling permits removing up to ~9.4x, so the hedge can reach its target
    comfortably and must be allowed.
    """
    rep = compute_exposure(
        [Position("XAUUSD.ecn", -1, 0.10, 4375.96, magic=0, logical="GOLD")],
        {"XAUUSD.ecn": GOLD_SPEC}, live_book(), EQUITY,
    )
    inst = HedgeInstrument("EURUSD", "EURUSD.ecn", EURUSD_SPEC, 1.15697, 1.15701)

    lev = rep.factor_leverage["USD"]
    assert lev > 1.50 * 1.25, "fixture must actually breach beyond hysteresis"

    plan = hedge_decide(rep, live_book(), caps(), [inst])

    assert plan.has_trades, f"a reachable breach must still be hedged: {plan.summary()}"
    a = plan.trades()[0]
    # And it must genuinely reach the target rather than squeak past the guard.
    per_lot = EURUSD_SPEC.notional_per_lot(1.15701)
    after = (rep.factor_exposure["USD"]
             + a.side * a.lots * per_lot * live_book().beta("EURUSD", "USD")) / EQUITY
    assert abs(after) < 1.50 * 1.05, f"projected USD {after:.2f}x should reach cap"


def test_min_excess_removed_is_validated():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="min_excess_removed"):
            HedgeCaps(factor_caps={"USD": 1.0}, min_excess_removed=bad)


def test_futility_does_not_fire_when_there_is_no_excess():
    """Guard must be inert when the book is inside its caps."""
    rep = compute_exposure(
        [Position("XAUUSD.ecn", -1, 0.001, 4375.96, magic=0, logical="GOLD")],
        {"XAUUSD.ecn": GOLD_SPEC}, live_book(), EQUITY,
    )
    inst = HedgeInstrument("EURUSD", "EURUSD.ecn", EURUSD_SPEC, 1.15697, 1.15701)
    plan = hedge_decide(rep, live_book(), caps(), [inst])
    assert plan.actions[0].action == "nothing"
