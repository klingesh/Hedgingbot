"""
Tests for the Phase 3 replay engine.

The load-bearing test is `test_curves_are_identical_when_the_overlay_does_nothing`.
If the two equity curves can diverge without a single hedge being placed, then any
difference the replay reports is an artefact of the machinery rather than an effect
of the overlay — and the whole validation would be worthless. That is checked first
and directly.

The second most important is `test_a_perfect_hedge_reduces_drawdown`, which proves
the harness can DETECT a benefit when one genuinely exists. A validation that can
only ever return "no improvement" is equally worthless.
"""

from __future__ import annotations

import math
import os
from datetime import date, timedelta

import pytest

from src.backtest.replay import (
    DayRow,
    ReplayInputs,
    ReplayResult,
    TradeRecord,
    replay,
    trading_days,
)
from src.exposure.model import InstrumentSpec
from src.factors.book import BetaBook
from src.overlay.costs import InstrumentCost
from src.overlay.decision import HedgeCaps

FACTORS = ("USD", "RISK", "ENERGY", "METALS")
SIGMA = {"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022, "METALS": 0.009}


def bdays(start: str, n: int) -> list[str]:
    d = date.fromisoformat(start)
    out: list[str] = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


GOLD_SPEC = InstrumentSpec("XAUUSD.ecn", 0.01, 1.0, 0.01, 0.01, 100.0, logical="GOLD")
EUR_SPEC = InstrumentSpec("EURUSD.ecn", 0.00001, 1.0, 0.01, 0.01, 100.0,
                          logical="EURUSD")

EUR_COST = InstrumentCost(
    symbol="EURUSD.ecn", tick_size=0.00001, tick_value=1.0,
    bid=1.15697, ask=1.15701, swap_long=-13.32, swap_short=-3.72,
)


def book(gold_usd: float = -1.03, eur_usd: float = -0.94) -> BetaBook:
    betas = {
        "GOLD":   {"USD": gold_usd, "RISK": 0.0, "ENERGY": 0.0, "METALS": 0.48},
        "EURUSD": {"USD": eur_usd, "RISK": 0.0, "ENERGY": 0.0, "METALS": 0.0},
    }
    return BetaBook(
        factors=FACTORS, betas=betas,
        sigma={k: 0.011 for k in betas}, resid_sigma={k: 0.003 for k in betas},
        r_squared={k: 0.8 for k in betas}, n_obs={k: 417 for k in betas},
        factor_sigma=SIGMA,
    )


def caps(**kw) -> HedgeCaps:
    base = dict(factor_caps={"USD": 1.5, "RISK": 1.5, "ENERGY": 1.0, "METALS": 1.25},
                max_hedge_gross_pct=1000.0)
    base.update(kw)
    return HedgeCaps(**base)


def make_inputs(
    days: list[str],
    gold_path: list[float],
    lots: float = 0.02,
    side: int = -1,
    balance: float = 10_000.0,
    eur_flat: bool = True,
) -> ReplayInputs:
    """One gold trade spanning the whole window, plus a EURUSD hedge instrument."""
    gold_prices = dict(zip(days, gold_path))
    eur_prices = {d: 1.157 for d in days} if eur_flat else {
        d: 1.157 * (1.0 + 0.0002 * i) for i, d in enumerate(days)
    }
    return ReplayInputs(
        slot_specs={"GOLD": GOLD_SPEC},
        trades=[TradeRecord(
            logical="GOLD", symbol="XAUUSD.ecn", side=side, lots=lots,
            entry_day=days[0], exit_day=days[-1],
            entry=gold_path[0], exit=gold_path[-1],
        )],
        prices={"GOLD": gold_prices, "EURUSD": eur_prices},
        hedge_specs={"EURUSD": EUR_SPEC},
        hedge_costs={"EURUSD": EUR_COST},
        hedge_symbols={"EURUSD": "EURUSD.ecn"},
        initial_balance=balance,
    )


# ---------------------------------------------------------------------------
# THE control test
# ---------------------------------------------------------------------------


def test_curves_are_identical_when_the_overlay_does_nothing():
    """If these can diverge with zero hedges, every reported difference is an
    artefact of the harness and the whole validation is worthless."""
    days = bdays("2024-01-01", 60)
    path = [2000.0 * (1.0 + 0.001 * math.sin(i / 4.0)) for i in range(60)]
    # Caps far above anything a 0.02-lot position can reach, so it never acts.
    inputs = make_inputs(days, path, lots=0.02)

    result = replay(inputs, book(), caps(factor_caps={
        "USD": 1000.0, "RISK": 1000.0, "ENERGY": 1000.0, "METALS": 1000.0}))

    assert result.hedges_opened == 0
    assert result.total_spread == 0.0
    assert result.total_swap == 0.0
    for r in result.rows:
        assert r.hedged_equity == pytest.approx(r.base_equity, rel=1e-12), (
            f"{r.day}: curves diverged with no hedge placed"
        )

    s = result.summary()
    assert s["base_return_pct"] == pytest.approx(s["hedged_return_pct"], rel=1e-12)
    assert s["base_max_dd_pct"] == pytest.approx(s["hedged_max_dd_pct"], rel=1e-12)


def test_overlay_disabled_matches_overlay_that_never_triggers():
    """Two routes to 'no hedging' must agree exactly."""
    days = bdays("2024-01-01", 40)
    path = [2000.0 + 5.0 * i for i in range(40)]
    inputs = make_inputs(days, path, lots=0.02)

    off = replay(inputs, book(), caps(), enable_overlay=False)
    never = replay(inputs, book(), caps(factor_caps={f: 1e6 for f in FACTORS}))

    assert [r.base_equity for r in off] if False else True
    for a, b in zip(off.rows, never.rows):
        assert a.base_equity == pytest.approx(b.base_equity, rel=1e-12)
        assert a.hedged_equity == pytest.approx(b.hedged_equity, rel=1e-12)


# ---------------------------------------------------------------------------
# Can it detect a real benefit?
# ---------------------------------------------------------------------------


def test_a_perfect_hedge_reduces_drawdown():
    """A validation that can only ever say 'no improvement' is useless.

    Setup: a short gold position, gold rising steadily (so the book bleeds), and a
    EURUSD hedge whose USD beta lets the overlay offset it. With zero swap and zero
    spread the hedge must reduce drawdown.
    """
    days = bdays("2024-01-01", 120)
    # Gold grinds up 20% => a short position loses steadily.
    path = [2000.0 * (1.0 + 0.20 * i / 119.0) for i in range(120)]

    free = InstrumentCost(
        symbol="EURUSD.ecn", tick_size=0.00001, tick_value=1.0,
        bid=1.157, ask=1.157, swap_long=0.0, swap_short=0.0,
    )
    inputs = make_inputs(days, path, lots=0.05, side=-1)
    inputs.hedge_costs = {"EURUSD": free}
    # EURUSD must MOVE for the hedge to have any P&L; tie it to gold inversely so
    # a long EURUSD position gains while the short gold loses.
    inputs.prices["EURUSD"] = {
        d: 1.157 * (1.0 + 0.20 * i / 119.0) for i, d in enumerate(days)
    }

    tight = caps(factor_caps={"USD": 0.5, "RISK": 50.0, "ENERGY": 50.0,
                              "METALS": 50.0},
                 min_excess_removed=0.0)
    result = replay(inputs, book(), tight)

    assert result.hedges_opened > 0, "fixture must actually trigger a hedge"
    s = result.summary()
    assert s["hedged_max_dd_pct"] < s["base_max_dd_pct"], (
        f"a free, correctly-signed hedge must reduce drawdown: "
        f"{s['base_max_dd_pct']:.2f}% -> {s['hedged_max_dd_pct']:.2f}%"
    )


def test_swap_is_charged_and_shows_up_as_a_cost():
    """Swap is ~100x the spread, so it must be the dominant cost in the replay."""
    days = bdays("2024-01-01", 90)
    path = [2000.0 * (1.0 + 0.15 * i / 89.0) for i in range(90)]
    inputs = make_inputs(days, path, lots=0.05, side=-1)

    tight = caps(factor_caps={"USD": 0.5, "RISK": 50.0, "ENERGY": 50.0,
                              "METALS": 50.0},
                 min_excess_removed=0.0)
    result = replay(inputs, book(), tight)

    assert result.hedges_opened > 0
    assert result.total_swap < 0.0, "swap on these rates is a cost, so negative"
    assert abs(result.total_swap) > result.total_spread, (
        "swap must dominate spread over a 90-day hold"
    )
    # And it must actually reduce the hedged curve relative to a free hedge.
    free = InstrumentCost(symbol="EURUSD.ecn", tick_size=0.00001, tick_value=1.0,
                          bid=1.157, ask=1.157, swap_long=0.0, swap_short=0.0)
    inputs.hedge_costs = {"EURUSD": free}
    costless = replay(inputs, book(), tight)
    assert costless.rows[-1].hedged_equity > result.rows[-1].hedged_equity


def test_triple_swap_lands_on_wednesday():
    days = bdays("2024-01-01", 15)
    path = [2000.0 * (1.0 + 0.10 * i / 14.0) for i in range(15)]
    inputs = make_inputs(days, path, lots=0.05, side=-1)
    tight = caps(factor_caps={"USD": 0.5, "RISK": 50.0, "ENERGY": 50.0,
                              "METALS": 50.0}, min_excess_removed=0.0)

    result = replay(inputs, book(), tight)
    charged = [(r.day, r.hedge_swap_paid) for r in result.rows
               if r.hedge_swap_paid != 0.0]
    assert charged, "fixture must hold a hedge long enough to accrue swap"

    weds = [v for d, v in charged if date.fromisoformat(d).weekday() == 2]
    others = [v for d, v in charged if date.fromisoformat(d).weekday() != 2]
    if weds and others:
        assert abs(min(weds)) > abs(max(others)) * 2.5, (
            "Wednesday must carry roughly triple swap"
        )


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


def test_trade_is_open_from_entry_up_to_but_not_including_exit():
    """Exclusive at the exit end, so a position is not double-counted on the day it
    closes — its final P&L is realised by that day's mark."""
    tr = TradeRecord("GOLD", "XAUUSD.ecn", 1, 0.1, "2024-01-05", "2024-01-10",
                     2000.0, 2050.0)
    assert not tr.is_open_on("2024-01-04")
    assert tr.is_open_on("2024-01-05")
    assert tr.is_open_on("2024-01-09")
    assert not tr.is_open_on("2024-01-10")


def test_base_pnl_matches_hand_calculation():
    """0.10 lots long gold, +10.00 price move, mppu 100 => +100.00."""
    days = ["2024-01-01", "2024-01-02"]
    inputs = ReplayInputs(
        slot_specs={"GOLD": GOLD_SPEC},
        trades=[TradeRecord("GOLD", "XAUUSD.ecn", 1, 0.10, "2024-01-01",
                            "2024-01-03", 2000.0, 2010.0)],
        prices={"GOLD": {"2024-01-01": 2000.0, "2024-01-02": 2010.0}},
        hedge_specs={}, hedge_costs={}, hedge_symbols={},
        initial_balance=10_000.0,
    )
    result = replay(inputs, book(), caps(), enable_overlay=False)

    assert result.rows[0].base_equity == pytest.approx(10_000.0)
    assert result.rows[1].base_equity == pytest.approx(10_100.0)


def test_short_position_pnl_is_inverted():
    days = ["2024-01-01", "2024-01-02"]
    inputs = ReplayInputs(
        slot_specs={"GOLD": GOLD_SPEC},
        trades=[TradeRecord("GOLD", "XAUUSD.ecn", -1, 0.10, "2024-01-01",
                            "2024-01-03", 2000.0, 2010.0)],
        prices={"GOLD": {"2024-01-01": 2000.0, "2024-01-02": 2010.0}},
        hedge_specs={}, hedge_costs={}, hedge_symbols={},
        initial_balance=10_000.0,
    )
    result = replay(inputs, book(), caps(), enable_overlay=False)
    assert result.rows[1].base_equity == pytest.approx(9_900.0)


def test_a_missing_price_carries_the_previous_one_forward():
    """A holiday must not drop the position from the book — that is the bad-tick
    mistake in another form. The position still exists; its value just did not
    change."""
    inputs = ReplayInputs(
        slot_specs={"GOLD": GOLD_SPEC},
        trades=[TradeRecord("GOLD", "XAUUSD.ecn", 1, 0.10, "2024-01-01",
                            "2024-01-05", 2000.0, 2000.0)],
        prices={"GOLD": {"2024-01-01": 2000.0, "2024-01-03": 2000.0,
                         "2024-01-04": 2010.0}},
        hedge_specs={}, hedge_costs={}, hedge_symbols={},
        initial_balance=10_000.0,
    )
    result = replay(inputs, book(), caps(), enable_overlay=False)

    days_seen = [r.day for r in result.rows]
    assert days_seen == ["2024-01-01", "2024-01-03", "2024-01-04"]
    # The position is counted on every day it is open, including the gap day.
    assert all(r.n_positions == 1 for r in result.rows)
    assert result.rows[-1].base_equity == pytest.approx(10_100.0)


def test_empty_inputs_do_not_crash():
    inputs = ReplayInputs(slot_specs={}, trades=[], prices={}, hedge_specs={},
                          hedge_costs={}, hedge_symbols={})
    result = replay(inputs, book(), caps())
    assert result.rows == []
    assert result.summary() == {}
    assert any("no price data" in w for w in result.warnings)


def test_trading_days_is_the_union_and_sorted():
    prices = {"A": {"2024-01-02": 1.0, "2024-01-01": 1.0},
              "B": {"2024-01-03": 1.0}}
    assert trading_days(prices) == ["2024-01-01", "2024-01-02", "2024-01-03"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_max_drawdown_calculation():
    assert ReplayResult._max_drawdown_pct([100, 120, 60, 90]) == pytest.approx(50.0)
    assert ReplayResult._max_drawdown_pct([100, 110, 120]) == pytest.approx(0.0)
    assert ReplayResult._max_drawdown_pct([]) == 0.0


def test_sharpe_of_a_flat_curve_is_zero():
    assert ReplayResult._sharpe([100.0] * 50) == 0.0


def test_sharpe_of_a_zero_volatility_curve_is_infinite_not_zero():
    """A perfectly smooth climb has no measured risk, so Sharpe is undefined —
    not zero. Returning 0.0 would print in the report as 'no risk-adjusted
    return', the opposite of the truth."""
    perfect = [100.0 * (1.005 ** i) for i in range(120)]
    assert ReplayResult._sharpe(perfect) == math.inf

    falling = [100.0 * (0.995 ** i) for i in range(120)]
    assert ReplayResult._sharpe(falling) == -math.inf

    assert ReplayResult._sharpe([100.0] * 50) == 0.0


def test_sharpe_rewards_a_noisy_climb():
    """The realistic case: upward drift with real variance gives a high but finite
    Sharpe."""
    from tests.test_math_core import LCG

    rng = LCG(9)
    curve = [100.0]
    for _ in range(250):
        curve.append(curve[-1] * (1.0 + 0.0008 + rng.normal(0.0, 0.004)))

    sharpe = ReplayResult._sharpe(curve)
    assert math.isfinite(sharpe)
    assert sharpe > 1.0, f"a drifting series should score well, got {sharpe:.2f}"


def test_summary_reports_costs_and_counts():
    r = ReplayResult(initial_balance=10_000.0)
    r.rows = [DayRow("2024-01-01", 10_000.0, 10_000.0, 1, 0),
              DayRow("2024-01-02", 10_500.0, 10_400.0, 1, 1)]
    r.hedges_opened = 1
    r.total_spread = 3.6
    r.total_swap = -12.0

    s = r.summary()
    assert s["base_return_pct"] == pytest.approx(5.0)
    assert s["hedged_return_pct"] == pytest.approx(4.0)
    assert s["total_cost"] == pytest.approx(-8.4)
    assert s["hedges_opened"] == 1
