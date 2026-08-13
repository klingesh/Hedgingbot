"""
Tests for hedge_decide().

Most of these tests exist to prove the overlay does NOTHING in situations where a
naive implementation would trade. That is the point: the six anti-churn guards in
overlay/decision.py are the actual product, and each one gets a test that fails if
the guard is removed.
"""

from __future__ import annotations

import pytest

from src.exposure.model import InstrumentSpec, Position, compute_exposure
from src.factors.book import BetaBook
from src.overlay.decision import (
    HedgeCaps,
    HedgeInstrument,
    hedge_decide,
)

FACTORS = ("USD", "RISK", "ENERGY", "METALS")
FACTOR_SIGMA = {"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022, "METALS": 0.009}

EQUITY = 10_000.0


def make_book(extra: dict[str, dict[str, float]] | None = None) -> BetaBook:
    betas = {
        "GOLD":   {"USD": -0.90, "RISK": 0.00, "ENERGY": 0.10, "METALS": 1.00},
        # A near-pure USD lever: tiny betas elsewhere.
        "EURUSD": {"USD": -1.00, "RISK": 0.02, "ENERGY": 0.00, "METALS": 0.01},
        # A pure RISK lever.
        "SP500":  {"USD": 0.00, "RISK": 1.00, "ENERGY": 0.00, "METALS": 0.00},
        # Deliberately dirty: strong USD beta but also strong RISK beta.
        "DIRTY":  {"USD": -1.00, "RISK": 1.20, "ENERGY": 0.00, "METALS": 0.00},
    }
    if extra:
        betas.update(extra)
    return BetaBook(
        factors=FACTORS, betas=betas,
        sigma={k: 0.012 for k in betas}, resid_sigma={k: 0.004 for k in betas},
        r_squared={k: 0.7 for k in betas}, n_obs={k: 2000 for k in betas},
        factor_sigma=FACTOR_SIGMA,
    )


GOLD_SPEC = InstrumentSpec("XAUUSD", 0.01, 1.00, 0.01, 0.01, 50.0, logical="GOLD")
EURUSD_SPEC = InstrumentSpec("EURUSD", 0.00001, 1.00, 0.01, 0.01, 100.0, logical="EURUSD")
SP500_SPEC = InstrumentSpec("US500", 0.1, 1.00, 0.01, 0.01, 100.0, logical="SP500")
DIRTY_SPEC = InstrumentSpec("DIRTY", 0.00001, 1.00, 0.01, 0.01, 100.0, logical="DIRTY")


def eurusd(bid: float = 1.08, ask: float = 1.0801) -> HedgeInstrument:
    return HedgeInstrument("EURUSD", "EURUSD", EURUSD_SPEC, bid, ask)


def sp500(bid: float = 5000.0, ask: float = 5000.5) -> HedgeInstrument:
    return HedgeInstrument("SP500", "US500", SP500_SPEC, bid, ask)


def dirty() -> HedgeInstrument:
    return HedgeInstrument("DIRTY", "DIRTY", DIRTY_SPEC, 1.08, 1.0801)


def gold_report(lots: float, book: BetaBook, equity: float = EQUITY, side: int = 1):
    """Exposure of a single gold position — the simplest breach generator.

    lots=0.1 gives USD leverage -2.16 and METALS +2.40 at 10k equity.
    """
    return compute_exposure(
        [Position("XAUUSD", side, lots, 2400.0, magic=990011, logical="GOLD")],
        {"XAUUSD": GOLD_SPEC}, book, equity,
    )


def caps(**kw) -> HedgeCaps:
    base = dict(
        factor_caps={"USD": 1.50, "RISK": 1.50, "ENERGY": 1.00, "METALS": 3.00},
        hysteresis=0.25, target_fraction=0.80, unwind_band=0.60,
        max_hedge_gross_pct=400.0, min_purity=0.50, min_abs_beta=0.20,
        max_actions_per_cycle=1, max_hedges_per_day=6, allow_unmapped=False,
    )
    base.update(kw)
    return HedgeCaps(**base)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_unwind_band_above_target_fraction_is_rejected():
    """The configuration that guarantees churn must be impossible to express."""
    with pytest.raises(ValueError, match="unwind_band"):
        HedgeCaps(factor_caps={"USD": 1.0}, target_fraction=0.5, unwind_band=0.9)


def test_empty_caps_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        HedgeCaps(factor_caps={})


def test_non_positive_cap_rejected():
    with pytest.raises(ValueError, match="must be finite and > 0"):
        HedgeCaps(factor_caps={"USD": 0.0})


# ---------------------------------------------------------------------------
# Guard 1: hysteresis
# ---------------------------------------------------------------------------


def test_within_caps_does_nothing():
    book = make_book()
    rep = gold_report(0.05, book)      # USD leverage -1.08, inside a 1.5 cap
    plan = hedge_decide(rep, book, caps(), [eurusd()])

    assert not plan.has_trades
    assert plan.actions[0].action == "nothing"
    assert "within caps" in plan.reason


def test_breach_inside_hysteresis_band_holds_and_explains_why():
    """USD leverage -1.7 breaches the 1.5 cap but is below 1.5*1.25 = 1.875."""
    book = make_book()
    rep = gold_report(0.0787, book)
    assert 1.5 < abs(rep.factor_leverage["USD"]) < 1.875

    plan = hedge_decide(rep, book, caps(), [eurusd()])

    assert not plan.has_trades
    assert plan.actions[0].action == "nothing"
    assert "hysteresis" in plan.actions[0].reason
    assert plan.breaches, "the breach must still be REPORTED even though unactioned"


def test_breach_beyond_hysteresis_triggers_a_hedge():
    book = make_book()
    rep = gold_report(0.10, book)      # USD leverage -2.16 > 1.875
    plan = hedge_decide(rep, book, caps(), [eurusd()])

    trades = plan.trades()
    assert len(trades) == 1
    a = trades[0]
    assert a.action == "open"
    assert a.factor == "USD"
    assert a.logical == "EURUSD"
    # Book is SHORT usd (leverage -2.16); EURUSD beta to USD is -1.0, so to add
    # positive USD exposure we must SHORT EURUSD.
    assert a.side == -1
    assert a.lots > 0


# ---------------------------------------------------------------------------
# Guard 2: target undershoot
# ---------------------------------------------------------------------------


def test_hedge_targets_below_the_cap_not_at_it():
    """After hedging, |leverage| must land near cap*target_fraction, not at cap."""
    book = make_book()
    rep = gold_report(0.10, book)
    c = caps()
    plan = hedge_decide(rep, book, c, [eurusd()])
    a = plan.trades()[0]

    per_lot = EURUSD_SPEC.notional_per_lot(EURUSD_SPEC and a.side > 0 and 1.0801 or 1.08)
    added = a.side * a.lots * per_lot
    after = (rep.factor_exposure["USD"] + added * book.beta("EURUSD", "USD")) / EQUITY

    target = c.cap("USD") * c.target_fraction        # 1.20
    # Rounding down means we may not fully reach the target, but we must never
    # overshoot past the cap in the opposite direction.
    assert abs(after) <= c.cap("USD") + 1e-9
    assert abs(after) < abs(rep.factor_leverage["USD"]), "hedge must reduce exposure"
    assert abs(after) == pytest.approx(target, abs=0.15)


def test_hedge_never_flips_the_book_into_the_opposite_breach():
    book = make_book()
    for lots in (0.10, 0.15, 0.22, 0.35, 0.5):
        rep = gold_report(lots, book)
        c = caps()
        plan = hedge_decide(rep, book, c, [eurusd()])
        trades = plan.trades()
        if not trades:
            continue
        a = trades[0]
        price = 1.0801 if a.side > 0 else 1.08
        added = a.side * a.lots * EURUSD_SPEC.notional_per_lot(price)
        after = (rep.factor_exposure["USD"] + added * book.beta("EURUSD", "USD")) / EQUITY
        before = rep.factor_leverage["USD"]
        assert abs(after) < abs(before), f"lots={lots}: hedge increased exposure"
        # Sign must not flip past the cap on the other side.
        assert not (before < 0 and after > c.cap("USD")), f"lots={lots}: overshot"


# ---------------------------------------------------------------------------
# Guard 3: unwind band
# ---------------------------------------------------------------------------


def test_stale_hedge_is_closed_when_exposure_falls_away():
    """Exposure back below cap*unwind_band => close the hedge."""
    book = make_book()
    # Small gold position: USD leverage -0.216, well under 1.5*0.6 = 0.9.
    positions = [
        Position("XAUUSD", 1, 0.01, 2400.0, magic=990011, logical="GOLD"),
        Position("EURUSD", -1, 0.05, 1.08, ticket=555, magic=990012, logical="EURUSD"),
    ]
    specs = {"XAUUSD": GOLD_SPEC, "EURUSD": EURUSD_SPEC}
    rep = compute_exposure(positions, specs, book, EQUITY)

    hedges = [p for p in positions if p.magic == 990012]
    plan = hedge_decide(rep, book, caps(), [eurusd()],
                        existing_hedges=hedges, hedge_specs=specs)

    trades = plan.trades()
    assert len(trades) == 1
    assert trades[0].action == "close"
    assert trades[0].ticket == 555
    assert "unwind band" in trades[0].reason


def test_hedge_is_kept_while_exposure_sits_in_the_dead_zone():
    """Between unwind_band and cap, do nothing at all — no open, no close.

    This dead zone is what stops the overlay oscillating.
    """
    book = make_book()
    # Target USD leverage ~ -1.2, i.e. above 0.9 (unwind) and below 1.5 (cap).
    positions = [
        Position("XAUUSD", 1, 0.0556, 2400.0, magic=990011, logical="GOLD"),
        Position("EURUSD", -1, 0.01, 1.08, ticket=777, magic=990012, logical="EURUSD"),
    ]
    specs = {"XAUUSD": GOLD_SPEC, "EURUSD": EURUSD_SPEC}
    rep = compute_exposure(positions, specs, book, EQUITY)
    assert 0.9 < abs(rep.factor_leverage["USD"]) < 1.5

    plan = hedge_decide(rep, book, caps(), [eurusd()],
                        existing_hedges=[positions[1]], hedge_specs=specs)

    assert not plan.has_trades, f"should sit still, got: {plan.summary()}"


def test_unwind_is_not_blocked_by_a_spent_daily_budget():
    """Running out of budget must never trap the account in a stale hedge."""
    book = make_book()
    positions = [
        Position("XAUUSD", 1, 0.01, 2400.0, magic=990011, logical="GOLD"),
        Position("EURUSD", -1, 0.05, 1.08, ticket=555, magic=990012, logical="EURUSD"),
    ]
    specs = {"XAUUSD": GOLD_SPEC, "EURUSD": EURUSD_SPEC}
    rep = compute_exposure(positions, specs, book, EQUITY)

    plan = hedge_decide(rep, book, caps(), [eurusd()],
                        existing_hedges=[positions[1]], hedge_specs=specs,
                        hedges_today=99)

    assert plan.trades()[0].action == "close"


# ---------------------------------------------------------------------------
# Guard 4: minimum lot refusal
# ---------------------------------------------------------------------------


def test_tiny_required_hedge_is_refused_not_rounded_up():
    book = make_book()
    # Huge equity so the required hedge is a fraction of a lot.
    rep = gold_report(0.10, book, equity=100_000.0)
    coarse = InstrumentSpec("EURUSD", 0.00001, 1.00, volume_min=1.0,
                            volume_step=1.0, volume_max=100.0, logical="EURUSD")
    inst = HedgeInstrument("EURUSD", "EURUSD", coarse, 1.08, 1.0801)

    c = caps(factor_caps={"USD": 0.10, "RISK": 1.5, "ENERGY": 1.0, "METALS": 3.0})
    plan = hedge_decide(rep, book, c, [inst])

    assert not plan.has_trades
    assert plan.actions[0].action == "skip"
    assert "volume_min" in plan.actions[0].reason
    assert "refusing to round up" in plan.actions[0].reason


def test_lots_are_rounded_down_to_the_volume_step():
    """0.3 lots of gold needs ~0.489 lots of EURUSD; on a 0.10 step that must
    become 0.40, never 0.50.

    Rounding DOWN matters: rounding up would remove more exposure than the breach
    justifies and could flip the book into the opposite breach.
    """
    book = make_book()
    rep = gold_report(0.30, book)
    # METALS/ENERGY caps lifted so USD is unambiguously the worst breach, and the
    # gross ceiling lifted so it is the STEP that binds, not the ceiling.
    c = caps(factor_caps={"USD": 1.50, "RISK": 1.50, "ENERGY": 100.0, "METALS": 100.0},
             max_hedge_gross_pct=5000.0)
    stepped = InstrumentSpec("EURUSD", 0.00001, 1.00, volume_min=0.10,
                             volume_step=0.10, volume_max=100.0, logical="EURUSD")

    plan = hedge_decide(rep, book, c,
                        [HedgeInstrument("EURUSD", "EURUSD", stepped, 1.08, 1.0801)])

    a = plan.trades()[0]
    assert a.factor == "USD"
    assert a.lots == pytest.approx(0.40), (
        f"expected 0.489 rounded DOWN to 0.40, got {a.lots}"
    )


def test_required_size_below_the_step_is_refused_not_rounded_up():
    """The companion case: 0.1 lots of gold needs only ~0.089 lots of EURUSD,
    which is below a 0.10 minimum. The overlay must decline, not round up."""
    book = make_book()
    rep = gold_report(0.10, book)
    stepped = InstrumentSpec("EURUSD", 0.00001, 1.00, volume_min=0.10,
                             volume_step=0.10, volume_max=100.0, logical="EURUSD")

    plan = hedge_decide(rep, book, caps(),
                        [HedgeInstrument("EURUSD", "EURUSD", stepped, 1.08, 1.0801)])

    assert not plan.has_trades
    assert plan.actions[0].action == "skip"
    assert "volume_min" in plan.actions[0].reason


# ---------------------------------------------------------------------------
# Guard 5: collateral damage
# ---------------------------------------------------------------------------


def test_dirty_hedge_that_would_breach_another_factor_is_rejected():
    """A hedge that fixes USD by pushing RISK into breach must not be used.

    DIRTY has beta -1.0 to USD (so it is a valid USD lever on beta alone) but
    +1.2 to RISK. Sizing it to fix USD would create a large RISK exposure.
    """
    book = make_book()
    rep = gold_report(0.10, book)
    # A tight RISK cap makes the collateral damage unacceptable.
    c = caps(factor_caps={"USD": 1.50, "RISK": 0.05, "ENERGY": 1.00, "METALS": 3.00},
             min_purity=0.0, min_abs_beta=0.0)

    plan = hedge_decide(rep, book, c, [dirty()])

    assert not plan.has_trades, f"dirty hedge should be refused, got {plan.summary()}"
    assert plan.actions[0].action == "skip"


def test_clean_hedge_is_preferred_over_dirty_when_both_available():
    book = make_book()
    rep = gold_report(0.10, book)
    c = caps(min_purity=0.0, min_abs_beta=0.0)

    plan = hedge_decide(rep, book, c, [dirty(), eurusd()])

    a = plan.trades()[0]
    assert a.logical == "EURUSD", "the purer lever must win the ranking"


def test_low_purity_candidate_is_filtered_by_min_purity():
    book = make_book()
    rep = gold_report(0.10, book)
    plan = hedge_decide(rep, book, caps(min_purity=0.95), [dirty()])

    assert not plan.has_trades
    assert "purity" in plan.actions[0].reason


def test_low_beta_candidate_is_filtered_by_min_abs_beta():
    book = make_book(extra={"WEAK": {"USD": 0.05, "RISK": 0.0,
                                     "ENERGY": 0.0, "METALS": 0.0}})
    rep = gold_report(0.10, book)
    weak_spec = InstrumentSpec("WEAK", 0.00001, 1.0, logical="WEAK")
    plan = hedge_decide(rep, book, caps(),
                        [HedgeInstrument("WEAK", "WEAK", weak_spec, 1.0, 1.0001)])

    assert not plan.has_trades
    assert "beta" in plan.actions[0].reason


# ---------------------------------------------------------------------------
# Guard 6: daily budget
# ---------------------------------------------------------------------------


def test_daily_budget_stops_opening_and_says_the_caps_may_be_wrong():
    book = make_book()
    rep = gold_report(0.10, book)
    plan = hedge_decide(rep, book, caps(max_hedges_per_day=3), [eurusd()],
                        hedges_today=3)

    assert not plan.has_trades
    assert plan.actions[0].action == "skip"
    assert "budget" in plan.actions[0].reason
    assert "caps are too tight" in plan.actions[0].reason


# ---------------------------------------------------------------------------
# Safety gates
# ---------------------------------------------------------------------------


def test_unmapped_symbol_blocks_all_hedging():
    """Refuse to hedge a book the model cannot fully see."""
    book = make_book()
    mystery = InstrumentSpec("MYST", 0.001, 1.0, logical="MYST")
    rep = compute_exposure(
        [Position("XAUUSD", 1, 0.10, 2400.0, logical="GOLD"),
         Position("MYST", 1, 1.0, 100.0, logical="MYST")],
        {"XAUUSD": GOLD_SPEC, "MYST": mystery}, book, EQUITY,
    )

    plan = hedge_decide(rep, book, caps(), [eurusd()])

    assert plan.blocked
    assert not plan.has_trades
    assert "MYST" in plan.reason


def test_allow_unmapped_overrides_the_block():
    book = make_book()
    mystery = InstrumentSpec("MYST", 0.001, 1.0, logical="MYST")
    rep = compute_exposure(
        [Position("XAUUSD", 1, 0.10, 2400.0, logical="GOLD"),
         Position("MYST", 1, 1.0, 100.0, logical="MYST")],
        {"XAUUSD": GOLD_SPEC, "MYST": mystery}, book, EQUITY,
    )
    plan = hedge_decide(rep, book, caps(allow_unmapped=True), [eurusd()])
    assert not plan.blocked


def test_symbols_held_by_the_trader_are_never_traded():
    """On a NETTING account an opposing same-symbol order would REDUCE the
    trader's position instead of hedging it."""
    book = make_book()
    rep = gold_report(0.10, book)

    plan = hedge_decide(rep, book, caps(), [eurusd()],
                        avoid_symbols=frozenset({"EURUSD"}))

    assert not plan.has_trades
    assert "held by trader" in plan.actions[0].reason


def test_flatten_closes_every_hedge_regardless_of_exposure():
    book = make_book()
    rep = gold_report(0.10, book)     # still breaching
    hedges = [
        Position("EURUSD", -1, 0.05, 1.08, ticket=1, magic=990012, logical="EURUSD"),
        Position("US500", 1, 0.02, 5000.0, ticket=2, magic=990012, logical="SP500"),
    ]
    plan = hedge_decide(rep, book, caps(), [eurusd()], existing_hedges=hedges,
                        flatten=True)

    assert len(plan.trades()) == 2
    assert all(a.action == "close" for a in plan.trades())
    assert {a.ticket for a in plan.trades()} == {1, 2}


def test_flatten_with_no_hedges_is_a_no_op_not_an_error():
    book = make_book()
    rep = gold_report(0.10, book)
    plan = hedge_decide(rep, book, caps(), [], flatten=True)
    assert not plan.has_trades
    assert "no hedges open" in plan.actions[0].reason


def test_no_candidates_reports_the_unhedged_breach():
    book = make_book()
    rep = gold_report(0.10, book)
    plan = hedge_decide(rep, book, caps(), [])

    assert not plan.has_trades
    assert plan.actions[0].action == "skip"
    assert "no candidates supplied" in plan.actions[0].reason


def test_missing_price_disqualifies_a_candidate():
    book = make_book()
    rep = gold_report(0.10, book)
    dead = HedgeInstrument("EURUSD", "EURUSD", EURUSD_SPEC, 0.0, 0.0)
    plan = hedge_decide(rep, book, caps(), [dead])

    assert not plan.has_trades
    assert "price" in plan.actions[0].reason


def test_gross_hedge_ceiling_blocks_further_hedging():
    book = make_book()
    rep = gold_report(0.10, book)
    # One existing hedge already worth ~108,000 notional vs a 50% * 10k = 5,000 cap.
    big = Position("EURUSD", -1, 1.0, 1.08, ticket=9, magic=990012, logical="EURUSD")
    plan = hedge_decide(
        rep, book, caps(max_hedge_gross_pct=50.0), [eurusd()],
        existing_hedges=[big], hedge_specs={"EURUSD": EURUSD_SPEC},
    )

    assert not plan.has_trades
    assert "ceiling" in plan.actions[0].reason


# ---------------------------------------------------------------------------
# Direction correctness — the thing that must never be wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side,expected_hedge_side", [(1, -1), (-1, 1)])
def test_hedge_direction_opposes_the_exposure(side, expected_hedge_side):
    """Long gold is short USD -> short EURUSD (long USD). And vice versa.

    A sign error here would double the risk instead of removing it, so it is
    tested from both directions.
    """
    book = make_book()
    rep = gold_report(0.10, book, side=side)
    plan = hedge_decide(rep, book, caps(), [eurusd()])

    a = plan.trades()[0]
    assert a.side == expected_hedge_side


def test_hedging_reduces_measured_portfolio_risk_end_to_end():
    """Full loop: decide a hedge, apply it, re-measure. Risk must fall.

    If this fails, the overlay is moving risk around rather than reducing it,
    which is the entire failure mode it exists to avoid.
    """
    book = make_book()
    specs = {"XAUUSD": GOLD_SPEC, "EURUSD": EURUSD_SPEC}
    positions = [Position("XAUUSD", 1, 0.10, 2400.0, magic=990011, logical="GOLD")]

    before = compute_exposure(positions, specs, book, EQUITY)
    plan = hedge_decide(before, book, caps(), [eurusd()])
    a = plan.trades()[0]

    after = compute_exposure(
        positions + [Position(a.symbol, a.side, a.lots,
                              1.0801 if a.side > 0 else 1.08,
                              magic=990012, logical=a.logical)],
        specs, book, EQUITY,
    )

    assert abs(after.factor_leverage["USD"]) < abs(before.factor_leverage["USD"])
    assert after.factor_daily_risk["USD"] < before.factor_daily_risk["USD"]
    assert not after.breaches(caps().factor_caps).get("USD"), (
        "the USD breach should be resolved after applying the hedge"
    )


def test_repeated_cycles_converge_and_stop_trading():
    """Simulate the overlay running repeatedly on a static book.

    It must hedge once (or twice) and then sit still forever. A churning
    implementation would keep trading, and this test is what catches that.
    """
    book = make_book()
    specs = {"XAUUSD": GOLD_SPEC, "EURUSD": EURUSD_SPEC}
    positions = [Position("XAUUSD", 1, 0.10, 2400.0, magic=990011, logical="GOLD")]
    hedges: list[Position] = []
    c = caps()

    trade_cycles = 0
    for cycle in range(12):
        rep = compute_exposure(positions + hedges, specs, book, EQUITY)
        plan = hedge_decide(rep, book, c, [eurusd()],
                            existing_hedges=hedges, hedge_specs=specs,
                            hedges_today=trade_cycles)
        trades = plan.trades()
        if not trades:
            continue
        trade_cycles += 1
        for a in trades:
            if a.action == "open":
                hedges.append(Position(a.symbol, a.side, a.lots,
                                       1.0801 if a.side > 0 else 1.08,
                                       ticket=100 + cycle, magic=990012,
                                       logical=a.logical))
            elif a.action == "close":
                hedges = [h for h in hedges if h.ticket != a.ticket]

    assert trade_cycles <= 2, (
        f"overlay traded {trade_cycles} times on a static book — it is churning"
    )
    final = compute_exposure(positions + hedges, specs, book, EQUITY)
    assert abs(final.factor_leverage["USD"]) <= c.cap("USD") + 1e-9
