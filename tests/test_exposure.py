"""
Tests for the exposure model.

Numbers here are chosen so every expected value can be verified by hand from the
docstring formulas — no magic constants. The scenario mirrors Tradingbot's actual
DEFAULT_PORTFOLIO so the concentration assertions mean something real.
"""

from __future__ import annotations

import math

import pytest

from src.exposure.model import (
    InstrumentSpec,
    Position,
    compute_exposure,
    signed_notional,
)
from src.factors.book import BetaBook

FACTORS = ("USD", "RISK", "ENERGY", "METALS")

# Daily factor vols, roughly realistic.
FACTOR_SIGMA = {"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022, "METALS": 0.009}


def make_book() -> BetaBook:
    """A BetaBook shaped like the real Tradingbot book plus hedge candidates."""
    betas = {
        "GOLD":     {"USD": -0.90, "RISK": 0.00, "ENERGY": 0.10, "METALS": 1.00},
        "SILVER":   {"USD": -1.00, "RISK": 0.30, "ENERGY": 0.10, "METALS": 0.80},
        "PLATINUM": {"USD": -0.80, "RISK": 0.40, "ENERGY": 0.20, "METALS": 0.50},
        "BRENT":    {"USD": -0.40, "RISK": 0.40, "ENERGY": 1.00, "METALS": 0.00},
        "NATGAS":   {"USD": -0.20, "RISK": 0.10, "ENERGY": 0.50, "METALS": 0.00},
        "AUDUSD":   {"USD": -0.90, "RISK": 0.50, "ENERGY": 0.10, "METALS": 0.20},
        "USDJPY":   {"USD": 0.80, "RISK": 0.30, "ENERGY": 0.00, "METALS": -0.20},
        "GBPJPY":   {"USD": 0.00, "RISK": 0.50, "ENERGY": 0.00, "METALS": -0.10},
        "EURUSD":   {"USD": -1.00, "RISK": 0.05, "ENERGY": 0.00, "METALS": 0.05},
        "SP500":    {"USD": -0.10, "RISK": 1.00, "ENERGY": 0.10, "METALS": 0.00},
    }
    sigma = {k: 0.012 for k in betas}
    resid = {k: 0.004 for k in betas}
    return BetaBook(
        factors=FACTORS, betas=betas, sigma=sigma, resid_sigma=resid,
        r_squared={k: 0.7 for k in betas}, n_obs={k: 2000 for k in betas},
        factor_sigma=FACTOR_SIGMA,
    )


# Gold CFD: 100 oz per lot => tick_value/tick_size = 1.00/0.01 = 100.
GOLD_SPEC = InstrumentSpec("XAUUSD", tick_size=0.01, tick_value=1.00,
                           volume_min=0.01, volume_step=0.01, volume_max=50.0,
                           logical="GOLD")
# EURUSD: 100k per lot => 1.00/0.00001 = 100_000.
EURUSD_SPEC = InstrumentSpec("EURUSD", tick_size=0.00001, tick_value=1.00,
                             volume_min=0.01, volume_step=0.01, volume_max=100.0,
                             logical="EURUSD")
SILVER_SPEC = InstrumentSpec("XAGUSD", tick_size=0.001, tick_value=5.00,
                             logical="SILVER")


# ---------------------------------------------------------------------------
# Specs and notional
# ---------------------------------------------------------------------------


def test_money_per_price_unit_is_tick_value_over_tick_size():
    assert GOLD_SPEC.money_per_price_unit_per_lot == pytest.approx(100.0)
    assert EURUSD_SPEC.money_per_price_unit_per_lot == pytest.approx(100_000.0)


def test_notional_per_lot_matches_hand_calculation():
    # 100 * 2400 = 240,000 per lot of gold
    assert GOLD_SPEC.notional_per_lot(2400.0) == pytest.approx(240_000.0)
    # 100,000 * 1.08 = 108,000 per lot of EURUSD
    assert EURUSD_SPEC.notional_per_lot(1.08) == pytest.approx(108_000.0)


def test_signed_notional_signs_shorts_negative():
    long = Position("XAUUSD", side=1, volume=0.1, price=2400.0, logical="GOLD")
    short = Position("XAUUSD", side=-1, volume=0.1, price=2400.0, logical="GOLD")

    assert signed_notional(long, GOLD_SPEC) == pytest.approx(24_000.0)
    assert signed_notional(short, GOLD_SPEC) == pytest.approx(-24_000.0)


def test_spec_rejects_nonsense():
    with pytest.raises(ValueError, match="tick_size"):
        InstrumentSpec("X", tick_size=0.0, tick_value=1.0)
    with pytest.raises(ValueError, match="tick_value"):
        InstrumentSpec("X", tick_size=0.01, tick_value=-1.0)
    with pytest.raises(ValueError, match="volume_min .* volume_max"):
        InstrumentSpec("X", tick_size=0.01, tick_value=1.0,
                       volume_min=5.0, volume_max=1.0)


def test_position_rejects_nonsense():
    with pytest.raises(ValueError, match="side"):
        Position("X", side=0, volume=1.0, price=1.0)
    with pytest.raises(ValueError, match="volume"):
        Position("X", side=1, volume=0.0, price=1.0)
    with pytest.raises(ValueError, match="price"):
        Position("X", side=1, volume=1.0, price=-1.0)


# ---------------------------------------------------------------------------
# Factor exposure
# ---------------------------------------------------------------------------


def test_single_position_leverage_is_hand_checkable():
    """0.1 lot long gold at 2400 on 10k equity.

    notional      = 0.1 * 100 * 2400            = 24,000
    USD exposure  = 24,000 * -0.90              = -21,600
    USD leverage  = -21,600 / 10,000            = -2.16
    """
    book = make_book()
    pos = [Position("XAUUSD", 1, 0.1, 2400.0, logical="GOLD")]
    specs = {"XAUUSD": GOLD_SPEC}

    rep = compute_exposure(pos, specs, book, equity=10_000.0)

    assert rep.gross_notional == pytest.approx(24_000.0)
    assert rep.net_notional == pytest.approx(24_000.0)
    assert rep.factor_exposure["USD"] == pytest.approx(-21_600.0)
    assert rep.factor_leverage["USD"] == pytest.approx(-2.16)
    assert rep.factor_leverage["METALS"] == pytest.approx(2.40)
    assert rep.factor_leverage["ENERGY"] == pytest.approx(0.24)
    assert rep.factor_leverage["RISK"] == pytest.approx(0.0)
    assert rep.gross_leverage == pytest.approx(2.4)


def test_shorting_flips_every_factor_exposure():
    book = make_book()
    specs = {"XAUUSD": GOLD_SPEC}
    long = compute_exposure([Position("XAUUSD", 1, 0.1, 2400.0, logical="GOLD")],
                            specs, book, 10_000.0)
    short = compute_exposure([Position("XAUUSD", -1, 0.1, 2400.0, logical="GOLD")],
                             specs, book, 10_000.0)

    for f in FACTORS:
        assert short.factor_leverage[f] == pytest.approx(-long.factor_leverage[f])


def test_opposing_factor_exposures_net_out():
    """Long gold (short USD) plus long USDJPY (long USD) must reduce net USD."""
    book = make_book()
    usdjpy = InstrumentSpec("USDJPY", tick_size=0.001, tick_value=0.67,
                            logical="USDJPY")
    specs = {"XAUUSD": GOLD_SPEC, "USDJPY": usdjpy}

    gold_only = compute_exposure(
        [Position("XAUUSD", 1, 0.1, 2400.0, logical="GOLD")], specs, book, 10_000.0
    )
    both = compute_exposure(
        [Position("XAUUSD", 1, 0.1, 2400.0, logical="GOLD"),
         Position("USDJPY", 1, 0.5, 150.0, logical="USDJPY")],
        specs, book, 10_000.0,
    )

    assert abs(both.factor_leverage["USD"]) < abs(gold_only.factor_leverage["USD"])


def test_factor_variance_shares_sum_to_one():
    book = make_book()
    specs = {"XAUUSD": GOLD_SPEC, "EURUSD": EURUSD_SPEC}
    rep = compute_exposure(
        [Position("XAUUSD", 1, 0.1, 2400.0, logical="GOLD"),
         Position("EURUSD", -1, 0.2, 1.08, logical="EURUSD")],
        specs, book, 10_000.0,
    )
    shares = rep.variance_shares()
    assert math.fsum(shares.values()) == pytest.approx(1.0)
    assert all(v >= 0.0 for v in shares.values())


def test_portfolio_risk_is_quadrature_of_factor_and_idio():
    book = make_book()
    specs = {"XAUUSD": GOLD_SPEC}
    rep = compute_exposure([Position("XAUUSD", 1, 0.1, 2400.0, logical="GOLD")],
                           specs, book, 10_000.0)

    factor_var = math.fsum(v * v for v in rep.factor_daily_risk.values())
    expected = math.sqrt(factor_var + rep.idio_daily_risk ** 2)
    assert rep.portfolio_daily_risk == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The concentration question — the reason this repo exists
# ---------------------------------------------------------------------------


def _full_book_positions():
    """All 8 Tradingbot slots, long, roughly equal notional."""
    specs = {}
    positions = []
    setup = [
        ("XAUUSD", "GOLD", 0.01, 2400.0, 100.0),
        ("XAGUSD", "SILVER", 0.05, 30.0, 5000.0),
        ("XPTUSD", "PLATINUM", 0.05, 1000.0, 100.0),
        ("NGAS", "NATGAS", 0.10, 3.0, 10000.0),
        ("UKOIL", "BRENT", 0.05, 80.0, 1000.0),
        ("GBPJPY", "GBPJPY", 0.05, 190.0, 670.0),
        ("AUDUSD", "AUDUSD", 0.05, 0.65, 100000.0),
        ("USDJPY", "USDJPY", 0.05, 150.0, 670.0),
    ]
    for sym, logical, vol, price, mppu in setup:
        # Build a spec whose tick_value/tick_size gives the intended mppu.
        specs[sym] = InstrumentSpec(sym, tick_size=0.001, tick_value=0.001 * mppu,
                                    logical=logical)
        positions.append(Position(sym, 1, vol, price, logical=logical))
    return positions, specs


def test_eight_correlated_longs_are_not_eight_independent_bets():
    """The headline claim of the whole project, asserted.

    Eight simultaneous longs across correlated commodities and FX must report an
    effective bet count well below 8. If this ever passes trivially, the factor
    model has stopped working.
    """
    book = make_book()
    positions, specs = _full_book_positions()
    rep = compute_exposure(positions, specs, book, equity=10_000.0)

    assert len(rep.positions) == 8
    assert rep.effective_bet_count < 5.0, (
        f"8 correlated longs reported {rep.effective_bet_count:.2f} effective bets; "
        "the concentration metric is not detecting overlap"
    )
    assert rep.diversification_ratio > 1.0 / math.sqrt(8), (
        "a correlated book cannot diversify better than 8 independent bets"
    )

    # SOME factor must dominate — that is the concentration. Which one is an
    # empirical question, and deliberately NOT asserted here.
    #
    # It is tempting to assume USD dominates a commodity+FX long book. With these
    # betas it does not: USDJPY is long-dollar and partially cancels the metals'
    # short-dollar, while NATGAS and BRENT stack ENERGY in the SAME direction and
    # ENERGY is the most volatile factor. That is precisely the kind of thing you
    # cannot eyeball from a position list, and the reason the measurement exists.
    dom, share = rep.dominant_factor()
    assert dom in FACTORS
    assert share > 0.2, f"no factor exceeded 20% of variance (largest: {dom} {share:.2f})"

    # USD must still be materially present even though it is not the largest.
    assert abs(rep.factor_leverage["USD"]) > 0.1


def test_a_genuinely_hedged_book_reports_more_diversification():
    """Adding an offsetting position must IMPROVE the diversification ratio.

    This is the mechanism the overlay exploits, so it needs a direct test.
    """
    book = make_book()
    positions, specs = _full_book_positions()
    unhedged = compute_exposure(positions, specs, book, 10_000.0)

    # The book is net short USD (long metals). Buy USD via short EURUSD.
    specs["EURUSD"] = EURUSD_SPEC
    hedged = compute_exposure(
        positions + [Position("EURUSD", -1, 0.15, 1.08, magic=990012, logical="EURUSD")],
        specs, book, 10_000.0,
    )

    assert abs(hedged.factor_leverage["USD"]) < abs(unhedged.factor_leverage["USD"])
    assert hedged.diversification_ratio < unhedged.diversification_ratio
    assert hedged.portfolio_daily_risk < unhedged.portfolio_daily_risk


# ---------------------------------------------------------------------------
# Breaches
# ---------------------------------------------------------------------------


def test_breaches_report_signed_excess_only_when_over_cap():
    book = make_book()
    specs = {"XAUUSD": GOLD_SPEC}
    # USD leverage = -2.16, METALS = +2.40
    rep = compute_exposure([Position("XAUUSD", 1, 0.1, 2400.0, logical="GOLD")],
                           specs, book, 10_000.0)

    br = rep.breaches({"USD": 1.5, "METALS": 3.0, "RISK": 1.0, "ENERGY": 1.0})

    assert set(br) == {"USD"}
    assert br["USD"] == pytest.approx(-(2.16 - 1.5))   # sign follows the exposure
    assert "METALS" not in br, "2.40 is inside a 3.0 cap"


def test_factor_without_a_cap_is_never_a_breach():
    book = make_book()
    specs = {"XAUUSD": GOLD_SPEC}
    rep = compute_exposure([Position("XAUUSD", 1, 1.0, 2400.0, logical="GOLD")],
                           specs, book, 10_000.0)
    assert rep.breaches({"USD": 1.5}) .keys() == {"USD"}
    assert "METALS" not in rep.breaches({"USD": 1.5})


# ---------------------------------------------------------------------------
# Safety: unknown instruments must never read as zero risk
# ---------------------------------------------------------------------------


def test_unmapped_symbol_is_flagged_and_charged_risk_not_ignored():
    book = make_book()
    mystery = InstrumentSpec("SHIBUSD", tick_size=0.001, tick_value=1.0,
                             logical="SHIBUSD")
    specs = {"SHIBUSD": mystery}

    rep = compute_exposure([Position("SHIBUSD", 1, 1.0, 10.0, logical="SHIBUSD")],
                           specs, book, 10_000.0, unmapped_daily_vol=0.02)

    assert rep.unmapped_symbols == ["SHIBUSD"]
    # Contributes NO factor exposure (we cannot attribute it)...
    assert all(v == 0.0 for v in rep.factor_exposure.values())
    # ...but it absolutely must contribute risk.
    assert rep.idio_daily_risk > 0.0
    assert rep.portfolio_daily_risk > 0.0
    notional = 1.0 * 1000.0 * 10.0
    assert rep.idio_daily_risk == pytest.approx(notional * 0.02)


def test_missing_spec_raises_rather_than_silently_skipping():
    book = make_book()
    with pytest.raises(KeyError, match="no InstrumentSpec"):
        compute_exposure([Position("XAUUSD", 1, 0.1, 2400.0, logical="GOLD")],
                         {}, book, 10_000.0)


def test_bad_equity_raises():
    book = make_book()
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="equity"):
            compute_exposure([], {}, book, bad)


def test_empty_book_is_all_zeros_not_an_error():
    book = make_book()
    rep = compute_exposure([], {}, book, 10_000.0)

    assert rep.gross_notional == 0.0
    assert rep.portfolio_daily_risk == 0.0
    assert rep.diversification_ratio == 0.0
    assert rep.effective_bet_count == 0.0
    assert rep.breaches({"USD": 1.0}) == {}
    assert all(v == 0.0 for v in rep.variance_shares().values())


def test_position_logical_overrides_spec_logical():
    """The caller can relabel a position without editing the spec."""
    book = make_book()
    specs = {"XAUUSD": GOLD_SPEC}
    rep = compute_exposure(
        [Position("XAUUSD", 1, 0.1, 2400.0, logical="SILVER")], specs, book, 10_000.0
    )
    assert rep.positions[0].logical == "SILVER"
    # SILVER's USD beta is -1.00, so exposure is -24,000 not -21,600.
    assert rep.factor_exposure["USD"] == pytest.approx(-24_000.0)
