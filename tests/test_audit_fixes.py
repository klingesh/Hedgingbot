"""
Regression tests for the findings in Hedgingbot_Deep_Architecture_Risk_Report.pdf
(read-only architecture and risk audit).

Every claim in that report was verified against the code before being fixed, and
all ten were accurate. Each test below names the audit item it pins, so a future
refactor that reintroduces one fails with an explanation rather than a diff.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from src.exposure.model import InstrumentSpec, Position, compute_exposure
from src.factors.book import BetaBook
from src.overlay._attempt import total_risk_after
from src.overlay.decision import (
    HedgeCaps,
    HedgeInstrument,
    _collateral_damage,
    hedge_decide,
)
from src.state.lock import AlreadyRunning, SingleInstance
from src.state.shared import SharedPortfolioState

FACTORS = ("USD", "RISK", "ENERGY", "METALS")
SIGMA = {"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022, "METALS": 0.009}
EQUITY = 10_000.0


def book(extra=None) -> BetaBook:
    betas = {
        "GOLD":   {"USD": -1.03, "RISK": 0.21, "ENERGY": 0.00, "METALS": 0.48},
        "EURUSD": {"USD": -0.94, "RISK": 0.00, "ENERGY": 0.00, "METALS": -0.01},
        "SP500":  {"USD": -0.65, "RISK": 0.95, "ENERGY": 0.00, "METALS": 0.00},
        "SILVER": {"USD": -1.89, "RISK": 0.94, "ENERGY": 0.02, "METALS": 1.07},
        # Deliberately dirty: a strong USD lever that also loads RISK heavily.
        "DIRTY":  {"USD": -1.00, "RISK": 1.50, "ENERGY": 0.00, "METALS": 0.00},
    }
    if extra:
        betas.update(extra)
    return BetaBook(
        factors=FACTORS, betas=betas,
        sigma={k: 0.012 for k in betas}, resid_sigma={k: 0.004 for k in betas},
        r_squared={k: 0.8 for k in betas}, n_obs={k: 417 for k in betas},
        factor_sigma=SIGMA,
    )


GOLD_SPEC = InstrumentSpec("XAUUSD.ecn", 0.01, 1.0, 0.01, 0.01, 100.0, logical="GOLD")
EUR_SPEC = InstrumentSpec("EURUSD.ecn", 0.00001, 1.0, 0.01, 0.01, 100.0,
                          logical="EURUSD")
SP_SPEC = InstrumentSpec("US500.ecn", 0.1, 0.1, 0.01, 0.01, 100.0, logical="SP500")
AG_SPEC = InstrumentSpec("XAGUSD.ecn", 0.001, 5.0, 0.01, 0.01, 100.0, logical="SILVER")


def caps(**kw) -> HedgeCaps:
    base = dict(factor_caps={"USD": 1.5, "RISK": 1.5, "ENERGY": 1.0, "METALS": 1.25},
                max_hedge_gross_pct=1000.0)
    base.update(kw)
    return HedgeCaps(**base)


def eur(bid=1.157, ask=1.15704) -> HedgeInstrument:
    return HedgeInstrument("EURUSD", "EURUSD.ecn", EUR_SPEC, bid, ask)


def sp500() -> HedgeInstrument:
    return HedgeInstrument("SP500", "US500.ecn", SP_SPEC, 7785.3, 7785.5)


def silver() -> HedgeInstrument:
    return HedgeInstrument("SILVER", "XAGUSD.ecn", AG_SPEC, 64.68, 64.704)


def gold_report(lots: float, side: int = -1, b=None):
    return compute_exposure(
        [Position("XAUUSD.ecn", side, lots, 4375.96, magic=0, logical="GOLD")],
        {"XAUUSD.ecn": GOLD_SPEC}, b or book(), EQUITY,
    )


# ---------------------------------------------------------------------------
# Audit item 2 — collateral check blind spot
# ---------------------------------------------------------------------------


def test_collateral_check_catches_a_new_breach():
    """Case 1, the only one the original version handled."""
    c = caps()
    before = {"USD": 3.0, "RISK": 0.5, "ENERGY": 0.0, "METALS": 0.0}
    after = {"USD": 1.2, "RISK": 2.0, "ENERGY": 0.0, "METALS": 0.0}
    assert _collateral_damage(before, after, c, exclude="USD") == ["RISK"]


def test_collateral_check_catches_a_WORSENED_existing_breach():
    """Case 2. The original test was `abs(before) <= cap and abs(after) > cap`, so
    a factor ALREADY in breach could be pushed arbitrarily further out in silence —
    and on a book with several simultaneous breaches that is the normal case."""
    c = caps()
    before = {"USD": 3.0, "RISK": 2.0, "ENERGY": 0.0, "METALS": 0.0}
    after = {"USD": 1.2, "RISK": 5.0, "ENERGY": 0.0, "METALS": 0.0}

    damage = _collateral_damage(before, after, c, exclude="USD")
    assert "RISK" in damage, (
        "a breach at 2.0x pushed to 5.0x must be flagged as damage"
    )


def test_collateral_check_allows_improving_an_existing_breach():
    """Reducing another breach is not damage, and must not block the hedge."""
    c = caps()
    before = {"USD": 3.0, "RISK": 2.0, "ENERGY": 0.0, "METALS": 0.0}
    after = {"USD": 1.2, "RISK": 1.8, "ENERGY": 0.0, "METALS": 0.0}
    assert _collateral_damage(before, after, c, exclude="USD") == []


def test_collateral_check_covers_UNCAPPED_factors():
    """Case 3. The original skipped any factor with no cap entirely, so a hedge
    could load up a factor nobody is watching. 'No cap configured' usually means
    nobody got round to it, not 'unlimited is fine'."""
    partial = HedgeCaps(factor_caps={"USD": 1.5})      # RISK/ENERGY/METALS uncapped
    before = {"USD": 3.0, "RISK": 0.1}
    after = {"USD": 1.2, "RISK": 4.0}

    assert "RISK" in _collateral_damage(before, after, partial, exclude="USD")


def test_collateral_check_tolerates_floating_point_dust():
    c = caps()
    before = {"USD": 3.0, "RISK": 2.0}
    after = {"USD": 1.2, "RISK": 2.0000000001}
    assert _collateral_damage(before, after, c, exclude="USD") == []


# ---------------------------------------------------------------------------
# Audit item 9 — breach starvation
# ---------------------------------------------------------------------------


def test_an_unhedgeable_worst_breach_no_longer_blocks_a_hedgeable_one():
    """THE starvation fix.

    Gold short at size: USD is enormous and unhedgeable (futile), while RISK also
    breaches and IS hedgeable with SP500. The original code took only the worst
    breach and returned, so it did nothing at all.
    """
    rep = gold_report(1.0)                      # USD ~ +45x, RISK ~ -9x

    assert abs(rep.factor_leverage["USD"]) > 20, "fixture: USD must be extreme"
    assert abs(rep.factor_leverage["RISK"]) > 1.5 * 1.25, "fixture: RISK must breach"

    plan = hedge_decide(rep, book(), caps(), [eur(), sp500()])

    assert plan.has_trades, f"should fall through to RISK, got: {plan.summary()}"
    a = plan.trades()[0]
    assert a.factor == "RISK"
    assert a.logical == "SP500"
    assert "could not hedge USD" in plan.reason


def test_when_nothing_is_hedgeable_the_worst_factor_is_still_the_headline():
    """Falling through must not bury the most important unaddressed risk."""
    rep = gold_report(1.0)
    # Only EURUSD available: it cannot help RISK or METALS (betas ~0).
    plan = hedge_decide(rep, book(), caps(), [eur()])

    assert not plan.has_trades
    action = plan.actions[0]
    assert action.factor == "USD", "the worst breach must lead the report"
    assert "REDUCE THE POSITION" in action.reason
    assert "futile" in plan.reason
    assert "Also unhedgeable" in action.reason, "the rest must still be reported"


def test_a_single_candidate_failing_does_not_stop_the_candidate_search():
    """Audit: 'Minimum-lot early return: failure to fit one candidate can stop the
    search before another viable candidate is considered.'"""
    rep = gold_report(0.10)          # modest USD breach, reachable
    coarse = InstrumentSpec("COARSE", 0.00001, 1.0, volume_min=50.0,
                            volume_step=50.0, volume_max=100.0, logical="EURUSD")
    unusable = HedgeInstrument("EURUSD", "COARSE", coarse, 1.157, 1.15704)

    # The coarse leg cannot fit; the fine one can. Order matters: coarse first.
    plan = hedge_decide(rep, book(), caps(), [unusable, eur()])

    assert plan.has_trades, f"must try the second candidate: {plan.summary()}"
    assert plan.trades()[0].symbol == "EURUSD.ecn"


# ---------------------------------------------------------------------------
# Audit section 6 — risk must actually decrease
# ---------------------------------------------------------------------------


def test_total_risk_after_matches_the_report_when_nothing_is_added():
    rep = gold_report(0.10)
    same = total_risk_after(rep, book(), 0.0, "EURUSD", SIGMA)
    assert same == pytest.approx(rep.portfolio_daily_risk, rel=1e-9)


def test_a_hedge_that_would_raise_total_risk_is_refused():
    """Cap compliance is a proxy; lower risk is the objective. A dirty lever can
    satisfy the USD cap while raising total risk, because contribution scales with
    each factor's own volatility rather than with its leverage number."""
    rep = gold_report(0.10)
    dirty = HedgeInstrument("DIRTY", "DIRTY", EUR_SPEC, 1.157, 1.15704)

    # Thresholds relaxed so purity/collateral do not reject it first — we want the
    # RISK check to be the thing that decides.
    permissive = caps(factor_caps={"USD": 1.5}, min_purity=0.0, min_abs_beta=0.0)
    plan = hedge_decide(rep, book(), permissive, [dirty])

    if plan.has_trades:
        # If it was allowed, risk must genuinely have fallen.
        a = plan.trades()[0]
        per_lot = EUR_SPEC.notional_per_lot(1.15704 if a.side > 0 else 1.157)
        after = total_risk_after(rep, book(), a.side * a.lots * per_lot, "DIRTY", SIGMA)
        assert after < rep.portfolio_daily_risk
    else:
        assert any(w in plan.reason for w in
                   ("not reduce total risk", "damage", "futile")), plan.reason


def test_an_effective_hedge_reduces_measured_total_risk():
    rep = gold_report(0.10)
    plan = hedge_decide(rep, book(), caps(), [eur()])

    a = plan.trades()[0]
    per_lot = EUR_SPEC.notional_per_lot(1.15704 if a.side > 0 else 1.157)
    after = total_risk_after(rep, book(), a.side * a.lots * per_lot, "EURUSD", SIGMA)
    assert after < rep.portfolio_daily_risk
    assert "total risk" in a.reason, "the reason must state the risk change"


# ---------------------------------------------------------------------------
# Audit section 3 — uncapped factors silently unlimited
# ---------------------------------------------------------------------------


def test_uncapped_factors_are_warned_about_not_swallowed():
    """'A configuration typo could remove a risk limit without an explicit
    failure.'"""
    rep = gold_report(0.10)
    partial = HedgeCaps(factor_caps={"USD": 1.5})

    plan = hedge_decide(rep, book(), partial, [eur()])

    joined = " ".join(plan.warnings)
    assert "no cap configured" in joined
    assert "UNLIMITED" in joined
    for f in ("RISK", "ENERGY", "METALS"):
        assert f in joined


def test_fully_capped_config_produces_no_cap_warning():
    plan = hedge_decide(gold_report(0.10), book(), caps(), [eur()])
    assert not any("no cap configured" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# Audit item 10 — unwind provenance
# ---------------------------------------------------------------------------


def test_unwind_uses_the_RECORDED_factor_not_the_current_ranking():
    """A hedge opened for USD must be judged on USD, even if a beta refresh makes
    its instrument rank highest for something else."""
    # SILVER ranks highest for METALS, but say it was opened as a USD hedge.
    positions = [
        Position("XAUUSD.ecn", -1, 0.001, 4375.96, magic=0, logical="GOLD"),
        Position("XAGUSD.ecn", 1, 0.05, 64.68, ticket=555, magic=990012,
                 logical="SILVER"),
    ]
    specs = {"XAUUSD.ecn": GOLD_SPEC, "XAGUSD.ecn": AG_SPEC}
    rep = compute_exposure(positions, specs, book(), EQUITY)
    hedges = [p for p in positions if p.magic == 990012]

    plan = hedge_decide(rep, book(), caps(), [silver()],
                        existing_hedges=hedges, hedge_specs=specs,
                        hedge_provenance={555: "USD"})

    # Whatever it decides, it must not have guessed the factor.
    assert not any("guessed" in w for w in plan.warnings)
    closes = [a for a in plan.actions if a.action == "close"]
    if closes:
        assert closes[0].factor == "USD", "must unwind against the RECORDED factor"


def test_missing_provenance_warns_that_the_factor_was_guessed():
    positions = [
        Position("XAUUSD.ecn", -1, 0.001, 4375.96, magic=0, logical="GOLD"),
        Position("XAGUSD.ecn", 1, 0.05, 64.68, ticket=777, magic=990012,
                 logical="SILVER"),
    ]
    specs = {"XAUUSD.ecn": GOLD_SPEC, "XAGUSD.ecn": AG_SPEC}
    rep = compute_exposure(positions, specs, book(), EQUITY)

    plan = hedge_decide(rep, book(), caps(), [silver()],
                        existing_hedges=[positions[1]], hedge_specs=specs)

    joined = " ".join(plan.warnings)
    assert "no recorded factor" in joined
    assert "777" in joined
    assert "order comment" in joined, "must say how Phase 2 should fix it"


def test_provenance_naming_a_factor_that_no_longer_exists_is_handled():
    positions = [
        Position("XAUUSD.ecn", -1, 0.001, 4375.96, magic=0, logical="GOLD"),
        Position("XAGUSD.ecn", 1, 0.05, 64.68, ticket=99, magic=990012,
                 logical="SILVER"),
    ]
    specs = {"XAUUSD.ecn": GOLD_SPEC, "XAGUSD.ecn": AG_SPEC}
    rep = compute_exposure(positions, specs, book(), EQUITY)

    plan = hedge_decide(rep, book(), caps(), [silver()],
                        existing_hedges=[positions[1]], hedge_specs=specs,
                        hedge_provenance={99: "CRYPTO"})

    joined = " ".join(plan.warnings)
    assert "not in the current factor set" in joined


# ---------------------------------------------------------------------------
# Audit item 3 — shared-state clobber
# ---------------------------------------------------------------------------


def test_a_stale_writer_cannot_clear_a_halt(tmp_path):
    """The precise failure: two processes hold copies, one halts, the other saves
    its stale copy and resurrects halted=False."""
    path = os.path.join(str(tmp_path), "portfolio_state.json")

    overlay = SharedPortfolioState.load(path)
    overlay.sync_baseline(10_000.0)
    overlay.save(path, by="overlay")

    stale = SharedPortfolioState.load(path)        # overlay's view, pre-halt

    trader = SharedPortfolioState.load(path)
    assert trader.halt("total drawdown 20%") is True
    trader.save(path, by="trader")

    stale.note_equity(10_100.0)
    stale.save(path, by="overlay")                 # would have clobbered

    final = SharedPortfolioState.load(path)
    assert final.halted is True, "a halt must survive a stale writer"
    assert final.halt_reason == "total drawdown 20%"
    # And the stale writer's own view is corrected rather than left disagreeing.
    assert stale.halted is True


def test_a_stale_writer_cannot_regress_the_high_water_mark(tmp_path):
    path = os.path.join(str(tmp_path), "s.json")

    a = SharedPortfolioState.load(path)
    a.sync_baseline(10_000.0)
    a.save(path)

    stale = SharedPortfolioState.load(path)

    b = SharedPortfolioState.load(path)
    b.note_equity(25_000.0)
    b.save(path)

    stale.note_equity(10_500.0)
    stale.save(path)

    assert SharedPortfolioState.load(path).peak_equity == 25_000.0


def test_a_stale_writer_cannot_hand_back_spent_hedge_budget(tmp_path):
    path = os.path.join(str(tmp_path), "s.json")

    a = SharedPortfolioState.load(path)
    a.roll_day("2026-08-16", 10_000.0)
    a.save(path)

    stale = SharedPortfolioState.load(path)

    b = SharedPortfolioState.load(path)
    b.roll_day("2026-08-16", 10_000.0)
    for _ in range(4):
        b.note_hedge()
    b.save(path)

    stale.roll_day("2026-08-16", 10_000.0)
    stale.note_hedge()
    stale.save(path)

    assert SharedPortfolioState.load(path).hedge_count_today >= 4


def test_a_new_day_legitimately_resets_the_budget(tmp_path):
    """The merge must not make hedge_count_today permanently sticky."""
    path = os.path.join(str(tmp_path), "s.json")

    a = SharedPortfolioState.load(path)
    a.roll_day("2026-08-16", 10_000.0)
    for _ in range(6):
        a.note_hedge()
    a.save(path)

    b = SharedPortfolioState.load(path)
    assert b.roll_day("2026-08-17", 10_000.0) is True
    b.save(path)

    assert SharedPortfolioState.load(path).hedge_count_today == 0


def test_day_halt_does_not_leak_across_days(tmp_path):
    path = os.path.join(str(tmp_path), "s.json")

    a = SharedPortfolioState.load(path)
    a.roll_day("2026-08-16", 10_000.0)
    a.halt_day("daily loss")
    a.save(path)

    b = SharedPortfolioState.load(path)
    b.roll_day("2026-08-17", 10_000.0)
    b.save(path)

    final = SharedPortfolioState.load(path)
    assert final.day == "2026-08-17"
    assert final.day_halted is False


# ---------------------------------------------------------------------------
# Audit item 7 — single-instance lock
# ---------------------------------------------------------------------------


def test_lock_is_exclusive(tmp_path):
    path = os.path.join(str(tmp_path), "hedge.lock")
    first = SingleInstance(path).acquire()
    try:
        with pytest.raises(AlreadyRunning, match="already holds"):
            SingleInstance(path).acquire()
    finally:
        first.release()


def test_lock_is_reacquirable_after_release(tmp_path):
    path = os.path.join(str(tmp_path), "hedge.lock")
    SingleInstance(path).acquire().release()
    second = SingleInstance(path).acquire()
    second.release()


def test_lock_survives_a_stale_file(tmp_path):
    """An OS lock is released by the kernel when a process dies, so a leftover
    lock FILE must not block startup the way a PID file would."""
    path = os.path.join(str(tmp_path), "hedge.lock")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("pid=999999\n")

    inst = SingleInstance(path).acquire()
    inst.release()


def test_lock_works_as_a_context_manager(tmp_path):
    path = os.path.join(str(tmp_path), "hedge.lock")
    with SingleInstance(path):
        with pytest.raises(AlreadyRunning):
            SingleInstance(path).acquire()
    SingleInstance(path).acquire().release()


def test_hold_keeps_a_module_level_reference(tmp_path):
    """Tradingbot shipped a bug where the lock object was discarded, garbage
    collected, its handle closed, and the kernel dropped the lock silently."""
    import gc

    from src.state import lock as lock_mod

    path = os.path.join(str(tmp_path), "hedge.lock")
    lock_mod.release()
    try:
        lock_mod.hold(path)
        gc.collect()
        assert lock_mod._held is not None
        with pytest.raises(AlreadyRunning):
            SingleInstance(path).acquire()
        # Idempotent within one process.
        assert lock_mod.hold(path) is lock_mod._held
    finally:
        lock_mod.release()


def test_lock_creates_missing_directories(tmp_path):
    path = os.path.join(str(tmp_path), "deep", "nested", "hedge.lock")
    inst = SingleInstance(path).acquire()
    try:
        assert os.path.exists(path)
    finally:
        inst.release()
