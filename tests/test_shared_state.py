"""
Tests for the shared portfolio state.

The high-water-mark tests are the important ones. Tradingbot shipped a real bug
where a 30-second auto-restart loop silently re-armed a spent kill switch, because
the drawdown baseline was re-seeded from current balance on startup. These tests
encode that lesson so the overlay cannot repeat it.
"""

from __future__ import annotations

import json
import os

import pytest

from src.state.shared import (
    SharedPortfolioState,
    read_trader_status,
    write_overlay_status,
)


# ---------------------------------------------------------------------------
# High-water mark — the anti-restart-bug guarantee
# ---------------------------------------------------------------------------


def test_peak_equity_ratchets_up_only():
    st = SharedPortfolioState()
    st.sync_baseline(10_000.0)
    assert st.peak_equity == 10_000.0
    assert st.start_balance == 10_000.0

    st.note_equity(12_000.0)
    assert st.peak_equity == 12_000.0

    st.note_equity(9_000.0)
    assert st.peak_equity == 12_000.0, "peak must never fall"


def test_sync_baseline_does_not_lower_the_peak_on_restart():
    """The exact Tradingbot bug: crash mid-drawdown, restart, and the kill switch
    must still know about the old peak."""
    st = SharedPortfolioState()
    st.sync_baseline(10_000.0)
    st.note_equity(15_000.0)

    # Simulated crash and restart with a drawn-down balance.
    st.sync_baseline(11_000.0)

    assert st.peak_equity == 15_000.0
    assert st.drawdown_percent(11_000.0) == pytest.approx(100 * 4000 / 15000)


def test_peak_survives_a_save_load_cycle(tmp_path):
    path = os.path.join(str(tmp_path), "state.json")
    st = SharedPortfolioState()
    st.sync_baseline(10_000.0)
    st.note_equity(20_000.0)
    st.save(path, by="test")

    reloaded = SharedPortfolioState.load(path)

    assert reloaded.peak_equity == 20_000.0
    assert reloaded.updated_by == "test"
    # And a fresh baseline sync after "restart" must not reset it.
    reloaded.sync_baseline(12_000.0)
    assert reloaded.peak_equity == 20_000.0


def test_zero_or_negative_balance_is_ignored_by_sync():
    st = SharedPortfolioState()
    st.sync_baseline(0.0)
    st.sync_baseline(-100.0)
    assert st.peak_equity == 0.0
    assert st.start_balance == 0.0


def test_drawdown_is_zero_before_any_baseline():
    st = SharedPortfolioState()
    assert st.drawdown_percent(5_000.0) == 0.0
    assert st.day_drawdown_percent(5_000.0) == 0.0


# ---------------------------------------------------------------------------
# Halts
# ---------------------------------------------------------------------------


def test_halt_returns_true_only_on_the_first_call():
    """So a 60-second poll loop does not write the same log line 1440x/day."""
    st = SharedPortfolioState()
    assert st.halt("drawdown 20%") is True
    assert st.halt("drawdown 21%") is False
    assert st.halt_reason == "drawdown 20%", "the FIRST reason must be preserved"
    assert st.halted_at


def test_halt_day_returns_true_only_once():
    st = SharedPortfolioState()
    assert st.halt_day("daily loss 6%") is True
    assert st.halt_day("daily loss 7%") is False


def test_roll_day_resets_daily_state_but_not_the_peak():
    st = SharedPortfolioState()
    st.sync_baseline(10_000.0)
    st.note_equity(12_000.0)
    st.roll_day("2026-08-12", 11_000.0)
    st.halt_day("daily loss")
    st.note_hedge()
    st.note_hedge()
    assert st.hedge_count_today == 2

    assert st.roll_day("2026-08-13", 10_500.0) is True

    assert st.day == "2026-08-13"
    assert st.day_start_equity == 10_500.0
    assert st.day_halted is False
    assert st.day_halt_reason == ""
    assert st.hedge_count_today == 0
    assert st.peak_equity == 12_000.0, "a new day must not reset the total peak"


def test_roll_day_is_idempotent_within_a_day():
    st = SharedPortfolioState()
    assert st.roll_day("2026-08-13", 10_000.0) is True
    assert st.roll_day("2026-08-13", 9_000.0) is False
    assert st.day_start_equity == 10_000.0, "must not re-seed mid-day"


def test_day_drawdown_percent():
    st = SharedPortfolioState()
    st.roll_day("2026-08-13", 10_000.0)
    assert st.day_drawdown_percent(9_400.0) == pytest.approx(6.0)


def test_halt_persists_across_a_reload(tmp_path):
    """Clearing a halt must require a human, not a restart."""
    path = os.path.join(str(tmp_path), "state.json")
    st = SharedPortfolioState()
    st.halt("total drawdown 20%")
    st.save(path)

    reloaded = SharedPortfolioState.load(path)
    assert reloaded.halted is True
    assert reloaded.halt_reason == "total drawdown 20%"
    assert reloaded.halt("something else") is False


# ---------------------------------------------------------------------------
# Robustness — a broken state file must never stop the bot
# ---------------------------------------------------------------------------


def test_missing_file_returns_fresh_state(tmp_path):
    st = SharedPortfolioState.load(os.path.join(str(tmp_path), "nope.json"))
    assert st.peak_equity == 0.0
    assert st.halted is False


def test_corrupt_json_returns_fresh_state_with_a_note(tmp_path):
    path = os.path.join(str(tmp_path), "state.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not json at all")

    st = SharedPortfolioState.load(path)

    assert st.halted is False
    assert any("unreadable" in n for n in st.notes)


def test_json_that_is_not_an_object_returns_fresh_state(tmp_path):
    path = os.path.join(str(tmp_path), "state.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([1, 2, 3], fh)

    st = SharedPortfolioState.load(path)
    assert any("not a JSON object" in n for n in st.notes)


def test_unknown_fields_are_ignored_for_forward_compatibility(tmp_path):
    path = os.path.join(str(tmp_path), "state.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"peak_equity": 5_000.0, "invented_field_from_the_future": 42}, fh)

    st = SharedPortfolioState.load(path)
    assert st.peak_equity == 5_000.0


def test_notes_are_capped():
    st = SharedPortfolioState()
    for i in range(50):
        st.note(f"event {i}", keep=10)
    assert len(st.notes) == 10
    assert "event 49" in st.notes[-1]


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = os.path.join(str(tmp_path), "state.json")
    st = SharedPortfolioState()
    st.sync_baseline(10_000.0)
    for _ in range(3):
        st.save(path)

    leftovers = [f for f in os.listdir(str(tmp_path)) if f.startswith(".tmp")]
    assert leftovers == []
    # And the file is valid JSON every time.
    with open(path, "r", encoding="utf-8") as fh:
        assert isinstance(json.load(fh), dict)


def test_save_creates_missing_directories(tmp_path):
    path = os.path.join(str(tmp_path), "deep", "nested", "state.json")
    SharedPortfolioState().save(path)
    assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Reading the trader's heartbeat
# ---------------------------------------------------------------------------


def test_read_trader_status_is_never_fatal(tmp_path):
    """The overlay must keep protecting the account even if the trader is dead."""
    assert read_trader_status(os.path.join(str(tmp_path), "absent.json")) == {}

    bad = os.path.join(str(tmp_path), "bad.json")
    with open(bad, "w", encoding="utf-8") as fh:
        fh.write("<html>not json</html>")
    assert read_trader_status(bad) == {}

    notdict = os.path.join(str(tmp_path), "list.json")
    with open(notdict, "w", encoding="utf-8") as fh:
        json.dump(["a"], fh)
    assert read_trader_status(notdict) == {}


def test_read_trader_status_returns_the_payload(tmp_path):
    path = os.path.join(str(tmp_path), "status.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"equity": 10_500.0, "halted": False}, fh)

    assert read_trader_status(path)["equity"] == 10_500.0


# ---------------------------------------------------------------------------
# Status publishing
# ---------------------------------------------------------------------------


def test_write_overlay_status_with_a_full_report(tmp_path):
    from src.exposure.model import InstrumentSpec, Position, compute_exposure
    from src.factors.book import BetaBook
    from src.overlay.decision import HedgeCaps, HedgeInstrument, hedge_decide

    factors = ("USD", "RISK", "ENERGY", "METALS")
    book = BetaBook(
        factors=factors,
        betas={"GOLD": {"USD": -0.9, "RISK": 0.0, "ENERGY": 0.1, "METALS": 1.0},
               "EURUSD": {"USD": -1.0, "RISK": 0.02, "ENERGY": 0.0, "METALS": 0.01}},
        sigma={"GOLD": 0.011, "EURUSD": 0.005},
        resid_sigma={"GOLD": 0.004, "EURUSD": 0.001},
        factor_sigma={"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022, "METALS": 0.009},
    )
    gold = InstrumentSpec("XAUUSD", 0.01, 1.0, logical="GOLD")
    eur = InstrumentSpec("EURUSD", 0.00001, 1.0, logical="EURUSD")
    rep = compute_exposure([Position("XAUUSD", 1, 0.1, 2400.0, logical="GOLD")],
                           {"XAUUSD": gold}, book, 10_000.0)
    caps = HedgeCaps(factor_caps={"USD": 1.5, "RISK": 1.5, "ENERGY": 1.0, "METALS": 3.0})
    plan = hedge_decide(rep, book, caps,
                        [HedgeInstrument("EURUSD", "EURUSD", eur, 1.08, 1.0801)])

    st = SharedPortfolioState()
    st.sync_baseline(10_000.0)
    path = os.path.join(str(tmp_path), "hedge_status.json")

    write_overlay_status(st, equity=10_000.0, balance=10_000.0, currency="USD",
                         report=rep, plan=plan, caps=caps.factor_caps,
                         dry_run=True, path=path)

    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload["dry_run"] is True
    assert payload["exposure"]["factor_leverage"]["USD"] == pytest.approx(-2.16)
    assert payload["exposure"]["open_positions"] == 1
    assert "USD" in payload["exposure"]["breaches"]
    assert payload["plan"]["actions"][0]["action"] == "open"
    assert payload["plan"]["summary"]


def test_write_overlay_status_tolerates_missing_report_and_plan(tmp_path):
    path = os.path.join(str(tmp_path), "s.json")
    write_overlay_status(SharedPortfolioState(), equity=1.0, balance=1.0,
                         currency="USD", report=None, plan=None, path=path)
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "exposure" not in payload
    assert "plan" not in payload
