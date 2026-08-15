"""
Tests for the read-only MT5 connector.

The MetaTrader5 package is Windows-only, so the parts that talk to a terminal
cannot be tested here. But the pure-logic parts CAN be, and one of them shipped a
crash that only appeared when a real account had open positions:

    specs_for() iterated a dict, which yields KEYS, so it passed strings where
    Position objects were expected:
        AttributeError: 'str' object has no attribute 'symbol'

That is exactly the class of bug a unit test catches for free, and it went
unnoticed because this module had no tests at all. These fill the gap for
everything that does not require a live terminal.
"""

from __future__ import annotations

import pytest

from src.connectors.mt5_reader import (
    MARGIN_MODE_NAMES,
    AccountState,
    MT5Reader,
    MT5Unavailable,
)
from src.exposure.model import InstrumentSpec, Position


def spec_for(symbol: str) -> InstrumentSpec:
    return InstrumentSpec(symbol, tick_size=0.01, tick_value=1.0, logical="")


# ---------------------------------------------------------------------------
# specs_for — the regression
# ---------------------------------------------------------------------------


def test_specs_for_handles_multiple_positions_on_one_symbol(monkeypatch):
    """The live case: four 0.25-lot gold tickets are four Positions, one symbol."""
    reader = MT5Reader()
    calls: list[str] = []

    def fake_spec(symbol, use_cache=True):
        calls.append(symbol)
        return spec_for(symbol)

    monkeypatch.setattr(reader, "spec", fake_spec)

    positions = [
        Position("XAUUSD.ecn", -1, 0.25, 4382.26, ticket=2278989980, magic=0),
        Position("XAUUSD.ecn", -1, 0.25, 4382.26, ticket=2278989987, magic=0),
        Position("XAUUSD.ecn", -1, 0.25, 4382.26, ticket=2278989993, magic=0),
        Position("XAUUSD.ecn", -1, 0.25, 4382.26, ticket=2278989997, magic=0),
    ]

    specs = reader.specs_for(positions)

    assert set(specs) == {"XAUUSD.ecn"}
    assert isinstance(specs["XAUUSD.ecn"], InstrumentSpec)
    assert calls == ["XAUUSD.ecn"], (
        f"symbol_info is a synchronous terminal call; expected 1 lookup, got {calls}"
    )


def test_specs_for_multiple_distinct_symbols(monkeypatch):
    reader = MT5Reader()
    monkeypatch.setattr(reader, "spec", lambda s, use_cache=True: spec_for(s))

    positions = [
        Position("XAUUSD.ecn", -1, 0.25, 4382.0),
        Position("EURUSD.ecn", 1, 0.10, 1.157),
        Position("XAUUSD.ecn", -1, 0.25, 4382.0),
    ]

    specs = reader.specs_for(positions)
    assert set(specs) == {"XAUUSD.ecn", "EURUSD.ecn"}
    for symbol, spec in specs.items():
        assert spec.symbol == symbol, "specs must be keyed by their own symbol"


def test_specs_for_empty_book():
    assert MT5Reader().specs_for([]) == {}


def test_specs_for_propagates_a_bad_symbol(monkeypatch):
    """A symbol the broker cannot value must not be silently dropped — that would
    understate risk, the one thing a risk system must never do."""
    reader = MT5Reader()

    def fake_spec(symbol, use_cache=True):
        if symbol == "BROKEN":
            raise ValueError("broker reported tick_value=0")
        return spec_for(symbol)

    monkeypatch.setattr(reader, "spec", fake_spec)

    with pytest.raises(ValueError, match="tick_value"):
        reader.specs_for([Position("BROKEN", 1, 1.0, 10.0)])


# ---------------------------------------------------------------------------
# AccountState
# ---------------------------------------------------------------------------


def account(**kw) -> AccountState:
    base = dict(
        login=1100219238, balance=9872.26, equity=10502.26, margin_free=9625.81,
        margin_level=1198.27, currency="USD", leverage=500, margin_mode=2,
        trade_allowed=True, trade_expert=True, server="JustMarkets-Demo2",
        company="Just Global Markets Ltd.",
    )
    base.update(kw)
    return AccountState(**base)


def test_margin_mode_names_cover_the_documented_values():
    assert MARGIN_MODE_NAMES[0] == "RETAIL_NETTING"
    assert MARGIN_MODE_NAMES[2] == "RETAIL_HEDGING"


def test_hedging_account_detection():
    assert account(margin_mode=2).is_hedging_account
    assert not account(margin_mode=0).is_hedging_account
    assert account(margin_mode=2).margin_mode_name == "RETAIL_HEDGING"
    assert account(margin_mode=0).margin_mode_name == "RETAIL_NETTING"


def test_unknown_margin_mode_is_labelled_not_hidden():
    assert "UNKNOWN(7)" in account(margin_mode=7).margin_mode_name


def test_algo_trading_needs_both_account_and_terminal_permission():
    assert account(trade_allowed=True, trade_expert=True).algo_trading_enabled
    assert not account(trade_allowed=True, trade_expert=False).algo_trading_enabled
    assert not account(trade_allowed=False, trade_expert=True).algo_trading_enabled


def test_server_and_company_default_to_empty(monkeypatch):
    """preflight.py printed acct.server before the field existed, crashing with
    AttributeError. Defaults keep an older MetaTrader5 build from doing it again."""
    minimal = AccountState(
        login=1, balance=1.0, equity=1.0, margin_free=1.0, margin_level=0.0,
        currency="USD", leverage=100, margin_mode=0,
        trade_allowed=False, trade_expert=False,
    )
    assert minimal.server == ""
    assert minimal.company == ""


# ---------------------------------------------------------------------------
# Symbol mapping and platform guard
# ---------------------------------------------------------------------------


def test_symbol_map_inverts_for_logical_labelling():
    reader = MT5Reader(symbol_map={"GOLD": "XAUUSD.ecn", "SILVER": "XAGUSD.ecn"})
    assert reader._logical_of["XAUUSD.ecn"] == "GOLD"
    assert reader._logical_of["XAGUSD.ecn"] == "SILVER"


def test_methods_raise_a_clear_error_without_the_package():
    """On Linux/Mac the guard must produce an explanatory error, not an
    ImportError from deep inside the module."""
    if MT5Reader.available():
        pytest.skip("MetaTrader5 is installed; the guard path cannot be exercised")

    reader = MT5Reader()
    with pytest.raises(MT5Unavailable, match="Windows"):
        reader.connect()
    with pytest.raises(MT5Unavailable):
        reader.account()


def test_shutdown_is_safe_when_never_connected():
    MT5Reader().shutdown()      # must not raise
