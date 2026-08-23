"""
READ-ONLY MetaTrader 5 connector for Phase 1 (observe mode).

Deliberately incapable of trading. There is no order_send in this file, no
TRADE_ACTION anything. Phase 1 measures and logs; it cannot place an order even
if misconfigured, which is a much stronger safety property than a `dry_run` flag
that could be flipped by a typo in a YAML file.

Order placement arrives in Phase 2 as a separate module, after the measurements
have shown the caps are calibrated correctly.

The `import MetaTrader5` is guarded exactly as Tradingbot's connector does it, so
every other module in this repo stays importable on Linux and Mac. Methods raise
a clear error if called without the package.

Two things this connector does that Tradingbot's does NOT
--------------------------------------------------------
1. `all_positions()` uses positions_get() WITHOUT filtering by magic, so the
   overlay sees the trader's book natively. Tradingbot's bot_positions() filters
   by magic, which is right for a strategy bot and wrong for a risk overlay.

2. `margin_mode()` reports NETTING vs HEDGING, which Tradingbot never queries and
   which determines whether same-symbol opposing positions are even possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..exposure.model import InstrumentSpec, Position

try:                                     # pragma: no cover - platform dependent
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:                      # pragma: no cover
    mt5 = None                           # type: ignore[assignment]
    _MT5_AVAILABLE = False


MARGIN_MODE_NAMES = {
    0: "RETAIL_NETTING",
    1: "EXCHANGE",
    2: "RETAIL_HEDGING",
}


class MT5Unavailable(RuntimeError):
    """The MetaTrader5 package is not installed (non-Windows, usually)."""


class UnvaluablePosition(RuntimeError):
    """A live position exists that cannot be priced.

    Raised rather than skipped. The previous code did:

        price = float(getattr(p, "price_current", 0.0) or p.price_open)
        if price <= 0:
            continue          # <-- silently drops a REAL position

    which removes live risk from the risk book with no warning at all. Every
    downstream number — factor leverage, portfolio risk, breach detection — then
    understates reality while looking perfectly healthy. That is the single worst
    failure mode a risk system can have.

    Skipping a CYCLE is safe (the next poll retries 60 seconds later). Skipping a
    POSITION is not. So this propagates, and the live loop catches it, reports it
    loudly, and measures nothing that cycle instead of measuring something wrong.
    """


@dataclass(frozen=True)
class AccountState:
    login: int
    balance: float
    equity: float
    margin_free: float
    margin_level: float
    currency: str
    leverage: int
    margin_mode: int
    trade_allowed: bool
    trade_expert: bool
    server: str = ""
    company: str = ""

    @property
    def margin_mode_name(self) -> str:
        return MARGIN_MODE_NAMES.get(self.margin_mode, f"UNKNOWN({self.margin_mode})")

    @property
    def is_hedging_account(self) -> bool:
        return self.margin_mode == 2

    @property
    def algo_trading_enabled(self) -> bool:
        return bool(self.trade_allowed and self.trade_expert)


class MT5Reader:
    """Read-only view of an MT5 terminal."""

    def __init__(self, symbol_map: dict[str, str] | None = None) -> None:
        """symbol_map maps LOGICAL names to BROKER symbols, e.g.
        {"GOLD": "XAUUSD.ecn"}. The inverse is used to label positions with the
        logical name the beta book is keyed on.
        """
        self.symbol_map = dict(symbol_map or {})
        self._logical_of = {v: k for k, v in self.symbol_map.items()}
        self._spec_cache: dict[str, InstrumentSpec] = {}
        self._connected = False
        #: Non-fatal data-quality problems found while reading broker specs.
        #: The caller is expected to surface these; they are conditions that
        #: distort numbers without stopping anything, which is exactly the kind
        #: of thing that goes unnoticed for weeks.
        self.spec_warnings: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    @staticmethod
    def available() -> bool:
        return _MT5_AVAILABLE

    def _require(self) -> None:
        if not _MT5_AVAILABLE:
            raise MT5Unavailable(
                "MetaTrader5 package not installed. The overlay's measurement and "
                "decision layers run anywhere, but reading a live account needs "
                "Windows + `pip install MetaTrader5`."
            )

    def connect(
        self, login: int | None = None, password: str | None = None,
        server: str | None = None, path: str | None = None,
    ) -> AccountState:
        """Attach to the terminal.

        With no credentials, attaches to whatever account the terminal is already
        logged into — the same convention as Tradingbot, and the reason no
        credential store exists in this repo.
        """
        self._require()
        kwargs: dict = {}
        if path:
            kwargs["path"] = path
        if login and password and server:
            kwargs.update(login=int(login), password=password, server=server)

        if not mt5.initialize(**kwargs):
            raise ConnectionError(f"mt5.initialize() failed: {mt5.last_error()}")
        self._connected = True
        return self.account()

    def shutdown(self) -> None:
        if _MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False

    def __enter__(self) -> "MT5Reader":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.shutdown()

    # -- account -----------------------------------------------------------

    def account(self) -> AccountState:
        self._require()
        a = mt5.account_info()
        if a is None:
            raise ConnectionError(f"account_info() failed: {mt5.last_error()}")
        return AccountState(
            login=int(a.login),
            balance=float(a.balance),
            equity=float(a.equity),
            margin_free=float(a.margin_free),
            margin_level=float(getattr(a, "margin_level", 0.0) or 0.0),
            currency=str(a.currency),
            leverage=int(a.leverage),
            margin_mode=int(getattr(a, "margin_mode", -1)),
            trade_allowed=bool(a.trade_allowed),
            trade_expert=bool(a.trade_expert),
            server=str(getattr(a, "server", "") or ""),
            company=str(getattr(a, "company", "") or ""),
        )

    def margin_mode(self) -> str:
        return self.account().margin_mode_name

    # -- symbols -----------------------------------------------------------

    def ensure_symbol(self, symbol: str) -> None:
        self._require()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise KeyError(f"unknown broker symbol {symbol!r}")
        if not info.visible:
            mt5.symbol_select(symbol, True)

    def spec(self, symbol: str, use_cache: bool = True) -> InstrumentSpec:
        """InstrumentSpec for a broker symbol.

        Cached because symbol_info is a synchronous terminal call and the contract
        details do not change intraday. Prices are NOT cached.
        """
        if use_cache and symbol in self._spec_cache:
            return self._spec_cache[symbol]

        self._require()
        self.ensure_symbol(symbol)
        info = mt5.symbol_info(symbol)

        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        if tick_size <= 0:
            # `point` is usually equal to trade_tick_size, but not always — on
            # some symbols a tick spans several points. Since notional is
            # tick_value/tick_size * price, substituting the wrong one scales
            # EVERY downstream leverage figure. The fallback is kept because
            # refusing outright would block symbols that work fine, but it is
            # recorded so it can never pass unnoticed.
            tick_size = float(getattr(info, "point", 0.0) or 0.0)
            if tick_size > 0:
                warning = (
                    f"{symbol}: broker reported no trade_tick_size; fell back to "
                    f"point={tick_size}. If a tick is not one point on this "
                    f"symbol, its notional and leverage are WRONG by that ratio. "
                    f"Verify with scripts/check_account_mode.py."
                )
                if warning not in self.spec_warnings:
                    self.spec_warnings.append(warning)

        tick_value = float(getattr(info, "trade_tick_value", 0.0))
        if tick_size <= 0 or tick_value <= 0:
            raise ValueError(
                f"{symbol}: broker reported tick_size={tick_size}, "
                f"tick_value={tick_value}. Cannot value this instrument, so it "
                "cannot be included in the risk model."
            )

        spec = InstrumentSpec(
            symbol=symbol,
            tick_size=tick_size,
            tick_value=tick_value,
            volume_min=float(info.volume_min),
            volume_step=float(info.volume_step),
            volume_max=float(info.volume_max),
            logical=self._logical_of.get(symbol, ""),
        )
        self._spec_cache[symbol] = spec
        return spec

    def tick(self, symbol: str) -> tuple[float, float]:
        """(bid, ask) for a broker symbol."""
        self._require()
        self.ensure_symbol(symbol)
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise ConnectionError(f"symbol_info_tick({symbol}) failed: {mt5.last_error()}")
        return float(t.bid), float(t.ask)

    # -- positions ---------------------------------------------------------

    def all_positions(self) -> list[Position]:
        """EVERY open position, regardless of magic number.

        This is the key difference from Tradingbot's bot_positions(): a risk
        overlay must see the whole account, including positions opened by the
        trader, by another EA, or by hand. positions_get() is unfiltered, so no
        shared state file is needed for position visibility.
        """
        self._require()
        raw = mt5.positions_get()
        if raw is None:
            return []

        out: list[Position] = []
        unvaluable: list[str] = []
        for p in raw:
            side = 1 if int(p.type) == int(mt5.POSITION_TYPE_BUY) else -1
            # price_current is the mark; price_open is a valid fallback because a
            # position always had a real entry price.
            price = float(getattr(p, "price_current", 0.0) or 0.0)
            if price <= 0:
                price = float(getattr(p, "price_open", 0.0) or 0.0)
            if price <= 0 or not math.isfinite(price):
                # Do NOT skip. See UnvaluablePosition.
                unvaluable.append(
                    f"ticket {getattr(p, 'ticket', '?')} {getattr(p, 'symbol', '?')} "
                    f"volume {getattr(p, 'volume', '?')} "
                    f"(price_current={getattr(p, 'price_current', None)!r}, "
                    f"price_open={getattr(p, 'price_open', None)!r})"
                )
                continue
            out.append(
                Position(
                    symbol=str(p.symbol),
                    side=side,
                    volume=float(p.volume),
                    price=price,
                    ticket=int(p.ticket),
                    magic=int(p.magic),
                    logical=self._logical_of.get(str(p.symbol), ""),
                )
            )
        if unvaluable:
            raise UnvaluablePosition(
                f"{len(unvaluable)} live position(s) cannot be priced, so the risk "
                f"book would be incomplete: {'; '.join(unvaluable)}. "
                "Refusing to report a partial book — measuring nothing is safer "
                "than measuring less than everything."
            )
        return out

    def positions_by_magic(self, magic: int) -> list[Position]:
        return [p for p in self.all_positions() if p.magic == magic]

    def specs_for(self, positions: list[Position]) -> dict[str, InstrumentSpec]:
        """Specs keyed by broker symbol, one lookup per DISTINCT symbol.

        Deduplicating matters: four separate 0.25-lot gold tickets are four
        Position objects on one symbol, and symbol_info is a synchronous terminal
        call.

        (The previous one-liner iterated a dict, which yields KEYS — so it passed
        strings where Position objects were expected and crashed with
        "'str' object has no attribute 'symbol'". Iterating a set of symbols
        directly makes that class of mistake impossible.)
        """
        return {symbol: self.spec(symbol) for symbol in sorted({p.symbol for p in positions})}

    def symbol_details(self, symbol: str) -> dict:
        """Extra facts useful for diagnostics and Phase 2 order placement."""
        self._require()
        self.ensure_symbol(symbol)
        info = mt5.symbol_info(symbol)
        bid, ask = self.tick(symbol)
        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "spread_points": int(info.spread),
            "digits": int(info.digits),
            "contract_size": float(info.trade_contract_size),
            "trade_stops_level": int(info.trade_stops_level),
            "freeze_level": int(getattr(info, "trade_freeze_level", 0)),
            "filling_mode_mask": int(info.filling_mode),
            "swap_long": float(info.swap_long),
            "swap_short": float(info.swap_short),
            "trade_mode": int(info.trade_mode),
        }
