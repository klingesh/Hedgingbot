"""
Answer the questions we cannot answer from Linux, on the machine where MT5 lives.

Run this FIRST, on your Windows box, with the JustMarkets terminal open and
logged in:

    python scripts/check_account_mode.py

It resolves, in order of importance:

  1. NETTING or HEDGING margin mode. This decides whether the overlay is even
     allowed to hold a position opposite to the trader's on the same symbol. On
     a NETTING account an opposing order REDUCES the existing position instead
     of hedging it, which would silently sabotage the strategy. The overlay is
     designed to avoid same-symbol hedging entirely for this reason, but you
     need to know which world you are in.

  2. Whether algorithmic trading is actually permitted right now — at both the
     account level (trade_expert) and the terminal level (the AutoTrading
     button). "My broker allows EAs" is a broker policy; these two flags are
     the runtime reality.

  3. Per-symbol facts the exposure model needs: trade_tick_value,
     trade_tick_size, contract_size, volume min/step/max, and the derived
     money-per-price-unit-per-lot that all notional maths depends on.

  4. trade_stops_level — the minimum stop distance. Tradingbot prints this in
     check_mt5.py but never enforces it, which is a known source of "Invalid
     stops" rejections.

  5. SWAP RATES on both directions. This is the recurring cost of carrying a
     hedge and it is the number most hedging discussions leave out. A hedge held
     for weeks pays swap on the hedge leg on top of the spread. If swap_long +
     swap_short is strongly negative for a pair, hedging with it is expensive to
     hold and the overlay should prefer another leg.

  6. A live order_check() dry validation and order_calc_margin() for one minimum
     lot, so you learn whether an order WOULD be accepted without sending one.

Nothing here places, modifies or closes any order. It is read-only apart from
order_check, which the terminal evaluates without sending.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed.")
    print("This script must run on Windows alongside the MT5 terminal:")
    print("    pip install MetaTrader5")
    sys.exit(1)


# --- MT5 enum decoding -----------------------------------------------------

MARGIN_MODES = {
    0: ("RETAIL_NETTING", "One net position per symbol. Opposing orders REDUCE "
                          "the existing position — they do NOT create a hedge."),
    1: ("EXCHANGE", "Exchange execution. Position handling follows the exchange."),
    2: ("RETAIL_HEDGING", "Multiple independent positions per symbol, including "
                          "opposing ones. Same-symbol hedging IS possible."),
}

TRADE_MODES = {
    0: "DISABLED (no trading)",
    1: "LONGONLY",
    2: "SHORTONLY",
    3: "CLOSEONLY",
    4: "FULL",
}


def _filling_modes(mask: int) -> list[str]:
    """Decode the SYMBOL_FILLING_* bitmask into names."""
    out = []
    if mask & 1:
        out.append("FOK")
    if mask & 2:
        out.append("IOC")
    if not out:
        out.append("RETURN (neither FOK nor IOC advertised)")
    return out


def _hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def describe_account() -> dict:
    acct = mt5.account_info()
    term = mt5.terminal_info()
    if acct is None:
        print(f"account_info() returned None — last_error={mt5.last_error()}")
        return {}

    _hr("ACCOUNT")
    print(f"  login              : {acct.login}")
    print(f"  server             : {acct.server}")
    print(f"  company            : {acct.company}")
    print(f"  currency           : {acct.currency}")
    print(f"  leverage           : 1:{acct.leverage}")
    print(f"  balance            : {acct.balance:,.2f}")
    print(f"  equity             : {acct.equity:,.2f}")
    print(f"  margin_free        : {acct.margin_free:,.2f}")
    print(f"  margin_level       : {getattr(acct, 'margin_level', 0.0):.2f}%")

    mode_id = int(getattr(acct, "margin_mode", -1))
    name, explanation = MARGIN_MODES.get(mode_id, (f"UNKNOWN({mode_id})", "?"))

    _hr("*** THE ANSWER: MARGIN MODE ***")
    print(f"  margin_mode        : {mode_id} = {name}")
    print(f"  {explanation}")
    if name == "RETAIL_HEDGING":
        print("\n  => Same-symbol opposing positions ARE possible.")
        print("     The overlay still avoids them by default (cleaner attribution,")
        print("     and it halves the spread bill), but you have the option.")
    elif name == "RETAIL_NETTING":
        print("\n  => Same-symbol opposing positions are NOT possible.")
        print("     The overlay MUST hedge with different-but-correlated instruments.")
        print("     Good news: that is exactly how it is already designed. Keep the")
        print("     trader's symbols in `trader_symbols` so it never touches them.")

    _hr("ALGORITHMIC TRADING PERMISSION")
    print(f"  account.trade_allowed : {acct.trade_allowed}"
          f"   <- broker permits trading on this account")
    print(f"  account.trade_expert  : {acct.trade_expert}"
          f"   <- broker permits EAs / algos")
    if term is not None:
        print(f"  terminal.trade_allowed: {term.trade_allowed}"
              f"   <- the AutoTrading button in the terminal")
        print(f"  terminal.connected    : {term.connected}")
    if not acct.trade_expert:
        print("\n  WARNING: trade_expert is False. Algo orders will be REJECTED")
        print("           regardless of what the broker's website says.")
    if term is not None and not term.trade_allowed:
        print("\n  WARNING: terminal trade_allowed is False. Press the AutoTrading")
        print("           button in the MT5 toolbar.")

    return {
        "login": acct.login,
        "server": acct.server,
        "currency": acct.currency,
        "leverage": acct.leverage,
        "balance": acct.balance,
        "equity": acct.equity,
        "margin_mode": mode_id,
        "margin_mode_name": name,
        "trade_allowed": acct.trade_allowed,
        "trade_expert": acct.trade_expert,
        "terminal_trade_allowed": bool(getattr(term, "trade_allowed", False)),
    }


def describe_symbol(symbol: str, currency: str) -> dict | None:
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"\n  {symbol:<14} NOT FOUND. Search for the right name with:")
        print(f"                 python scripts/check_account_mode.py --search "
              f"{symbol[:3]}")
        return None

    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)

    tick = mt5.symbol_info_tick(symbol)
    bid = getattr(tick, "bid", 0.0) or 0.0
    ask = getattr(tick, "ask", 0.0) or 0.0
    mid = 0.5 * (bid + ask) if (bid and ask) else 0.0

    tick_size = float(getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.0))
    tick_value = float(getattr(info, "trade_tick_value", 0.0))
    mppu = (tick_value / tick_size) if tick_size > 0 else 0.0
    notional_per_lot = mppu * mid

    # Margin for one minimum lot, straight from the broker's own calculation.
    margin_min_lot = None
    try:
        margin_min_lot = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY, symbol, info.volume_min, ask or mid
        )
    except Exception:
        pass

    print(f"\n  {symbol}")
    print(f"    trade_mode        : {TRADE_MODES.get(int(info.trade_mode), info.trade_mode)}")
    print(f"    bid / ask         : {bid} / {ask}")
    spread_pct = f"   ({(ask - bid) / mid * 100:.4f}% of price)" if mid > 0 else ""
    print(f"    spread            : {info.spread} points{spread_pct}")
    print(f"    digits            : {info.digits}")
    print(f"    contract_size     : {info.trade_contract_size:,.2f}")
    print(f"    trade_tick_size   : {tick_size}")
    print(f"    trade_tick_value  : {tick_value} {currency}")
    print(f"    -> money per 1.0 price move per lot = tick_value/tick_size "
          f"= {mppu:,.2f} {currency}")
    print(f"    -> NOTIONAL PER 1.0 LOT = {notional_per_lot:,.2f} {currency}")
    print(f"    volume min/step/max: {info.volume_min} / {info.volume_step} / "
          f"{info.volume_max}")
    print(f"    trade_stops_level : {info.trade_stops_level} points"
          f"   <- minimum SL/TP distance; Tradingbot does not enforce this")
    print(f"    freeze_level      : {getattr(info, 'trade_freeze_level', 0)} points")
    print(f"    filling modes     : {', '.join(_filling_modes(int(info.filling_mode)))}")
    if margin_min_lot is not None:
        print(f"    margin for {info.volume_min} lot: {margin_min_lot:,.2f} {currency}")
    print(f"    swap_long / short : {info.swap_long} / {info.swap_short}"
          f"   (mode={getattr(info, 'swap_mode', '?')})")
    carry = float(info.swap_long) + float(info.swap_short)
    print(f"    -> HEDGE CARRY COST proxy (swap_long+swap_short) = {carry:+.2f}")
    if carry < 0:
        print(f"       Negative: holding both directions bleeds carry. Expensive to")
        print(f"       hold a long-term hedge on this symbol.")

    return {
        "symbol": symbol,
        "trade_mode": int(info.trade_mode),
        "bid": bid,
        "ask": ask,
        "spread_points": int(info.spread),
        "digits": int(info.digits),
        "contract_size": float(info.trade_contract_size),
        "tick_size": tick_size,
        "tick_value": tick_value,
        "money_per_price_unit_per_lot": mppu,
        "notional_per_lot": notional_per_lot,
        "volume_min": float(info.volume_min),
        "volume_step": float(info.volume_step),
        "volume_max": float(info.volume_max),
        "trade_stops_level": int(info.trade_stops_level),
        "filling_mode_mask": int(info.filling_mode),
        "margin_min_lot": margin_min_lot,
        "swap_long": float(info.swap_long),
        "swap_short": float(info.swap_short),
    }


def dry_check_order(symbol: str) -> None:
    """Ask the terminal to validate a minimum-lot market order WITHOUT sending it.

    order_check() runs the broker's own validation: margin, filling mode, stop
    levels, trade permissions. If this returns retcode 0 (DONE) the order would
    be accepted. This is the closest thing to a safe live test and it is how you
    verify the overlay could actually trade before letting it.
    """
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        print(f"  {symbol}: cannot order_check (symbol or tick unavailable)")
        return

    mask = int(info.filling_mode)
    filling = mt5.ORDER_FILLING_IOC if (mask & 2) else (
        mt5.ORDER_FILLING_FOK if (mask & 1) else mt5.ORDER_FILLING_RETURN
    )

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(info.volume_min),
        "type": mt5.ORDER_TYPE_BUY,
        "price": float(tick.ask),
        "deviation": 20,
        "magic": 990012,
        "comment": "hedge_precheck",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    result = mt5.order_check(request)
    if result is None:
        print(f"  {symbol}: order_check returned None — last_error={mt5.last_error()}")
        return

    ok = int(result.retcode) == 0
    print(f"  {symbol:<14} retcode={result.retcode} "
          f"{'WOULD BE ACCEPTED' if ok else 'WOULD BE REJECTED'}  "
          f"margin={getattr(result, 'margin', 0.0):,.2f} "
          f"free_after={getattr(result, 'margin_free', 0.0):,.2f}  "
          f"{result.comment}")


def search_symbols(fragment: str) -> None:
    """List every broker symbol containing `fragment` (case-insensitive).

    Broker symbol names are never what you expect — JustMarkets uses suffixes
    like `.ecn`, and gold might be XAUUSD, GOLD, or XAUUSD.m depending on the
    account type.
    """
    frag = fragment.upper()
    allsyms = mt5.symbols_get()
    if allsyms is None:
        print(f"symbols_get() failed — last_error={mt5.last_error()}")
        return
    hits = sorted(s.name for s in allsyms if frag in s.name.upper())
    print(f"\n{len(hits)} symbol(s) matching {fragment!r} (of {len(allsyms)} total):")
    for name in hits:
        print(f"  {name}")


DEFAULT_SYMBOLS = [
    # Tradingbot's live book — adjust suffixes to match your account.
    "XAUUSD", "XAGUSD", "XPTUSD", "NGAS", "UKOIL", "GBPJPY", "AUDUSD", "USDJPY",
    # Hedge candidates.
    "EURUSD", "US500", "USOIL", "USDCHF",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Report MT5 account margin mode, algo permissions and "
                    "per-symbol trading specs. Read-only."
    )
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="broker symbols to inspect (default: a sensible starter list)")
    ap.add_argument("--suffix", default="",
                    help="appended to every symbol, e.g. --suffix .ecn")
    ap.add_argument("--search", default=None,
                    help="list broker symbols containing this fragment, then exit")
    ap.add_argument("--order-check", action="store_true",
                    help="also run a non-sending order_check on each symbol")
    ap.add_argument("--json", default=None,
                    help="write the collected facts to this JSON path")
    ap.add_argument("--path", default=None, help="explicit terminal64.exe path")
    args = ap.parse_args()

    init_kwargs = {"path": args.path} if args.path else {}
    if not mt5.initialize(**init_kwargs):
        print(f"mt5.initialize() failed — last_error={mt5.last_error()}")
        print("Is the MT5 terminal running and logged in?")
        return 1

    try:
        if args.search:
            search_symbols(args.search)
            return 0

        account = describe_account()
        currency = account.get("currency", "")

        names = args.symbols if args.symbols else DEFAULT_SYMBOLS
        names = [f"{n}{args.suffix}" for n in names]

        _hr("SYMBOL SPECIFICATIONS")
        print("  (money_per_price_unit_per_lot = tick_value/tick_size is the number")
        print("   the exposure model uses for ALL notional maths — it already")
        print("   includes the account-currency conversion.)")
        symbols: list[dict] = []
        for name in names:
            got = describe_symbol(name, currency)
            if got:
                symbols.append(got)

        if args.order_check:
            _hr("ORDER PRE-CHECK (nothing is sent)")
            for s in symbols:
                dry_check_order(s["symbol"])

        _hr("OPEN POSITIONS (all magics — this is what the overlay will see)")
        positions = mt5.positions_get()
        if not positions:
            print("  none")
        else:
            print(f"  {'ticket':>10} {'symbol':<14} {'side':>5} {'volume':>8} "
                  f"{'price':>12} {'profit':>12} {'magic':>8}")
            for p in positions:
                side = "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell"
                print(f"  {p.ticket:>10} {p.symbol:<14} {side:>5} {p.volume:>8.2f} "
                      f"{p.price_open:>12} {p.profit:>12,.2f} {p.magic:>8}")
            print("\n  NOTE: positions_get() is unfiltered by magic. This is exactly")
            print("        why the overlay can see the trader's book with no shared file.")

        _hr("NEXT STEPS")
        mode = account.get("margin_mode_name", "?")
        print(f"  1. Margin mode is {mode}. Record it in config/hedge_config.yaml notes.")
        print("  2. Copy the broker symbol names printed above into")
        print("     config/hedge_config.yaml (hedge_instruments + trader_symbols).")
        print("  3. Run: python scripts/estimate_betas.py   (needs internet)")
        print("  4. Run: python scripts/measure_exposure.py --loop")
        print("     and leave it in observe mode for a few weeks.")

        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump({"account": account, "symbols": symbols}, fh, indent=2)
            print(f"\n  Wrote {args.json}")

        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
