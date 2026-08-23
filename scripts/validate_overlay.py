"""
PHASE 3: does the overlay actually reduce drawdown, after costs?

    python scripts\\validate_overlay.py

Needs data/trades_export.json (from tools/export_tradingbot_trades.py run inside
the Tradingbot repo) and betas/betas_latest.json. Needs internet for daily prices.
Uses MT5 for real hedge swap rates if available.

This is the question that decides whether Phase 2 is worth building. If the answer
is no, the correct outcome is to NOT build order execution — which is a good
result, arrived at cheaply.

Read the honest-limitations section in src/backtest/replay.py before quoting any
number from this. Two of the four limitations FLATTER the overlay, which is
deliberate: a negative result under favourable assumptions is conclusive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.replay import (                                        # noqa: E402
    ReplayInputs,
    TradeRecord,
    replay,
)
from src.data.yahoo import fetch_daily                                   # noqa: E402
from src.exposure.model import InstrumentSpec                            # noqa: E402
from src.factors.definitions import HEDGE_CANDIDATES                     # noqa: E402
from src.factors.store import StaleBetasError, load_betas                # noqa: E402
from src.overlay.costs import InstrumentCost                             # noqa: E402
from src.overlay.decision import HedgeCaps                               # noqa: E402

#: Swap rates measured on the live JustMarkets ECN account. Used only when MT5 is
#: unavailable. Every one of these was read from the broker, not estimated —
#: guessing swap would invalidate the entire exercise, since swap is ~100x the
#: spread and therefore IS the cost of hedging.
FALLBACK_HEDGE_COSTS: dict[str, dict] = {
    "EURUSD": dict(symbol="EURUSD.ecn", tick_size=0.00001, tick_value=1.0,
                   bid=1.15697, ask=1.15701, swap_long=-13.32, swap_short=-3.72),
    "SP500":  dict(symbol="US500.ecn", tick_size=0.1, tick_value=0.1,
                   bid=7785.3, ask=7785.5, swap_long=-5.0, swap_short=-7.5),
    "SILVER": dict(symbol="XAGUSD.ecn", tick_size=0.001, tick_value=5.0,
                   bid=64.68, ask=64.704, swap_long=-8.52, swap_short=-1.08),
    "WTI":    dict(symbol="WTI.ecn", tick_size=0.01, tick_value=10.0,
                   bid=63.0, ask=63.03, swap_long=-12.0, swap_short=-14.0),
}


def _hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load_hedge_costs(use_mt5: bool, symbols: dict) -> tuple[dict, str]:
    """Real swap/spread per hedge candidate. Prefers live MT5."""
    costs: dict = {}
    if use_mt5:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            mt5 = None
        if mt5 is not None and mt5.initialize():
            try:
                for logical, sym in symbols.items():
                    info = mt5.symbol_info(sym)
                    if info is None:
                        mt5.symbol_select(sym, True)
                        info = mt5.symbol_info(sym)
                    tick = mt5.symbol_info_tick(sym) if info is not None else None
                    if info is None or tick is None:
                        continue
                    ts = float(getattr(info, "trade_tick_size", 0.0)
                               or getattr(info, "point", 0.0))
                    tv = float(getattr(info, "trade_tick_value", 0.0))
                    bid, ask = float(tick.bid), float(tick.ask)
                    if min(ts, tv, bid, ask) <= 0:
                        continue
                    try:
                        costs[logical] = InstrumentCost(
                            symbol=sym, tick_size=ts, tick_value=tv,
                            bid=bid, ask=ask,
                            swap_long=float(info.swap_long),
                            swap_short=float(info.swap_short),
                            swap_mode=int(getattr(info, "swap_mode",
                                                  1)) or 1,
                        )
                        print(f"  {logical:<8} {sym:<12} spread "
                              f"{costs[logical].spread_fraction * 100:.4f}%  "
                              f"swap {info.swap_long}/{info.swap_short}")
                    except ValueError as exc:
                        print(f"  {logical:<8} {sym:<12} unusable: {exc}")
            finally:
                mt5.shutdown()
            if costs:
                return costs, "MetaTrader5 (live)"

    print("  MT5 unavailable — using swap rates measured on the live account.")
    for logical, kw in FALLBACK_HEDGE_COSTS.items():
        costs[logical] = InstrumentCost(**kw)
    return costs, "measured fallback"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the overlay on real trades.")
    ap.add_argument("--trades", default=os.path.join("data", "trades_export.json"))
    ap.add_argument("--betas", default=os.path.join("betas", "betas_latest.json"))
    ap.add_argument("--config", default=os.path.join("config", "hedge_config.yaml"))
    ap.add_argument("--balance", type=float, default=None,
                    help="override the export's balance")
    ap.add_argument("--no-mt5", action="store_true")
    ap.add_argument("--allow-stale-betas", action="store_true")
    ap.add_argument("--out", default=os.path.join("logs", "validation.json"))
    ap.add_argument("--caps", default=None,
                    help="override caps, e.g. USD=2,RISK=2,ENERGY=1.5,METALS=2")
    args = ap.parse_args()

    # ---- trades ---------------------------------------------------------
    if not os.path.exists(args.trades):
        print(f"No trade export at {args.trades}")
        print("Run this inside the Tradingbot repo, then copy the file here:")
        print("    python export_tradingbot_trades.py --balance 10000 --risk 2.0")
        print(r"    copy trades_export.json C:\Hedgingbot\data\ ")
        return 1

    with open(args.trades, "r", encoding="utf-8") as fh:
        export = json.load(fh)

    _hr("TRADE EXPORT")
    print(f"  generated  : {export.get('generated_at')}")
    print(f"  specs from : {export.get('spec_source')}")
    print(f"  balance    : {export.get('balance'):,.2f}  "
          f"risk {export.get('risk_percent_per_trade')}%/trade")
    missing = export.get("missing_slots") or []
    if missing:
        print(f"  WARNING: slots missing from the export: {', '.join(missing)}")
        print("  Total exposure is UNDERSTATED; fix the export before trusting this.")

    slot_specs: dict = {}
    hedge_specs: dict = {}
    trades: list[TradeRecord] = []
    yahoo_of: dict = {}
    print()
    for slot in export.get("slots", []):
        lg = slot["logical"]
        sp = slot["spec"]
        spec = InstrumentSpec(
            symbol=sp["symbol"], tick_size=sp["tick_size"],
            tick_value=sp["tick_value"], volume_min=sp["volume_min"],
            volume_step=sp["volume_step"], volume_max=sp["volume_max"],
            logical=lg,
        )
        slot_specs[lg] = spec
        yahoo_of[lg] = slot["yahoo_symbol"]
        for t in slot["trades"]:
            trades.append(TradeRecord(
                logical=lg, symbol=sp["symbol"], side=int(t["side"]),
                lots=float(t["lots"]),
                entry_day=str(t["entry_time"])[:10],
                exit_day=str(t["exit_time"])[:10],
                entry=float(t["entry"]), exit=float(t["exit"]),
            ))
        print(f"  {lg:<10} {sp['symbol']:<12} {len(slot['trades']):>4} trades  "
              f"mppu {sp['tick_value'] / sp['tick_size']:>9,.0f}")

    if not trades:
        print("\nNo trades in the export. Nothing to replay.")
        return 1

    balance = args.balance or float(export.get("balance", 10_000.0))

    # ---- betas ----------------------------------------------------------
    try:
        book, meta = load_betas(args.betas, max_age_days=None if args.allow_stale_betas
                               else 30.0, allow_stale=args.allow_stale_betas)
    except FileNotFoundError:
        print(f"\nNo beta file at {args.betas}. Run scripts\\update_betas.bat")
        return 1
    except StaleBetasError as exc:
        print(f"\n{exc}")
        return 1

    _hr("BETAS")
    print(f"  {len(book.instruments())} instruments, created {meta.get('created_at')}")
    print(f"  return period: {meta.get('return_period', 'daily')}  "
          f"sigma basis: {meta.get('sigma_basis', 'unknown')}")

    unmapped = [lg for lg in slot_specs if not book.has(lg)]
    if unmapped:
        print(f"\n  FATAL: no betas for {', '.join(unmapped)}. The overlay would")
        print("  refuse to hedge an incomplete book, so the replay would be")
        print("  meaningless. Re-run scripts\\update_betas.bat")
        return 1

    # ---- hedge instruments ----------------------------------------------
    _hr("HEDGE INSTRUMENT COSTS")
    hedge_symbols = {lg: sym for lg, (_y, sym, _f) in HEDGE_CANDIDATES.items()
                     if book.has(lg)}
    costs, cost_source = load_hedge_costs(not args.no_mt5, hedge_symbols)
    hedge_symbols = {lg: s for lg, s in hedge_symbols.items() if lg in costs}
    for lg, c in costs.items():
        hedge_specs[lg] = InstrumentSpec(
            symbol=c.symbol, tick_size=c.tick_size, tick_value=c.tick_value,
            volume_min=0.01, volume_step=0.01, volume_max=100.0, logical=lg,
        )
        yahoo_of.setdefault(lg, HEDGE_CANDIDATES[lg][0])
    print(f"\n  source: {cost_source}")

    # ---- prices ---------------------------------------------------------
    _hr("FETCHING DAILY PRICES")
    prices: dict = {}
    for lg, ysym in yahoo_of.items():
        try:
            bars = fetch_daily(ysym, years=4.0)
            prices[lg] = bars.as_map()
            print(f"  {lg:<10} {ysym:<12} {len(bars):>5} days")
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {lg:<10} {ysym:<12} FAILED: {exc!r}")

    for lg in list(slot_specs):
        if lg not in prices:
            print(f"\n  FATAL: no prices for traded slot {lg}. Cannot mark the book.")
            return 1

    # ---- caps -----------------------------------------------------------
    from scripts.measure_exposure import caps_from_config, load_config

    cfg = load_config(args.config)
    caps = caps_from_config(cfg)
    if args.caps:
        overrides = {}
        for part in args.caps.split(","):
            k, _, v = part.partition("=")
            if k.strip() and v.strip():
                overrides[k.strip().upper()] = float(v)
        caps = HedgeCaps(
            factor_caps=overrides, hysteresis=caps.hysteresis,
            target_fraction=caps.target_fraction, unwind_band=caps.unwind_band,
            max_hedge_gross_pct=caps.max_hedge_gross_pct,
            min_purity=caps.min_purity, min_abs_beta=caps.min_abs_beta,
            max_actions_per_cycle=caps.max_actions_per_cycle,
            max_hedges_per_day=caps.max_hedges_per_day,
            allow_unmapped=caps.allow_unmapped,
            min_excess_removed=caps.min_excess_removed,
        )

    inputs = ReplayInputs(
        slot_specs=slot_specs, trades=trades, prices=prices,
        hedge_specs=hedge_specs, hedge_costs=costs,
        hedge_symbols=hedge_symbols, initial_balance=balance,
    )

    # ---- run ------------------------------------------------------------
    _hr("REPLAY")
    print(f"  caps: " + "  ".join(f"{k}={v}" for k, v in caps.factor_caps.items()))
    print(f"  balance {balance:,.2f}   {len(trades)} trades   "
          f"{len(prices)} instruments")

    result = replay(inputs, book, caps, enable_overlay=True)
    s = result.summary()
    if not s:
        print("\n  Replay produced no rows. Check the date ranges overlap.")
        return 1

    # ---- report ---------------------------------------------------------
    _hr("RESULT")
    print(f"  {'':<26}{'UNHEDGED':>14}{'HEDGED':>14}{'change':>14}")
    print("  " + "-" * 66)

    def row(label, a, b, unit="%", better_lower=False):
        d = b - a
        arrow = ""
        if abs(d) > 1e-9:
            good = (d < 0) if better_lower else (d > 0)
            arrow = "  better" if good else "  worse"
        print(f"  {label:<26}{a:>13.2f}{unit}{b:>13.2f}{unit}{d:>+13.2f}{unit}{arrow}")

    row("Total return", s["base_return_pct"], s["hedged_return_pct"])
    row("Max drawdown", s["base_max_dd_pct"], s["hedged_max_dd_pct"],
        better_lower=True)
    print(f"  {'Sharpe (daily, ann.)':<26}{s['base_sharpe']:>14.2f}"
          f"{s['hedged_sharpe']:>14.2f}"
          f"{s['hedged_sharpe'] - s['base_sharpe']:>+14.2f}")

    print()
    print(f"  Days replayed        : {int(s['days'])}")
    print(f"  Hedges opened/closed : {int(s['hedges_opened'])} / "
          f"{int(s['hedges_closed'])}")
    print(f"  Spread paid          : {s['total_spread']:>12,.2f}")
    print(f"  Swap paid            : {s['total_swap']:>12,.2f}")
    print(f"  TOTAL HEDGING COST   : {s['total_cost']:>12,.2f}"
          f"   ({s['total_cost'] / balance * 100:+.2f}% of starting equity)")
    if result.blocked_days:
        print(f"  Days overlay blocked : {result.blocked_days}")
    if result.skipped_days:
        print(f"  Days skipped         : {result.skipped_days}")

    # ---- verdict --------------------------------------------------------
    _hr("VERDICT")
    dd_better = s["base_max_dd_pct"] - s["hedged_max_dd_pct"]
    ret_cost = s["base_return_pct"] - s["hedged_return_pct"]

    if s["hedges_opened"] == 0:
        print("  The overlay never hedged. Either the caps are above anything this")
        print("  book ever reached, or no usable hedge instrument qualified. Try")
        print("  tighter caps with --caps to find where it starts acting:")
        print("      --caps USD=1,RISK=1,ENERGY=0.75,METALS=0.75")
    elif dd_better > 0 and s["hedged_sharpe"] >= s["base_sharpe"]:
        print(f"  Drawdown improved by {dd_better:.2f} points AND risk-adjusted")
        print(f"  return did not worsen. Cost {ret_cost:+.2f} points of return.")
        print("  This is the case where the overlay is worth running.")
    elif dd_better > 0:
        print(f"  Drawdown improved by {dd_better:.2f} points, but it cost")
        print(f"  {ret_cost:.2f} points of return and Sharpe fell.")
        print("  Whether that trade is worth making is YOUR call, not the model's:")
        print("  it depends on how much you dislike drawdown versus return.")
    else:
        print(f"  Drawdown did NOT improve ({dd_better:+.2f} points) and the hedging")
        print(f"  cost {s['total_cost']:,.2f}.")
        print()
        print("  On this evidence the overlay is not worth running on this book at")
        print("  this account size, and Phase 2 (order execution) should NOT be")
        print("  built yet. Better position sizing is the cheaper answer.")
        print()
        print("  Before accepting that, check whether the caps ever bound at all —")
        print("  a cap above the observed maximum does nothing. Re-run with")
        print("  tighter caps to see if there is a setting where it does help.")

    print()
    print("  Read src/backtest/replay.py for the four limitations. Two of them")
    print("  FLATTER the overlay (daily decisions, no slippage), so a negative")
    print("  result here is strong evidence rather than a modelling artefact.")

    # ---- persist --------------------------------------------------------
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "summary": s,
                "caps": caps.factor_caps,
                "balance": balance,
                "cost_source": cost_source,
                "beta_meta": {k: meta.get(k) for k in
                              ("created_at", "return_period", "sigma_basis")},
                "warnings": result.warnings[:50],
                "days": [
                    {"day": r.day, "base": round(r.base_equity, 2),
                     "hedged": round(r.hedged_equity, 2),
                     "n_positions": r.n_positions, "n_hedges": r.n_hedges,
                     "risk_pct": round(r.base_risk_pct, 3),
                     "action": r.action, "leverage": r.leverage}
                    for r in result.rows
                ],
            }, fh, indent=1)
        print(f"  Wrote {args.out}")
    except OSError as exc:
        print(f"  Could not write {args.out}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
