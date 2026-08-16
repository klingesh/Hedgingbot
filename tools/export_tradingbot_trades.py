"""
Export Tradingbot's backtest trades so Hedgingbot can replay the overlay on them.

WHERE THIS RUNS
---------------
This script imports Tradingbot's own engine, strategies and data loaders, so it
must live in and run from the TRADINGBOT repo, not this one:

    copy C:\\Hedgingbot\\tools\\export_tradingbot_trades.py C:\\Tradingbot\\
    cd C:\\Tradingbot
    python export_tradingbot_trades.py

Needs internet (Yahoo). Uses MT5 for real contract specs if available, and falls
back to a table measured from a live JustMarkets account if not.

Output: trades_export.json — copy it to C:\\Hedgingbot\\data\\

WHY THE SPECS MATTER MORE THAN YOU WOULD THINK
----------------------------------------------
Tradingbot's research experiments deliberately use a FICTIONAL spec:

    RESEARCH_SPEC = SymbolSpec(tick_size=1.0, tick_value=1.0,
                               volume_min=1e-6, volume_step=1e-6, volume_max=1e12)

That is correct for comparing strategies — "lots" becomes a pure risk unit and
every instrument is on one scale. It is USELESS for this export. The overlay
computes exposure as

    notional = lots * (tick_value / tick_size) * price

so a fake tick_value produces fake notional, fake factor leverage, and a
validation result that means nothing. This script therefore uses REAL broker
specs, preferring live MT5 values.

THE ONE APPROXIMATION, STATED PLAINLY
-------------------------------------
Each slot is backtested INDEPENDENTLY, then the trades are merged into one
timeline. So slot A's drawdown does not shrink slot B's position size.

That is not a shortcut, it mirrors what Tradingbot actually does live: in
src/live/trader.py every slot sizes off `acct.balance`, the shared account
balance, with up to `max_open_trades` positions open at once. Balance moves
slowly relative to a single trade, so independent sizing is close to the live
behaviour. It is still an approximation and the replay report repeats the caveat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas as pd
except ImportError:
    print("This script runs inside the Tradingbot repo, which needs pandas.")
    print("    pip install pandas")
    sys.exit(1)

# --- Tradingbot imports. Fail loudly if we are in the wrong directory. -------
try:
    from src.backtest.engine import BacktestConfig, CostModel, run_backtest
    from src.data.yahoo_loader import INSTRUMENTS, fetch_yahoo_h4
    from src.live.portfolio import DEFAULT_PORTFOLIO
    from src.risk.position_sizing import MinLotPolicy, RiskParams, SymbolSpec
    from src.risk.vol_target import volatility_target_scalar
except ImportError as exc:
    print(f"Could not import Tradingbot modules: {exc}")
    print()
    print("This script must be COPIED INTO and RUN FROM the Tradingbot repo,")
    print("because it uses Tradingbot's own engine and strategies:")
    print()
    print("    copy C:\\Hedgingbot\\tools\\export_tradingbot_trades.py C:\\Tradingbot\\")
    print("    cd C:\\Tradingbot")
    print("    python export_tradingbot_trades.py")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Real broker specs
# ---------------------------------------------------------------------------

#: Measured with Hedgingbot's scripts/check_account_mode.py on a live
#: JustMarkets ECN demo account (login 1100219238). Used only when MT5 is
#: unavailable. logical -> (broker symbol, tick_size, tick_value)
FALLBACK_SPECS: dict[str, tuple[str, float, float]] = {
    "GOLD":     ("XAUUSD.ecn", 0.01,    1.0),
    "SILVER":   ("XAGUSD.ecn", 0.001,   5.0),
    "PLATINUM": ("XPTUSD.ecn", 0.01,    1.0),
    "NATGAS":   ("XNGUSD.ecn", 0.001,   1.0),    # NOT verified — confirm on your account
    "BRENT":    ("BRENT.ecn",  0.01,    1.0),    # NOT verified
    "GBPJPY":   ("GBPJPY.ecn", 0.001,   0.6276675872457946),
    "AUDUSD":   ("AUDUSD.ecn", 0.00001, 1.0),
    "USDJPY":   ("USDJPY.ecn", 0.001,   0.6276675872457946),
}

#: Spread assumptions per asset class, matching Tradingbot's own experiments.
SPREAD_BY_CLASS = {"forex": 0.0001, "commodity": 0.0004, "index": 0.0002}

#: DEFAULT_PORTFOLIO slot name -> yahoo_loader.INSTRUMENTS key.
#:
#: These two tables inside Tradingbot do not agree, and the disagreement is
#: silent. src/live/portfolio.py names its slots GOLD and SILVER, while
#: src/data/yahoo_loader.py keys the same instruments XAUUSD and XAGUSD. The live
#: trader never notices because it maps logical -> broker symbol through
#: config/live_config.yaml; only backtest tooling touches INSTRUMENTS directly.
#:
#: Without this map, a naive `INSTRUMENTS[slot.logical]` lookup skips GOLD and
#: SILVER — the two largest metals positions, which are precisely the
#: concentration this whole validation exists to measure. It would have looked
#: like a successful export of 6 slots.
SLOT_TO_INSTRUMENT: dict[str, str] = {
    "GOLD": "XAUUSD",
    "SILVER": "XAGUSD",
    # The rest already match by name.
    "PLATINUM": "PLATINUM",
    "NATGAS": "NATGAS",
    "BRENT": "BRENT",
    "GBPJPY": "GBPJPY",
    "AUDUSD": "AUDUSD",
    "USDJPY": "USDJPY",
}


def load_specs(use_mt5: bool = True) -> tuple[dict[str, dict], str]:
    """Real contract specs per logical name. Prefers live MT5 values.

    Returns (specs, source) where specs is
    {logical: {symbol, tick_size, tick_value, volume_min, volume_step, volume_max}}
    """
    specs: dict[str, dict] = {}

    if use_mt5:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            mt5 = None
        if mt5 is not None and mt5.initialize():
            try:
                for logical, (symbol, _ts, _tv) in FALLBACK_SPECS.items():
                    info = mt5.symbol_info(symbol)
                    if info is None:
                        mt5.symbol_select(symbol, True)
                        info = mt5.symbol_info(symbol)
                    if info is None:
                        print(f"  {logical:<10} {symbol:<12} NOT FOUND at broker")
                        continue
                    tick_size = float(getattr(info, "trade_tick_size", 0.0)
                                      or getattr(info, "point", 0.0))
                    tick_value = float(getattr(info, "trade_tick_value", 0.0))
                    if tick_size <= 0 or tick_value <= 0:
                        print(f"  {logical:<10} {symbol:<12} unusable "
                              f"(tick_size={tick_size}, tick_value={tick_value})")
                        continue
                    specs[logical] = {
                        "symbol": symbol,
                        "tick_size": tick_size,
                        "tick_value": tick_value,
                        "volume_min": float(info.volume_min),
                        "volume_step": float(info.volume_step),
                        "volume_max": float(info.volume_max),
                    }
                    print(f"  {logical:<10} {symbol:<12} tick_size={tick_size:<10} "
                          f"tick_value={tick_value:<8.4f} "
                          f"-> {tick_value / tick_size:,.0f} per price unit per lot")
            finally:
                mt5.shutdown()
            if specs:
                return specs, "MetaTrader5 (live)"

    print("  MT5 unavailable — using the measured fallback table.")
    print("  WARNING: NATGAS and BRENT specs in that table are NOT verified.")
    for logical, (symbol, tick_size, tick_value) in FALLBACK_SPECS.items():
        specs[logical] = {
            "symbol": symbol, "tick_size": tick_size, "tick_value": tick_value,
            "volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0,
        }
    return specs, "fallback table (unverified for NATGAS/BRENT)"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export(
    balance: float,
    risk_percent: float,
    days: int,
    use_mt5: bool,
    out_path: str,
) -> int:
    print("=" * 78)
    print("CONTRACT SPECS")
    print("=" * 78)
    specs, spec_source = load_specs(use_mt5)

    risk = RiskParams(
        risk_percent_per_trade=risk_percent,
        max_risk_percent_per_trade=5.0,
        # MIN, not SKIP: for a research export we want the trade recorded even
        # when the minimum lot exceeds the risk budget, so the position timeline
        # stays complete. The overlay is being validated on EXPOSURE, and a
        # silently missing trade would understate it.
        on_min_lot_exceeds_risk=MinLotPolicy.MIN,
    )
    # bars_per_year for H4: 6 bars/day * 252 trading days.
    config = BacktestConfig(initial_balance=balance, bars_per_year=6 * 252,
                            allow_short=True)

    print()
    print("=" * 78)
    print("BACKTESTING EACH SLOT")
    print("=" * 78)
    print(f"  balance={balance:,.2f}  risk={risk_percent}%/trade  "
          f"H4 history={days} days")
    print()

    slots_out: list[dict] = []
    total_trades = 0

    skipped: list[str] = []
    for slot in DEFAULT_PORTFOLIO:
        logical = slot.logical
        inst_key = SLOT_TO_INSTRUMENT.get(logical, logical)
        if inst_key not in INSTRUMENTS:
            print(f"  {logical:<10} SKIPPED — no INSTRUMENTS entry "
                  f"(tried {inst_key!r}); add it to SLOT_TO_INSTRUMENT")
            skipped.append(logical)
            continue
        if logical not in specs:
            print(f"  {logical:<10} SKIPPED — no contract spec")
            skipped.append(logical)
            continue

        ysym, _broker, asset_class = INSTRUMENTS[inst_key]
        spec_d = specs[logical]

        try:
            df = fetch_yahoo_h4(ysym)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {logical:<10} FETCH FAILED: {exc!r}")
            continue
        if len(df) < 200:
            print(f"  {logical:<10} SKIPPED — only {len(df)} H4 bars")
            continue

        strategy = slot.build()
        symbol = SymbolSpec(
            tick_size=spec_d["tick_size"], tick_value=spec_d["tick_value"],
            volume_min=spec_d["volume_min"], volume_step=spec_d["volume_step"],
            volume_max=spec_d["volume_max"],
        )
        cost = CostModel(spread_frac=SPREAD_BY_CLASS.get(asset_class, 0.0004))

        vol_scalar = None
        if slot.use_vol_target:
            vol_scalar = volatility_target_scalar(
                df["close"], lookback=14, median_window=500,
                min_scale=0.5, max_scale=1.5,
            )

        try:
            report, trades, _equity = run_backtest(
                df, strategy, symbol, risk, cost=cost, config=config,
                vol_scalar=vol_scalar,
            )
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {logical:<10} BACKTEST FAILED: {exc!r}")
            continue

        rows = []
        for t in trades:
            rows.append({
                "entry_time": pd.Timestamp(t.entry_time).isoformat(),
                "exit_time": pd.Timestamp(t.exit_time).isoformat(),
                "side": int(t.side),
                "entry": float(t.entry),
                "exit": float(t.exit),
                "lots": float(t.lots),
                "pnl": float(t.pnl),
                "r_multiple": float(t.r_multiple),
                "reason": str(t.reason),
            })

        slots_out.append({
            "logical": logical,
            "broker_symbol": spec_d["symbol"],
            "yahoo_symbol": ysym,
            "asset_class": asset_class,
            "timeframe": slot.timeframe,
            "strategy": type(strategy).__name__,
            "params": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                           else str(v))
                       for k, v in slot.params.items()},
            "use_vol_target": bool(slot.use_vol_target),
            "spec": spec_d,
            "spread_frac": cost.spread_frac,
            "bars": int(len(df)),
            "bar_range": [pd.Timestamp(df.index[0]).isoformat(),
                          pd.Timestamp(df.index[-1]).isoformat()],
            "report": {
                "num_trades": int(report.num_trades),
                "win_rate": float(report.win_rate),
                "profit_factor": float(report.profit_factor),
                "expectancy_r": float(report.expectancy_r),
                "total_return_pct": float(report.total_return_pct),
                "max_drawdown_pct": float(report.max_drawdown_pct),
                "sharpe": float(report.sharpe),
            },
            "trades": rows,
        })
        total_trades += len(rows)
        print(f"  {logical:<10} {type(strategy).__name__:<24} "
              f"{len(rows):>4} trades  "
              f"expR={report.expectancy_r:+.3f}  "
              f"ret={report.total_return_pct:+7.1f}%  "
              f"maxDD={report.max_drawdown_pct:5.1f}%")

    if not slots_out:
        print("\nFATAL: no slot exported. Nothing was written.")
        return 1

    # A partial export is a trap: 6 of 8 slots looks like success but silently
    # drops exposure, and the whole point is measuring TOTAL concentration.
    expected = [s.logical for s in DEFAULT_PORTFOLIO
                if s.logical in SLOT_TO_INSTRUMENT]
    got = [s["logical"] for s in slots_out]
    missing = [s for s in expected if s not in got]
    if missing:
        print()
        print(f"  WARNING: {len(missing)} expected slot(s) missing: "
              f"{', '.join(missing)}")
        print("  The replay will UNDERSTATE total exposure. Fix these before")
        print("  trusting any validation result.")

    payload = {
        "schema": 1,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": "Tradingbot DEFAULT_PORTFOLIO backtest",
        "spec_source": spec_source,
        "balance": balance,
        "risk_percent_per_trade": risk_percent,
        "min_lot_policy": "MIN",
        "expected_slots": expected,
        "missing_slots": missing,
        "caveat": (
            "Each slot was backtested INDEPENDENTLY, so one slot's drawdown does "
            "not shrink another's position size. This mirrors Tradingbot's live "
            "behaviour (every slot sizes off the shared acct.balance with up to "
            "max_open_trades concurrent positions), but it remains an "
            "approximation."
        ),
        "slots": slots_out,
    }

    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print()
    print("=" * 78)
    print("EXPORTED")
    print("=" * 78)
    print(f"  {out_path}")
    print(f"  {len(slots_out)} slots, {total_trades} trades")
    print(f"  contract specs from: {spec_source}")
    print()
    print("  Next: copy this file to Hedgingbot and replay the overlay on it:")
    print(f"      copy {os.path.basename(out_path)} C:\\Hedgingbot\\data\\")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export Tradingbot backtest trades for Hedgingbot validation."
    )
    ap.add_argument("--balance", type=float, default=10_000.0,
                    help="starting balance for sizing (use your real account size, "
                         "since hedge viability depends on it)")
    ap.add_argument("--risk", type=float, default=2.0,
                    help="risk %% per trade, matching live_config.yaml")
    ap.add_argument("--days", type=int, default=729,
                    help="H4 history days (Yahoo caps intraday at ~730)")
    ap.add_argument("--no-mt5", action="store_true",
                    help="skip MT5 and use the measured fallback specs")
    ap.add_argument("--out", default="trades_export.json")
    args = ap.parse_args()

    return export(
        balance=args.balance, risk_percent=args.risk, days=args.days,
        use_mt5=not args.no_mt5, out_path=args.out,
    )


if __name__ == "__main__":
    sys.exit(main())
