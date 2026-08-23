"""
OBSERVE MODE: measure the live factor exposure of the account and log it.
Places no orders — it physically cannot, the connector it uses has no order code.

    python scripts/measure_exposure.py                 # one snapshot
    python scripts/measure_exposure.py --loop           # poll forever
    python scripts/measure_exposure.py --demo           # no MT5 needed, fake book

Run this for a few weeks before enabling any hedging. It costs nothing and it
answers the questions you cannot answer any other way:

  * What factor leverage does the book ACTUALLY run, day to day?
  * How often would each candidate cap have been breached?
  * How many effective independent bets are the 8 slots really making?
  * What gross notional does the book carry, so max_hedge_gross_pct can be set
    from data rather than guessed?

Only after those are known can the caps be chosen sensibly. A cap above the
observed maximum does nothing; a cap far below it hedges constantly and bleeds
spread. Both failures are invisible until measured.

--demo generates a synthetic 8-slot book with plausible betas so the output format
can be inspected on any machine, including Linux with no MT5.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exposure.model import InstrumentSpec, Position, compute_exposure   # noqa: E402
from src.exposure.report import render, render_positions                     # noqa: E402
from src.factors.book import BetaBook                                        # noqa: E402
from src.factors.definitions import FACTORS                                  # noqa: E402
from src.factors.store import StaleBetasError, load_betas                    # noqa: E402
from src.overlay.decision import (                                           # noqa: E402
    HedgeCaps,
    HedgeInstrument,
    hedge_decide,
)
from src.state.shared import (                                               # noqa: E402
    SharedPortfolioState,
    read_trader_status,
    write_overlay_status,
)


# ---------------------------------------------------------------------------
# Demo fixtures — a synthetic version of Tradingbot's DEFAULT_PORTFOLIO
# ---------------------------------------------------------------------------

DEMO_BETAS = {
    "GOLD":     {"USD": -0.90, "RISK": 0.00, "ENERGY": 0.10, "METALS": 1.00},
    "SILVER":   {"USD": -1.00, "RISK": 0.30, "ENERGY": 0.10, "METALS": 0.80},
    "PLATINUM": {"USD": -0.80, "RISK": 0.40, "ENERGY": 0.20, "METALS": 0.50},
    "NATGAS":   {"USD": -0.20, "RISK": 0.10, "ENERGY": 0.50, "METALS": 0.00},
    "BRENT":    {"USD": -0.40, "RISK": 0.40, "ENERGY": 1.00, "METALS": 0.00},
    "GBPJPY":   {"USD": 0.00, "RISK": 0.50, "ENERGY": 0.00, "METALS": -0.10},
    "AUDUSD":   {"USD": -0.90, "RISK": 0.50, "ENERGY": 0.10, "METALS": 0.20},
    "USDJPY":   {"USD": 0.80, "RISK": 0.30, "ENERGY": 0.00, "METALS": -0.20},
    "EURUSD":   {"USD": -1.00, "RISK": 0.02, "ENERGY": 0.00, "METALS": 0.01},
    "SP500":    {"USD": -0.10, "RISK": 1.00, "ENERGY": 0.10, "METALS": 0.00},
    "WTI":      {"USD": -0.40, "RISK": 0.35, "ENERGY": 1.00, "METALS": 0.00},
}

DEMO_FACTOR_SIGMA = {"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022, "METALS": 0.009}

DEMO_INSTRUMENT_VOL = {
    "GOLD": 0.010, "SILVER": 0.018, "PLATINUM": 0.016, "NATGAS": 0.035,
    "BRENT": 0.021, "GBPJPY": 0.007, "AUDUSD": 0.006, "USDJPY": 0.005,
    "EURUSD": 0.005, "SP500": 0.011, "WTI": 0.022,
}

# (logical, broker symbol, lots, price, money-per-price-unit-per-lot)
DEMO_BOOK = [
    ("GOLD",     "XAUUSD", 0.02, 2400.0, 100.0),
    ("SILVER",   "XAGUSD", 0.10, 30.0, 5000.0),
    ("PLATINUM", "XPTUSD", 0.05, 1000.0, 100.0),
    ("NATGAS",   "NGAS",   0.20, 3.0, 10000.0),
    ("BRENT",    "UKOIL",  0.08, 80.0, 1000.0),
    ("GBPJPY",   "GBPJPY", 0.10, 190.0, 670.0),
    ("AUDUSD",   "AUDUSD", 0.10, 0.65, 100000.0),
    ("USDJPY",   "USDJPY", 0.10, 150.0, 670.0),
]

DEMO_HEDGE_CANDIDATES = [
    ("EURUSD", "EURUSD", 1.08, 100000.0),
    ("SP500",  "US500",  5000.0, 1.0),
    ("WTI",    "USOIL",  78.0, 1000.0),
]


def demo_book() -> BetaBook:
    resid = {}
    for inst, bv in DEMO_BETAS.items():
        total_var = DEMO_INSTRUMENT_VOL[inst] ** 2
        explained = math.fsum(
            (b * DEMO_FACTOR_SIGMA[f]) ** 2 for f, b in bv.items()
        )
        resid[inst] = math.sqrt(max(total_var - explained, 0.25 * total_var))
    return BetaBook(
        factors=FACTORS, betas=DEMO_BETAS,
        sigma=dict(DEMO_INSTRUMENT_VOL), resid_sigma=resid,
        r_squared={k: 0.6 for k in DEMO_BETAS},
        n_obs={k: 2000 for k in DEMO_BETAS},
        factor_sigma=DEMO_FACTOR_SIGMA,
    )


def demo_positions() -> tuple[list[Position], dict[str, InstrumentSpec]]:
    positions, specs = [], {}
    for logical, sym, lots, price, mppu in DEMO_BOOK:
        specs[sym] = InstrumentSpec(sym, tick_size=0.001, tick_value=0.001 * mppu,
                                    volume_min=0.01, volume_step=0.01,
                                    volume_max=100.0, logical=logical)
        positions.append(Position(sym, 1, lots, price, ticket=hash(sym) % 100000,
                                  magic=990011, logical=logical))
    return positions, specs


def demo_candidates() -> list[HedgeInstrument]:
    out = []
    for logical, sym, price, mppu in DEMO_HEDGE_CANDIDATES:
        spec = InstrumentSpec(sym, tick_size=0.001, tick_value=0.001 * mppu,
                              volume_min=0.01, volume_step=0.01, volume_max=100.0,
                              logical=logical)
        half = price * 0.00005
        out.append(HedgeInstrument(logical, sym, spec, price - half, price + half))
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


#: Phase 1 has no order-placement path at all, so nothing can be sent regardless
#: of config. Named explicitly so the budget-accounting branch below reads as a
#: deliberate no-op rather than dead code.
DRY_RUN_ALWAYS = True

#: Tradingbot's magic number (src/live/portfolio.py / live_config.yaml). Sharing it
#: would make the overlay mistake the trader's positions for its own hedges.
TRADINGBOT_MAGIC = 990011


def _beta_age_days(path: str) -> float | None:
    """Age of the beta file in days, or None if it cannot be determined."""
    import json as _json
    from datetime import datetime, timezone

    try:
        with open(path, "r", encoding="utf-8") as fh:
            created = str(_json.load(fh).get("created_at", ""))
    except (OSError, ValueError):
        return None
    if not created:
        return None
    try:
        ts = datetime.fromisoformat(created)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0


def _load_provenance(path: str) -> dict:
    """ticket -> originating factor, for hedges this overlay opened.

    Unwind decisions must use the factor a hedge was OPENED for, not whichever
    factor its instrument currently ranks highest for (audit item 10). Betas are
    re-estimated monthly and rankings move, so a hedge opened against USD can
    later look like a METALS hedge and be unwound on the wrong signal.

    Phase 1 opens nothing, so this is normally empty. Phase 2 must write an entry
    when it places a hedge, and additionally stamp the factor into the MT5 order
    comment so the mapping survives losing this file.
    """
    import json as _json

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = _json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: str(v) for k, v in raw.items() if isinstance(v, str)}


def load_config(path: str) -> dict:
    """Load YAML config, falling back to a minimal parser if PyYAML is absent."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        print(f"  NOTE: PyYAML not installed; using defaults instead of {path}")
        return {}


def caps_from_config(cfg: dict) -> HedgeCaps:
    caps = cfg.get("caps") or {}
    bands = cfg.get("bands") or {}
    limits = cfg.get("limits") or {}
    return HedgeCaps(
        factor_caps={k: float(v) for k, v in caps.items()} or
                    {"USD": 1.5, "RISK": 1.5, "ENERGY": 1.0, "METALS": 1.25},
        hysteresis=float(bands.get("hysteresis", 0.25)),
        target_fraction=float(bands.get("target_fraction", 0.80)),
        unwind_band=float(bands.get("unwind_band", 0.60)),
        max_hedge_gross_pct=float(limits.get("max_hedge_gross_pct", 1000.0)),
        min_purity=float(limits.get("min_purity", 0.50)),
        min_abs_beta=float(limits.get("min_abs_beta", 0.20)),
        max_actions_per_cycle=int(limits.get("max_actions_per_cycle", 1)),
        max_hedges_per_day=int(limits.get("max_hedges_per_day", 6)),
        allow_unmapped=bool(limits.get("allow_unmapped", False)),
        min_excess_removed=float(limits.get("min_excess_removed", 0.50)),
    )


# ---------------------------------------------------------------------------
# One measurement cycle
# ---------------------------------------------------------------------------


def measure_once(
    positions: list[Position],
    specs: dict[str, InstrumentSpec],
    book: BetaBook,
    equity: float,
    caps: HedgeCaps,
    candidates: list[HedgeInstrument],
    currency: str = "",
    hedge_magic: int = 990012,
    trader_symbols: frozenset[str] = frozenset(),
    verbose: bool = True,
    hedges_today: int = 0,
    hedge_provenance: dict | None = None,
) -> tuple[object, object]:
    report = compute_exposure(positions, specs, book, equity)

    hedges = [p for p in positions if p.magic == hedge_magic]

    # AVOID LIST FROM THE LIVE BOOK, not just from config.
    #
    #     5. Static avoid list / SILVER conflict — Live manual or other-EA
    #        positions may not be protected; SILVER has conflicting roles.
    #
    # Two distinct problems. First, deriving `avoid` purely from `trader_symbols`
    # misses anything the config does not know about: manual trades, another EA,
    # a slot added to Tradingbot but not here. Those are real positions the
    # overlay must not trade against. Second, SILVER is both a trader slot and the
    # METALS hedge candidate, so on a netting account a "hedge" would silently
    # reduce the trader's own silver position instead of hedging anything.
    #
    # Taking the union of configured symbols and every symbol currently held by
    # someone else fixes both: any symbol another party holds is off-limits this
    # cycle, whatever the config says.
    others_hold = {p.symbol for p in positions if p.magic != hedge_magic}
    avoid = frozenset(trader_symbols) | others_hold

    plan = hedge_decide(
        report, book, caps, candidates,
        existing_hedges=hedges, hedge_specs=specs,
        avoid_symbols=avoid,
        hedges_today=hedges_today,
        hedge_provenance=hedge_provenance,
    )

    blocked_by_live = sorted(others_hold - frozenset(trader_symbols))
    if blocked_by_live and verbose:
        print(f"\n  NOTE: {', '.join(blocked_by_live)} held by another party "
              f"(manual or other EA) — excluded as hedge instruments this cycle "
              f"even though config permits them.")

    if verbose:
        print(render(report, caps.factor_caps, currency))
        print()
        print(render_positions(report))
        print()
        print("=" * 78)
        print("WHAT THE OVERLAY WOULD DO (observe mode — nothing is sent)")
        print("=" * 78)
        for w in getattr(plan, "warnings", []):
            print(f"  WARNING: {w}")
        if getattr(plan, "warnings", []):
            print()
        print(f"  {plan.summary()}")
        if plan.has_trades:
            print("\n  Proposed order(s):")
            for a in plan.trades():
                verb = "SELL" if a.side < 0 else "BUY"
                print(f"    {a.action.upper():<6} {verb} {a.lots:.2f} lots {a.symbol}"
                      f"  [{a.factor}]")
        print("=" * 78)

    return report, plan


def append_history(path: str, report, plan, equity: float) -> None:
    """One JSON object per line — trivially greppable, and safe to append to
    while another process reads it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "equity": round(equity, 2),
        "leverage": {k: round(v, 4) for k, v in report.factor_leverage.items()},
        "variance_shares": {k: round(v, 4) for k, v in report.variance_shares().items()},
        "portfolio_daily_risk": round(report.portfolio_daily_risk, 2),
        "portfolio_daily_risk_pct": round(report.portfolio_daily_risk_pct, 3),
        "effective_bets": round(report.effective_bet_count, 3),
        "gross_notional": round(report.gross_notional, 2),
        "gross_leverage": round(report.gross_leverage, 3),
        "n_positions": len(report.positions),
        "plan": plan.summary(),
        "would_trade": plan.has_trades,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure live factor exposure. Places no orders."
    )
    ap.add_argument("--config", default=os.path.join("config", "hedge_config.yaml"))
    ap.add_argument("--betas", default=None, help="override the beta file path")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic book with plausible betas; no MT5 required")
    ap.add_argument("--loop", action="store_true", help="poll continuously")
    ap.add_argument("--interval", type=int, default=None, help="poll seconds")
    ap.add_argument("--equity", type=float, default=10_000.0,
                    help="equity to assume in --demo mode")
    ap.add_argument("--allow-stale-betas", action="store_true")
    ap.add_argument("--history", default=os.path.join("logs", "exposure_history.jsonl"))
    ap.add_argument("--quiet", action="store_true", help="only append history")
    args = ap.parse_args()

    cfg = load_config(args.config)
    caps = caps_from_config(cfg)
    run = cfg.get("run") or {}
    interval = args.interval or int(run.get("poll_seconds", 60))
    hedge_magic = int(run.get("magic_number", 990012))

    # MAGIC COLLISION IS A RUNTIME CHECK, not just a preflight one.
    #
    #     "Hedge identification relies on the configured magic number, with no
    #      runtime validation that it differs from the trader's magic number."
    #
    # If both bots share a magic, the overlay counts the trader's positions as its
    # own hedges — and would then "unwind" them. Refuse to start.
    if hedge_magic == TRADINGBOT_MAGIC:
        print(f"FATAL: run.magic_number is {hedge_magic}, the same as Tradingbot's.")
        print("The overlay would treat the trader's positions as its own hedges and")
        print("could close them. Use a different magic, e.g. 990012.")
        return 1

    trader_symbols = frozenset((cfg.get("trader_symbols") or {}).values())

    # ---- demo path -------------------------------------------------------
    if args.demo:
        print("DEMO MODE — synthetic positions and betas. Numbers are illustrative,")
        print("not measurements of your account.\n")
        book = demo_book()
        positions, specs = demo_positions()
        candidates = demo_candidates()
        for h in candidates:
            specs.setdefault(h.symbol, h.spec)
        report, plan = measure_once(
            positions, specs, book, args.equity, caps, candidates,
            currency="USD", hedge_magic=hedge_magic,
            trader_symbols=frozenset(s for _l, s, *_r in DEMO_BOOK),
            verbose=not args.quiet,
        )
        append_history(args.history, report, plan, args.equity)
        print(f"\nAppended a row to {args.history}")
        return 0

    # ---- live path -------------------------------------------------------
    beta_cfg = cfg.get("betas") or {}
    beta_path = args.betas or beta_cfg.get("path") or os.path.join("betas",
                                                                  "betas_latest.json")
    try:
        book, meta = load_betas(
            beta_path,
            max_age_days=float(beta_cfg.get("max_age_days", 30)),
            allow_stale=args.allow_stale_betas,
        )
    except FileNotFoundError:
        print(f"No beta file at {beta_path}.")
        print("Run: python scripts/estimate_betas.py")
        print("Or preview the output format with: "
              "python scripts/measure_exposure.py --demo")
        return 1
    except StaleBetasError as exc:
        print(f"{exc}")
        return 1

    print(f"Loaded betas from {beta_path} "
          f"(created {meta.get('created_at')}, age {meta.get('age_days', 0):.1f}d, "
          f"{len(book.instruments())} instruments)")

    from src.connectors.mt5_reader import (
        MT5Reader,
        MT5Unavailable,
        UnvaluablePosition,
    )

    symbol_map = dict(cfg.get("trader_symbols") or {})
    symbol_map.update(cfg.get("hedge_instruments") or {})

    # SINGLE INSTANCE. The config documented lock_path but nothing ever acquired
    # it (audit item 7). Two observers are only wasteful; two EXECUTORS would each
    # decide the same breach needs hedging and place the hedge twice, then see each
    # other's position and potentially over-correct. Acquire it now so the
    # behaviour is established before Phase 2 depends on it.
    from src.state.lock import AlreadyRunning, hold

    try:
        hold(str(run.get("lock_path", os.path.join("logs", "hedge.lock"))))
    except AlreadyRunning as exc:
        print(f"FATAL: {exc}")
        return 1

    reader = MT5Reader(symbol_map=symbol_map)
    if not reader.available():
        print("\nMetaTrader5 package not installed — cannot read a live account.")
        print("Preview the output format instead with: "
              "python scripts/measure_exposure.py --demo")
        return 1

    conn_cfg = cfg.get("connection") or {}
    try:
        acct = reader.connect(
            login=conn_cfg.get("login"), password=conn_cfg.get("password"),
            server=conn_cfg.get("server"), path=conn_cfg.get("path"),
        )
    except (ConnectionError, MT5Unavailable) as exc:
        print(f"Could not attach to the MT5 terminal: {exc}")
        return 1

    print(f"Account {acct.login} on {acct.currency} — margin mode "
          f"{acct.margin_mode_name}, equity {acct.equity:,.2f}")
    if not acct.algo_trading_enabled:
        print("  NOTE: algo trading is currently DISABLED on this account/terminal. "
              "Harmless in observe mode, blocking in Phase 2.")

    state = SharedPortfolioState.load(str(run.get("state_path",
                                                  os.path.join("logs",
                                                               "portfolio_state.json"))))
    state.sync_baseline(acct.balance)
    state.overlay_mode = "observe"

    beta_max_age = float(beta_cfg.get("max_age_days", 30))
    provenance_path = str(run.get("provenance_path",
                                  os.path.join("logs", "hedge_provenance.json")))
    reported_spec_warnings: set[str] = set()

    try:
        while True:
            acct = reader.account()
            state.note_equity(acct.equity)
            state.roll_day(equity=acct.equity)

            # BETA STALENESS IS RE-CHECKED EVERY CYCLE, not once at startup.
            #
            #     4. One-time beta staleness check — A long-running process can
            #        continue using arbitrarily old betas.
            #
            # Exactly the situation observe mode creates: it is meant to run for
            # WEEKS. Loading betas once and validating them once means that after
            # 31 days it is happily sizing decisions from a file its own rules
            # would refuse to load.
            age_days = _beta_age_days(beta_path)
            if age_days is not None and age_days > beta_max_age:
                print(f"\n  BETAS ARE STALE: {age_days:.1f} days old "
                      f"(limit {beta_max_age}). Measurement continues — knowing "
                      f"your exposure with old betas beats knowing nothing — but "
                      f"the numbers are drifting and NO hedge should be placed. "
                      f"Run scripts\\update_betas.bat")
                stale_betas = True
            else:
                stale_betas = False

            try:
                positions = reader.all_positions()
            except UnvaluablePosition as exc:
                # A position we cannot price means the book is incomplete. Skip the
                # CYCLE, never the position — see UnvaluablePosition.
                print(f"\n  SKIPPING THIS CYCLE: {exc}")
                if not args.loop:
                    return 1
                time.sleep(interval)
                continue

            specs = reader.specs_for(positions)

            for w in reader.spec_warnings:
                if w not in reported_spec_warnings:
                    print(f"\n  BROKER DATA WARNING: {w}")
                    reported_spec_warnings.add(w)

            candidates: list[HedgeInstrument] = []
            for logical, sym in (cfg.get("hedge_instruments") or {}).items():
                try:
                    spec = reader.spec(sym)
                    bid, ask = reader.tick(sym)
                    candidates.append(HedgeInstrument(logical, sym, spec, bid, ask))
                    specs.setdefault(sym, spec)
                except (KeyError, ValueError, ConnectionError) as exc:
                    print(f"  hedge candidate {logical} ({sym}) unavailable: {exc}")

            report, plan = measure_once(
                positions, specs, book, acct.equity, caps, candidates,
                currency=acct.currency, hedge_magic=hedge_magic,
                trader_symbols=trader_symbols, verbose=not args.quiet,
                hedges_today=state.hedge_count_today,
                hedge_provenance=_load_provenance(provenance_path),
            )

            if stale_betas:
                plan.warnings.append(
                    f"betas are {age_days:.1f} days old (limit {beta_max_age}); "
                    "treat any proposed hedge as advisory only"
                )

            # Record executed hedges against the daily budget. In observe mode
            # nothing is sent, so nothing is counted — but the wiring is here and
            # exercised, rather than being a counter that exists and is never
            # incremented (audit item 1: the daily budget was cosmetic because
            # hedges_today was never passed in and note_hedge() was never called).
            if not DRY_RUN_ALWAYS and plan.has_trades:
                for _a in plan.trades():
                    state.note_hedge()

            # Monitoring writes are individually guarded — the principle from
            # Tradingbot's HARDENING_LOG: monitoring must never be able to stop
            # the thing it monitors.
            for label, fn in (
                ("history", lambda: append_history(args.history, report, plan,
                                                   acct.equity)),
                ("status", lambda: write_overlay_status(
                    state, equity=acct.equity, balance=acct.balance,
                    currency=acct.currency, report=report, plan=plan,
                    caps=caps.factor_caps, dry_run=True,
                    path=str(run.get("status_path",
                                     os.path.join("logs", "hedge_status.json"))))),
                ("state", lambda: state.save(
                    str(run.get("state_path",
                                os.path.join("logs", "portfolio_state.json"))),
                    by="overlay-observe")),
            ):
                try:
                    fn()
                except Exception as exc:                       # noqa: BLE001
                    print(f"  WARNING: {label} write failed: {exc!r}")

            trader = read_trader_status(str(run.get("trader_status_path",
                                                    os.path.join("logs",
                                                                 "status.json"))))
            if trader and not args.quiet:
                print(f"  trader heartbeat: equity={trader.get('equity')} "
                      f"halted={trader.get('halted')} "
                      f"updated={trader.get('updated_at')}")

            if not args.loop:
                return 0
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        reader.shutdown()


if __name__ == "__main__":
    sys.exit(main())
