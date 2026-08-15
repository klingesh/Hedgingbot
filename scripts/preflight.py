"""
Check everything the overlay needs, in dependency order, and say exactly what to
run next. Read-only and safe: it opens no orders and writes no files.

    python scripts/preflight.py

Run this FIRST, and again any time something does not work. It exists because the
three setup steps have non-obvious ordering dependencies:

    check_account_mode.py  needs MT5, needs NO config, needs NO internet
    estimate_betas.py      needs internet, needs NO MT5, needs NO config
    measure_exposure.py    needs MT5 AND config AND a fresh beta file

So running them in the wrong order, or on the wrong machine, fails in ways whose
error messages do not obviously point at the real cause. This script checks each
prerequisite and prints the one command that fixes the first thing that is broken.

Every check degrades gracefully: on Linux or Mac the MT5 checks report SKIP rather
than failing, because the measurement and decision layers are testable anywhere and
only the live account reading needs Windows.
"""

from __future__ import annotations

import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

_MARK = {PASS: "[ ok ]", FAIL: "[FAIL]", WARN: "[warn]", SKIP: "[skip]"}


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.next_steps: list[str] = []

    def add(self, status: str, name: str, detail: str = "") -> str:
        self.rows.append((status, name, detail))
        print(f"  {_MARK[status]} {name}")
        for line in str(detail).splitlines():
            if line.strip():
                print(f"         {line}")
        return status

    def suggest(self, command: str, why: str) -> None:
        self.next_steps.append((command, why))

    @property
    def failed(self) -> int:
        return sum(1 for s, _n, _d in self.rows if s == FAIL)

    @property
    def warned(self) -> int:
        return sum(1 for s, _n, _d in self.rows if s == WARN)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------


def check_environment(c: Checks) -> None:
    hr("0. ENVIRONMENT")

    v = sys.version_info
    if v >= (3, 9):
        c.add(PASS, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        c.add(FAIL, f"Python {v.major}.{v.minor}.{v.micro} is too old",
              "Need 3.9 or newer. Tradingbot already requires Python on this "
              "machine, so check you are using the same interpreter.")

    c.add(PASS, f"Platform {platform.system()} {platform.machine()}")

    # Third-party imports are OPTIONAL by design.
    try:
        import yaml            # noqa: F401
        c.add(PASS, "PyYAML installed (config parsing)")
    except ImportError:
        c.add(WARN, "PyYAML not installed",
              "The overlay falls back to defaults instead of reading your config, "
              "which is almost certainly not what you want.\n"
              "Fix:  pip install PyYAML")
        c.suggest("pip install PyYAML", "so your config file is actually read")

    for mod in ("pandas", "numpy"):
        try:
            __import__(mod)
            c.add(PASS, f"{mod} present (not required, but harmless)")
        except ImportError:
            c.add(PASS, f"{mod} absent — fine, this project needs neither")

    # The pure layers must import cleanly regardless of platform.
    try:
        from src.exposure.model import compute_exposure      # noqa: F401
        from src.factors.math_core import build_orthogonal_factors  # noqa: F401
        from src.overlay.decision import hedge_decide        # noqa: F401
        c.add(PASS, "Core modules import (factors, exposure, overlay)")
    except Exception as exc:                                  # noqa: BLE001
        c.add(FAIL, "Core modules failed to import", repr(exc))


# ---------------------------------------------------------------------------
# 1. MT5
# ---------------------------------------------------------------------------


def check_mt5(c: Checks) -> object | None:
    hr("1. METATRADER 5  (needed by check_account_mode.py and measure_exposure.py)")

    try:
        import MetaTrader5 as mt5       # noqa: F401
    except ImportError:
        c.add(SKIP, "MetaTrader5 package not installed",
              "Expected on Linux/Mac — the package is Windows-only.\n"
              "On the Windows machine that runs Tradingbot:\n"
              "    pip install MetaTrader5\n"
              "Steps 1 and 3 must run there. Step 2 (betas) can run anywhere "
              "with internet.")
        return None

    c.add(PASS, "MetaTrader5 package installed")

    from src.connectors.mt5_reader import MT5Reader

    reader = MT5Reader()
    try:
        acct = reader.connect()
    except Exception as exc:                                  # noqa: BLE001
        c.add(FAIL, "Could not attach to the MT5 terminal", 
              f"{exc}\n"
              "Checklist:\n"
              "  * Is the MT5 terminal actually RUNNING and logged in?\n"
              "  * Is it the same Windows user as this Python process?\n"
              "  * Try passing an explicit path:\n"
              "      python scripts/check_account_mode.py "
              "--path \"C:\\Program Files\\MetaTrader 5\\terminal64.exe\"")
        return None

    c.add(PASS, f"Attached to account {acct.login} on {acct.server}",
          f"currency={acct.currency} leverage=1:{acct.leverage} "
          f"balance={acct.balance:,.2f} equity={acct.equity:,.2f}")

    # THE question we could not answer from Linux.
    mode = acct.margin_mode_name
    if mode == "RETAIL_HEDGING":
        c.add(PASS, f"Margin mode: {mode}",
              "Same-symbol opposing positions ARE possible. The overlay still "
              "avoids them by default (cleaner attribution, half the spread "
              "bill), but you have the option.")
    elif mode == "RETAIL_NETTING":
        c.add(PASS, f"Margin mode: {mode}",
              "Same-symbol opposing positions are NOT possible — an opposing "
              "order would REDUCE the trader's position. The overlay is already "
              "designed for this: keep every Tradingbot symbol listed under "
              "trader_symbols so it never touches them.")
    else:
        c.add(WARN, f"Margin mode: {mode}",
              "Unexpected value. Report it and we will handle it explicitly.")

    if acct.algo_trading_enabled:
        c.add(PASS, "Algo trading permitted (account + terminal)")
    else:
        c.add(WARN, "Algo trading is currently DISABLED",
              f"account.trade_allowed={acct.trade_allowed} "
              f"account.trade_expert={acct.trade_expert}\n"
              "Harmless for Phase 1 (observe mode places no orders), but it "
              "would block Phase 2. Press the AutoTrading button in the MT5 "
              "toolbar, and confirm the broker allows EAs on this account.")

    positions = reader.all_positions()
    if positions:
        magics = sorted({p.magic for p in positions})
        c.add(PASS, f"{len(positions)} open position(s) visible, magics={magics}",
              "positions_get() is unfiltered by magic, which is why the overlay "
              "can see Tradingbot's book with no shared file.")
    else:
        c.add(WARN, "No open positions right now",
              "Not a problem, but observe mode has nothing to measure until "
              "Tradingbot opens something. Leave it looping.")

    return reader


# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------


def check_config(c: Checks, reader) -> dict:
    hr("2. CONFIG  (needed by measure_exposure.py only)")

    path = rel("config", "hedge_config.yaml")
    example = rel("config", "hedge_config.example.yaml")

    if not os.path.exists(path):
        c.add(FAIL, "config/hedge_config.yaml is missing",
              "Copy the example and edit it:\n"
              "    Windows:  copy config\\hedge_config.example.yaml "
              "config\\hedge_config.yaml\n"
              "    Linux/Mac: cp config/hedge_config.example.yaml "
              "config/hedge_config.yaml\n"
              "It is gitignored, so your broker details stay local.")
        c.suggest("copy config\\hedge_config.example.yaml config\\hedge_config.yaml",
                  "create your local config")
        return {}

    c.add(PASS, "config/hedge_config.yaml exists")

    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except ImportError:
        c.add(WARN, "Cannot parse the config without PyYAML", "pip install PyYAML")
        return {}
    except Exception as exc:                                  # noqa: BLE001
        c.add(FAIL, "config/hedge_config.yaml is not valid YAML", repr(exc))
        return {}

    mode = cfg.get("mode")
    if mode == "observe":
        c.add(PASS, "mode: observe",
              "Correct for Phase 1. Nothing can be traded.")
    else:
        c.add(WARN, f"mode: {mode!r}",
              "Phase 1 has no order code at all, so this has no effect yet. "
              "Set it back to 'observe' to avoid confusion.")

    run = cfg.get("run") or {}
    magic = run.get("magic_number")
    if magic == 990011:
        c.add(FAIL, f"magic_number is {magic} — the SAME as Tradingbot",
              "The two bots could not tell their positions apart. Use 990012.")
    elif magic:
        c.add(PASS, f"magic_number {magic} differs from Tradingbot's 990011")
    else:
        c.add(WARN, "No run.magic_number set", "Defaults to 990012.")

    lock = str(run.get("lock_path", ""))
    if lock and "bot.lock" in lock and "hedge" not in lock:
        c.add(FAIL, f"lock_path {lock!r} collides with Tradingbot's logs/bot.lock",
              "The overlay would refuse to start alongside the trader. "
              "Use logs/hedge.lock.")
    else:
        c.add(PASS, f"lock_path {lock or 'logs/hedge.lock (default)'} is distinct")

    # Bands must not encode a churning configuration.
    try:
        from scripts.measure_exposure import caps_from_config
        caps = caps_from_config(cfg)
        c.add(PASS, "Caps and bands are internally consistent",
              "  ".join(f"{k}={v}" for k, v in caps.factor_caps.items())
              + f"\nhysteresis={caps.hysteresis} target={caps.target_fraction} "
              f"unwind={caps.unwind_band} max_hedge_gross_pct="
              f"{caps.max_hedge_gross_pct}")
    except ValueError as exc:
        c.add(FAIL, "Caps/bands are inconsistent", str(exc))
    except Exception as exc:                                  # noqa: BLE001
        c.add(WARN, "Could not validate caps", repr(exc))

    # ---- do the configured broker symbols actually exist? ----------------
    trader_symbols = dict(cfg.get("trader_symbols") or {})
    hedge_symbols = dict(cfg.get("hedge_instruments") or {})

    if not trader_symbols:
        c.add(WARN, "trader_symbols is empty",
              "The overlay would not know which symbols to avoid. On a NETTING "
              "account that matters a lot.")
    if not hedge_symbols:
        c.add(FAIL, "hedge_instruments is empty",
              "With no candidates the overlay can measure but never hedge.")

    if reader is None:
        c.add(SKIP, "Broker symbol names not verified",
              "Needs a live MT5 connection. Re-run this on the Windows machine.")
        return cfg

    bad: list[str] = []
    good = 0
    for label, mapping in (("trader", trader_symbols), ("hedge", hedge_symbols)):
        for logical, sym in mapping.items():
            try:
                reader.spec(str(sym))
                good += 1
            except Exception as exc:                          # noqa: BLE001
                bad.append(f"{label}/{logical} -> {sym!r}: {exc}")

    if bad:
        c.add(FAIL, f"{len(bad)} configured symbol(s) do not exist at your broker",
              "\n".join(bad)
              + "\n\nFind the real names — brokers use suffixes like .ecn:\n"
              "    python scripts/check_account_mode.py --search XAU\n"
              "    python scripts/check_account_mode.py --search EUR")
        c.suggest("python scripts/check_account_mode.py --search XAU",
                  "discover your broker's real symbol names")
    else:
        c.add(PASS, f"All {good} configured symbols resolve at your broker")

    return cfg


# ---------------------------------------------------------------------------
# 3. Betas
# ---------------------------------------------------------------------------


def check_betas(c: Checks, cfg: dict, reader) -> None:
    hr("3. BETAS  (needed by measure_exposure.py only)")

    beta_cfg = cfg.get("betas") or {}
    path = beta_cfg.get("path") or os.path.join("betas", "betas_latest.json")
    if not os.path.isabs(path):
        path = rel(path)
    max_age = float(beta_cfg.get("max_age_days", 30))

    if not os.path.exists(path):
        c.add(FAIL, f"No beta file at {os.path.relpath(path, ROOT)}",
              "Run the study (needs internet, does NOT need MT5, so it can run "
              "on any machine):\n"
              "    python scripts/estimate_betas.py")
        c.suggest("python scripts/estimate_betas.py", "measure your factor betas")
        return

    from src.factors.store import StaleBetasError, load_betas

    try:
        book, meta = load_betas(path, max_age_days=max_age)
        c.add(PASS, f"Beta file loaded ({os.path.relpath(path, ROOT)})",
              f"created {meta.get('created_at')} — {meta.get('age_days', 0):.1f} "
              f"days old (limit {max_age})")
    except StaleBetasError as exc:
        c.add(FAIL, "Beta file is STALE", 
              f"{exc}\nRe-run:  python scripts/estimate_betas.py")
        c.suggest("python scripts/estimate_betas.py", "refresh stale betas")
        return
    except Exception as exc:                                  # noqa: BLE001
        c.add(FAIL, "Beta file is unreadable", repr(exc))
        return

    c.add(PASS, f"{len(book.instruments())} instruments, "
                f"factors={list(book.factors)}")

    if not book.factor_sigmas():
        c.add(FAIL, "Beta file has no factor sigmas",
              "Purity and risk decomposition need them. Re-run the study.")

    err = float(meta.get("orthogonality_error", 0.0) or 0.0)
    if err < 1e-6:
        c.add(PASS, f"Factors are orthogonal (max off-diagonal |corr| = {err:.1e})")
    else:
        c.add(WARN, f"Orthogonality error is {err:.2e}", "Expected ~1e-16.")

    # ---- coverage: can every open/configured symbol be measured? --------
    logical_names: set[str] = set()
    for mapping in (cfg.get("trader_symbols") or {}, cfg.get("hedge_instruments") or {}):
        logical_names.update(str(k) for k in mapping)

    missing = sorted(n for n in logical_names if not book.has(n))
    if missing:
        known = ", ".join(book.instruments())
        c.add(FAIL,
              f"No betas for configured logical name(s): {', '.join(missing)}",
              "MOST LIKELY CAUSE: the config KEY does not match the logical name "
              "the betas are indexed by. The key is not cosmetic — it is the "
              "lookup key.\n"
              f"  Names that DO have betas: {known}\n"
              "  Common mistakes: US500 should be SP500, USOIL should be WTI, "
              "XAGUSD should be SILVER.\n"
              "  In config, key = logical name, value = your broker symbol:\n"
              "      SP500: US500.ecn\n"
              "A mismatch here silently makes the candidate invisible, and the "
              "overlay then reports 'no usable hedge instrument' for a factor it "
              "could hedge perfectly well.")
    elif logical_names:
        c.add(PASS, f"All {len(logical_names)} configured names have betas")

    # ---- is there a usable hedge lever for every capped factor? ---------
    candidates = [str(k) for k in (cfg.get("hedge_instruments") or {})]
    if candidates:
        limits = cfg.get("limits") or {}
        min_p = float(limits.get("min_purity", 0.50))
        min_b = float(limits.get("min_abs_beta", 0.20))
        for f in (cfg.get("caps") or {}):
            best = book.best_hedge_for(f, candidates, min_purity=min_p,
                                       min_abs_beta=min_b)
            if best:
                c.add(PASS, f"{f}: hedgeable with {best}",
                      f"beta {book.beta(best, f):+.2f}, "
                      f"purity {book.purity(best, f):.2f}")
            else:
                c.add(WARN, f"{f}: NO usable hedge instrument",
                      f"No candidate clears min_purity={min_p} and "
                      f"min_abs_beta={min_b}. The overlay will report breaches "
                      f"on {f} that it can never act on. Either add a suitable "
                      f"instrument, loosen the thresholds, or remove the "
                      f"{f} cap.")

    # ---- open positions with no betas ------------------------------------
    if reader is not None:
        try:
            open_pos = reader.all_positions()
        except Exception:                                     # noqa: BLE001
            open_pos = []
        symbol_to_logical = {}
        for mapping in (cfg.get("trader_symbols") or {},
                        cfg.get("hedge_instruments") or {}):
            for logical, sym in mapping.items():
                symbol_to_logical[str(sym)] = str(logical)

        unmapped = sorted({
            p.symbol for p in open_pos
            if not book.has(symbol_to_logical.get(p.symbol, p.symbol))
        })
        if unmapped:
            c.add(FAIL, f"OPEN positions with no betas: {', '.join(unmapped)}",
                  "Observe mode still reports (charging them a pessimistic "
                  "placeholder vol), but hedging is BLOCKED while these are "
                  "open. Add them to the config and to definitions.py, then "
                  "re-run the study.")
        elif open_pos:
            c.add(PASS, f"All {len(open_pos)} open position(s) have betas")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(c: Checks, reader) -> int:
    hr("SUMMARY")
    print(f"  {len(c.rows)} checks: "
          f"{sum(1 for s, _, _ in c.rows if s == PASS)} ok, "
          f"{c.failed} failed, {c.warned} warnings, "
          f"{sum(1 for s, _, _ in c.rows if s == SKIP)} skipped")

    hr("WHAT TO RUN NEXT")

    have_mt5 = reader is not None
    if not have_mt5:
        print("  You are NOT on a machine with a live MT5 connection.")
        print("  Only step 2 can run here:\n")
        print("    python scripts/estimate_betas.py        # needs internet only")
        print("\n  Steps 1 and 3 must run on the Windows box with the MT5 terminal.")
        print("  Meanwhile you can preview the output format anywhere:\n")
        print("    python scripts/measure_exposure.py --demo")
        return 0 if c.failed == 0 else 1

    if c.failed:
        print("  Fix the FAIL items above first. In order:\n")
        for cmd, why in c.next_steps:
            print(f"    {cmd}\n        -> {why}\n")
        if not c.next_steps:
            print("    (see the FAIL details above)")
        return 1

    print("  Everything needed is in place. Start observing:\n")
    print("    python scripts/measure_exposure.py --loop\n")
    print("  Leave it running for weeks. It writes one JSON line per cycle to")
    print("  logs/exposure_history.jsonl. Do NOT choose caps until you have seen")
    print("  what leverage your book actually runs — a cap above your observed")
    print("  maximum does nothing, and one far below it bleeds spread.")
    return 0


def main() -> int:
    print("Hedgingbot preflight — read-only, opens no orders, writes no files.")
    c = Checks()
    check_environment(c)
    reader = check_mt5(c)
    try:
        cfg = check_config(c, reader)
        check_betas(c, cfg, reader)
        return summarize(c, reader)
    finally:
        if reader is not None:
            reader.shutdown()


if __name__ == "__main__":
    sys.exit(main())
