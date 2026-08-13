"""
Shared, crash-safe state so the trader and the overlay cannot disagree about
whether the portfolio is halted.

Why this exists when MT5 already shows both books
-------------------------------------------------
Position visibility is free: MT5's positions_get() returns every position
regardless of magic number, so the overlay can see the trader's book natively
(Tradingbot's bot_positions() filters by magic; get_open_positions() does not).
What is NOT shared is *decisions*. Two processes both computing "am I in
drawdown?" from the same equity will drift the moment one restarts, because the
high-water mark lives in memory until it is written down. Tradingbot already
learned this the hard way — its own state.py exists because a 30-second restart
loop kept silently re-arming a spent kill switch.

So this file owns exactly one thing: the PORTFOLIO-LEVEL halt, with a
high-water-mark peak equity that only ever rises. Both processes read it, either
may set it, and neither can clear it without a human deleting the file.

Deliberately reuses Tradingbot's proven mechanics:
  * atomic write via mkstemp + os.replace, so a reader never sees a half file
  * peak equity as a high-water mark that survives restarts
  * halt() returns True only the FIRST time, so logs are not spammed 1440x/day
  * every write wrapped by the caller — monitoring must never kill the thing it
    monitors
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

DEFAULT_PATH = os.path.join("logs", "portfolio_state.json")

#: Path to Tradingbot's status heartbeat. Read-only from here — the overlay
#: never writes into the trader's files.
TRADER_STATUS_PATH = os.path.join("logs", "status.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _write_atomic(path: str, payload: str) -> None:
    """Write via a temp file in the SAME directory, then os.replace.

    Same directory matters: os.replace is only atomic within one filesystem.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass
class SharedPortfolioState:
    """Portfolio-level risk state shared by the trader and the overlay."""

    # High-water mark. Rises with equity, NEVER falls on restart.
    peak_equity: float = 0.0
    start_balance: float = 0.0

    # Portfolio kill switch.
    halted: bool = False
    halt_reason: str = ""
    halted_at: str = ""

    # Daily loss halt.
    day: str = ""
    day_start_equity: float = 0.0
    day_halted: bool = False
    day_halt_reason: str = ""

    # Overlay bookkeeping.
    overlay_enabled: bool = True
    overlay_mode: str = "observe"     # "observe" | "live"
    last_hedge_at: str = ""
    hedge_count_today: int = 0

    # Provenance, so a confusing state file can be traced to a writer.
    updated_at: str = ""
    updated_by: str = ""
    notes: list[str] = field(default_factory=list)

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "SharedPortfolioState":
        """Load state, or return a fresh one. A corrupt file is never fatal —
        it is replaced by a fresh state carrying a note about what happened."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return cls()
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            fresh = cls()
            fresh.note(f"state file at {path} was unreadable ({exc!r}); started fresh")
            return fresh

        if not isinstance(raw, dict):
            fresh = cls()
            fresh.note(f"state file at {path} was not a JSON object; started fresh")
            return fresh

        allowed = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in raw.items() if k in allowed}
        try:
            return cls(**clean)
        except TypeError as exc:
            fresh = cls()
            fresh.note(f"state file at {path} had bad field types ({exc!r}); started fresh")
            return fresh

    def save(self, path: str = DEFAULT_PATH, by: str = "") -> None:
        self.updated_at = _utc_now_iso()
        if by:
            self.updated_by = by
        _write_atomic(path, json.dumps(asdict(self), indent=2, sort_keys=True))

    # -- equity tracking ---------------------------------------------------

    def sync_baseline(self, balance: float) -> None:
        """Seed the baseline on first run and let it RATCHET UP only.

        The ratchet is the whole point: if peak_equity were re-seeded from
        current balance on restart, a bot that crashed mid-drawdown would come
        back believing it was at its peak and would happily resume trading with
        the kill switch disarmed.
        """
        if balance <= 0:
            return
        if self.start_balance <= 0:
            self.start_balance = float(balance)
        if balance > self.peak_equity:
            self.peak_equity = float(balance)

    def note_equity(self, equity: float) -> None:
        if equity > self.peak_equity:
            self.peak_equity = float(equity)

    def drawdown_percent(self, equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return 100.0 * (self.peak_equity - equity) / self.peak_equity

    def day_drawdown_percent(self, equity: float) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return 100.0 * (self.day_start_equity - equity) / self.day_start_equity

    # -- halts -------------------------------------------------------------

    def halt(self, reason: str) -> bool:
        """Trip the portfolio kill switch. True only on the FIRST call."""
        if self.halted:
            return False
        self.halted = True
        self.halt_reason = reason
        self.halted_at = _utc_now_iso()
        return True

    def halt_day(self, reason: str) -> bool:
        if self.day_halted:
            return False
        self.day_halted = True
        self.day_halt_reason = reason
        return True

    def roll_day(self, today: str | None = None, equity: float | None = None) -> bool:
        """Start a new UTC day. True if the day actually rolled."""
        today = today or _utc_today()
        if self.day == today:
            return False
        self.day = today
        if equity is not None and equity > 0:
            self.day_start_equity = float(equity)
        self.day_halted = False
        self.day_halt_reason = ""
        self.hedge_count_today = 0
        return True

    def note_hedge(self) -> None:
        self.last_hedge_at = _utc_now_iso()
        self.hedge_count_today += 1

    def note(self, message: str, keep: int = 20) -> None:
        self.notes.append(f"{_utc_now_iso()} {message}")
        if len(self.notes) > keep:
            self.notes = self.notes[-keep:]


def read_trader_status(path: str = TRADER_STATUS_PATH) -> dict:
    """Read Tradingbot's logs/status.json heartbeat, read-only and never fatal.

    Returns {} if absent or unreadable. The overlay uses this only for context
    in its own logs and status output; it must NOT depend on the trader being up
    in order to protect the account. If the trader is dead but positions remain
    open (broker SL/TP still live), the overlay must keep managing exposure.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


def write_overlay_status(
    state: SharedPortfolioState,
    *,
    equity: float,
    balance: float,
    currency: str,
    report,                      # ExposureReport | None
    plan,                        # HedgePlan | None
    caps: dict[str, float] | None = None,
    dry_run: bool = True,
    path: str = os.path.join("logs", "hedge_status.json"),
) -> None:
    """Publish the overlay heartbeat for monitoring.

    Mirrors Tradingbot's write_status(). Wrapped defensively by callers: a
    monitoring write must never be able to stop the bot.
    """
    payload: dict = {
        "updated_at": _utc_now_iso(),
        "dry_run": dry_run,
        "balance": round(float(balance), 2),
        "equity": round(float(equity), 2),
        "currency": currency,
        "peak_equity": round(float(state.peak_equity), 2),
        "drawdown_percent": round(state.drawdown_percent(equity), 3),
        "halted": state.halted,
        "halt_reason": state.halt_reason,
        "day_halted": state.day_halted,
        "overlay_mode": state.overlay_mode,
        "hedge_count_today": state.hedge_count_today,
        "last_hedge_at": state.last_hedge_at,
    }

    if report is not None:
        payload["exposure"] = {
            "factor_leverage": {k: round(v, 4) for k, v in report.factor_leverage.items()},
            "factor_daily_risk": {k: round(v, 2) for k, v in report.factor_daily_risk.items()},
            "variance_shares": {k: round(v, 4) for k, v in report.variance_shares().items()},
            "portfolio_daily_risk": round(report.portfolio_daily_risk, 2),
            "portfolio_daily_risk_pct": round(report.portfolio_daily_risk_pct, 3),
            "diversification_ratio": round(report.diversification_ratio, 4),
            "effective_bet_count": round(report.effective_bet_count, 3),
            "open_positions": len(report.positions),
            "gross_notional": round(report.gross_notional, 2),
            "unmapped_symbols": list(report.unmapped_symbols),
        }
        if caps:
            payload["exposure"]["caps"] = dict(caps)
            payload["exposure"]["breaches"] = {
                k: round(v, 4) for k, v in report.breaches(caps).items()
            }

    if plan is not None:
        payload["plan"] = {
            "blocked": plan.blocked,
            "reason": plan.reason,
            "summary": plan.summary(),
            "actions": [
                {
                    "action": a.action, "symbol": a.symbol, "logical": a.logical,
                    "side": a.side, "lots": a.lots, "factor": a.factor, "reason": a.reason,
                }
                for a in plan.actions
            ],
        }

    _write_atomic(path, json.dumps(payload, indent=2, sort_keys=True))
