"""
Replay the overlay against a real position timeline and compare equity curves.

This is Phase 3: the question "does the overlay actually reduce drawdown, after
costs?" — which decides whether Phase 2 (order execution) is worth building at all.

How it works
------------
Input is a trade timeline exported from Tradingbot's own backtest
(tools/export_tradingbot_trades.py), so the book being protected is the real one
rather than one invented here.

For each trading day:
  1. Determine which slots are open, and mark them at that day's close.
  2. Compute factor exposure from the measured betas.
  3. Ask hedge_decide() what to do.
  4. Apply it, and carry the hedge forward.
  5. Accrue P&L for BOTH books from the day's price moves, charging the hedge
     spread on open/close and swap every night.

Both curves are computed the SAME way from the same daily marks, so the difference
between them is the overlay's effect and nothing else. That matters more than
either curve being a perfect reproduction of Tradingbot's own reported equity.

Four honest limitations, all of which push the same way
------------------------------------------------------
1. DAILY DECISIONS. The live overlay polls every 60s; this decides once per day.
   Fewer decisions means less churn and less cost, so the replay FLATTERS the
   overlay slightly on cost and understates its responsiveness.

2. INDEPENDENT SLOT SIZING. The export backtests each slot separately, so one
   slot's drawdown does not shrink another's position size. This mirrors
   Tradingbot's live behaviour (each slot sizes off the shared acct.balance) but
   is still an approximation.

3. NO INTRADAY PATH. Marking at the close misses intraday extremes, so measured
   max drawdown is a lower bound for both books.

4. SLIPPAGE NOT MODELLED beyond the quoted spread.

Because (1) and (4) flatter the overlay, a NEGATIVE result here is strong evidence:
if it fails to help even under favourable assumptions, it will not help live.

Pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from ..exposure.model import InstrumentSpec, Position, compute_exposure
from ..overlay.costs import InstrumentCost
from ..overlay.decision import HedgeCaps, HedgeInstrument, hedge_decide


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeRecord:
    """One completed trade from the Tradingbot export."""

    logical: str
    symbol: str
    side: int
    lots: float
    entry_day: str          # ISO date
    exit_day: str           # ISO date
    entry: float
    exit: float

    def is_open_on(self, day: str) -> bool:
        """Open from the entry day up to but NOT including the exit day.

        Exclusive at the exit end so a position is not double-counted on the day
        it closes: its final P&L is realised by that day's mark.
        """
        return self.entry_day <= day < self.exit_day


@dataclass
class ReplayInputs:
    slot_specs: Dict[str, InstrumentSpec]          # logical -> spec
    trades: List[TradeRecord]
    prices: Dict[str, Dict[str, float]]            # logical -> {day: close}
    hedge_specs: Dict[str, InstrumentSpec]         # logical -> spec
    hedge_costs: Dict[str, InstrumentCost]         # logical -> costs
    hedge_symbols: Dict[str, str]                  # logical -> broker symbol
    initial_balance: float = 10_000.0


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass
class DayRow:
    day: str
    base_equity: float
    hedged_equity: float
    n_positions: int
    n_hedges: int
    leverage: Dict[str, float] = field(default_factory=dict)
    base_risk_pct: float = 0.0
    action: str = ""
    hedge_spread_paid: float = 0.0
    hedge_swap_paid: float = 0.0


@dataclass
class ReplayResult:
    rows: List[DayRow] = field(default_factory=list)
    hedges_opened: int = 0
    hedges_closed: int = 0
    total_spread: float = 0.0
    total_swap: float = 0.0
    blocked_days: int = 0
    skipped_days: int = 0
    initial_balance: float = 10_000.0
    warnings: List[str] = field(default_factory=list)

    # -- metrics -----------------------------------------------------------

    @staticmethod
    def _max_drawdown_pct(curve: List[float]) -> float:
        peak = curve[0] if curve else 0.0
        worst = 0.0
        for v in curve:
            peak = max(peak, v)
            if peak > 0:
                worst = max(worst, (peak - v) / peak * 100.0)
        return worst

    @staticmethod
    def _daily_returns(curve: List[float]) -> List[float]:
        out = []
        for a, b in zip(curve, curve[1:]):
            if a > 0:
                out.append(b / a - 1.0)
        return out

    @classmethod
    def _sharpe(cls, curve: List[float]) -> float:
        """Annualized Sharpe from daily returns.

        A zero-volatility curve returns +/-inf rather than 0.0. Returning 0.0 for a
        curve that grows smoothly forever would read in the report as "no
        risk-adjusted return", when the truth is the opposite — undefined because
        there is no measured risk. inf prints as "inf" and is unmistakably
        degenerate, which is what a reader needs to see.
        """
        rets = cls._daily_returns(curve)
        if len(rets) < 2:
            return 0.0
        mean = math.fsum(rets) / len(rets)
        var = math.fsum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sd = math.sqrt(var)
        if sd <= 1e-12:
            if abs(mean) <= 1e-15:
                return 0.0
            return math.inf if mean > 0 else -math.inf
        return (mean / sd) * math.sqrt(252.0)

    def base_curve(self) -> List[float]:
        return [r.base_equity for r in self.rows]

    def hedged_curve(self) -> List[float]:
        return [r.hedged_equity for r in self.rows]

    def summary(self) -> Dict[str, float]:
        base, hedged = self.base_curve(), self.hedged_curve()
        if not base:
            return {}
        b0 = self.initial_balance
        return {
            "days": len(self.rows),
            "base_return_pct": (base[-1] / b0 - 1.0) * 100.0,
            "hedged_return_pct": (hedged[-1] / b0 - 1.0) * 100.0,
            "base_max_dd_pct": self._max_drawdown_pct(base),
            "hedged_max_dd_pct": self._max_drawdown_pct(hedged),
            "base_sharpe": self._sharpe(base),
            "hedged_sharpe": self._sharpe(hedged),
            "hedges_opened": self.hedges_opened,
            "hedges_closed": self.hedges_closed,
            "total_spread": self.total_spread,
            "total_swap": self.total_swap,
            "total_cost": self.total_spread + self.total_swap,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def trading_days(prices: Dict[str, Dict[str, float]]) -> List[str]:
    """Union of every day any instrument has a price for, sorted."""
    days: set = set()
    for series in prices.values():
        days.update(series)
    return sorted(days)


def _price_on(series: Dict[str, float], day: str, last: Optional[float]) -> Optional[float]:
    """Close for a day, falling back to the last known price.

    Carrying the previous price forward over a holiday is right: the position
    still exists and its value did not change on a day with no print. Returning
    None would drop it from the book, which is the bad-tick mistake in another
    form.
    """
    v = series.get(day)
    if v is not None and math.isfinite(v) and v > 0:
        return v
    return last


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------


def replay(
    inputs: ReplayInputs,
    betas,
    caps: HedgeCaps,
    enable_overlay: bool = True,
    hedge_magic: int = 990012,
) -> ReplayResult:
    """Run the overlay over the timeline. Returns both equity curves.

    `enable_overlay=False` produces the base book only, which is used to prove the
    two curves are identical when the overlay does nothing — a check that the
    machinery itself is not introducing a difference.
    """
    result = ReplayResult(initial_balance=inputs.initial_balance)
    days = trading_days(inputs.prices)
    if not days:
        result.warnings.append("no price data; nothing to replay")
        return result

    base_equity = inputs.initial_balance
    hedged_equity = inputs.initial_balance

    last_price: Dict[str, float] = {}
    prev_price: Dict[str, float] = {}
    open_hedges: List[Tuple[Position, str, float]] = []   # (position, logical, entry)
    hedge_provenance: Dict[int, str] = {}
    next_ticket = 900_000
    hedges_today = 0
    current_day_key = ""

    for day in days:
        # ---- mark every instrument ---------------------------------------
        for logical, series in inputs.prices.items():
            p = _price_on(series, day, last_price.get(logical))
            if p is not None:
                last_price[logical] = p

        # ---- which slots are open, and their P&L today -------------------
        book: List[Position] = []
        base_pnl = 0.0
        for tr in inputs.trades:
            if not tr.is_open_on(day):
                continue
            spec = inputs.slot_specs.get(tr.logical)
            price = last_price.get(tr.logical)
            if spec is None or price is None:
                continue
            book.append(
                Position(tr.symbol, tr.side, tr.lots, price, magic=990011,
                         logical=tr.logical)
            )
            prev = prev_price.get(tr.logical)
            if prev is not None:
                base_pnl += (tr.side * tr.lots
                             * spec.money_per_price_unit_per_lot * (price - prev))

        # ---- hedge P&L and carry -----------------------------------------
        hedge_pnl = 0.0
        swap_today = 0.0
        spread_today = 0.0
        weekday = date.fromisoformat(day).weekday()
        for pos, logical, _entry in open_hedges:
            price = last_price.get(logical)
            spec = inputs.hedge_specs.get(logical)
            if price is None or spec is None:
                continue
            prev = prev_price.get(logical)
            if prev is not None:
                hedge_pnl += (pos.side * pos.volume
                              * spec.money_per_price_unit_per_lot * (price - prev))
            cost = inputs.hedge_costs.get(logical)
            if cost is not None:
                swap_today += cost.carry_for_weekday(pos.volume, pos.side, weekday)

        # ---- decide -------------------------------------------------------
        action_txt = ""
        if enable_overlay and book:
            day_key = day[:10]
            if day_key != current_day_key:
                current_day_key = day_key
                hedges_today = 0

            # The overlay sees the HEDGED equity, because that is the account it
            # would actually be running on.
            equity_for_decision = max(hedged_equity, 1.0)

            hedge_positions = [
                Position(p.symbol, p.side, p.volume,
                         last_price.get(lg, p.price), ticket=p.ticket,
                         magic=hedge_magic, logical=lg)
                for p, lg, _e in open_hedges
                if last_price.get(lg) is not None
            ]

            specs_by_symbol: Dict[str, InstrumentSpec] = {}
            for tr in inputs.trades:
                s = inputs.slot_specs.get(tr.logical)
                if s is not None:
                    specs_by_symbol[tr.symbol] = s
            for lg, sym in inputs.hedge_symbols.items():
                s = inputs.hedge_specs.get(lg)
                if s is not None:
                    specs_by_symbol[sym] = s

            try:
                report = compute_exposure(
                    book + hedge_positions, specs_by_symbol, betas,
                    equity_for_decision,
                )
            except (KeyError, ValueError) as exc:
                result.skipped_days += 1
                result.warnings.append(f"{day}: exposure failed: {exc}")
                report = None

            if report is not None:
                candidates: List[HedgeInstrument] = []
                for lg, sym in inputs.hedge_symbols.items():
                    spec = inputs.hedge_specs.get(lg)
                    cost = inputs.hedge_costs.get(lg)
                    price = last_price.get(lg)
                    if spec is None or cost is None or price is None:
                        continue
                    half = price * cost.spread_fraction / 2.0
                    candidates.append(
                        HedgeInstrument(lg, sym, spec, price - half, price + half)
                    )

                held_by_others = {p.symbol for p in book}
                plan = hedge_decide(
                    report, betas, caps, candidates,
                    existing_hedges=hedge_positions,
                    hedge_specs=specs_by_symbol,
                    avoid_symbols=frozenset(held_by_others),
                    hedges_today=hedges_today,
                    hedge_provenance=hedge_provenance,
                )
                if plan.blocked:
                    result.blocked_days += 1

                for a in plan.trades():
                    cost = inputs.hedge_costs.get(a.logical)
                    if a.action == "open":
                        spec = inputs.hedge_specs.get(a.logical)
                        price = last_price.get(a.logical)
                        if spec is None or price is None or cost is None:
                            continue
                        next_ticket += 1
                        open_hedges.append((
                            Position(a.symbol, a.side, a.lots, price,
                                     ticket=next_ticket, magic=hedge_magic,
                                     logical=a.logical),
                            a.logical, price,
                        ))
                        hedge_provenance[next_ticket] = a.factor
                        spread_today += cost.spread_cost(a.lots)
                        result.hedges_opened += 1
                        hedges_today += 1
                        action_txt = f"open {a.lots:.2f} {a.logical} [{a.factor}]"
                    elif a.action == "close":
                        for idx, (pos, lg, _e) in enumerate(open_hedges):
                            if pos.ticket == a.ticket:
                                if cost is not None:
                                    spread_today += cost.spread_cost(pos.volume)
                                open_hedges.pop(idx)
                                hedge_provenance.pop(pos.ticket, None)
                                result.hedges_closed += 1
                                action_txt = f"close {lg} [{a.factor}]"
                                break

        # ---- accrue -------------------------------------------------------
        base_equity += base_pnl
        hedged_equity += base_pnl + hedge_pnl + swap_today - spread_today
        result.total_spread += spread_today
        result.total_swap += swap_today

        lev = {}
        risk_pct = 0.0
        if book:
            try:
                specs_by_symbol = {}
                for tr in inputs.trades:
                    s = inputs.slot_specs.get(tr.logical)
                    if s is not None:
                        specs_by_symbol[tr.symbol] = s
                base_report = compute_exposure(
                    book, specs_by_symbol, betas, max(base_equity, 1.0)
                )
                lev = {k: round(v, 3) for k, v in base_report.factor_leverage.items()}
                risk_pct = base_report.portfolio_daily_risk_pct
            except (KeyError, ValueError):
                pass

        result.rows.append(DayRow(
            day=day, base_equity=base_equity, hedged_equity=hedged_equity,
            n_positions=len(book), n_hedges=len(open_hedges),
            leverage=lev, base_risk_pct=risk_pct, action=action_txt,
            hedge_spread_paid=spread_today, hedge_swap_paid=swap_today,
        ))

        # Today's marks become tomorrow's reference.
        for logical, p in last_price.items():
            prev_price[logical] = p

    return result
