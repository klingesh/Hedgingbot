"""
hedge_decide() — the pure decision layer of the overlay.

Deliberately the sibling of Tradingbot's src/live/decision.py: no I/O, no broker,
no clock, no logging, standard library only. Feed it an ExposureReport, get back a
HedgePlan. That is what makes it unit-testable, and it is why the same function
can be driven by the live loop and by a backtester with zero changes.

The six guards that stop this being a spread-donation machine
-------------------------------------------------------------
A naive overlay ("if exposure > cap, hedge to cap") churns: it hedges, the market
wiggles, it unhedges, and it pays the spread both ways forever. Six guards prevent
that, and they ARE the product:

1. HYSTERESIS. Open only once leverage exceeds cap * (1 + hysteresis). The cap is
   not a trigger, it is the edge of a band.

2. TARGET UNDERSHOOT. Aim for cap * target_fraction, not the cap itself. Hedging
   exactly to the boundary guarantees re-breaching on the next tick.

3. UNWIND BAND. Close a hedge only once leverage falls below cap * unwind_band.
   Combined with (1) this creates a wide dead zone where nothing happens.

4. MIN-LOT REFUSAL. If the required size rounds below volume_min, do nothing and
   say why. Never round UP into a hedge larger than the risk it removes — the
   same philosophy as Tradingbot's MinLotPolicy.SKIP.

5. COLLATERAL-DAMAGE CHECK. Every candidate is simulated against ALL factors
   before it is allowed. A hedge that fixes USD by pushing RISK into breach is
   shrunk until it stops doing that, or rejected. Hand-rolled hedging bots never
   have this guard, and it is the difference between reducing risk and moving it.

6. DAILY ACTION BUDGET. A hard ceiling on hedges per day. If the overlay wants to
   trade more than a handful of times a day, the caps are wrong — and the budget
   makes that visible instead of expensive.

7. FUTILITY CHECK. If the largest hedge available cannot remove a meaningful share
   of the excess leverage, do nothing and say the position is too big. Found on a
   real account: a 43x USD breach where the best possible hedge reached 33.7x,
   removing 22% of the excess, costing ~3.4% of equity per month, and leaving the
   book 22x over cap. A token hedge is worse than none, because it bills you
   monthly for the impression that the risk was handled.

One design decision worth defending
-----------------------------------
A drawdown kill switch (`halt_new` in Tradingbot) does NOT block hedging.
Tradingbot's kill switch exists to stop the account taking on NEW RISK. A hedge
removes risk, so blocking it during a drawdown would be exactly backwards — that
is the moment the overlay is most needed. `flatten` is the separate, opposite
instruction: close the hedges as part of taking the whole book flat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..exposure.model import ExposureReport, InstrumentSpec, Position


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HedgeCaps:
    """Risk limits and anti-churn band settings.

    factor_caps is in LEVERAGE units: 1.5 means "a 1% adverse move in this factor
    may cost at most 1.5% of equity".
    """

    factor_caps: dict[str, float]
    hysteresis: float = 0.25            # breach must exceed cap by 25% to act
    target_fraction: float = 0.80       # hedge down to 80% of cap, not to 100%
    unwind_band: float = 0.60           # close hedge below 60% of cap
    # Total hedge notional ceiling, as a percentage of EQUITY. This must be a big
    # number and that is not a mistake: leveraged CFD notional is inherently a
    # large multiple of equity. One lot of EURUSD is ~108,000 of notional, so on a
    # 10,000 account a ceiling of 200% (=20,000) would cap the overlay at 0.18
    # lots and it would spend its life refusing to hedge. Calibrate this against
    # the gross_notional that observe mode reports for your own book.
    max_hedge_gross_pct: float = 1000.0
    min_purity: float = 0.50            # candidate must be this single-factor
    min_abs_beta: float = 0.20          # and at least this factor-sensitive
    max_actions_per_cycle: int = 1      # one change per cycle, then let it settle
    max_hedges_per_day: int = 6         # hard brake on churn
    allow_unmapped: bool = False        # refuse to act on an incomplete book
    # A hedge must remove at least this fraction of the EXCESS leverage to be
    # worth placing. See the futility check in hedge_decide() for why.
    min_excess_removed: float = 0.50

    def __post_init__(self) -> None:
        if not self.factor_caps:
            raise ValueError("factor_caps must not be empty")
        for f, c in self.factor_caps.items():
            c = float(c)
            if not math.isfinite(c) or c <= 0:
                raise ValueError(f"cap for {f} must be finite and > 0, got {c!r}")
        if not 0.0 <= self.hysteresis <= 5.0:
            raise ValueError(f"hysteresis must be in [0, 5], got {self.hysteresis}")
        if not 0.0 < self.target_fraction <= 1.0:
            raise ValueError(f"target_fraction must be in (0, 1], got {self.target_fraction}")
        if not 0.0 <= self.unwind_band <= 1.0:
            raise ValueError(f"unwind_band must be in [0, 1], got {self.unwind_band}")
        if self.unwind_band > self.target_fraction:
            raise ValueError(
                f"unwind_band ({self.unwind_band}) must be <= target_fraction "
                f"({self.target_fraction}), otherwise the overlay closes a hedge it "
                "has just opened — guaranteed churn"
            )
        if self.max_actions_per_cycle < 1:
            raise ValueError("max_actions_per_cycle must be >= 1")
        if not 0.0 <= self.min_excess_removed <= 1.0:
            raise ValueError("min_excess_removed must be in [0, 1]")

    def cap(self, factor: str) -> float:
        v = self.factor_caps.get(factor)
        return float(v) if v is not None else float("inf")

    def has_cap(self, factor: str) -> bool:
        return factor in self.factor_caps


@dataclass(frozen=True)
class HedgeInstrument:
    """A tradable hedge leg with live prices."""

    logical: str
    symbol: str
    spec: InstrumentSpec
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_fraction(self) -> float:
        """Spread as a fraction of price — the cost of using this leg."""
        m = self.mid
        if m <= 0:
            return float("inf")
        return (self.ask - self.bid) / m

    def entry_price(self, side: int) -> float:
        """Buy at the ask, sell at the bid. Always the adverse side."""
        return self.ask if side > 0 else self.bid

    def has_price(self) -> bool:
        return (self.bid > 0 and self.ask > 0 and self.ask >= self.bid
                and math.isfinite(self.bid) and math.isfinite(self.ask))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HedgeAction:
    action: str            # "open" | "close" | "nothing" | "skip"
    symbol: str = ""
    logical: str = ""
    side: int = 0
    lots: float = 0.0
    factor: str = ""
    ticket: int = 0        # set for close
    reason: str = ""

    @property
    def is_trade(self) -> bool:
        return self.action in ("open", "close") and self.lots > 0


@dataclass
class HedgePlan:
    actions: list[HedgeAction] = field(default_factory=list)
    breaches: dict[str, float] = field(default_factory=dict)
    blocked: bool = False
    reason: str = ""

    @property
    def has_trades(self) -> bool:
        return any(a.is_trade for a in self.actions)

    def trades(self) -> list[HedgeAction]:
        return [a for a in self.actions if a.is_trade]

    def summary(self) -> str:
        if self.blocked:
            return f"BLOCKED: {self.reason}"
        if not self.actions:
            return f"no action: {self.reason}"
        parts = []
        for a in self.actions:
            if a.is_trade:
                parts.append(
                    f"{a.action} {a.logical or a.symbol} side={a.side:+d} "
                    f"{a.lots:.2f}lot [{a.factor}] — {a.reason}"
                )
            else:
                parts.append(f"{a.action} — {a.reason}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_down_to_step(volume: float, spec: InstrumentSpec) -> float:
    """Round lots DOWN to the broker's volume step, then clamp to volume_max.

    Always down. Rounding a hedge up would remove more exposure than the breach
    justifies and could flip the book into the opposite breach.
    """
    if volume <= 0:
        return 0.0
    steps = math.floor(volume / spec.volume_step + 1e-9)
    v = steps * spec.volume_step
    v = min(v, spec.volume_max)
    return round(v, 8)   # kill float dust like 0.30000000000000004


def _simulate_leverage(
    report: ExposureReport, betas, added_notional: float, hedge_key: str,
) -> dict[str, float]:
    """Leverage across all factors after adding `added_notional` of `hedge_key`."""
    out: dict[str, float] = {}
    for f in report.factors:
        delta = added_notional * betas.beta(hedge_key, f)
        out[f] = (report.factor_exposure.get(f, 0.0) + delta) / report.equity
    return out


def _new_breaches(
    before: dict[str, float], after: dict[str, float], caps: HedgeCaps, exclude: str,
) -> list[str]:
    """Factors inside their cap before and outside it after."""
    bad: list[str] = []
    for f, lev_after in after.items():
        if f == exclude or not caps.has_cap(f):
            continue
        c = caps.cap(f)
        if abs(before.get(f, 0.0)) <= c and abs(lev_after) > c:
            bad.append(f)
    return bad


def _score_candidate(betas, key: str, factor: str, inst: HedgeInstrument) -> float:
    """Rank hedge candidates. Higher is better.

    Purity dominates because a dirty hedge just creates the next problem; it is
    squared to make it decisive rather than advisory. Larger |beta| helps, since
    less notional is needed and that means less margin and less spread. Spread is
    a mild penalty.
    """
    p = betas.purity(key, factor)
    b = abs(betas.beta(key, factor))
    sp = inst.spread_fraction
    if not math.isfinite(sp):
        return 0.0
    return (p * p) * b / (1.0 + 50.0 * max(sp, 0.0))


def _primary_factor(betas, key: str, factors: tuple[str, ...]) -> str:
    """The factor a given instrument is mainly a lever for."""
    best, best_score = "", -1.0
    for f in factors:
        score = betas.purity(key, f) * abs(betas.beta(key, f))
        if score > best_score:
            best, best_score = f, score
    return best


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def hedge_decide(
    report: ExposureReport,
    betas,
    caps: HedgeCaps,
    candidates: list[HedgeInstrument],
    existing_hedges: list[Position] | None = None,
    hedge_specs: dict[str, InstrumentSpec] | None = None,
    flatten: bool = False,
    avoid_symbols=(),
    hedges_today: int = 0,
) -> HedgePlan:
    """Decide what the overlay should do right now.

    report          : exposure of the WHOLE book — trader positions AND existing
                      hedges. Leverage must already be net of current hedges,
                      or the overlay will double-hedge.
    betas           : BetaBook (duck-typed: .beta, .purity, .has, .instruments).
    caps            : limits and anti-churn bands.
    candidates      : tradable hedge legs with live bid/ask.
    existing_hedges : the overlay's OWN positions, magic-filtered by the caller.
                      Needed to unwind and to enforce the gross hedge ceiling.
    hedge_specs     : specs for existing hedge symbols, for notional maths.
    flatten         : emergency — close every hedge and stop.
    avoid_symbols   : broker symbols the overlay must not trade, normally the
                      symbols the trader already holds. On a NETTING account an
                      opposing order on the same symbol REDUCES the trader's
                      position instead of hedging, silently sabotaging the
                      strategy it is meant to protect.
    hedges_today    : actions already taken today, for the daily budget.

    Never raises for ordinary conditions. An unactionable situation comes back as
    a plan with a reason string, so the reason always reaches the journal.
    """
    existing_hedges = list(existing_hedges or [])
    hedge_specs = dict(hedge_specs or {})
    avoid = frozenset(avoid_symbols)
    plan = HedgePlan()

    # ---- emergency flatten (highest priority, no other gate applies) ------
    if flatten:
        plan.reason = "flatten requested"
        for h in existing_hedges:
            plan.actions.append(
                HedgeAction(
                    action="close", symbol=h.symbol, logical=h.logical or h.symbol,
                    side=h.side, lots=h.volume, ticket=h.ticket, reason="flatten",
                )
            )
        if not plan.actions:
            plan.actions.append(HedgeAction(action="nothing", reason="flatten: no hedges open"))
        return plan

    # ---- refuse to act on an incomplete risk picture ----------------------
    if report.unmapped_symbols and not caps.allow_unmapped:
        plan.blocked = True
        plan.reason = (
            f"no betas for open symbol(s) {', '.join(report.unmapped_symbols)} — "
            "cannot measure total exposure, so refusing to hedge a book it cannot see"
        )
        return plan

    breaches = report.breaches(caps.factor_caps)
    plan.breaches = breaches
    before = dict(report.factor_leverage)

    # ---- 1. unwinding takes priority over opening ------------------------
    # A hedge that is no longer needed is pure drag: it costs spread, margin and
    # swap. Closing it also frees headroom under the gross ceiling. So check this
    # before considering anything new. Unwinding is NOT charged against the daily
    # budget — running out of budget must never trap the account in a stale hedge.
    for h in existing_hedges:
        key = h.logical or h.symbol
        factor = _primary_factor(betas, key, report.factors)
        if not factor or not caps.has_cap(factor):
            continue
        c = caps.cap(factor)
        lev = abs(before.get(factor, 0.0))
        if lev < c * caps.unwind_band:
            plan.actions.append(
                HedgeAction(
                    action="close", symbol=h.symbol, logical=key, side=h.side,
                    lots=h.volume, factor=factor, ticket=h.ticket,
                    reason=(f"{factor} leverage {lev:.2f}x fell below the unwind band "
                            f"{c * caps.unwind_band:.2f}x (cap {c:.2f}x) — hedge no "
                            "longer needed"),
                )
            )
            if len(plan.actions) >= caps.max_actions_per_cycle:
                break

    if plan.actions:
        plan.reason = "unwinding stale hedge(s)"
        return plan

    # ---- 2. anything actually breaching? ---------------------------------
    if not breaches:
        plan.actions.append(
            HedgeAction(action="nothing", reason="all factor leverages within caps")
        )
        plan.reason = "within caps"
        return plan

    actionable = {
        f: excess for f, excess in breaches.items()
        if abs(before.get(f, 0.0)) > caps.cap(f) * (1.0 + caps.hysteresis)
    }
    if not actionable:
        worst = max(breaches, key=lambda f: abs(breaches[f]))
        c = caps.cap(worst)
        plan.actions.append(
            HedgeAction(
                action="nothing", factor=worst,
                reason=(f"{worst} leverage {before.get(worst, 0.0):+.2f}x breaches cap "
                        f"{c:.2f}x but is inside the {caps.hysteresis:.0%} hysteresis "
                        f"band (acts at {c * (1 + caps.hysteresis):.2f}x) — holding to "
                        "avoid churn"),
            )
        )
        plan.reason = "breach inside hysteresis band"
        return plan

    # ---- 3. daily action budget ------------------------------------------
    if hedges_today >= caps.max_hedges_per_day:
        worst = max(actionable, key=lambda f: abs(actionable[f]))
        plan.actions.append(
            HedgeAction(
                action="skip", factor=worst,
                reason=(f"daily hedge budget spent ({hedges_today}/"
                        f"{caps.max_hedges_per_day}); {worst} breach "
                        f"{before.get(worst, 0.0):+.2f}x left unhedged. Repeated hits "
                        "here mean the caps are too tight for this book."),
            )
        )
        plan.reason = "daily hedge budget spent"
        return plan

    # Worst breach first. One factor per cycle.
    factor = max(actionable, key=lambda f: abs(actionable[f]))
    lev = before.get(factor, 0.0)
    c = caps.cap(factor)

    # Exposure to REMOVE, in account currency per 1.0 factor move.
    sign = 1.0 if lev >= 0 else -1.0
    target_lev = sign * c * caps.target_fraction
    exposure_to_remove = (lev - target_lev) * report.equity

    # ---- 4. gross hedge ceiling ------------------------------------------
    hedge_gross = 0.0
    for h in existing_hedges:
        spec = hedge_specs.get(h.symbol)
        if spec is None:
            continue
        hedge_gross += abs(h.volume * spec.money_per_price_unit_per_lot * h.price)
    gross_ceiling = report.equity * caps.max_hedge_gross_pct / 100.0

    if hedge_gross >= gross_ceiling:
        plan.actions.append(
            HedgeAction(
                action="skip", factor=factor,
                reason=(f"hedge gross notional {hedge_gross:,.0f} is at the ceiling "
                        f"{gross_ceiling:,.0f} ({caps.max_hedge_gross_pct:.0f}% of "
                        f"equity) — {factor} breach {lev:+.2f}x left unhedged"),
            )
        )
        plan.reason = "hedge gross ceiling reached"
        return plan

    # ---- 5. choose the hedge instrument ----------------------------------
    usable: list[tuple[float, HedgeInstrument, str]] = []
    rejected: list[str] = []
    for inst in candidates:
        name = inst.logical or inst.symbol
        if inst.symbol in avoid:
            rejected.append(f"{name}(symbol held by trader)")
            continue
        if not betas.has(name):
            rejected.append(f"{name}(no betas)")
            continue
        b = betas.beta(name, factor)
        if abs(b) < caps.min_abs_beta:
            rejected.append(f"{name}(|beta|={abs(b):.2f} < {caps.min_abs_beta})")
            continue
        p = betas.purity(name, factor)
        if p < caps.min_purity:
            rejected.append(f"{name}(purity={p:.2f} < {caps.min_purity})")
            continue
        if not inst.has_price():
            rejected.append(f"{name}(no usable price)")
            continue
        usable.append((_score_candidate(betas, name, factor, inst), inst, name))

    if not usable:
        plan.actions.append(
            HedgeAction(
                action="skip", factor=factor,
                reason=(f"{factor} leverage {lev:+.2f}x breaches cap {c:.2f}x but no "
                        f"usable hedge instrument: "
                        f"{'; '.join(rejected) or 'no candidates supplied'}"),
            )
        )
        plan.reason = "no usable hedge instrument"
        return plan

    usable.sort(key=lambda t: t[0], reverse=True)

    # ---- 6. size it, then verify it does not break something else --------
    last_collateral: list[str] = []
    for _score, inst, key in usable:
        b = betas.beta(key, factor)
        # Adding notional N shifts factor exposure by N * b; we want the shift to
        # equal -exposure_to_remove.
        target_notional = -exposure_to_remove / b
        side = 1 if target_notional > 0 else -1
        per_lot = inst.spec.notional_per_lot(inst.entry_price(side))
        if per_lot <= 0:
            rejected.append(f"{key}(zero notional per lot)")
            continue

        raw_lots = abs(target_notional) / per_lot
        lots = _round_down_to_step(raw_lots, inst.spec)

        if lots < inst.spec.volume_min:
            plan.actions.append(
                HedgeAction(
                    action="skip", symbol=inst.symbol, logical=key, factor=factor,
                    side=side,
                    reason=(f"{factor} breach {lev:+.2f}x needs only {raw_lots:.4f} lots "
                            f"of {key}, below volume_min {inst.spec.volume_min} — "
                            "refusing to round up into an oversized hedge"),
                )
            )
            plan.reason = "required hedge below minimum lot"
            return plan

        # Respect the gross ceiling for the new leg too.
        room = gross_ceiling - hedge_gross
        if lots * per_lot > room:
            capped = _round_down_to_step(room / per_lot, inst.spec)
            if capped < inst.spec.volume_min:
                plan.actions.append(
                    HedgeAction(
                        action="skip", symbol=inst.symbol, logical=key, factor=factor,
                        reason=(f"only {room:,.0f} of hedge notional headroom remains, "
                                f"below one minimum lot of {key} "
                                f"({per_lot * inst.spec.volume_min:,.0f})"),
                    )
                )
                plan.reason = "insufficient hedge headroom"
                return plan
            lots = capped

        # Collateral-damage check: shrink while the hedge creates a NEW breach.
        collateral = _new_breaches(
            before, _simulate_leverage(report, betas, side * lots * per_lot, key),
            caps, exclude=factor,
        )
        shrunk = False
        while collateral and lots >= inst.spec.volume_min:
            smaller = _round_down_to_step(lots - inst.spec.volume_step, inst.spec)
            if smaller < inst.spec.volume_min or smaller >= lots:
                lots = 0.0
                break
            lots = smaller
            shrunk = True
            collateral = _new_breaches(
                before, _simulate_leverage(report, betas, side * lots * per_lot, key),
                caps, exclude=factor,
            )

        if lots < inst.spec.volume_min or collateral:
            last_collateral = collateral
            rejected.append(
                f"{key}(cannot size without breaching {','.join(collateral) or '?'})"
            )
            continue

        after = _simulate_leverage(report, betas, side * lots * per_lot, key)

        # ---- FUTILITY CHECK ------------------------------------------------
        # Found by running against a real account. A 1-lot short gold position on
        # a 10.5k account produced USD leverage of +43.07x against a 1.50x cap.
        # The gross-notional ceiling limited the hedge to 0.90 lots of EURUSD,
        # which moved leverage to +33.72x — removing 22% of the excess while
        # leaving the book 22x over its cap, and costing ~3.4% of equity a month
        # in swap plus the spread.
        #
        # That is not risk management. It is paying a recurring fee for a
        # cosmetic improvement, and worse, it creates the impression the risk has
        # been dealt with. When a breach is this far beyond what a hedge can
        # reach, the honest answer is that the POSITION is too big — and the
        # overlay cannot fix position sizing, so it must say so and stop.
        excess_before = abs(lev) - c
        excess_after = abs(after.get(factor, 0.0)) - c
        if excess_before > 1e-9:
            removed = 1.0 - max(excess_after, 0.0) / excess_before
            if removed < caps.min_excess_removed:
                plan.actions.append(
                    HedgeAction(
                        action="skip", symbol=inst.symbol, logical=key,
                        factor=factor, side=side,
                        reason=(
                            f"{factor} leverage {lev:+.2f}x vs cap {c:.2f}x, but the "
                            f"largest available hedge ({lots:.2f} lots {key}) only "
                            f"reaches {after.get(factor, 0.0):+.2f}x — removing "
                            f"{removed:.0%} of the excess, below the "
                            f"{caps.min_excess_removed:.0%} minimum. Hedging here "
                            f"would pay spread and swap for a cosmetic improvement "
                            f"while leaving the book "
                            f"{abs(after.get(factor, 0.0)) / c:.0f}x over cap. "
                            f"REDUCE THE POSITION instead — an overlay cannot fix "
                            f"position sizing."
                        ),
                    )
                )
                plan.reason = "hedge would be futile; position is too large"
                return plan

        note = " (shrunk to avoid a collateral breach)" if shrunk else ""
        plan.actions.append(
            HedgeAction(
                action="open", symbol=inst.symbol, logical=key, side=side, lots=lots,
                factor=factor,
                reason=(f"{factor} leverage {lev:+.2f}x vs cap {c:.2f}x -> "
                        f"{lots:.2f} lots {key} (beta {b:+.2f}, purity "
                        f"{betas.purity(key, factor):.2f}); projected {factor} "
                        f"{after.get(factor, 0.0):+.2f}x{note}"),
            )
        )
        plan.reason = f"hedging {factor}"
        return plan

    plan.actions.append(
        HedgeAction(
            action="skip", factor=factor,
            reason=(f"{factor} leverage {lev:+.2f}x breaches cap {c:.2f}x but every "
                    f"candidate was unusable: {'; '.join(rejected)}"
                    + (f" (last collateral breach: {','.join(last_collateral)})"
                       if last_collateral else "")),
        )
    )
    plan.reason = "all candidates unusable"
    return plan
