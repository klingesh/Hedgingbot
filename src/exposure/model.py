"""
Positions -> signed notional -> factor exposure -> risk decomposition.

Pure standard library. Nothing here imports MetaTrader5, pandas or numpy, so the
whole thing runs and is testable anywhere.

The one number that matters
---------------------------
Everything funnels into FACTOR LEVERAGE:

    leverage[f] = factor_exposure[f] / equity

read as: "a 1% adverse move in factor f costs leverage[f] percent of my equity."

So leverage[USD] = 3.2 means a 1% dollar rally costs 3.2% of the account. It is a
cap you can reason about, it is unit-free, and it stays meaningful as the account
grows or shrinks. The overlay's entire job is keeping |leverage[f]| under a
configured ceiling for every f.

Why notional is tick_value/tick_size and not contract_size * price
------------------------------------------------------------------
MT5 gives `trade_tick_value` (account currency per tick, per 1.0 lot) and
`trade_tick_size` (the price increment of one tick). Their ratio

    money_per_price_unit_per_lot = tick_value / tick_size

is the account-currency P&L of a 1.0 price move on a 1-lot position, with the
cross-currency conversion ALREADY APPLIED by the broker. Using it means this
module needs no FX rate table, no knowledge of the quote currency, and it works
unchanged on a USD, EUR or cent (USC) account. It is the same currency-agnostic
reasoning Tradingbot's position_sizing.py uses, so the two repos cannot drift
apart on valuation.

Signed notional is then

    notional = side * volume * (tick_value / tick_size) * price

which is exactly the account-currency P&L of a +100% move in that instrument.
Factor exposure is the beta-weighted sum of those.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstrumentSpec:
    """The broker facts needed to value one instrument.

    Mirrors Tradingbot's SymbolSpec plus a logical name, so an MT5 symbol_spec()
    result maps across with no translation layer.
    """

    symbol: str            # broker symbol, e.g. "XAUUSD.ecn"
    tick_size: float       # price increment of one tick
    tick_value: float      # account currency per tick per 1.0 lot
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 100.0
    logical: str = ""      # e.g. "GOLD" — the key used to look up betas

    def __post_init__(self) -> None:
        for name in ("tick_size", "tick_value", "volume_min", "volume_step", "volume_max"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{self.symbol}: {name} must be finite and > 0, got {v!r}")
        if self.volume_min > self.volume_max:
            raise ValueError(
                f"{self.symbol}: volume_min {self.volume_min} > volume_max {self.volume_max}"
            )

    @property
    def money_per_price_unit_per_lot(self) -> float:
        """Account-currency P&L for a 1.0 price move on a 1-lot position."""
        return self.tick_value / self.tick_size

    @property
    def key(self) -> str:
        """Beta-lookup key: logical name when present, else the raw symbol."""
        return self.logical or self.symbol

    def notional_per_lot(self, price: float) -> float:
        return self.money_per_price_unit_per_lot * price


@dataclass(frozen=True)
class Position:
    """One open broker position, reduced to what risk maths needs.

    `price` is the CURRENT price, not the entry price — exposure is a
    mark-to-market question. `magic` is retained so the overlay can distinguish
    the trader's positions from its own hedges; it must never hedge its own hedge.
    """

    symbol: str
    side: int          # +1 long, -1 short
    volume: float      # lots
    price: float       # current price
    ticket: int = 0
    magic: int = 0
    logical: str = ""

    def __post_init__(self) -> None:
        if self.side not in (1, -1):
            raise ValueError(f"side must be +1 or -1, got {self.side!r}")
        for name in ("volume", "price"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{self.symbol}: {name} must be finite and > 0, got {v!r}")


def signed_notional(pos: Position, spec: InstrumentSpec) -> float:
    """Account-currency P&L this position shows on a +100% price move.

    Signed: negative for shorts. This IS the position's delta.
    """
    return pos.side * pos.volume * spec.money_per_price_unit_per_lot * pos.price


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class PositionExposure:
    """Per-position breakdown, kept for reporting and debugging."""

    symbol: str
    logical: str
    side: int
    volume: float
    notional: float
    standalone_daily_risk: float          # |notional| * instrument daily vol
    factor_contribution: dict[str, float] = field(default_factory=dict)
    mapped: bool = True                   # False => no betas, charged to idio


@dataclass
class ExposureReport:
    """The complete risk picture of a book at one instant."""

    equity: float
    factor_exposure: dict[str, float]     # account ccy per 1.0 (=100%) factor move
    factor_leverage: dict[str, float]     # exposure / equity  <-- the cap variable
    factor_daily_risk: dict[str, float]   # |exposure| * factor daily vol
    idio_daily_risk: float
    portfolio_daily_risk: float           # 1 sigma, 1 day, account currency
    standalone_daily_risk_sum: float
    gross_notional: float
    net_notional: float
    positions: list[PositionExposure] = field(default_factory=list)
    unmapped_symbols: list[str] = field(default_factory=list)
    factors: tuple[str, ...] = ()

    # -- derived views -----------------------------------------------------

    @property
    def diversification_ratio(self) -> float:
        """portfolio risk / sum of standalone risks.

        1.0        = no diversification at all (one bet wearing N costumes)
        1/sqrt(N)  = N perfectly independent bets
        below that = genuinely offsetting positions

        This is the number Tradingbot's PROJECT_REPORT.md:143 flags as
        unquantified when it calls the diversification blend "optimistic".
        """
        if self.standalone_daily_risk_sum <= _EPS:
            return 0.0
        return self.portfolio_daily_risk / self.standalone_daily_risk_sum

    @property
    def effective_bet_count(self) -> float:
        """How many INDEPENDENT bets the book is really making.

        N independent equal-risk bets give a diversification ratio of 1/sqrt(N),
        so N = 1/ratio^2. Eight positions reporting 2.1 effective bets states the
        concentration problem in one number.
        """
        dr = self.diversification_ratio
        return (1.0 / (dr * dr)) if dr > 1e-9 else 0.0

    @property
    def portfolio_daily_risk_pct(self) -> float:
        if self.equity <= _EPS:
            return 0.0
        return 100.0 * self.portfolio_daily_risk / self.equity

    @property
    def gross_leverage(self) -> float:
        if self.equity <= _EPS:
            return 0.0
        return self.gross_notional / self.equity

    def variance_shares(self) -> dict[str, float]:
        """Fraction of total portfolio VARIANCE from each factor, plus 'IDIO'.

        Shares sum to 1.0. The largest share is the concentration metric: USD at
        0.71 means seven-tenths of day-to-day P&L variance is one dollar bet.
        """
        parts = {f: self.factor_daily_risk.get(f, 0.0) ** 2 for f in self.factors}
        parts["IDIO"] = self.idio_daily_risk ** 2
        total = math.fsum(parts.values())
        if total <= 1e-18:
            return {k: 0.0 for k in parts}
        return {k: v / total for k, v in parts.items()}

    def dominant_factor(self) -> tuple[str, float]:
        """(factor, variance share) of the largest contributor, excluding IDIO."""
        shares = self.variance_shares()
        ranked = sorted(
            ((f, s) for f, s in shares.items() if f != "IDIO"),
            key=lambda kv: kv[1], reverse=True,
        )
        return ranked[0] if ranked else ("", 0.0)

    def breaches(self, caps: dict[str, float]) -> dict[str, float]:
        """Factors whose |leverage| exceeds its cap -> the SIGNED excess.

        The sign is kept because the hedge needs to know which way to lean; the
        magnitude is the amount of LEVERAGE to remove, not the total exposure.
        """
        out: dict[str, float] = {}
        for f, lev in self.factor_leverage.items():
            cap = caps.get(f)
            if cap is None:
                continue
            cap = float(cap)
            if not math.isfinite(cap) or cap < 0:
                continue
            if abs(lev) > cap:
                excess = abs(lev) - cap
                out[f] = excess if lev >= 0 else -excess
        return out


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------


def compute_exposure(
    positions: list[Position],
    specs: dict[str, InstrumentSpec],
    betas,                        # factors.book.BetaBook (duck-typed for tests)
    equity: float,
    factor_sigma: dict[str, float] | None = None,
    unmapped_daily_vol: float = 0.015,
) -> ExposureReport:
    """Aggregate open positions into factor exposures and a risk decomposition.

    specs is keyed by BROKER symbol, matching Position.symbol.

    A position whose beta-lookup key is absent from `betas` is NOT dropped.
    Dropping it would understate risk, which is the one failure mode a risk
    system must never have. It is charged in full to idiosyncratic risk and its
    symbol is recorded in report.unmapped_symbols, so the decision layer can
    refuse to hedge a book it cannot fully see.

    unmapped_daily_vol is the pessimistic placeholder vol charged to positions
    with no betas (default 1.5%/day, roughly a volatile commodity CFD). It exists
    so the reported numbers stay honest rather than reading zero risk for an
    unrecognised symbol.

    An assumption worth stating plainly: idiosyncratic risks are treated as
    mutually independent and therefore added in quadrature. If two residuals are
    in fact correlated — gold and silver residuals very likely are — true
    portfolio risk is HIGHER than reported. The factor exposures and leverages
    are unaffected; only the idio term, and hence portfolio_daily_risk and the
    diversification ratio, are optimistic.
    """
    equity = float(equity)
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError(f"equity must be finite and > 0, got {equity!r}")

    factors = tuple(getattr(betas, "factors", ()) or tuple((factor_sigma or {}).keys()))
    if factor_sigma is None:
        getter = getattr(betas, "factor_sigmas", None)
        factor_sigma = getter() if callable(getter) else {}

    def fsig(f: str) -> float:
        v = factor_sigma.get(f, 0.0) if factor_sigma else 0.0
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return 0.0
        return fv if math.isfinite(fv) and fv >= 0 else 0.0

    factor_exposure = {f: 0.0 for f in factors}
    idio_var = 0.0
    gross = 0.0
    net = 0.0
    standalone_sum = 0.0
    out_positions: list[PositionExposure] = []
    unmapped: list[str] = []

    has = getattr(betas, "has", None)

    for pos in positions:
        spec = specs.get(pos.symbol)
        if spec is None:
            raise KeyError(
                f"no InstrumentSpec for open position symbol {pos.symbol!r}; "
                "refusing to compute exposure from an incomplete book"
            )

        key = pos.logical or spec.logical or spec.symbol
        notional = signed_notional(pos, spec)
        gross += abs(notional)
        net += notional

        mapped = bool(has(key)) if callable(has) else True
        if not mapped and pos.symbol not in unmapped:
            unmapped.append(pos.symbol)

        contribution: dict[str, float] = {}
        if mapped:
            for f in factors:
                c = notional * betas.beta(key, f)
                contribution[f] = c
                factor_exposure[f] += c

        if mapped:
            inst_vol = float(betas.sigma(key))
            ivol = float(betas.resid_sigma(key))
        else:
            # We know nothing about this instrument. The tempting answer is 0
            # vol, and it is the WRONG answer: a risk system that reports zero
            # risk for something it does not recognise is worse than one that
            # errors, because it looks fine. So charge a deliberately
            # pessimistic placeholder vol, entirely to idiosyncratic risk (we
            # cannot attribute it, so it must not net against anything).
            # report.unmapped_symbols still blocks hedging outright — this only
            # ensures the OBSERVED numbers do not lie in the meantime.
            inst_vol = ivol = unmapped_daily_vol

        idio_var += (abs(notional) * ivol) ** 2
        standalone = abs(notional) * inst_vol
        standalone_sum += standalone

        out_positions.append(
            PositionExposure(
                symbol=pos.symbol, logical=key, side=pos.side, volume=pos.volume,
                notional=notional, standalone_daily_risk=standalone,
                factor_contribution=contribution, mapped=mapped,
            )
        )

    # Factors are orthogonal by construction, so total factor variance is a plain
    # sum of squares with no covariance cross-terms. That is the payoff from
    # build_orthogonal_factors().
    factor_daily_risk = {f: abs(factor_exposure[f]) * fsig(f) for f in factors}
    factor_var = math.fsum(v * v for v in factor_daily_risk.values())
    portfolio_risk = math.sqrt(factor_var + idio_var)

    return ExposureReport(
        equity=equity,
        factor_exposure=factor_exposure,
        factor_leverage={f: factor_exposure[f] / equity for f in factors},
        factor_daily_risk=factor_daily_risk,
        idio_daily_risk=math.sqrt(idio_var),
        portfolio_daily_risk=portfolio_risk,
        standalone_daily_risk_sum=standalone_sum,
        gross_notional=gross,
        net_notional=net,
        positions=out_positions,
        unmapped_symbols=unmapped,
        factors=factors,
    )
