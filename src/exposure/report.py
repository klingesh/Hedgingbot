"""Human-readable rendering of an ExposureReport.

Kept separate from model.py so the maths stays free of formatting concerns, and
so the same numbers can later feed a JSON status file for monitoring without
duplicating logic.
"""

from __future__ import annotations

from .model import ExposureReport


def _bar(share: float, width: int = 24) -> str:
    filled = int(round(max(0.0, min(1.0, share)) * width))
    return "#" * filled + "." * (width - filled)


def render(report: ExposureReport, caps: dict[str, float] | None = None, currency: str = "") -> str:
    """Full text report. `caps` adds a breach column when supplied."""
    cur = f" {currency}" if currency else ""
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("FACTOR EXPOSURE REPORT")
    add("=" * 78)
    add(f"Equity{cur:>10}: {report.equity:>14,.2f}")
    add(f"Gross notional  : {report.gross_notional:>14,.2f}   "
        f"({report.gross_notional / report.equity:.1f}x equity)")
    add(f"Net notional    : {report.net_notional:>14,.2f}")
    add("")

    # ---- the headline table -------------------------------------------------
    add("FACTOR LEVERAGE  — 'a 1% adverse move in this factor costs N% of equity'")
    add("")
    header = f"  {'factor':<8}{'exposure':>16}{'leverage':>11}{'1d risk':>13}{'var share':>11}"
    if caps:
        header += f"{'cap':>8}{'status':>10}"
    add(header)
    add("  " + "-" * (len(header) - 2))

    shares = report.variance_shares()
    breaches = report.breaches(caps) if caps else {}

    for f in report.factors:
        exp = report.factor_exposure.get(f, 0.0)
        lev = report.factor_leverage.get(f, 0.0)
        risk = report.factor_daily_risk.get(f, 0.0)
        share = shares.get(f, 0.0)
        row = f"  {f:<8}{exp:>16,.0f}{lev:>10.2f}x{risk:>13,.0f}{share * 100:>10.1f}%"
        if caps:
            cap = caps.get(f)
            row += f"{cap:>8.2f}" if cap is not None else f"{'-':>8}"
            row += f"{'BREACH':>10}" if f in breaches else f"{'ok':>10}"
        add(row)

    add(f"  {'IDIO':<8}{'-':>16}{'-':>11}{report.idio_daily_risk:>13,.0f}"
        f"{shares.get('IDIO', 0.0) * 100:>10.1f}%")
    add("")

    # ---- variance concentration -------------------------------------------
    add("VARIANCE CONCENTRATION")
    for k, v in sorted(shares.items(), key=lambda kv: kv[1], reverse=True):
        add(f"  {k:<8} {_bar(v)} {v * 100:>5.1f}%")
    add("")

    # ---- the diversification verdict --------------------------------------
    n = len(report.positions)
    add("DIVERSIFICATION")
    add(f"  Open positions              : {n}")
    add(f"  Portfolio 1-day 1-sigma risk: {report.portfolio_daily_risk:>12,.2f}"
        f"  ({report.portfolio_daily_risk_pct:.2f}% of equity)")
    add(f"  Sum of standalone risks     : {report.standalone_daily_risk_sum:>12,.2f}")
    add(f"  Diversification ratio       : {report.diversification_ratio:>12.3f}")
    add(f"  EFFECTIVE INDEPENDENT BETS  : {report.effective_bet_count:>12.2f}"
        f"   (out of {n} positions)")
    if n >= 2:
        ideal = 1.0 / (n ** 0.5)
        add(f"  (ratio would be {ideal:.3f} if all {n} positions were independent, "
            f"1.000 if they were one bet)")
    add("")

    dom, dom_share = report.dominant_factor()
    if dom:
        add(f"VERDICT: {dom} accounts for {dom_share * 100:.0f}% of portfolio variance.")
        if dom_share > 0.5:
            add("         More than half of day-to-day P&L is a single factor bet.")
    if report.unmapped_symbols:
        add("")
        add("WARNING: no betas available for these open symbols — charged to")
        add("         idiosyncratic risk and NOT hedgeable:")
        for s in report.unmapped_symbols:
            add(f"           {s}")

    if caps and breaches:
        add("")
        add("CAP BREACHES (leverage units to remove):")
        for f, excess in sorted(breaches.items(), key=lambda kv: abs(kv[1]), reverse=True):
            direction = "long" if excess > 0 else "short"
            add(f"  {f:<8} {excess:+.2f}x  (book is too {direction} this factor)")

    add("=" * 78)
    return "\n".join(lines)


def render_positions(report: ExposureReport) -> str:
    """Per-position detail, including each position's factor contributions."""
    lines: list[str] = []
    add = lines.append
    add("PER-POSITION DETAIL")
    header = (f"  {'symbol':<14}{'logical':<10}{'side':>5}{'lots':>8}{'notional':>14}"
              f"{'1d risk':>11}  " + "".join(f"{f:>10}" for f in report.factors))
    add(header)
    add("  " + "-" * (len(header) - 2))
    for p in report.positions:
        side = "long" if p.side > 0 else "short"
        row = (f"  {p.symbol:<14}{p.logical:<10}{side:>5}{p.volume:>8.2f}"
               f"{p.notional:>14,.0f}{p.standalone_daily_risk:>11,.0f}  ")
        row += "".join(f"{p.factor_contribution.get(f, 0.0):>10,.0f}" for f in report.factors)
        if not p.mapped:
            row += "  [UNMAPPED]"
        add(row)
    return "\n".join(lines)
