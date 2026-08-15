"""
Data-alignment diagnostics.

Why this module exists
----------------------
The first real beta run on live data produced this pattern:

    instrument   source      R2
    GOLD         GC=F        1.00
    WTI          CL=F        1.00
    SP500        ES=F        0.96
    BRENT        BZ=F        0.88
    SILVER       SI=F        0.62
    PLATINUM     PL=F        0.43
    EURUSD       EURUSD=X    0.28     <-- every `=X` FX series
    AUDUSD       AUDUSD=X    0.16
    USDJPY       USDJPY=X    0.14
    GBPJPY       GBPJPY=X    0.02

Every futures/index series is well explained and every FX series is nearly
unexplained. That split is by DATA SOURCE, not by economics, which makes it a
bug rather than a finding. EURUSD is ~58% of the dollar index by weight, so a
0.28 R^2 against a dollar factor is not credible -- it should be ~0.9.

The likely cause is date labelling. Yahoo timestamps each daily bar at the start
of the session in exchange time. FX (`=X`) rolls at 22:00 UTC (5pm New York)
while futures roll on a different boundary, so bucketing both to a UTC calendar
day can put the SAME trading day under DIFFERENT labels. Aligning on those
labels then correlates Monday FX against Tuesday gold, which destroys the
relationship while leaving same-source pairs untouched.

This module measures that instead of assuming it: it counts how many dates each
series shares with a reference calendar, and how that count changes if the
series' labels are shifted by a day or two. If a non-zero shift dramatically
improves the overlap, the labels are misaligned and the betas built from them
are not trustworthy.

Pure stdlib. No network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

Series = List[Optional[float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def observed_dates(dates: List[str], column: Series) -> List[str]:
    """Dates at which a column actually has a value."""
    return [d for d, v in zip(dates, column) if v is not None]


def shift_labels(day_strings: List[str], days: int) -> set:
    """Shift ISO date labels by `days`, skipping anything unparseable."""
    if days == 0:
        return set(day_strings)
    out = set()
    delta = timedelta(days=days)
    for s in day_strings:
        try:
            out.add((date.fromisoformat(s) + delta).isoformat())
        except ValueError:
            continue
    return out


def overlap_profile(
    series_dates: List[str],
    reference_dates: List[str],
    max_offset: int = 2,
) -> Dict[int, int]:
    """How many dates the series shares with the reference at each label shift.

    A healthy same-calendar series peaks sharply at offset 0. A series whose
    labels are off by a day peaks at -1 or +1.
    """
    ref = set(reference_dates)
    return {
        off: len(shift_labels(series_dates, off) & ref)
        for off in range(-max_offset, max_offset + 1)
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class SeriesAlignment:
    name: str
    observations: int
    counts: Dict[int, int] = field(default_factory=dict)

    @property
    def at_zero(self) -> int:
        return self.counts.get(0, 0)

    @property
    def best_offset(self) -> int:
        if not self.counts:
            return 0
        # Prefer offset 0 on ties: never report a shift we do not need.
        return max(sorted(self.counts, key=lambda o: (abs(o), o)),
                   key=lambda o: self.counts[o])

    @property
    def best_count(self) -> int:
        return self.counts.get(self.best_offset, 0)

    @property
    def gain(self) -> float:
        """Relative improvement from shifting. 0.0 means offset 0 is already best."""
        if self.at_zero <= 0:
            return 1.0 if self.best_count > 0 else 0.0
        return (self.best_count - self.at_zero) / float(self.at_zero)

    @property
    def coverage(self) -> float:
        """Share of the series' own observations that land on a reference date."""
        if self.observations <= 0:
            return 0.0
        return self.at_zero / float(self.observations)

    def verdict(self, min_gain: float, min_coverage: float) -> str:
        if self.best_offset != 0 and self.gain >= min_gain:
            return "MISALIGNED"
        if self.coverage < min_coverage:
            return "LOW COVERAGE"
        return "ok"


@dataclass
class AlignmentReport:
    reference: str
    reference_days: int
    series: List[SeriesAlignment] = field(default_factory=list)
    min_gain: float = 0.10
    min_coverage: float = 0.80

    def misaligned(self) -> List[SeriesAlignment]:
        return [s for s in self.series
                if s.verdict(self.min_gain, self.min_coverage) == "MISALIGNED"]

    def low_coverage(self) -> List[SeriesAlignment]:
        return [s for s in self.series
                if s.verdict(self.min_gain, self.min_coverage) == "LOW COVERAGE"]

    @property
    def healthy(self) -> bool:
        return not self.misaligned() and not self.low_coverage()

    def render(self) -> str:
        offsets = sorted(self.series[0].counts) if self.series else []
        lines = [
            f"   Reference calendar: {self.reference} ({self.reference_days} days)",
            "",
            "   Shared dates at each label shift. A healthy series peaks at 0;",
            "   a peak at -1/+1 means the date labels disagree by a day.",
            "",
            "   " + f"{'series':<18}{'obs':>7}"
            + "".join(f"{('off ' + str(o)):>9}" for o in offsets)
            + f"{'cover':>8}{'verdict':>14}",
        ]
        lines.append("   " + "-" * (len(lines[-1]) - 3))
        for s in sorted(self.series, key=lambda x: x.name):
            row = f"   {s.name:<18}{s.observations:>7}"
            for o in offsets:
                mark = "*" if o == s.best_offset else " "
                row += f"{s.counts.get(o, 0):>8}{mark}"
            row += f"{s.coverage * 100:>7.0f}%"
            row += f"{s.verdict(self.min_gain, self.min_coverage):>14}"
            lines.append(row)

        lines.append("")
        lines.append("   * = best offset for that series")

        bad = self.misaligned()
        if bad:
            lines += [
                "",
                "   *** DATE LABELS ARE MISALIGNED ***",
                "",
                "   These series share far more dates with the reference after a shift,",
                "   which means the SAME trading day is filed under DIFFERENT labels:",
                "",
            ]
            for s in bad:
                lines.append(
                    f"     {s.name:<12} offset {s.best_offset:+d} raises overlap "
                    f"{s.at_zero} -> {s.best_count} ({s.gain * 100:+.0f}%)"
                )
            lines += [
                "",
                "   Any beta computed for these is NOT trustworthy: the regression is",
                "   pairing one day's return against another day's factor. Expect",
                "   spuriously LOW R^2 and betas biased toward zero.",
            ]

        weak = self.low_coverage()
        if weak:
            lines += [
                "",
                "   LOW COVERAGE (aligned, but sparse against the reference calendar):",
            ]
            for s in weak:
                lines.append(
                    f"     {s.name:<12} only {s.coverage * 100:.0f}% of its "
                    f"{s.observations} observations land on a reference date"
                )
            lines.append(
                "   Usually a genuinely different trading calendar (a holiday set, or"
            )
            lines.append(
                "   an instrument that trades weekends). Betas are still usable, but"
            )
            lines.append("   estimated on fewer points than the row count suggests.")

        if self.healthy:
            lines += ["", "   All series agree with the reference calendar."]

        return "\n".join(lines)


def diagnose_alignment(
    dates: List[str],
    frame: Dict[str, Series],
    reference: str,
    max_offset: int = 2,
    min_gain: float = 0.10,
    min_coverage: float = 0.80,
) -> AlignmentReport:
    """Check every column in `frame` against one reference column's calendar.

    reference should be the column the factor model is anchored on — normally a
    factor proxy, since the factors define the sample everything else is
    regressed against.
    """
    if reference not in frame:
        raise KeyError(f"reference column {reference!r} not in frame")

    ref_dates = observed_dates(dates, frame[reference])

    report = AlignmentReport(
        reference=reference, reference_days=len(ref_dates),
        min_gain=min_gain, min_coverage=min_coverage,
    )
    for name, column in frame.items():
        if name == reference:
            continue
        obs = observed_dates(dates, column)
        report.series.append(
            SeriesAlignment(
                name=name,
                observations=len(obs),
                counts=overlap_profile(obs, ref_dates, max_offset),
            )
        )
    return report


# ---------------------------------------------------------------------------
# Circularity: a traded instrument that IS its own factor proxy
# ---------------------------------------------------------------------------


def find_circular_instruments(
    factor_proxy_symbols: Dict[str, str],
    instrument_symbols: Dict[str, str],
) -> Dict[str, str]:
    """Instruments whose price series is also a factor proxy.

    Returns {instrument_name: factor_name}.

    This is a real modelling problem, not a cosmetic one. If GC=F is both the
    METALS proxy and the GOLD instrument, then GOLD regressed on the factors is
    regressed partly on ITSELF: R^2 comes out 1.00 and idiosyncratic vol comes
    out 0.00. The risk model then believes gold has no unexplained risk and is
    perfectly hedgeable, which is false and dangerous -- it is exactly the
    overconfidence a risk system must not have.

    The standard fix is to build each factor from a BASKET so no single traded
    instrument is the factor.
    """
    by_symbol = {sym: fac for fac, sym in factor_proxy_symbols.items()}
    return {
        inst: by_symbol[sym]
        for inst, sym in instrument_symbols.items()
        if sym in by_symbol
    }
