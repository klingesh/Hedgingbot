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
        """MISALIGNED > SUSPECT > LOW COVERAGE > ok.

        SUSPECT exists because of a real failure on live data: the Yahoo FX
        series peaked at offset +1 with only a 4.3% overlap gain, which fell
        under a 10% threshold, so the report concluded "all series agree". A
        check that observes the anomaly and then declares everything fine is
        worse than no check at all. ANY non-zero best offset is now surfaced;
        only the severity is graded.
        """
        if self.best_offset != 0:
            return "MISALIGNED" if self.gain >= min_gain else "SUSPECT"
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

    def _with_verdict(self, want: str) -> List[SeriesAlignment]:
        return [s for s in self.series
                if s.verdict(self.min_gain, self.min_coverage) == want]

    def misaligned(self) -> List[SeriesAlignment]:
        return self._with_verdict("MISALIGNED")

    def suspect(self) -> List[SeriesAlignment]:
        """Best offset is non-zero but the overlap gain is small.

        Usually means the series is NOT simply shifted a whole day, but is
        snapshotted at a different time WITHIN the day — which no label shift can
        fix. Check synchronicity_profile() next.
        """
        return self._with_verdict("SUSPECT")

    def low_coverage(self) -> List[SeriesAlignment]:
        return self._with_verdict("LOW COVERAGE")

    @property
    def healthy(self) -> bool:
        return not (self.misaligned() or self.suspect() or self.low_coverage())

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

        maybe = self.suspect()
        if maybe:
            lines += [
                "",
                "   SUSPECT — best offset is non-zero but the gain is small:",
                "",
            ]
            for s in maybe:
                lines.append(
                    f"     {s.name:<12} offset {s.best_offset:+d} raises overlap "
                    f"{s.at_zero} -> {s.best_count} ({s.gain * 100:+.1f}%)"
                )
            lines += [
                "",
                "   A small gain means these are probably NOT shifted a whole day.",
                "   The more likely cause is a different SNAPSHOT TIME within the day",
                "   (e.g. a futures settlement print vs an FX spot snapshot hours",
                "   later). No label shift can fix that, and it ATTENUATES betas",
                "   toward zero while leaving row counts and date coverage healthy.",
                "",
                "   Check the synchronicity report below before trusting these betas.",
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



# ---------------------------------------------------------------------------
# Synchronicity: are two series snapshotted at the same MOMENT?
# ---------------------------------------------------------------------------
#
# Date alignment asks "are these filed under the same day?". Synchronicity asks
# the harder question: "were they PRICED at the same moment?"
#
# The live run showed EURUSD with a 0.28 R^2 against the dollar factor. EURUSD is
# ~58% of DXY by weight, so its true daily correlation with the dollar is around
# -0.95, implying R^2 ~ 0.9. And the date-alignment check found only a 4.3%
# overlap gain from shifting, far too small for a whole-day shift to explain a
# collapse that severe.
#
# The remaining explanation is intraday: `DX-Y.NYB` is an index print on a US
# futures settlement clock, while `EURUSD=X` is a spot snapshot taken hours later.
# Both land on the same calendar date, so every date check passes, but they
# measure different windows of market time. Correlation is then split ACROSS
# adjacent days instead of concentrated on the same day.
#
# That is exactly what this measures. For non-synchronous series, |corr| at lag
# +/-1 is non-trivial and the sum across lags greatly exceeds the lag-0 value.
# The standard remedies are (a) longer return periods, where a few hours of
# offset stops mattering, and (b) summing the lagged betas -- the Dimson
# aggregated-coefficient correction.


@dataclass
class Synchronicity:
    name: str
    factor: str
    correlations: Dict[int, float] = field(default_factory=dict)
    betas: Dict[int, float] = field(default_factory=dict)
    n_obs: int = 0

    @property
    def contemporaneous(self) -> float:
        return self.correlations.get(0, 0.0)

    @property
    def best_lag(self) -> int:
        if not self.correlations:
            return 0
        return max(sorted(self.correlations, key=lambda k: (abs(k), k)),
                   key=lambda k: abs(self.correlations[k]))

    @property
    def leakage(self) -> float:
        """Share of total absolute correlation sitting at NON-zero lags.

        ~0 for synchronous data. Large means the relationship is smeared across
        adjacent days, i.e. the two series are priced at different moments.
        """
        total = sum(abs(v) for v in self.correlations.values())
        if total <= 1e-12:
            return 0.0
        return (total - abs(self.contemporaneous)) / total

    @property
    def dimson_beta(self) -> float:
        """Sum of betas across all lags — the Dimson aggregated coefficient.

        For non-synchronous data this recovers the economic sensitivity that the
        lag-0 beta alone understates.
        """
        return sum(self.betas.values())

    def verdict(self, max_leakage: float = 0.45) -> str:
        if self.best_lag != 0 and abs(self.correlations.get(self.best_lag, 0.0)) > \
                abs(self.contemporaneous) * 1.15:
            return "ASYNCHRONOUS"
        if self.leakage > max_leakage:
            return "SMEARED"
        return "ok"


def synchronicity_profile(
    instrument_returns: List[float],
    factor_returns: List[float],
    max_lag: int = 2,
    name: str = "",
    factor: str = "",
) -> Synchronicity:
    """Correlation and beta of an instrument against a factor at several lags.

    Both lists must be aligned and the same length, with no None values.

    A POSITIVE lag means the FACTOR is shifted forward in time, i.e. lag +1
    correlates the instrument today against the factor yesterday.
    """
    from ..factors.math_core import normalized_weights, wcorr, wcov, wvar

    if len(instrument_returns) != len(factor_returns):
        raise ValueError(
            f"length mismatch: instrument {len(instrument_returns)}, "
            f"factor {len(factor_returns)}"
        )

    out = Synchronicity(name=name, factor=factor, n_obs=len(instrument_returns))

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            y = instrument_returns[lag:]
            x = factor_returns[:-lag] if lag else factor_returns
        elif lag < 0:
            y = instrument_returns[:lag]
            x = factor_returns[-lag:]
        else:
            y = list(instrument_returns)
            x = list(factor_returns)

        if len(y) < 30 or len(y) != len(x):
            out.correlations[lag] = 0.0
            out.betas[lag] = 0.0
            continue

        w = normalized_weights(len(y), None)
        out.correlations[lag] = wcorr(y, x, w)
        vx = wvar(x, w)
        out.betas[lag] = (wcov(y, x, w) / vx) if vx > 1e-18 else 0.0

    return out


def render_synchronicity(
    profiles: List[Synchronicity], max_leakage: float = 0.45
) -> str:
    """Table of lagged correlations, one row per instrument."""
    if not profiles:
        return "   (nothing to check)"

    lags = sorted(profiles[0].correlations)
    lines = [
        f"   Correlation with the {profiles[0].factor} factor at several lags.",
        "   Synchronous data concentrates correlation at lag 0. Correlation",
        "   leaking to +/-1 means the two series are priced at different moments,",
        "   which ATTENUATES the lag-0 beta toward zero.",
        "",
        "   " + f"{'instrument':<14}"
        + "".join(f"{('lag ' + str(l)):>9}" for l in lags)
        + f"{'leak':>7}{'beta0':>8}{'dimson':>8}{'verdict':>15}",
    ]
    lines.append("   " + "-" * (len(lines[-1]) - 3))

    for p in sorted(profiles, key=lambda q: q.name):
        row = f"   {p.name:<14}"
        for l in lags:
            mark = "*" if l == p.best_lag else " "
            row += f"{p.correlations.get(l, 0.0):>8.2f}{mark}"
        row += f"{p.leakage * 100:>6.0f}%"
        row += f"{p.betas.get(0, 0.0):>8.2f}"
        row += f"{p.dimson_beta:>8.2f}"
        row += f"{p.verdict(max_leakage):>15}"
        lines.append(row)

    lines += ["", "   * = lag with the strongest |correlation|",
              "   beta0  = beta at lag 0 (what the model currently uses)",
              "   dimson = sum of betas across all lags (the Dimson correction,",
              "            which is the economically meaningful sensitivity when",
              "            the series are not synchronous)"]

    bad = [p for p in profiles if p.verdict(max_leakage) != "ok"]
    if bad:
        lines += [
            "",
            "   *** NON-SYNCHRONOUS DATA ***",
            "",
        ]
        for p in bad:
            lines.append(
                f"     {p.name:<12} {p.verdict(max_leakage):<13} "
                f"beta0 {p.betas.get(0, 0.0):+.2f} vs dimson "
                f"{p.dimson_beta:+.2f}  (leak {p.leakage * 100:.0f}%)"
            )
        lines += [
            "",
            "   These betas are ATTENUATED: the true sensitivity is closer to the",
            "   dimson column. Sizing a hedge from beta0 would UNDER-hedge.",
            "",
            "   Fix by lengthening the return period so a few hours of offset stops",
            "   mattering:",
            "       python scripts/estimate_betas.py --return-period weekly",
            "",
            "   A weekly period trades sample size for accuracy (about 1/5 the",
            "   observations), which is the right trade when the daily numbers are",
            "   biased rather than merely noisy.",
        ]
    else:
        lines += ["", "   All series look synchronous with the factor."]

    return "\n".join(lines)
