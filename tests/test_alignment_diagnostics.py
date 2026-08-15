"""
Tests for the date-alignment diagnostics.

The central test, `test_reproduces_the_live_fx_symptom`, manufactures the exact
pathology seen on the first real beta run: a series that is genuinely perfectly
correlated with a factor, but whose date labels are shifted by one day, producing
a near-zero R^2. It asserts that

  (a) the naive regression really does collapse to ~0 R^2 (so we know the
      diagnostic is chasing a real effect, not a hypothetical one), and
  (b) the diagnostic detects the shift, and correcting it restores the beta.

Without (a) this suite could pass while the bug remained.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.data.diagnostics import (
    AlignmentReport,
    SeriesAlignment,
    diagnose_alignment,
    find_circular_instruments,
    observed_dates,
    overlap_profile,
    shift_labels,
)
from src.factors.estimate import build_factors, estimate_betas, returns_from_prices
from tests.test_math_core import LCG

FACTOR_ORDER = ("USD", "RISK", "ENERGY", "METALS")


def business_days(start: str, n: int) -> list[str]:
    """n consecutive weekdays from `start`."""
    d = date.fromisoformat(start)
    out: list[str] = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_shift_labels_moves_dates():
    assert shift_labels(["2020-01-02"], 1) == {"2020-01-03"}
    assert shift_labels(["2020-01-02"], -1) == {"2020-01-01"}
    assert shift_labels(["2020-03-01"], 0) == {"2020-03-01"}


def test_shift_labels_handles_month_and_year_boundaries():
    assert shift_labels(["2020-02-28"], 1) == {"2020-02-29"}   # leap year
    assert shift_labels(["2020-12-31"], 1) == {"2021-01-01"}
    assert shift_labels(["2021-01-01"], -1) == {"2020-12-31"}


def test_shift_labels_skips_garbage_without_raising():
    assert shift_labels(["not-a-date", "2020-01-01"], 1) == {"2020-01-02"}


def test_observed_dates_ignores_gaps():
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert observed_dates(dates, [1.0, None, 3.0]) == ["2020-01-01", "2020-01-03"]


def test_overlap_profile_peaks_at_zero_when_aligned():
    ref = business_days("2020-01-01", 50)
    counts = overlap_profile(ref, ref, max_offset=2)
    assert counts[0] == 50
    assert counts[0] > counts[1] and counts[0] > counts[-1]


def test_overlap_profile_peaks_at_the_true_shift():
    ref = business_days("2020-01-01", 60)
    shifted = sorted(shift_labels(ref, -1))
    counts = overlap_profile(shifted, ref, max_offset=2)
    assert max(counts, key=lambda o: counts[o]) == 1, (
        "labels one day early must be corrected by +1"
    )


# ---------------------------------------------------------------------------
# Report behaviour
# ---------------------------------------------------------------------------


def make_frame(n: int = 200):
    dates = business_days("2020-01-01", n)
    rng = LCG(4)
    return dates, {
        "__factor_USD": rng.normals(n, 100.0, 1.0),
        "aligned": rng.normals(n, 50.0, 1.0),
    }


def test_aligned_frame_reports_healthy():
    dates, frame = make_frame()
    report = diagnose_alignment(dates, frame, "__factor_USD")

    assert report.healthy
    assert not report.misaligned()
    assert "agree with the reference calendar" in report.render()


def test_misaligned_column_is_detected_and_named():
    dates, frame = make_frame(200)
    # Shift the column DOWN one row: the same values now sit on later labels.
    frame["fx_like"] = [None] + frame["aligned"][:-1]
    # And blank it on the reference calendar's own dates in a shifted way by
    # rebuilding it purely on shifted labels.
    shifted_dates = sorted(shift_labels(dates, 1))
    combined = sorted(set(dates) | set(shifted_dates))
    idx = {d: i for i, d in enumerate(combined)}
    col_ref = [None] * len(combined)
    col_fx = [None] * len(combined)
    for i, d in enumerate(dates):
        col_ref[idx[d]] = 100.0 + i
    for i, d in enumerate(shifted_dates):
        col_fx[idx[d]] = 50.0 + i

    report = diagnose_alignment(
        combined, {"__factor_USD": col_ref, "fx_like": col_fx}, "__factor_USD"
    )

    bad = report.misaligned()
    assert [s.name for s in bad] == ["fx_like"]
    assert bad[0].best_offset == -1
    text = report.render()
    assert "MISALIGNED" in text
    assert "fx_like" in text


def test_missing_reference_raises():
    dates, frame = make_frame(60)
    with pytest.raises(KeyError, match="reference column"):
        diagnose_alignment(dates, frame, "__factor_NOPE")


def test_best_offset_prefers_zero_on_a_tie():
    s = SeriesAlignment(name="x", observations=10,
                        counts={-1: 5, 0: 5, 1: 5})
    assert s.best_offset == 0
    assert s.gain == 0.0
    assert s.verdict(0.1, 0.8) != "MISALIGNED"


def test_low_coverage_is_reported_separately_from_misalignment():
    """A series on a genuinely different calendar is not 'misaligned' — its
    labels are right, there are just fewer of them."""
    s = SeriesAlignment(name="sparse", observations=100,
                        counts={-1: 10, 0: 50, 1: 10})
    assert s.best_offset == 0
    assert s.coverage == pytest.approx(0.5)
    assert s.verdict(0.1, 0.8) == "LOW COVERAGE"

    report = AlignmentReport(reference="ref", reference_days=100, series=[s])
    assert report.low_coverage() == [s]
    assert not report.misaligned()
    assert "LOW COVERAGE" in report.render()


def test_zero_overlap_series_is_flagged():
    s = SeriesAlignment(name="disjoint", observations=50, counts={-1: 0, 0: 0, 1: 40})
    assert s.best_offset == 1
    assert s.gain == 1.0
    assert s.verdict(0.1, 0.8) == "MISALIGNED"


# ---------------------------------------------------------------------------
# THE test: reproduce the live symptom end to end
# ---------------------------------------------------------------------------


def test_reproduces_the_live_fx_symptom():
    """A perfectly-correlated series with one-day-shifted labels must (a) really
    produce a near-zero R^2, and (b) be caught by the diagnostic."""
    n = 900
    rng = LCG(2024)

    factor_days = business_days("2019-01-01", n)
    factor_rets = {f: rng.normals(n, 0.0, 0.01) for f in FACTOR_ORDER}

    # An instrument that IS 0.9 * the USD factor. Its true R^2 is ~1.
    truth_beta = 0.9
    inst_rets = [truth_beta * r for r in factor_rets["USD"]]

    def to_prices(rets):
        px = [100.0]
        for r in rets:
            px.append(px[-1] * (1.0 + r))
        return px

    # --- case A: correctly aligned -> beta recovered ---------------------
    dates_a = business_days("2019-01-01", n + 1)
    frame_a = {f"__factor_{f}": to_prices(r) for f, r in factor_rets.items()}
    frame_a["FXLIKE"] = to_prices(inst_rets)

    proxies = returns_from_prices(
        {f: frame_a[f"__factor_{f}"] for f in FACTOR_ORDER}, 0.0)
    factors, kept = build_factors(proxies, FACTOR_ORDER, None)
    book_a, _sk = estimate_betas(
        returns_from_prices({"FXLIKE": frame_a["FXLIKE"]}, 0.0),
        factors, kept, None, min_obs=100)

    assert book_a.r_squared("FXLIKE") > 0.99
    assert book_a.beta("FXLIKE", "USD") == pytest.approx(truth_beta, abs=0.01)

    report_a = diagnose_alignment(dates_a, frame_a, "__factor_USD")
    assert report_a.healthy, "correctly aligned data must not be flagged"

    # --- case B: identical data, labels shifted by one day ---------------
    shifted = sorted(shift_labels(dates_a, 1))
    combined = sorted(set(dates_a) | set(shifted))
    idx = {d: i for i, d in enumerate(combined)}

    frame_b: dict[str, list] = {k: [None] * len(combined) for k in frame_a}
    for f in FACTOR_ORDER:
        col = frame_a[f"__factor_{f}"]
        for i, d in enumerate(dates_a):
            frame_b[f"__factor_{f}"][idx[d]] = col[i]
    for i, d in enumerate(shifted):
        frame_b["FXLIKE"][idx[d]] = frame_a["FXLIKE"][i]

    proxies_b = returns_from_prices(
        {f: frame_b[f"__factor_{f}"] for f in FACTOR_ORDER}, 0.0)
    factors_b, kept_b = build_factors(proxies_b, FACTOR_ORDER, None)
    book_b, _sk2 = estimate_betas(
        returns_from_prices({"FXLIKE": frame_b["FXLIKE"]}, 0.0),
        factors_b, kept_b, None, min_obs=100)

    # (a) the damage is real, and it looks like an innocent low-R2 result
    assert book_b.r_squared("FXLIKE") < 0.15, (
        f"expected the shift to destroy R^2, got {book_b.r_squared('FXLIKE'):.2f}"
    )
    assert abs(book_b.beta("FXLIKE", "USD")) < 0.4 * truth_beta, (
        "a shifted series should bias the beta toward zero"
    )

    # (b) the diagnostic catches it and names the correct offset
    report_b = diagnose_alignment(combined, frame_b, "__factor_USD")
    bad = report_b.misaligned()
    assert [s.name for s in bad] == ["FXLIKE"]
    assert bad[0].best_offset == -1, (
        "labels one day LATE must be corrected by -1"
    )


# ---------------------------------------------------------------------------
# Circularity
# ---------------------------------------------------------------------------


def test_detects_instrument_that_is_its_own_factor_proxy():
    """GOLD (GC=F) and the METALS proxy (GC=F) are the same series — the cause of
    the R^2 = 1.00, idio = 0.00 result on live data."""
    circular = find_circular_instruments(
        {"USD": "DX-Y.NYB", "METALS": "GC=F", "ENERGY": "CL=F", "RISK": "^GSPC"},
        {"GOLD": "GC=F", "WTI": "CL=F", "SILVER": "SI=F", "AUDUSD": "AUDUSD=X"},
    )
    assert circular == {"GOLD": "METALS", "WTI": "ENERGY"}


def test_no_circularity_when_proxies_are_distinct():
    assert find_circular_instruments(
        {"METALS": "METALS-BASKET"}, {"GOLD": "GC=F"}
    ) == {}


def test_shipped_definitions_are_checked_for_circularity():
    """Documents the CURRENT state of the shipped config, so a future change to
    basket proxies visibly updates this expectation."""
    from src.factors.definitions import FACTOR_PROXIES, TRADINGBOT_PORTFOLIO

    circular = find_circular_instruments(
        {f: sym for f, (sym, _s, _d) in FACTOR_PROXIES.items()},
        {n: ysym for n, (ysym, _b, _c) in TRADINGBOT_PORTFOLIO.items()},
    )
    # GOLD is GC=F and so is the METALS proxy; BRENT is BZ=F, ENERGY is CL=F.
    assert circular == {"GOLD": "METALS"}, (
        f"shipped circularity changed: {circular}. If proxies moved to baskets "
        "this should now be empty — update the assertion deliberately."
    )
