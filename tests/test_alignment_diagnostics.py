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



# ---------------------------------------------------------------------------
# SUSPECT: the case the first version silently passed
# ---------------------------------------------------------------------------


def test_small_gain_nonzero_offset_is_suspect_not_ok():
    """The exact live numbers that were wrongly reported as "ok".

    Yahoo FX peaked at offset +1 with 1853 vs 1777 shared dates — a 4.3% gain,
    under the 10% MISALIGNED threshold. The first version therefore printed
    "All series agree with the reference calendar", which is the worst possible
    outcome: a check that sees the anomaly and declares everything fine.
    """
    s = SeriesAlignment(
        name="EURUSD", observations=2083,
        counts={-2: 1189, -1: 1364, 0: 1778, 1: 1853, 2: 1458},
    )

    assert s.best_offset == 1
    assert s.gain == pytest.approx((1853 - 1778) / 1778, rel=1e-6)
    assert s.gain < 0.10, "this is the sub-threshold case, by construction"
    assert s.verdict(0.10, 0.80) == "SUSPECT", (
        "a non-zero best offset must never be reported as 'ok'"
    )

    report = AlignmentReport(reference="__factor_USD", reference_days=2012,
                             series=[s])
    assert not report.healthy
    assert report.suspect() == [s]
    assert report.misaligned() == [], "a 4.3% gain is not a whole-day shift"

    text = report.render()
    assert "SUSPECT" in text
    assert "SNAPSHOT TIME" in text, "must point at the real likely cause"
    assert "All series agree" not in text, (
        "the misleading all-clear must not appear alongside a SUSPECT series"
    )


def test_suspect_does_not_escalate_to_misaligned():
    """SUSPECT warns; it must not stop the study, because a label shift is not
    the fix for an intraday timing offset."""
    s = SeriesAlignment(name="fx", observations=100,
                        counts={-1: 40, 0: 80, 1: 84})
    assert s.verdict(0.10, 0.80) == "SUSPECT"
    assert s.verdict(0.01, 0.80) == "MISALIGNED", (
        "a lower threshold should escalate the same data"
    )


# ---------------------------------------------------------------------------
# Synchronicity
# ---------------------------------------------------------------------------


def test_synchronous_data_concentrates_correlation_at_lag_zero():
    from src.data.diagnostics import synchronicity_profile

    rng = LCG(51)
    f = rng.normals(1500, 0.0, 0.01)
    inst = [0.8 * v + rng.normal(0.0, 0.002) for v in f]

    p = synchronicity_profile(inst, f, max_lag=2, name="clean", factor="USD")

    assert p.best_lag == 0
    assert p.correlations[0] > 0.9
    assert abs(p.correlations[1]) < 0.2
    assert abs(p.correlations[-1]) < 0.2
    assert p.leakage < 0.35
    assert p.verdict() == "ok"
    assert p.betas[0] == pytest.approx(0.8, abs=0.03)
    assert p.dimson_beta == pytest.approx(0.8, abs=0.1)


def test_asynchronous_data_smears_correlation_and_attenuates_beta():
    """An instrument that absorbs half of today's factor move and half of
    yesterday's — the signature of a different snapshot time."""
    from src.data.diagnostics import synchronicity_profile

    rng = LCG(52)
    f = rng.normals(3000, 0.0, 0.01)
    true_beta = 1.0
    inst = [true_beta * (0.5 * f[i] + 0.5 * f[i - 1]) if i else true_beta * f[i]
            for i in range(len(f))]

    p = synchronicity_profile(inst, f, max_lag=2, name="fx", factor="USD")

    # The lag-0 beta is roughly HALF the truth: that is the attenuation.
    assert p.betas[0] == pytest.approx(0.5, abs=0.08)
    # Dimson recovers it.
    assert p.dimson_beta == pytest.approx(true_beta, abs=0.12)
    assert p.leakage > 0.30
    assert p.verdict(max_leakage=0.30) != "ok"


def test_synchronicity_rejects_length_mismatch():
    from src.data.diagnostics import synchronicity_profile

    with pytest.raises(ValueError, match="length mismatch"):
        synchronicity_profile([0.1] * 10, [0.1] * 9)


def test_render_synchronicity_flags_and_advises():
    from src.data.diagnostics import render_synchronicity, synchronicity_profile

    rng = LCG(53)
    f = rng.normals(2000, 0.0, 0.01)
    inst = [0.5 * f[i] + 0.5 * f[i - 1] if i else f[i] for i in range(len(f))]

    text = render_synchronicity(
        [synchronicity_profile(inst, f, 2, "fx", "USD")], max_leakage=0.30
    )

    assert "NON-SYNCHRONOUS DATA" in text
    assert "--return-period weekly" in text
    assert "dimson" in text
    assert "UNDER-hedge" in text


def test_render_synchronicity_all_clear():
    from src.data.diagnostics import render_synchronicity, synchronicity_profile

    rng = LCG(54)
    f = rng.normals(1500, 0.0, 0.01)
    inst = [0.9 * v + rng.normal(0.0, 0.001) for v in f]

    text = render_synchronicity([synchronicity_profile(inst, f, 2, "x", "USD")])
    assert "look synchronous" in text
    assert "NON-SYNCHRONOUS" not in text


# ---------------------------------------------------------------------------
# Return-period resampling
# ---------------------------------------------------------------------------


def test_to_period_daily_is_identity():
    from src.data.yahoo import to_period

    dates = business_days("2020-01-01", 10)
    frame = {"a": [float(i) for i in range(10)]}
    kept, out = to_period(dates, frame, "daily")
    assert kept == dates
    assert out["a"] == frame["a"]


def test_to_period_weekly_keeps_the_last_observation_per_iso_week():
    from src.data.yahoo import to_period

    # 2020-01-01 is a Wednesday. Week 1: Wed-Fri. Week 2: Mon-Fri.
    dates = business_days("2020-01-01", 8)
    frame = {"a": [float(i) for i in range(8)]}

    kept, out = to_period(dates, frame, "weekly")

    assert kept == ["2020-01-03", "2020-01-10"]
    assert out["a"] == [2.0, 7.0], "must take the LAST value in each week"


def test_to_period_walks_back_over_a_holiday_at_the_week_end():
    """If the bucket's final day is missing for one series, use the most recent
    real value INSIDE that bucket rather than blanking the whole period."""
    from src.data.yahoo import to_period

    dates = business_days("2020-01-06", 5)          # Mon..Fri, one ISO week
    frame = {"a": [1.0, 2.0, 3.0, 4.0, None]}       # Friday holiday for `a`

    kept, out = to_period(dates, frame, "weekly")

    assert kept == ["2020-01-10"]
    assert out["a"] == [4.0], "should fall back to Thursday, not None"


def test_to_period_blanks_a_bucket_with_no_data_at_all():
    from src.data.yahoo import to_period

    dates = business_days("2020-01-06", 5)
    frame = {"a": [None] * 5}
    kept, out = to_period(dates, frame, "weekly")
    assert out["a"] == [None]


def test_to_period_reduces_sample_size_about_fivefold():
    from src.data.yahoo import to_period

    dates = business_days("2020-01-01", 500)
    frame = {"a": [float(i) for i in range(500)]}
    kept, _out = to_period(dates, frame, "weekly")
    assert 90 < len(kept) < 110, f"500 weekdays should give ~100 weeks, got {len(kept)}"


def test_to_period_monthly_and_biweekly_are_available():
    from src.data.yahoo import to_period

    dates = business_days("2020-01-01", 250)
    frame = {"a": [float(i) for i in range(250)]}

    weekly, _ = to_period(dates, frame, "weekly")
    biweekly, _ = to_period(dates, frame, "biweekly")
    monthly, _ = to_period(dates, frame, "monthly")

    assert len(weekly) > len(biweekly) > len(monthly)


def test_to_period_rejects_unknown_period():
    from src.data.yahoo import to_period

    with pytest.raises(ValueError, match="unknown period"):
        to_period(["2020-01-01"], {"a": [1.0]}, "fortnightly")


def test_weekly_resampling_repairs_an_asynchronous_beta():
    """The end-to-end justification for --return-period weekly.

    An instrument split half across today and yesterday has its DAILY beta
    attenuated to about half the truth. On weekly returns, a one-day smear is a
    small fraction of the period, so the beta should come much closer to the
    real value.
    """
    from src.data.yahoo import to_period
    from src.factors.estimate import build_factors, estimate_betas, returns_from_prices

    n = 2600
    rng = LCG(2025)
    true_beta = 1.0

    factor_rets = {f: rng.normals(n, 0.0, 0.01) for f in FACTOR_ORDER}
    usd = factor_rets["USD"]
    inst_rets = [true_beta * (0.5 * usd[i] + 0.5 * usd[i - 1]) if i
                 else true_beta * usd[i] for i in range(n)]

    def to_prices(rets):
        px = [100.0]
        for r in rets:
            px.append(px[-1] * (1.0 + r))
        return px

    dates = business_days("2016-01-04", n + 1)
    frame = {f"__factor_{f}": to_prices(r) for f, r in factor_rets.items()}
    frame["FXLIKE"] = to_prices(inst_rets)

    def beta_at(period: str) -> float:
        d, fr = to_period(dates, frame, period)
        proxies = returns_from_prices(
            {f: fr[f"__factor_{f}"] for f in FACTOR_ORDER}, 0.0)
        factors, kept = build_factors(proxies, FACTOR_ORDER, None)
        book, _sk = estimate_betas(
            returns_from_prices({"FXLIKE": fr["FXLIKE"]}, 0.0),
            factors, kept, None, min_obs=50)
        return book.beta("FXLIKE", "USD")

    daily = beta_at("daily")
    weekly = beta_at("weekly")

    assert daily == pytest.approx(0.5, abs=0.12), (
        f"daily beta should be attenuated to ~half, got {daily:.2f}"
    )
    assert weekly > daily, "weekly must recover some of the attenuation"
    assert weekly == pytest.approx(true_beta, abs=0.25), (
        f"weekly beta should approach the true {true_beta}, got {weekly:.2f}"
    )
