"""
End-to-end tests: synthetic PRICE series -> BetaBook -> JSON -> BetaBook.

test_end_to_end_from_prices is the integration test that matters. It starts from
prices (what the data loader actually returns), goes through returns,
winsorization, alignment, orthogonalization and regression, and checks the betas
that come out the far end are the ones that went in. If it passes, the whole
estimation pipeline is wired up correctly, not just the individual pieces.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from src.factors.book import BetaBook
from src.factors.definitions import FACTORS, ORTHOGONALIZATION_ORDER, priors_for
from src.factors.estimate import (
    build_factors,
    estimate_betas,
    estimate_from_prices,
    returns_from_prices,
)
from src.factors.store import StaleBetasError, load_betas, save_betas
from tests.test_math_core import LCG

ORDER = ORTHOGONALIZATION_ORDER


def prices_from_returns(rets: list[float], start: float = 100.0) -> list[float]:
    px = [start]
    for r in rets:
        px.append(px[-1] * math.exp(r))
    return px


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_returns_from_prices_handles_gaps():
    prices = {"A": [100.0, 101.0, None, 103.0], "B": [10.0, 10.5, 11.0, 11.5]}
    rets = returns_from_prices(prices, winsorize_fraction=0.0)

    assert rets["A"][0] is None
    assert rets["A"][1] == pytest.approx(math.log(101 / 100))
    assert rets["A"][2] is None and rets["A"][3] is None
    assert all(v is not None for v in rets["B"][1:])


def test_end_to_end_from_prices():
    """Synthetic prices built from known betas must yield those betas back."""
    n = 9000
    rng = LCG(2024)
    vols = {"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022, "METALS": 0.009}
    factor_rets = {f: rng.normals(n, 0.0, v) for f, v in vols.items()}

    truth = {
        "GOLD":   {"USD": -0.90, "RISK": 0.00, "ENERGY": 0.10, "METALS": 1.00},
        "EURUSD": {"USD": -1.00, "RISK": 0.05, "ENERGY": 0.00, "METALS": 0.05},
    }

    factor_prices = {f: prices_from_returns(r) for f, r in factor_rets.items()}
    instrument_prices = {}
    for inst, bv in truth.items():
        series = [0.0] * n
        for f, b in bv.items():
            fr = factor_rets[f]
            series = [s + b * fr[i] for i, s in enumerate(series)]
        series = [s + rng.normal(0.0, 0.002) for s in series]
        instrument_prices[inst] = prices_from_returns(series)

    book, factors, skipped = estimate_from_prices(
        factor_prices, instrument_prices, ORDER,
        halflife=None, winsorize_fraction=0.0, min_obs=500,
    )

    assert skipped == {}
    assert factors.orthogonality_error() < 1e-9
    assert set(book.instruments()) == {"GOLD", "EURUSD"}

    for inst, bv in truth.items():
        for f, expected in bv.items():
            assert book.beta(inst, f) == pytest.approx(expected, abs=0.06), (
                f"{inst}/{f}: expected {expected:+.2f}, got {book.beta(inst, f):+.2f}"
            )

    # Sanity: EURUSD must be an almost-pure USD lever.
    assert book.purity("EURUSD", "USD") > 0.55
    assert book.r_squared("GOLD") > 0.8


def test_short_history_instrument_is_skipped_with_a_count():
    n = 2000
    rng = LCG(5)
    factor_prices = {f: prices_from_returns(rng.normals(n, 0.0, 0.01)) for f in FACTORS}

    good = prices_from_returns(rng.normals(n, 0.0, 0.01))
    short = [None] * (n - 50) + prices_from_returns(rng.normals(50, 0.0, 0.01))

    book, _factors, skipped = estimate_from_prices(
        factor_prices, {"GOOD": good, "SHORT": short}, ORDER, min_obs=500,
    )

    assert "GOOD" in book.instruments()
    assert "SHORT" not in book.instruments()
    assert "SHORT" in skipped
    assert skipped["SHORT"] < 500


def test_instrument_with_partial_history_uses_its_own_overlap():
    """A shorter instrument must not truncate the sample for the others."""
    n = 3000
    rng = LCG(8)
    factor_rets = {f: rng.normals(n, 0.0, 0.01) for f in FACTORS}
    factor_prices = {f: prices_from_returns(r) for f, r in factor_rets.items()}

    full = prices_from_returns([0.5 * r for r in factor_rets["USD"]])
    half_start = n // 2
    partial_rets = [0.5 * r for r in factor_rets["USD"]]
    partial = [None] * half_start + prices_from_returns(partial_rets[half_start:])

    book, _f, skipped = estimate_from_prices(
        factor_prices, {"FULL": full, "PARTIAL": partial}, ORDER,
        winsorize_fraction=0.0, min_obs=500,
    )

    assert skipped == {}
    assert book.n_obs("FULL") > book.n_obs("PARTIAL")
    # Both should still measure the same underlying beta.
    assert book.beta("FULL", "USD") == pytest.approx(0.5, abs=0.02)
    assert book.beta("PARTIAL", "USD") == pytest.approx(0.5, abs=0.05)


def test_all_instruments_too_short_raises_with_detail():
    n = 600
    rng = LCG(4)
    factor_prices = {f: prices_from_returns(rng.normals(n, 0.0, 0.01)) for f in FACTORS}
    tiny = [None] * (n - 5) + prices_from_returns(rng.normals(5, 0.0, 0.01))

    with pytest.raises(ValueError, match="no instrument had"):
        estimate_from_prices(factor_prices, {"TINY": tiny}, ORDER, min_obs=500)


def test_missing_factor_price_column_raises():
    n = 600
    rng = LCG(6)
    factor_prices = {f: prices_from_returns(rng.normals(n, 0.0, 0.01))
                     for f in FACTORS if f != "METALS"}
    inst = {"X": prices_from_returns(rng.normals(n, 0.0, 0.01))}

    with pytest.raises(ValueError, match="missing factor"):
        estimate_from_prices(factor_prices, inst, ORDER, min_obs=100)


def test_build_factors_returns_row_indices_that_line_up():
    rets = {
        "USD": [None] + [0.01] * 60,
        "RISK": [None, None] + [0.01] * 59,
        "ENERGY": [None] + [0.01] * 60,
        "METALS": [None] + [0.01] * 60,
    }
    # Perfectly collinear input, but we only care about index bookkeeping here.
    factors, kept = build_factors(rets, ORDER)
    assert kept[0] == 2, "first usable row is index 2 (RISK missing at 0 and 1)"
    assert len(kept) == factors.n_obs


# ---------------------------------------------------------------------------
# BetaBook
# ---------------------------------------------------------------------------


def make_book() -> BetaBook:
    return BetaBook(
        factors=FACTORS,
        betas={
            "GOLD": {"USD": -0.9, "RISK": 0.0, "ENERGY": 0.1, "METALS": 1.0},
            "EURUSD": {"USD": -1.0, "RISK": 0.02, "ENERGY": 0.0, "METALS": 0.01},
            "SP500": {"USD": -0.1, "RISK": 1.0, "ENERGY": 0.1, "METALS": 0.0},
        },
        r_squared={"GOLD": 0.8, "EURUSD": 0.9, "SP500": 0.85},
        sigma={"GOLD": 0.011, "EURUSD": 0.005, "SP500": 0.012},
        resid_sigma={"GOLD": 0.004, "EURUSD": 0.001, "SP500": 0.004},
        n_obs={"GOLD": 2000, "EURUSD": 2000, "SP500": 2000},
        factor_sigma={"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022, "METALS": 0.009},
    )


def test_unknown_instrument_returns_zero_but_reports_absence():
    """The distinction that keeps the risk model honest."""
    book = make_book()
    assert book.beta("NOPE", "USD") == 0.0
    assert not book.has("NOPE")
    assert "NOPE" not in book
    assert book.has("GOLD") and "GOLD" in book


def test_best_hedge_for_picks_the_pure_lever():
    book = make_book()
    assert book.best_hedge_for("USD", ["EURUSD", "SP500", "GOLD"]) == "EURUSD"
    assert book.best_hedge_for("RISK", ["EURUSD", "SP500", "GOLD"]) == "SP500"


def test_best_hedge_for_respects_thresholds_and_can_return_none():
    book = make_book()
    assert book.best_hedge_for("USD", ["EURUSD"], min_purity=0.999) is None
    assert book.best_hedge_for("ENERGY", ["EURUSD"], min_abs_beta=0.5) is None
    assert book.best_hedge_for("USD", ["UNKNOWN"]) is None


def test_sign_disagreement_catches_an_inverted_series():
    """The check that catches a wrong ticker before it sizes a trade."""
    book = BetaBook(
        factors=FACTORS,
        betas={"GOLD": {"USD": +0.9, "RISK": 0.0, "ENERGY": 0.1, "METALS": 1.0}},
        factor_sigma={f: 0.01 for f in FACTORS},
    )
    flags = book.sign_disagreements(priors_for(["GOLD"]))

    assert any(f[0] == "GOLD" and f[1] == "USD" for f in flags), (
        "gold measuring POSITIVE dollar beta must be flagged against the prior"
    )


def test_sign_disagreement_ignores_immaterial_betas():
    book = BetaBook(
        factors=FACTORS,
        betas={"GOLD": {"USD": -0.9, "RISK": 0.01, "ENERGY": -0.01, "METALS": 1.0}},
        factor_sigma={f: 0.01 for f in FACTORS},
    )
    # ENERGY prior is +0.1, measured -0.01: both below the 0.15 materiality floor.
    assert book.sign_disagreements(priors_for(["GOLD"])) == []


def test_summary_renders_without_error():
    book = make_book()
    text = book.summary(priors=priors_for(book.instruments()))
    assert "GOLD" in text and "instrument" in text


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    book = make_book()
    path = os.path.join(str(tmp_path), "betas.json")
    save_betas(book, path, meta={"years": 8})

    loaded, meta = load_betas(path)

    assert loaded.instruments() == book.instruments()
    assert loaded.factors == book.factors
    for inst in book.instruments():
        for f in book.factors:
            assert loaded.beta(inst, f) == pytest.approx(book.beta(inst, f))
        assert loaded.resid_sigma(inst) == pytest.approx(book.resid_sigma(inst))
        assert loaded.n_obs(inst) == book.n_obs(inst)
    assert loaded.factor_sigmas() == book.factor_sigmas()
    assert meta["years"] == 8
    assert meta["age_days"] < 1.0


def test_factor_sigmas_travel_with_the_betas(tmp_path):
    """Purity depends on factor sigmas, so they must survive the round trip or
    every hedge choice silently changes."""
    book = make_book()
    path = os.path.join(str(tmp_path), "betas.json")
    save_betas(book, path)
    loaded, _ = load_betas(path)

    assert loaded.purity("EURUSD", "USD") == pytest.approx(book.purity("EURUSD", "USD"))


def test_stale_betas_are_refused_by_default(tmp_path):
    path = os.path.join(str(tmp_path), "betas.json")
    save_betas(make_book(), path)

    # Backdate the file.
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    raw["created_at"] = "2020-01-01T00:00:00+00:00"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh)

    with pytest.raises(StaleBetasError, match="days old"):
        load_betas(path, max_age_days=30)

    book, meta = load_betas(path, max_age_days=30, allow_stale=True)
    assert book.instruments()
    assert meta["age_days"] > 1000


def test_missing_created_at_counts_as_infinitely_stale(tmp_path):
    path = os.path.join(str(tmp_path), "betas.json")
    save_betas(make_book(), path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    del raw["created_at"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh)

    with pytest.raises(StaleBetasError):
        load_betas(path, max_age_days=30)


def test_malformed_beta_file_raises_clearly(tmp_path):
    path = os.path.join(str(tmp_path), "bad.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"hello": "world"}, fh)

    with pytest.raises(ValueError, match="not a beta file"):
        load_betas(path)


def test_beta_file_without_factors_raises(tmp_path):
    path = os.path.join(str(tmp_path), "bad.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"betas": {"X": {"USD": 1.0}}, "created_at": "2099-01-01T00:00:00+00:00"}, fh)

    with pytest.raises(ValueError, match="no factor list"):
        load_betas(path)


def test_save_is_atomic_leaving_no_temp_files(tmp_path):
    path = os.path.join(str(tmp_path), "betas.json")
    save_betas(make_book(), path)
    save_betas(make_book(), path)     # overwrite

    leftovers = [f for f in os.listdir(str(tmp_path)) if f.startswith(".tmp")]
    assert leftovers == [], f"atomic write left temp files behind: {leftovers}"



# ---------------------------------------------------------------------------
# best_hedge_for must not recommend a lever the overlay would reject
# ---------------------------------------------------------------------------


def test_best_hedge_for_rejects_a_near_zero_lever_when_thresholds_are_given():
    """A tiny-but-nonzero purity beat a starting score of 0.0 and got reported as
    the best lever ("RISK -> EURUSD beta +0.01, purity 0.00"). With the overlay's
    real thresholds applied it must come back None instead."""
    book = BetaBook(
        factors=FACTORS,
        betas={"EURUSD": {"USD": -0.90, "RISK": 0.01,
                          "ENERGY": -0.02, "METALS": 0.001}},
        sigma={"EURUSD": 0.005}, resid_sigma={"EURUSD": 0.001},
        factor_sigma={"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022,
                      "METALS": 0.009},
    )

    # Unfiltered, it picks the only candidate for every factor.
    assert book.best_hedge_for("RISK", ["EURUSD"]) == "EURUSD"

    # With the overlay's thresholds it correctly refuses.
    from src.overlay.decision import HedgeCaps
    d = HedgeCaps(factor_caps={"USD": 1.0})
    assert book.best_hedge_for("RISK", ["EURUSD"],
                               min_purity=d.min_purity,
                               min_abs_beta=d.min_abs_beta) is None
    assert book.best_hedge_for("METALS", ["EURUSD"],
                               min_purity=d.min_purity,
                               min_abs_beta=d.min_abs_beta) is None
    # But USD is a genuine lever and must survive.
    assert book.best_hedge_for("USD", ["EURUSD"],
                               min_purity=d.min_purity,
                               min_abs_beta=d.min_abs_beta) == "EURUSD"
