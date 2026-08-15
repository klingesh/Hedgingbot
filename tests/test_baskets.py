"""
Tests for basket factor construction.

The headline test is `test_basket_fixes_the_degenerate_self_regression`, which
manufactures a metals complex with a KNOWN idiosyncratic component and shows that:

  * a single-series proxy (the old behaviour) reports R2 = 1.00 and idio = 0.00
    for the instrument that IS the proxy — pure artefact, and it tells the risk
    model that instrument has no unhedgeable risk;
  * a basket proxy recovers a real R2 and a real idio close to the truth.

That is the entire justification for this module, so it is asserted from both
sides rather than argued.
"""

from __future__ import annotations

import math

import pytest

from src.factors.baskets import (
    build_basket,
    circularity_report,
    inverse_vol_weights,
    self_weight,
)
from src.factors.estimate import build_factors, estimate_betas
from src.factors.math_core import normalized_weights, wcorr, wstd
from tests.test_math_core import LCG

ORDER = ("USD", "RISK", "ENERGY", "METALS")


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------


def test_inverse_vol_weights_give_the_quiet_series_more_weight():
    rng = LCG(1)
    n = 800
    components = {
        "quiet": rng.normals(n, 0.0, 0.002),
        "loud": rng.normals(n, 0.0, 0.020),
    }
    w = inverse_vol_weights(components, list(range(n)))

    assert math.fsum(w.values()) == pytest.approx(1.0)
    assert w["quiet"] > w["loud"]
    # Ratio of weights should be ~ the inverse ratio of vols (10x here).
    assert w["quiet"] / w["loud"] == pytest.approx(10.0, rel=0.15)


def test_inverse_vol_weights_equalize_risk_contribution():
    """The point of the scheme: each component contributes the same risk."""
    rng = LCG(2)
    n = 1200
    components = {
        "a": rng.normals(n, 0.0, 0.003),
        "b": rng.normals(n, 0.0, 0.012),
        "c": rng.normals(n, 0.0, 0.030),
    }
    rows = list(range(n))
    w = inverse_vol_weights(components, rows)
    ew = normalized_weights(n, None)

    contributions = [
        w[name] * wstd([float(v) for v in col], ew)
        for name, col in components.items()
    ]
    assert max(contributions) / min(contributions) == pytest.approx(1.0, rel=0.15)


def test_zero_variance_component_gets_no_weight_not_infinite_weight():
    rng = LCG(3)
    n = 200
    components = {"real": rng.normals(n, 0.0, 0.01), "flat": [0.0] * n}
    w = inverse_vol_weights(components, list(range(n)))

    assert w["flat"] == 0.0
    assert w["real"] == pytest.approx(1.0)


def test_all_degenerate_components_fall_back_to_equal_weights():
    w = inverse_vol_weights({"a": [0.0] * 50, "b": [0.0] * 50}, list(range(50)))
    assert w == {"a": 0.5, "b": 0.5}


# ---------------------------------------------------------------------------
# Basket construction
# ---------------------------------------------------------------------------


def test_basket_preserves_length_and_gaps():
    rng = LCG(4)
    n = 300
    a = rng.normals(n, 0.0, 0.01)
    b = rng.normals(n, 0.0, 0.01)
    a[5] = None                                   # type: ignore[index]
    b[9] = None                                   # type: ignore[index]

    series, weights = build_basket({"a": a, "b": b})

    assert len(series) == n
    assert series[5] is None, "a gap in ANY component must blank the basket"
    assert series[9] is None
    assert series[20] is not None
    assert math.fsum(weights.values()) == pytest.approx(1.0)


def test_basket_of_one_is_a_passthrough_up_to_scale():
    rng = LCG(5)
    a = rng.normals(400, 0.0, 0.01)
    series, weights = build_basket({"a": list(a)})
    assert weights == {"a": 1.0}
    # Single component, no rescale applied (len == 1).
    assert series == pytest.approx(a)


def test_basket_is_rescaled_to_mean_component_volatility():
    """Inverse-vol weighting diversifies, so the raw basket is quieter than any
    member. Rescaling keeps '1% METALS' comparable in size to '1% gold', which is
    what makes the exposure report legible."""
    rng = LCG(6)
    n = 1500
    components = {
        "a": rng.normals(n, 0.0, 0.010),
        "b": rng.normals(n, 0.0, 0.010),
        "c": rng.normals(n, 0.0, 0.010),
    }
    rows = list(range(n))
    ew = normalized_weights(n, None)

    scaled, _w = build_basket(components, rescale_to_mean_component_vol=True)
    raw, _w2 = build_basket(components, rescale_to_mean_component_vol=False)

    mean_component = math.fsum(
        wstd([float(v) for v in col], ew) for col in components.values()
    ) / 3
    assert wstd([float(v) for v in scaled], ew) == pytest.approx(
        mean_component, rel=0.02
    )
    # Independent components: the raw basket is ~1/sqrt(3) as volatile.
    assert wstd([float(v) for v in raw], ew) < wstd(
        [float(v) for v in scaled], ew
    )


def test_rescaling_does_not_change_direction():
    """Scaling is a change of units, so correlation with anything is untouched."""
    rng = LCG(7)
    n = 800
    components = {"a": rng.normals(n, 0.0, 0.01), "b": rng.normals(n, 0.0, 0.02)}
    other = rng.normals(n, 0.0, 0.01)
    ew = normalized_weights(n, None)

    scaled, _ = build_basket(components, rescale_to_mean_component_vol=True)
    raw, _ = build_basket(components, rescale_to_mean_component_vol=False)

    assert wcorr([float(v) for v in scaled], other, ew) == pytest.approx(
        wcorr([float(v) for v in raw], other, ew), rel=1e-9
    )


def test_explicit_weights_are_normalized_and_respected():
    rng = LCG(8)
    n = 400
    components = {"a": rng.normals(n, 0.0, 0.01), "b": rng.normals(n, 0.0, 0.01)}
    _series, weights = build_basket(components, weights={"a": 3.0, "b": 1.0})
    assert weights["a"] == pytest.approx(0.75)
    assert weights["b"] == pytest.approx(0.25)


def test_missing_explicit_weight_raises():
    rng = LCG(9)
    n = 200
    components = {"a": rng.normals(n, 0.0, 0.01), "b": rng.normals(n, 0.0, 0.01)}
    with pytest.raises(ValueError, match="no weight supplied"):
        build_basket(components, weights={"a": 1.0})


def test_empty_and_too_short_baskets_raise():
    with pytest.raises(ValueError, match="no basket components"):
        build_basket({})
    with pytest.raises(ValueError, match="at least 30"):
        build_basket({"a": [0.01] * 10, "b": [0.01] * 10})


def test_basket_needs_overlapping_components():
    """Two components that never coexist cannot form a factor."""
    a = [0.01] * 50 + [None] * 50          # type: ignore[list-item]
    b = [None] * 50 + [0.01] * 50          # type: ignore[list-item]
    with pytest.raises(ValueError, match="at least 30"):
        build_basket({"a": a, "b": b})


# ---------------------------------------------------------------------------
# Circularity measurement
# ---------------------------------------------------------------------------


def test_self_weight_reports_only_nonzero_contributions():
    weights = {
        "METALS": {"GC=F": 0.43, "SI=F": 0.28, "PL=F": 0.29},
        "USD": {"DX-Y.NYB": 1.0},
        "ENERGY": {"CL=F": 1.0},
    }
    assert self_weight("GC=F", weights) == {"METALS": pytest.approx(0.43)}
    assert self_weight("CL=F", weights) == {"ENERGY": pytest.approx(1.0)}
    assert self_weight("EURUSD=X", weights) == {}


def test_circularity_report_distinguishes_traded_from_hedge_role():
    """The distinction matters: a traded instrument that IS its factor gets a fake
    R2 of 1.00 (bad), while a hedge candidate that IS its factor is a perfectly
    pure lever (good). Reporting both identically would invite 'fixing' the
    beneficial case."""
    weights = {
        "METALS": {"GC=F": 0.43, "SI=F": 0.28, "PL=F": 0.29},
        "ENERGY": {"CL=F": 1.0},
    }
    text = circularity_report(
        {"GOLD": "GC=F", "SILVER": "SI=F", "WTI": "CL=F"},
        weights,
        traded=["GOLD", "SILVER"],
    )

    assert "GOLD" in text and "WTI" in text
    assert "pure lever" in text, "a hedge candidate at 1.00 must be marked fine"
    assert "R2 inflated" in text, "a traded member at 0.43 must be flagged"
    # No traded instrument is at >0.90 here, so the hard warning must not fire.
    assert "A TRADED instrument IS a factor" not in text


def test_circularity_report_escalates_a_traded_instrument_that_is_a_factor():
    """The original live situation: METALS was GC=F and GOLD trades GC=F."""
    text = circularity_report(
        {"GOLD": "GC=F"}, {"METALS": {"GC=F": 1.0}}, traded=["GOLD"]
    )
    assert "A TRADED instrument IS a factor" in text
    assert "IS the factor" in text
    assert "FACTOR_BASKETS" in text


def test_circularity_report_with_no_overlap():
    text = circularity_report({"AUDUSD": "AUDUSD=X"}, {"USD": {"DX-Y.NYB": 1.0}})
    assert "No circularity" in text


# ---------------------------------------------------------------------------
# THE test: does the basket actually fix the degenerate regression?
# ---------------------------------------------------------------------------


def _metals_complex(n: int = 2600, seed: int = 2026):
    """A metals complex with a KNOWN common driver and known per-metal idio."""
    rng = LCG(seed)
    usd = rng.normals(n, 0.0, 0.004)
    risk = rng.normals(n, 0.0, 0.010)
    oil = rng.normals(n, 0.0, 0.022)
    common = rng.normals(n, 0.0, 0.009)          # the TRUE metals factor

    gold_idio = 0.004
    gold = [-0.9 * usd[i] + 1.00 * common[i] + rng.normal(0.0, gold_idio)
            for i in range(n)]
    silver = [-1.0 * usd[i] + 1.30 * common[i] + rng.normal(0.0, 0.011)
              for i in range(n)]
    plat = [-0.8 * usd[i] + 0.90 * common[i] + rng.normal(0.0, 0.013)
            for i in range(n)]

    # True gold R^2 = 1 - idio_var / total_var
    total_var = (0.9 * 0.004) ** 2 + (1.00 * 0.009) ** 2 + gold_idio ** 2
    true_r2 = 1.0 - gold_idio ** 2 / total_var

    return {
        "usd": usd, "risk": risk, "oil": oil,
        "gold": gold, "silver": silver, "plat": plat,
        "true_gold_idio": gold_idio, "true_gold_r2": true_r2,
    }


def _fit_gold(metals_proxy, data, n):
    proxies = {
        "USD": list(data["usd"]),
        "RISK": list(data["risk"]),
        "ENERGY": list(data["oil"]),
        "METALS": metals_proxy,
    }
    factors, kept = build_factors(proxies, ORDER, None)
    book, _sk = estimate_betas({"GOLD": list(data["gold"])}, factors, kept,
                               None, min_obs=200)
    return book


def test_basket_fixes_the_degenerate_self_regression():
    n = 2600
    d = _metals_complex(n=n)

    # --- OLD behaviour: METALS proxied by gold itself --------------------
    single = _fit_gold(list(d["gold"]), d, n)

    assert single.r_squared("GOLD") == pytest.approx(1.0, abs=1e-9), (
        "a single-series proxy must reproduce the degenerate result"
    )
    assert single.resid_sigma("GOLD") == pytest.approx(0.0, abs=1e-9), (
        "and report zero unhedgeable risk, which is the dangerous part"
    )

    # --- NEW behaviour: METALS as an equal-risk basket -------------------
    basket, weights = build_basket({
        "GC=F": list(d["gold"]),
        "SI=F": list(d["silver"]),
        "PL=F": list(d["plat"]),
    })
    multi = _fit_gold(basket, d, n)

    # Gold's own weight has dropped from 100% to roughly a third.
    assert weights["GC=F"] < 0.55
    assert self_weight("GC=F", {"METALS": weights})["METALS"] == weights["GC=F"]

    # R^2 is now a real number, not 1.00.
    assert multi.r_squared("GOLD") < 0.95
    # And idiosyncratic risk is no longer zero.
    assert multi.resid_sigma("GOLD") > 0.5 * d["true_gold_idio"]

    # It should land near the truth, and err on the CONSERVATIVE side:
    # reporting slightly MORE unhedgeable risk than reality is the correct
    # direction for a risk system to be wrong in.
    assert multi.resid_sigma("GOLD") == pytest.approx(
        d["true_gold_idio"], rel=0.6
    ), (f"idio {multi.resid_sigma('GOLD'):.5f} vs true {d['true_gold_idio']:.5f}")
    assert multi.r_squared("GOLD") == pytest.approx(d["true_gold_r2"], abs=0.15), (
        f"R2 {multi.r_squared('GOLD'):.2f} vs true {d['true_gold_r2']:.2f}"
    )


def test_basket_keeps_the_usd_relationship_intact():
    """Fixing METALS must not damage the other factors' betas."""
    n = 2600
    d = _metals_complex(n=n)
    basket, _w = build_basket({
        "GC=F": list(d["gold"]),
        "SI=F": list(d["silver"]),
        "PL=F": list(d["plat"]),
    })
    book = _fit_gold(basket, d, n)

    # Gold was built with a -0.9 loading on the dollar.
    assert book.beta("GOLD", "USD") == pytest.approx(-0.9, abs=0.2)
    # And no spurious energy or equity exposure.
    assert abs(book.beta("GOLD", "ENERGY")) < 0.1
    assert abs(book.beta("GOLD", "RISK")) < 0.15


def test_shipped_metals_basket_has_three_members_and_others_are_single():
    """Pins the deliberate design decision so a future edit is conscious.

    METALS is a basket because GOLD, SILVER and PLATINUM are all TRADED and gold
    alone was the proxy. ENERGY stays single because WTI is only a hedge candidate
    (its circularity makes it a pure lever) and adding BZ=F would make the traded
    BRENT slot *more* circular, not less.
    """
    from src.factors.definitions import FACTOR_BASKETS

    assert FACTOR_BASKETS["METALS"][0] == ("GC=F", "SI=F", "PL=F")
    assert len(FACTOR_BASKETS["USD"][0]) == 1
    assert len(FACTOR_BASKETS["RISK"][0]) == 1
    assert len(FACTOR_BASKETS["ENERGY"][0]) == 1
    assert "BZ=F" not in FACTOR_BASKETS["ENERGY"][0], (
        "BRENT is traded; adding BZ=F to ENERGY would increase its circularity"
    )
