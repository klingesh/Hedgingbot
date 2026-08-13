"""
Tests for the factor mathematics.

The central test is `test_recovers_known_betas`: we MANUFACTURE returns from betas
we chose, then check the estimator gets them back. If that passes, the regression
machinery is sound and any strange live number is a data problem, not a maths
problem. Everything downstream — exposure, hedge sizing — is arithmetic on top of
these betas, so this is the test that matters most.

No numpy: a small deterministic LCG generates the synthetic data, so results are
identical on every machine and Python version.
"""

from __future__ import annotations

import math

import pytest

from src.factors.math_core import (
    align,
    build_orthogonal_factors,
    fit_instrument,
    log_return_series,
    normalized_weights,
    purity,
    wcorr,
    wcov,
    winsorize,
    wmean,
    wstd,
    wvar,
)

FACTOR_ORDER = ("USD", "RISK", "ENERGY", "METALS")


class LCG:
    """Deterministic linear congruential generator + Box-Muller normals.

    Hand-rolled so the tests need no numpy and give byte-identical results
    everywhere.
    """

    def __init__(self, seed: int = 12345) -> None:
        self.s = seed & 0xFFFFFFFF
        self._spare: float | None = None

    def uniform(self) -> float:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return (self.s + 1) / 2147483649.0

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        if self._spare is not None:
            z, self._spare = self._spare, None
            return mu + sigma * z
        while True:
            u1, u2 = self.uniform(), self.uniform()
            if u1 > 1e-12:
                break
        r = math.sqrt(-2.0 * math.log(u1))
        self._spare = r * math.sin(2.0 * math.pi * u2)
        return mu + sigma * (r * math.cos(2.0 * math.pi * u2))

    def normals(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> list[float]:
        return [self.normal(mu, sigma) for _ in range(n)]


def independent_proxies(n: int = 3000, seed: int = 7) -> dict[str, list[float]]:
    """Four mutually (near-)independent proxy series with realistic vols."""
    rng = LCG(seed)
    vols = {"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022, "METALS": 0.009}
    return {f: rng.normals(n, 0.0, v) for f, v in vols.items()}


# ---------------------------------------------------------------------------
# Weighted moments
# ---------------------------------------------------------------------------


def test_equal_weights_reduce_to_unbiased_sample_stats():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    w = normalized_weights(len(x), None)

    assert wmean(x, w) == pytest.approx(3.0)
    # Unbiased sample variance of 1..5 is 2.5
    assert wvar(x, w) == pytest.approx(2.5)
    assert wstd(x, w) == pytest.approx(math.sqrt(2.5))


def test_weights_sum_to_one_and_decay():
    w = normalized_weights(100, halflife=10)
    assert math.fsum(w) == pytest.approx(1.0)
    assert w[-1] > w[0], "most recent observation must carry the most weight"
    # A point one half-life back should carry ~half the weight of the newest.
    assert w[-11] / w[-1] == pytest.approx(0.5, rel=1e-9)


def test_zero_and_negative_halflife_fall_back_to_equal_weights():
    for hl in (None, 0, -5):
        w = normalized_weights(10, hl)
        assert all(wi == pytest.approx(0.1) for wi in w)


def test_covariance_and_correlation_are_consistent():
    rng = LCG(3)
    x = rng.normals(2000, 0.0, 1.0)
    y = [2.0 * xi + rng.normal(0.0, 0.5) for xi in x]
    w = normalized_weights(len(x), None)

    assert wcov(x, y, w) == pytest.approx(2.0 * wvar(x, w), rel=0.05)
    assert wcorr(x, y, w) == pytest.approx(
        wcov(x, y, w) / (wstd(x, w) * wstd(y, w)), rel=1e-12
    )
    assert wcorr(x, x, w) == pytest.approx(1.0)


def test_correlation_of_constant_series_is_zero_not_nan():
    x = [1.0] * 50
    y = list(range(50))
    w = normalized_weights(50, None)
    assert wcorr(x, y, w) == 0.0


# ---------------------------------------------------------------------------
# Returns / winsorize / align
# ---------------------------------------------------------------------------


def test_log_returns_basic():
    rets = log_return_series([100.0, 110.0, 99.0])
    assert rets[0] is None
    assert rets[1] == pytest.approx(math.log(1.1))
    assert rets[2] == pytest.approx(math.log(99.0 / 110.0))


def test_log_returns_survive_bad_ticks():
    rets = log_return_series([100.0, 0.0, -3.0, 105.0, None, 110.0])
    assert rets[0] is None
    assert rets[1] is None and rets[2] is None and rets[3] is None
    assert rets[4] is None and rets[5] is None
    # And nothing raised, which is the point.


def test_winsorize_clips_both_tails():
    values = [0.0] * 98 + [50.0, -50.0]
    clipped = winsorize(values, 0.02)
    assert max(clipped) < 50.0
    assert min(clipped) > -50.0
    assert len(clipped) == len(values)


def test_winsorize_zero_fraction_is_identity():
    values = [1.0, -8.0, 3.0]
    assert winsorize(values, 0.0) == values


def test_winsorize_tames_a_natgas_style_spike():
    prices = [10.0] * 200
    prices[100] = 40.0     # a +300% print
    prices[101] = 10.0
    rets = [r for r in log_return_series(prices) if r is not None]

    raw_max = max(abs(r) for r in rets)
    clipped = winsorize(rets, 0.02)
    assert raw_max > 1.3
    assert max(abs(r) for r in clipped) < raw_max


def test_align_drops_rows_where_any_series_is_missing():
    series = {
        "a": [1.0, 2.0, None, 4.0],
        "b": [1.0, None, 3.0, 4.0],
        "c": [1.0, 2.0, 3.0, 4.0],
    }
    keep, aligned = align(series)
    assert keep == [0, 3]
    assert aligned["a"] == [1.0, 4.0]
    assert aligned["b"] == [1.0, 4.0]


def test_align_rejects_ragged_input():
    with pytest.raises(ValueError, match="length"):
        align({"a": [1.0, 2.0], "b": [1.0]})


# ---------------------------------------------------------------------------
# Orthogonalization
# ---------------------------------------------------------------------------


def test_independent_proxies_pass_through_orthogonal():
    fs = build_orthogonal_factors(independent_proxies(), FACTOR_ORDER)
    assert fs.orthogonality_error() < 1e-9
    # Nothing to remove, so almost all variance survives.
    for f in FACTOR_ORDER:
        assert fs.variance_retained[f] > 0.90


def test_orthogonalization_removes_real_correlation():
    """Gold and the dollar are strongly negatively correlated. After
    orthogonalization the factors must be independent."""
    n = 3000
    rng = LCG(11)
    usd = rng.normals(n, 0.0, 0.004)
    rho = -0.85
    resid_scale = 0.009 * math.sqrt(1 - rho * rho)
    gold = [rho * (u / 0.004) * 0.009 + rng.normal(0.0, resid_scale) for u in usd]
    risk = [0.5 * (u / 0.004) * 0.011 + rng.normal(0.0, 0.010) for u in usd]
    energy = [0.3 * (r / 0.011) * 0.022 + rng.normal(0.0, 0.020) for r in risk]

    proxies = {"USD": usd, "RISK": risk, "ENERGY": energy, "METALS": gold}
    w = normalized_weights(n, None)
    assert abs(wcorr(usd, gold, w)) > 0.6, "test setup must actually be correlated"

    fs = build_orthogonal_factors(proxies, FACTOR_ORDER)

    assert fs.orthogonality_error() < 1e-9
    assert fs.variance_retained["USD"] == pytest.approx(1.0, abs=1e-9)
    # METALS is last, so it sheds the most variance to the earlier factors.
    assert fs.variance_retained["METALS"] < 0.6
    # Rescaling restores the proxy's own scale.
    assert fs.sigma["METALS"] == pytest.approx(wstd(gold, w), rel=0.02)


def test_orthogonalization_order_changes_the_model():
    """Order is a modelling choice, not a fact. Assert that visibly."""
    n = 2000
    rng = LCG(3)
    a = rng.normals(n, 0.0, 0.01)
    b = [0.8 * ai + rng.normal(0.0, 0.006) for ai in a]
    proxies = {
        "USD": a, "RISK": b,
        "ENERGY": rng.normals(n, 0.0, 0.02),
        "METALS": rng.normals(n, 0.0, 0.009),
    }

    usd_first = build_orthogonal_factors(proxies, ("USD", "RISK", "ENERGY", "METALS"))
    risk_first = build_orthogonal_factors(proxies, ("RISK", "USD", "ENERGY", "METALS"))

    assert usd_first.variance_retained["USD"] == pytest.approx(1.0, abs=1e-9)
    assert usd_first.variance_retained["RISK"] < 0.7
    assert risk_first.variance_retained["RISK"] == pytest.approx(1.0, abs=1e-9)
    assert risk_first.variance_retained["USD"] < 0.7


def test_rescale_off_leaves_residual_scale():
    proxies = independent_proxies(n=500, seed=9)
    scaled = build_orthogonal_factors(proxies, FACTOR_ORDER, rescale_to_proxy=True)
    raw = build_orthogonal_factors(proxies, FACTOR_ORDER, rescale_to_proxy=False)
    # Orthogonality holds either way; only the scale differs.
    assert scaled.orthogonality_error() < 1e-9
    assert raw.orthogonality_error() < 1e-9


def test_missing_proxy_raises():
    proxies = independent_proxies(n=200)
    del proxies["METALS"]
    with pytest.raises(ValueError, match="missing factor"):
        build_orthogonal_factors(proxies, FACTOR_ORDER)


def test_too_few_observations_raises():
    with pytest.raises(ValueError, match="at least 30"):
        build_orthogonal_factors(independent_proxies(n=20), FACTOR_ORDER)


def test_ragged_proxies_raise():
    proxies = independent_proxies(n=200)
    proxies["USD"] = proxies["USD"][:100]
    with pytest.raises(ValueError, match="differing lengths"):
        build_orthogonal_factors(proxies, FACTOR_ORDER)


def test_none_in_proxies_raises_rather_than_silently_skewing():
    proxies = independent_proxies(n=200)
    proxies["USD"][5] = None            # type: ignore[index]
    with pytest.raises(ValueError, match="None/non-finite"):
        build_orthogonal_factors(proxies, FACTOR_ORDER)


# ---------------------------------------------------------------------------
# Beta recovery — THE important test
# ---------------------------------------------------------------------------


def test_recovers_known_betas():
    """Manufacture instruments from known betas, then recover them."""
    n = 12000
    fs = build_orthogonal_factors(independent_proxies(n=n, seed=42), FACTOR_ORDER)

    truth = {
        "GOLD":   {"USD": -0.90, "RISK": 0.00, "ENERGY": 0.10, "METALS": 1.00},
        "AUDUSD": {"USD": -0.90, "RISK": 0.50, "ENERGY": 0.10, "METALS": 0.20},
        "USDJPY": {"USD": 0.80, "RISK": 0.30, "ENERGY": 0.00, "METALS": -0.20},
    }

    rng = LCG(99)
    for inst, bv in truth.items():
        y = [0.0] * n
        for f, b in bv.items():
            fser = fs.series[f]
            y = [yi + b * fser[i] for i, yi in enumerate(y)]
        y = [yi + rng.normal(0.0, 0.003) for yi in y]   # idiosyncratic noise

        fit = fit_instrument(y, fs)
        for f, expected in bv.items():
            got = fit.betas[f]
            assert got == pytest.approx(expected, abs=0.06), (
                f"{inst}/{f}: expected {expected:+.2f}, measured {got:+.2f}"
            )


def test_exact_recovery_with_no_noise():
    """With zero idiosyncratic noise the betas must be recovered essentially
    exactly, and R^2 must be 1."""
    n = 2000
    fs = build_orthogonal_factors(independent_proxies(n=n, seed=5), FACTOR_ORDER)
    truth = {"USD": -0.75, "RISK": 0.4, "ENERGY": 0.0, "METALS": 1.2}

    y = [0.0] * n
    for f, b in truth.items():
        fser = fs.series[f]
        y = [yi + b * fser[i] for i, yi in enumerate(y)]

    fit = fit_instrument(y, fs)
    for f, b in truth.items():
        assert fit.betas[f] == pytest.approx(b, abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit.resid_sigma == pytest.approx(0.0, abs=1e-9)


def test_r_squared_and_resid_reconcile():
    n = 4000
    fs = build_orthogonal_factors(independent_proxies(n=n, seed=5), FACTOR_ORDER)
    rng = LCG(17)
    pure = list(fs.series["USD"])
    noisy = [p + rng.normal(0.0, 0.02) for p in pure]

    for y, expect_high in ((pure, True), (noisy, False)):
        fit = fit_instrument(y, fs)
        assert 0.0 <= fit.r_squared <= 1.0 + 1e-12
        total = fit.sigma ** 2
        explained = total * fit.r_squared
        assert fit.resid_sigma ** 2 == pytest.approx(total - explained, rel=1e-9, abs=1e-18)
        if expect_high:
            assert fit.r_squared > 0.999
        else:
            assert fit.r_squared < 0.5


def test_flat_instrument_produces_zeros_not_nan():
    n = 500
    fs = build_orthogonal_factors(independent_proxies(n=n, seed=8), FACTOR_ORDER)
    fit = fit_instrument([0.0] * n, fs)

    assert fit.r_squared == 0.0
    assert fit.sigma == pytest.approx(0.0)
    assert all(math.isfinite(b) for b in fit.betas.values())


def test_fit_rejects_length_mismatch():
    n = 200
    fs = build_orthogonal_factors(independent_proxies(n=n, seed=2), FACTOR_ORDER)
    with pytest.raises(ValueError, match="length mismatch"):
        fit_instrument([0.0] * (n - 1), fs)


def test_halflife_weighting_tracks_a_regime_change():
    """Betas must follow a regime shift when a half-life is supplied."""
    n = 4000
    rng = LCG(77)
    f = rng.normals(n, 0.0, 0.01)
    proxies = {
        "USD": f,
        "RISK": rng.normals(n, 0.0, 0.01),
        "ENERGY": rng.normals(n, 0.0, 0.02),
        "METALS": rng.normals(n, 0.0, 0.009),
    }
    # Beta is 0.2 in the first half, 1.8 in the second.
    y = [(0.2 if i < n // 2 else 1.8) * f[i] + rng.normal(0.0, 0.001)
         for i in range(n)]

    fs_eq = build_orthogonal_factors(proxies, FACTOR_ORDER, halflife=None)
    flat = fit_instrument(y, fs_eq, halflife=None).betas["USD"]

    fs_w = build_orthogonal_factors(proxies, FACTOR_ORDER, halflife=200)
    recent = fit_instrument(y, fs_w, halflife=200).betas["USD"]

    assert flat == pytest.approx(1.0, abs=0.2), "equal weight averages the regimes"
    assert recent > 1.4, "half-life weighting must track the recent regime"


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_purity_identifies_the_clean_single_factor_lever():
    sig = {"USD": 0.004, "RISK": 0.011, "ENERGY": 0.022, "METALS": 0.009}
    clean = {"USD": 1.0, "RISK": 0.0, "ENERGY": 0.0, "METALS": 0.0}
    mixed = {"USD": 1.0, "RISK": 1.0, "ENERGY": 0.0, "METALS": 0.0}

    assert purity(clean, "USD", sig) == pytest.approx(1.0)
    # RISK has ~2.75x the vol of USD, so a 1:1 beta mix is dominated by RISK.
    assert purity(mixed, "USD", sig) < 0.2
    assert purity(mixed, "RISK", sig) > 0.8


def test_purity_is_variance_weighted_not_beta_weighted():
    """A 'small' beta to a volatile factor is not small in risk terms."""
    sig = {"USD": 0.004, "RISK": 0.040}
    betas = {"USD": 1.0, "RISK": 0.5}

    beta_only = purity(betas, "USD", None)
    var_weighted = purity(betas, "USD", sig)

    assert beta_only == pytest.approx(1.0 / 1.25)     # 1^2 / (1^2 + 0.5^2)
    assert var_weighted < 0.05, (
        "RISK is 10x more volatile, so 0.5 beta to it dwarfs 1.0 beta to USD"
    )


def test_purity_of_all_zero_betas_is_zero():
    assert purity({"USD": 0.0, "RISK": 0.0}, "USD", {"USD": 0.01, "RISK": 0.01}) == 0.0
