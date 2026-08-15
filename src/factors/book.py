"""
BetaBook — the container the whole overlay reads betas from.

Plain dicts, pure stdlib, no pandas. This is the object that estimate.py
produces, store.py serializes, and exposure/overlay consume. Keeping it
dependency-free means the LIVE path (measure exposure, decide, trade) needs
nothing beyond the standard library, so it runs unchanged on a bare VPS.

Every lookup is total: an unknown instrument or factor returns 0.0 rather than
raising. That is deliberate for `beta()` because a missing beta must degrade to
"no measured relationship" rather than crashing a live risk loop — but callers
must ALSO check `has(instrument)` before trusting a zero, because "beta is
genuinely zero" and "I have never heard of this instrument" mean very different
things for risk. exposure.compute_exposure() does exactly that check and routes
unknown instruments to idiosyncratic risk plus a loud warning.
"""

from __future__ import annotations

import math


class BetaBook:
    """Factor betas and risk statistics for a set of instruments."""

    __slots__ = ("factors", "_betas", "_r2", "_sigma", "_resid", "_nobs", "_factor_sigma")

    def __init__(
        self,
        factors: tuple[str, ...],
        betas: dict[str, dict[str, float]],
        r_squared: dict[str, float] | None = None,
        sigma: dict[str, float] | None = None,
        resid_sigma: dict[str, float] | None = None,
        n_obs: dict[str, int] | None = None,
        factor_sigma: dict[str, float] | None = None,
    ) -> None:
        self.factors = tuple(factors)
        self._betas = {k: dict(v) for k, v in betas.items()}
        self._r2 = dict(r_squared or {})
        self._sigma = dict(sigma or {})
        self._resid = dict(resid_sigma or {})
        self._nobs = dict(n_obs or {})
        self._factor_sigma = dict(factor_sigma or {})

    # -- membership --------------------------------------------------------

    def instruments(self) -> list[str]:
        return sorted(self._betas)

    def has(self, instrument: str) -> bool:
        return instrument in self._betas

    def __contains__(self, instrument: object) -> bool:
        return isinstance(instrument, str) and instrument in self._betas

    def __len__(self) -> int:
        return len(self._betas)

    # -- lookups -----------------------------------------------------------

    def beta(self, instrument: str, factor: str) -> float:
        v = self._betas.get(instrument, {}).get(factor, 0.0)
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0

    def beta_vector(self, instrument: str) -> dict[str, float]:
        return {f: self.beta(instrument, f) for f in self.factors}

    def r_squared(self, instrument: str) -> float:
        return float(self._r2.get(instrument, 0.0))

    def sigma(self, instrument: str) -> float:
        """Total daily vol of the instrument."""
        return float(self._sigma.get(instrument, 0.0))

    def resid_sigma(self, instrument: str) -> float:
        """Idiosyncratic (unexplained) daily vol."""
        return float(self._resid.get(instrument, 0.0))

    def n_obs(self, instrument: str) -> int:
        return int(self._nobs.get(instrument, 0))

    def factor_sigma(self, factor: str = "") -> float | dict[str, float]:
        if not factor:
            return dict(self._factor_sigma)
        return float(self._factor_sigma.get(factor, 0.0))

    def factor_sigmas(self) -> dict[str, float]:
        return dict(self._factor_sigma)

    # -- derived -----------------------------------------------------------

    def purity(self, instrument: str, factor: str) -> float:
        """Share of the instrument's explained variance from one factor.

        Variance-weighted when factor sigmas are present (the normal case), else
        a beta-only approximation. See math_core.purity for the reasoning.
        """
        bv = self.beta_vector(instrument)
        if self._factor_sigma:
            contrib = {
                f: (b * self._factor_sigma.get(f, 0.0)) ** 2 for f, b in bv.items()
            }
        else:
            contrib = {f: b * b for f, b in bv.items()}
        denom = math.fsum(contrib.values())
        if denom <= 1e-18:
            return 0.0
        return contrib.get(factor, 0.0) / denom

    def best_hedge_for(
        self, factor: str, candidates: list[str], min_purity: float = 0.0,
        min_abs_beta: float = 0.0,
    ) -> str | None:
        """Highest purity^2 * |beta| candidate for a factor, or None."""
        best, best_score = None, 0.0
        for c in candidates:
            if not self.has(c):
                continue
            b = abs(self.beta(c, factor))
            p = self.purity(c, factor)
            if b < min_abs_beta or p < min_purity:
                continue
            score = p * p * b
            if score > best_score:
                best, best_score = c, score
        return best

    # -- reporting ---------------------------------------------------------

    def summary(self, priors: dict[str, dict[str, float]] | None = None) -> str:
        head = (f"{'instrument':<12}" + "".join(f"{f:>9}" for f in self.factors)
                + f"{'R2':>7}{'vol':>8}{'idio':>8}{'n':>7}")
        lines = [head, "-" * len(head)]
        for inst in self.instruments():
            row = f"{inst:<12}"
            row += "".join(f"{self.beta(inst, f):>9.2f}" for f in self.factors)
            row += f"{self.r_squared(inst):>7.2f}"
            row += f"{self.sigma(inst) * 100:>7.2f}%"
            row += f"{self.resid_sigma(inst) * 100:>7.2f}%"
            row += f"{self.n_obs(inst):>7}"
            lines.append(row)

        if priors is not None:
            flags = self.sign_disagreements(priors)
            lines.append("")
            if flags:
                lines.append("SIGN DISAGREEMENTS vs prior expectations (investigate):")
                for inst, fac, measured, prior in flags:
                    lines.append(
                        f"  {inst:<10} {fac:<8} measured={measured:+.2f}  prior={prior:+.2f}"
                    )
            else:
                lines.append("No sign disagreements vs prior expectations.")
        return "\n".join(lines)

    def sign_disagreements(
        self, priors: dict[str, dict[str, float]], min_abs: float = 0.15
    ) -> list[tuple[str, str, float, float]]:
        """Measured betas whose SIGN contradicts the prior.

        Betas too small to matter in either series are ignored. A hit here
        usually means a data problem — wrong ticker, inverted quote convention —
        rather than a market insight, which is exactly why it is worth surfacing
        before the number is used to size a trade.
        """
        out: list[tuple[str, str, float, float]] = []
        for inst in self.instruments():
            prior_row = priors.get(inst)
            if not prior_row:
                continue
            for fac in self.factors:
                m = self.beta(inst, fac)
                p = float(prior_row.get(fac, 0.0))
                if abs(m) < min_abs or abs(p) < min_abs:
                    continue
                if (m > 0) != (p > 0):
                    out.append((inst, fac, m, p))
        return out

    def scale_sigmas(self, factor: float) -> "BetaBook":
        """Return a copy with every volatility multiplied by `factor`.

        Used to convert volatilities estimated over a multi-day return period into
        DAILY-equivalent numbers, by scaling with 1/sqrt(period_days).

        This exists because BETAS and VOLATILITIES behave differently across
        horizons. A beta is a ratio of covariance to variance, so it is roughly
        horizon-invariant — which is exactly why estimating on weekly returns to
        defeat an intraday timing offset is legitimate. A volatility is NOT
        horizon-invariant: a weekly sigma is about sqrt(5) times a daily one.

        The exposure model consumes sigmas as DAILY (factor_daily_risk,
        portfolio_daily_risk). Handing it weekly sigmas silently inflates every
        risk figure by ~2.24x, so the conversion happens once here, at the point
        the betas are persisted, and the basis is recorded in the file's metadata.
        """
        if not (factor > 0) or not math.isfinite(factor):
            raise ValueError(f"scale factor must be finite and > 0, got {factor!r}")
        out = BetaBook(
            factors=self.factors,
            betas={k: dict(v) for k, v in self._betas.items()},
            r_squared=dict(self._r2),
            sigma={k: v * factor for k, v in self._sigma.items()},
            resid_sigma={k: v * factor for k, v in self._resid.items()},
            n_obs=dict(self._nobs),
            factor_sigma={k: v * factor for k, v in self._factor_sigma.items()},
        )
        return out

    def to_dict(self) -> dict:
        return {
            "factors": list(self.factors),
            "betas": {k: dict(v) for k, v in self._betas.items()},
            "r_squared": dict(self._r2),
            "sigma": dict(self._sigma),
            "resid_sigma": dict(self._resid),
            "n_obs": dict(self._nobs),
            "factor_sigma": dict(self._factor_sigma),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "BetaBook":
        return cls(
            factors=tuple(raw.get("factors", ())),
            betas=raw.get("betas", {}) or {},
            r_squared=raw.get("r_squared", {}),
            sigma=raw.get("sigma", {}),
            resid_sigma=raw.get("resid_sigma", {}),
            n_obs=raw.get("n_obs", {}),
            factor_sigma=raw.get("factor_sigma", {}),
        )
