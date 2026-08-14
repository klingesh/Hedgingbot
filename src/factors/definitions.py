"""
What the factors ARE, which market series proxies them, and a set of
hand-written PRIOR betas used only as a sanity check against the measured ones.

Design note — why four factors and not twenty
---------------------------------------------
The Tradingbot book is 8 CFD slots: 5 commodities (GOLD, SILVER, PLATINUM,
NATGAS, BRENT) and 3 FX legs (GBPJPY, AUDUSD, USDJPY). Six of the eight have a
USD leg. That book does not need a 20-factor risk model; it needs to know
whether it is secretly making the same bet six times. Four factors is enough to
answer that and few enough that the betas are estimated from hundreds of
observations each rather than dozens.

    USD      Broad dollar direction. The dominant shared driver of this book.
    RISK     Risk-on / risk-off sentiment (equity beta).
    ENERGY   Oil & gas complex, after removing USD and RISK.
    METALS   Gold complex / real-rates, after removing USD, RISK and ENERGY.

Factors are ORTHOGONALIZED (see estimate.build_factors). That matters: raw gold
and raw DXY returns are strongly negatively correlated, so unorthogonalized
betas double-count the same risk and are numerically unstable. After
orthogonalization each factor is statistically independent of the ones before
it, which means (a) betas are stable, (b) univariate and multivariate betas
coincide, and (c) portfolio variance is a plain sum of squares with no
covariance cross-terms.

Because orthogonalization is SEQUENTIAL, the ORDER MATTERS and is a modelling
choice, not a fact. We order from most macro-primitive to most idiosyncratic:
USD -> RISK -> ENERGY -> METALS. Consequence to keep in mind: METALS is defined
as "the part of gold that USD, RISK and ENERGY do not explain". So a plain
dollar-driven gold rally shows up in the USD factor, not in METALS. That is the
behaviour we want for a book this metals-heavy.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Factor names, in orthogonalization order.
# ---------------------------------------------------------------------------

USD = "USD"
RISK = "RISK"
ENERGY = "ENERGY"
METALS = "METALS"

FACTORS: tuple[str, ...] = (USD, RISK, ENERGY, METALS)

#: Sequential Gram-Schmidt order. Earlier factors keep their full variance;
#: later ones are residualized against all earlier ones.
ORTHOGONALIZATION_ORDER: tuple[str, ...] = (USD, RISK, ENERGY, METALS)

# ---------------------------------------------------------------------------
# Market series that proxy each factor (Yahoo Finance symbols, so the existing
# Tradingbot yahoo_loader can fetch them with no new dependency).
# ---------------------------------------------------------------------------

#: factor -> (yahoo symbol, sign, human description)
#:
#: `sign` flips the proxy so that a POSITIVE factor return always means the
#: economically positive direction of the factor name. DXY already rises when
#: the dollar strengthens, so USD is +1.
FACTOR_PROXIES: dict[str, tuple[str, int, str]] = {
    USD:    ("DX-Y.NYB", +1, "US Dollar Index (DXY) — dollar strength"),
    RISK:   ("^GSPC",    +1, "S&P 500 — risk-on sentiment"),
    ENERGY: ("CL=F",     +1, "WTI crude — energy complex"),
    METALS: ("GC=F",     +1, "Gold — metals / real-rates complex"),
}

#: If DXY is unavailable (it is occasionally flaky on Yahoo), synthesize a
#: dollar proxy from FX majors instead: a trade-weighted-ish basket where each
#: leg is signed so positive == stronger dollar.
USD_SYNTHETIC_BASKET: dict[str, tuple[int, float]] = {
    # yahoo symbol: (sign, weight)   sign=-1 because EURUSD FALLS when USD rises
    "EURUSD=X": (-1, 0.576),
    "USDJPY=X": (+1, 0.136),
    "GBPUSD=X": (-1, 0.119),
    "USDCAD=X": (+1, 0.091),
    "USDCHF=X": (+1, 0.078),
}

# ---------------------------------------------------------------------------
# The instruments actually traded by Tradingbot's DEFAULT_PORTFOLIO, mapped to
# the Yahoo series used to estimate their betas.
#
# Logical name -> (yahoo symbol, JustMarkets-style broker symbol, asset class)
# Kept identical to Tradingbot/src/data/yahoo_loader.py INSTRUMENTS so the two
# repos never disagree about what "GOLD" means.
# ---------------------------------------------------------------------------

TRADINGBOT_PORTFOLIO: dict[str, tuple[str, str, str]] = {
    "GOLD":     ("GC=F",     "XAUUSD", "commodity"),
    "SILVER":   ("SI=F",     "XAGUSD", "commodity"),
    "PLATINUM": ("PL=F",     "XPTUSD", "commodity"),
    "NATGAS":   ("NG=F",     "NGAS",   "commodity"),
    "BRENT":    ("BZ=F",     "UKOIL",  "commodity"),
    "GBPJPY":   ("GBPJPY=X", "GBPJPY", "forex"),
    "AUDUSD":   ("AUDUSD=X", "AUDUSD", "forex"),
    "USDJPY":   ("USDJPY=X", "USDJPY", "forex"),
}

# ---------------------------------------------------------------------------
# Hedge instrument candidates.
#
# A good hedge instrument is (a) liquid with a tight spread, (b) high |beta| to
# the factor we want to neutralize, and (c) LOW beta to everything else, so
# hedging one factor does not silently create exposure to another. Property (c)
# is called "purity" in overlay/decision.py and is scored from the MEASURED
# betas, never from the priors below.
#
# `preferred_for` is only a hint that seeds the candidate search; the actual
# choice is made numerically at decision time.
# ---------------------------------------------------------------------------

#: logical -> (yahoo symbol, broker symbol, preferred_for factor or None)
HEDGE_CANDIDATES: dict[str, tuple[str, str, str | None]] = {
    # Cleanest single-factor USD expression available as a retail CFD.
    "EURUSD": ("EURUSD=X", "EURUSD", USD),
    # Backup / cross-check USD leg.
    "USDCHF": ("USDCHF=X", "USDCHF", USD),
    # Equity index for RISK. US500 is the tightest-spread index CFD at most brokers.
    "SP500":  ("ES=F",     "US500",  RISK),
    # WTI for ENERGY (BRENT is already a book position, so hedging with WTI
    # avoids netting against the trader's own leg on the same symbol).
    "WTI":    ("CL=F",     "USOIL",  ENERGY),
    # Silver is the liquid metals leg that is NOT gold, so it can offset a
    # METALS breach without colliding with the GOLD slot.
    "SILVER": ("SI=F",     "XAGUSD", METALS),
}

# ---------------------------------------------------------------------------
# PRIOR betas — SANITY CHECK ONLY. NOT USED FOR SIZING.
#
# These are hand-written economic expectations. Their single purpose is to be
# compared against the empirically measured betas so that a data or sign error
# is caught loudly instead of quietly mis-hedging. If a measured beta disagrees
# with its prior in SIGN, scripts/estimate_betas.py flags it.
#
# Read as: "a +1% move in this factor moves this instrument by X%", for a LONG
# position of the instrument. Values are deliberately round numbers — they are
# expectations, not measurements.
# ---------------------------------------------------------------------------

PRIOR_BETAS: dict[str, dict[str, float]] = {
    # Metals: strongly short the dollar, mildly risk-sensitive.
    "GOLD":     {USD: -0.9, RISK:  0.0, ENERGY: 0.1, METALS: 1.0},
    "SILVER":   {USD: -1.0, RISK:  0.3, ENERGY: 0.1, METALS: 0.8},
    "PLATINUM": {USD: -0.8, RISK:  0.4, ENERGY: 0.2, METALS: 0.5},
    # Energy: oil is only weakly a dollar trade; gas is nearly its own asset class.
    "BRENT":    {USD: -0.4, RISK:  0.4, ENERGY: 1.0, METALS: 0.0},
    "NATGAS":   {USD: -0.2, RISK:  0.1, ENERGY: 0.5, METALS: 0.0},
    # FX. AUD is the commodity/risk currency; USDJPY is long dollar AND risk-on
    # (yen is the funding/haven currency); GBPJPY is a pure risk barometer with
    # little direct dollar content.
    "AUDUSD":   {USD: -0.9, RISK:  0.5, ENERGY: 0.1, METALS: 0.2},
    "USDJPY":   {USD:  0.8, RISK:  0.3, ENERGY: 0.0, METALS: -0.2},
    "GBPJPY":   {USD:  0.0, RISK:  0.5, ENERGY: 0.0, METALS: -0.1},
    # Hedge candidates.
    "EURUSD":   {USD: -1.0, RISK:  0.1, ENERGY: 0.0, METALS: 0.1},
    "USDCHF":   {USD:  0.8, RISK:  0.0, ENERGY: 0.0, METALS: -0.2},
    "SP500":    {USD: -0.1, RISK:  1.0, ENERGY: 0.1, METALS: 0.0},
    "WTI":      {USD: -0.4, RISK:  0.4, ENERGY: 1.0, METALS: 0.0},
}


def priors_for(instruments: list[str] | None = None) -> dict[str, dict[str, float]]:
    """PRIOR betas as a nested dict, restricted to `instruments` if given.

    Sanity-check use only. Never size a hedge from this — it is fed to
    BetaBook.sign_disagreements() so that a wrong ticker or an inverted quote
    convention is caught loudly instead of quietly mis-hedging.
    """
    names = instruments if instruments is not None else list(PRIOR_BETAS)
    return {
        n: {f: PRIOR_BETAS.get(n, {}).get(f, 0.0) for f in FACTORS}
        for n in names if n in PRIOR_BETAS
    }
