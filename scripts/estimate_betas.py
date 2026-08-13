"""
Measure the factor betas of the Tradingbot portfolio, and answer the question
PROJECT_REPORT.md:143 leaves open: how correlated is that book, really?

    python scripts/estimate_betas.py

Needs internet (Yahoo Finance). Writes betas/betas_latest.json, which the overlay
loads. Run it monthly — betas decay, and factors/store.py refuses to trade on a
file older than max_age_days.

Output, in order:
  1. Raw correlation of the traded instruments. The concentration, unfiltered,
     plus an equal-risk diversification ratio and effective-bet count that are
     properties of the INSTRUMENT SET rather than of your current position sizes.
  2. Factor construction diagnostics: how much overlap orthogonalization removed.
  3. Per-instrument betas, R^2, idiosyncratic vol, and prior-sign checks.
  4. Split-sample stability: betas on the first half vs the second half. A beta
     that flips sign or halves between halves is not a number to size a hedge
     from, and this check tells you before you find out live.
  5. Hedge-candidate purity per factor: which instrument is the cleanest lever.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.yahoo import align_price_frame                            # noqa: E402
from src.factors.definitions import (                                    # noqa: E402
    FACTOR_PROXIES,
    FACTORS,
    HEDGE_CANDIDATES,
    ORTHOGONALIZATION_ORDER,
    TRADINGBOT_PORTFOLIO,
    priors_for,
)
from src.factors.estimate import (                                       # noqa: E402
    build_factors,
    estimate_betas,
    returns_from_prices,
)
from src.factors.math_core import (                                      # noqa: E402
    align,
    normalized_weights,
    wcorr,
)
from src.factors.store import save_betas                                 # noqa: E402


def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Concentration of the traded book
# ---------------------------------------------------------------------------


def describe_concentration(inst_returns: dict[str, list], portfolio: list[str]) -> None:
    cols = [c for c in portfolio if c in inst_returns]
    if len(cols) < 2:
        print("   Fewer than 2 traded instruments fetched; skipping.")
        return

    keep, aligned = align({c: inst_returns[c] for c in cols})
    n = len(keep)
    if n < 60:
        print(f"   Only {n} overlapping days across all {len(cols)} instruments; "
              "too few for a correlation study.")
        return

    w = normalized_weights(n, None)

    _hr("1. RAW CORRELATION OF THE TRADED BOOK")
    print(f"   {len(cols)} instruments, {n} overlapping days\n")

    width = max(len(c) for c in cols) + 1
    print("   " + " " * width + "".join(f"{c[:7]:>8}" for c in cols))
    corr: dict[tuple[str, str], float] = {}
    for a in cols:
        row = f"   {a:<{width}}"
        for b in cols:
            c = 1.0 if a == b else wcorr(aligned[a], aligned[b], w)
            corr[(a, b)] = c
            row += f"{c:>8.2f}"
        print(row)

    pairs = [corr[(a, b)] for i, a in enumerate(cols) for b in cols[i + 1:]]
    mean_c = math.fsum(pairs) / len(pairs)
    mean_abs = math.fsum(abs(p) for p in pairs) / len(pairs)
    print(f"\n   Mean pairwise correlation      : {mean_c:+.3f}")
    print(f"   Mean ABS pairwise correlation  : {mean_abs:+.3f}")
    print(f"   Max  pairwise correlation      : {max(pairs):+.3f}")
    print(f"   Min  pairwise correlation      : {min(pairs):+.3f}")

    # Equal-risk diversification ratio: how much risk cancels if you held equal
    # RISK in each instrument. Independent of position sizing, so it is a property
    # of the instrument SET itself.
    k = len(cols)
    port_var = math.fsum(corr[(a, b)] for a in cols for b in cols) / (k * k)
    ratio = math.sqrt(max(port_var, 0.0))
    print(f"\n   Equal-risk diversification ratio: {ratio:.3f}")
    if ratio > 1e-9:
        print(f"   Effective independent bets      : {1.0 / (ratio ** 2):.2f}  out of {k}")
    print(f"   (would be {k}.00 if independent, 1.00 if they were all one bet)")

    strong = sorted(
        ((abs(corr[(a, b)]), a, b)
         for i, a in enumerate(cols) for b in cols[i + 1:] if abs(corr[(a, b)]) > 0.5),
        reverse=True,
    )
    if strong:
        print("\n   Pairs correlated above 0.5 — these are NOT separate bets:")
        for c, a, b in strong:
            print(f"     {a:<10} {b:<10} {corr[(a, b)]:+.3f}")


# ---------------------------------------------------------------------------
# 4. Stability
# ---------------------------------------------------------------------------


def stability_check(
    inst_returns: dict[str, list],
    proxy_returns: dict[str, list],
    halflife: float | None,
    min_obs: int,
) -> None:
    _hr("4. SPLIT-SAMPLE BETA STABILITY")
    keep, _ = align({f: proxy_returns[f] for f in ORTHOGONALIZATION_ORDER})
    if len(keep) < 4 * min_obs:
        print(f"   Only {len(keep)} factor observations; need {4 * min_obs} for a "
              "meaningful split. Skipped.")
        return

    mid_row = keep[len(keep) // 2]
    half_min = max(30, min_obs // 3)

    def slice_series(series: dict[str, list], lo: int, hi: int) -> dict[str, list]:
        return {k: v[lo:hi] for k, v in series.items()}

    halves = {}
    for label, (lo, hi) in (("early", (0, mid_row)), ("late", (mid_row, None))):
        try:
            pr = slice_series(proxy_returns, lo, hi)
            ir = slice_series(inst_returns, lo, hi)
            fs, kept = build_factors(pr, ORTHOGONALIZATION_ORDER, halflife)
            book, _sk = estimate_betas(ir, fs, kept, halflife, min_obs=half_min)
            halves[label] = book
        except ValueError as exc:
            print(f"   {label} half failed: {exc}")
            return

    early, late = halves["early"], halves["late"]
    common = [i for i in early.instruments() if i in late.instruments()]
    if not common:
        print("   No instrument had enough data in both halves. Skipped.")
        return

    header = f"   {'instrument':<12}" + "".join(f"{f + ' early->late':>20}" for f in FACTORS)
    print(header)
    print("   " + "-" * (len(header) - 3))

    unstable: list[str] = []
    for inst in common:
        row = f"   {inst:<12}"
        for f in FACTORS:
            a, b = early.beta(inst, f), late.beta(inst, f)
            flag = " "
            if abs(a) > 0.2 or abs(b) > 0.2:
                if (a > 0) != (b > 0):
                    flag = "!"
                    unstable.append(f"{inst}/{f} FLIPPED SIGN {a:+.2f} -> {b:+.2f}")
                elif abs(a) > 1e-9 and not (0.5 <= abs(b) / abs(a) <= 2.0):
                    flag = "~"
                    unstable.append(f"{inst}/{f} changed >2x {a:+.2f} -> {b:+.2f}")
            row += f"{a:>9.2f} ->{b:>7.2f}{flag}"
        print(row)

    print("\n   ! = sign flip between halves     ~ = magnitude changed more than 2x")
    if unstable:
        print("\n   UNSTABLE BETAS — treat hedges driven by these with suspicion:")
        for u in unstable:
            print(f"     {u}")
        print("\n   Consider a SHORTER half-life so the live betas track the recent")
        print("   regime, and widen the caps for the affected factors.")
    else:
        print("\n   All material betas kept their sign and rough magnitude. Good.")


# ---------------------------------------------------------------------------
# 5. Hedge candidate quality
# ---------------------------------------------------------------------------


def hedge_purity_table(book, candidates: list[str]) -> None:
    _hr("5. HEDGE CANDIDATE QUALITY")
    print("   purity = share of the instrument's EXPLAINED variance from that factor,")
    print("   variance-weighted. A good hedge for factor F has HIGH |beta| to F and")
    print("   HIGH purity in F, so neutralizing F does not create exposure elsewhere.\n")

    header = f"   {'candidate':<12}" + "".join(f"{f + ' beta/pur':>19}" for f in FACTORS)
    print(header)
    print("   " + "-" * (len(header) - 3))
    for c in candidates:
        if not book.has(c):
            print(f"   {c:<12}  (no betas — fetch failed or too little history)")
            continue
        row = f"   {c:<12}"
        for f in FACTORS:
            row += f"{book.beta(c, f):>11.2f}/{book.purity(c, f):<7.2f}"
        print(row)

    print("\n   Best available lever per factor (score = purity^2 * |beta|):")
    for f in FACTORS:
        best = book.best_hedge_for(f, candidates)
        if best is None:
            print(f"     {f:<8} -> NO USABLE CANDIDATE")
            continue
        print(f"     {f:<8} -> {best:<10} beta {book.beta(best, f):+.2f}, "
              f"purity {book.purity(best, f):.2f}")

    print("\n   If a factor has no usable candidate, the overlay CANNOT hedge it.")
    print("   Either add a suitable instrument to HEDGE_CANDIDATES, or remove that")
    print("   factor's cap so the overlay does not repeatedly report an")
    print("   unactionable breach.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate factor betas for the overlay.")
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--halflife", type=float, default=252.0,
                    help="exponential weighting half-life in days; 0 => equal weight")
    ap.add_argument("--winsorize", type=float, default=0.005)
    ap.add_argument("--min-obs", type=int, default=250)
    ap.add_argument("--out", default=os.path.join("betas", "betas_latest.json"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--skip-stability", action="store_true")
    args = ap.parse_args()

    halflife = args.halflife if args.halflife and args.halflife > 0 else None

    # ---- fetch -----------------------------------------------------------
    _hr("FETCHING DAILY HISTORY")
    wanted: dict[str, str] = {}
    for f, (sym, _sign, _desc) in FACTOR_PROXIES.items():
        wanted[f"__factor_{f}"] = sym
    for name, (ysym, _b, _c) in TRADINGBOT_PORTFOLIO.items():
        wanted[name] = ysym
    for name, (ysym, _b, _f) in HEDGE_CANDIDATES.items():
        wanted.setdefault(name, ysym)

    try:
        dates, frame = align_price_frame(wanted, years=args.years,
                                         use_cache=not args.no_cache)
    except RuntimeError as exc:
        # Almost always a network problem, and a stack trace tells the user
        # nothing useful about it.
        print(f"\nFATAL: {exc}")
        print("\nEvery price fetch failed. Usual causes, in order of likelihood:")
        print("  1. No internet access, or a proxy/firewall blocking "
              "query1.finance.yahoo.com")
        print("  2. Yahoo rate-limiting this IP — wait a few minutes and retry")
        print("  3. Yahoo changed its chart API (the loader would need updating)")
        print("\nNothing was written, so any existing beta file is untouched.")
        print("\nTo confirm the maths itself is fine without any network:")
        print("    python -m pytest tests/ -q")
        print("The estimator is verified against synthetic data with known betas,")
        print("so a failure here is a DATA problem, not a model problem.")
        return 1

    if len(dates) < 2:
        print(f"\nFATAL: only {len(dates)} date(s) fetched. Nothing to estimate.")
        return 1

    print(f"\n  Common date axis: {len(dates)} days "
          f"({dates[0]} -> {dates[-1]})")

    factor_prices: dict[str, list] = {}
    for f, (_sym, sign, _desc) in FACTOR_PROXIES.items():
        col = frame.get(f"__factor_{f}")
        if col is None:
            continue
        # Apply the proxy sign so a POSITIVE factor return always means the
        # economically positive direction of the factor's name.
        factor_prices[f] = ([1.0 / p if (p and p > 0) else None for p in col]
                            if sign < 0 else col)

    missing = [f for f in ORTHOGONALIZATION_ORDER if f not in factor_prices]
    if missing:
        print(f"\nFATAL: factor proxy fetch failed for {missing}. Cannot build the "
              "factor model. Re-run, or edit FACTOR_PROXIES in "
              "src/factors/definitions.py.")
        return 1

    instrument_prices = {k: v for k, v in frame.items() if not k.startswith("__factor_")}

    # ---- returns ---------------------------------------------------------
    proxy_rets = returns_from_prices(factor_prices, args.winsorize)
    inst_rets = returns_from_prices(instrument_prices, args.winsorize)

    describe_concentration(inst_rets, list(TRADINGBOT_PORTFOLIO))

    # ---- factors ---------------------------------------------------------
    try:
        factors, kept = build_factors(proxy_rets, ORTHOGONALIZATION_ORDER, halflife)
    except ValueError as exc:
        print(f"\nFATAL: could not build the factor model: {exc}")
        print("\nThis means the factor proxies had too little OVERLAPPING history.")
        print("The factors must share one common sample, so a single proxy with a")
        print("short series truncates all of them. Try:")
        print("  * --years 15   (fetch more history)")
        print("  * --no-cache   (a stale/truncated cache file can cause this)")
        print("  * check which proxy is short in the fetch listing above")
        return 1
    _hr("2. FACTOR CONSTRUCTION")
    print(f"   observations={factors.n_obs}  order={' -> '.join(factors.names)}")
    print(f"   halflife={halflife}  winsorize={args.winsorize}")
    print(f"   orthogonality error (max |corr| off-diagonal) = "
          f"{factors.orthogonality_error():.2e}\n")
    print(f"   {'factor':<8}{'daily vol':>11}{'ann vol':>10}{'var retained':>14}")
    print("   " + "-" * 41)
    for f in factors.names:
        s = factors.sigma[f]
        print(f"   {f:<8}{s * 100:>10.3f}%{s * math.sqrt(252) * 100:>9.1f}%"
              f"{factors.variance_retained[f] * 100:>13.1f}%")

    print("\n   Raw proxy correlations BEFORE orthogonalization — the overlap the")
    print("   factor model removes:")
    names = list(factors.names)
    print("   " + " " * 9 + "".join(f"{n[:7]:>9}" for n in names))
    for a in names:
        row = f"   {a:<9}"
        for b in names:
            c = 1.0 if a == b else factors.raw_correlation.get((a, b), 0.0)
            row += f"{c:>9.2f}"
        print(row)

    print("\n   'var retained' is the share of each proxy's variance left after")
    print("   removing the earlier factors. A LOW number means that proxy was")
    print("   mostly explained by the ones before it — precisely the double-")
    print("   counting an unorthogonalized model would have hidden.")

    # ---- betas -----------------------------------------------------------
    try:
        book, skipped = estimate_betas(inst_rets, factors, kept, halflife, args.min_obs)
    except ValueError as exc:
        print(f"\nFATAL: could not estimate betas: {exc}")
        print(f"\nEvery instrument had fewer than --min-obs ({args.min_obs}) "
              "observations overlapping the factor sample.")
        print("Try --years 15, or lower --min-obs — but be aware that betas from a")
        print("few dozen observations are noise, and the overlay would size real")
        print("hedges from them.")
        return 1
    _hr("3. MEASURED BETAS")
    print(book.summary(priors=priors_for(book.instruments())))
    if skipped:
        print("\n   SKIPPED (too little overlapping history):")
        for k, v in sorted(skipped.items()):
            print(f"     {k:<10} {v} obs (need {args.min_obs})")
    print("\n   R2   = share of the instrument's variance explained by the 4 factors.")
    print("   idio = daily vol left unexplained. High idio is GOOD for")
    print("          diversification and BAD for hedgeability: the overlay cannot")
    print("          hedge what the factor model cannot see.")

    if not args.skip_stability:
        stability_check(inst_rets, proxy_rets, halflife, args.min_obs)

    hedge_purity_table(book, list(HEDGE_CANDIDATES))

    # ---- persist ---------------------------------------------------------
    meta = {
        "years": args.years,
        "halflife_days": halflife,
        "winsorize": args.winsorize,
        "min_obs": args.min_obs,
        "n_factor_obs": factors.n_obs,
        "date_range": [dates[0], dates[-1]],
        "orthogonalization_order": list(ORTHOGONALIZATION_ORDER),
        "factor_proxies": {f: s for f, (s, _sg, _d) in FACTOR_PROXIES.items()},
        "variance_retained": dict(factors.variance_retained),
        "orthogonality_error": factors.orthogonality_error(),
        "skipped": skipped,
    }
    save_betas(book, args.out, meta=meta)

    _hr("SAVED")
    print(f"  {args.out}")
    print(f"  {len(book.instruments())} instruments x {len(book.factors)} factors, "
          f"{factors.n_obs} factor observations")
    print("\n  Next: python scripts/measure_exposure.py --loop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
