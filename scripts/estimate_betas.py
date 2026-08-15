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

from src.data.diagnostics import (                                       # noqa: E402
    diagnose_alignment,
    find_circular_instruments,
)
from src.data.diagnostics import (                                       # noqa: E402
    render_synchronicity,
    synchronicity_profile,
)
from src.data.yahoo import align_price_frame, to_period                  # noqa: E402
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


def _apply_offsets(dates, frame, report) -> dict:
    """Re-file each misaligned series onto the reference calendar, in place.

    Only the date LABEL a price is filed under changes; no price is altered,
    invented or interpolated. This corrects a source-metadata artefact — Yahoo
    timestamps FX and futures bars on different session boundaries — rather than
    massaging data to fit.

    Returns {series_name: applied_offset}.
    """
    from datetime import date as _date
    from datetime import timedelta as _timedelta

    index = {d: i for i, d in enumerate(dates)}
    applied: dict = {}

    for s in report.misaligned():
        column = frame.get(s.name)
        if column is None:
            continue
        shifted = [None] * len(dates)
        delta = _timedelta(days=s.best_offset)
        for i, value in enumerate(column):
            if value is None:
                continue
            try:
                target = (_date.fromisoformat(dates[i]) + delta).isoformat()
            except ValueError:
                continue
            j = index.get(target)
            if j is not None:
                shifted[j] = value
        frame[s.name] = shifted
        applied[s.name] = s.best_offset

    return applied


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

    # Apply the SAME thresholds the overlay enforces (HedgeCaps defaults).
    # Without them this table happily reported "RISK -> EURUSD beta +0.01,
    # purity 0.00" as the best lever — technically the highest score, but an
    # instrument the overlay would reject outright. A report that disagrees with
    # the thing it is reporting on is worse than no report.
    from src.overlay.decision import HedgeCaps

    defaults = HedgeCaps(factor_caps={"USD": 1.0})
    min_p, min_b = defaults.min_purity, defaults.min_abs_beta

    print(f"\n   Best available lever per factor (score = purity^2 * |beta|),")
    print(f"   applying the overlay's own thresholds: purity >= {min_p}, "
          f"|beta| >= {min_b}")
    for f in FACTORS:
        best = book.best_hedge_for(f, candidates, min_purity=min_p,
                                   min_abs_beta=min_b)
        if best is None:
            near = book.best_hedge_for(f, candidates)
            extra = ""
            if near:
                extra = (f"  (closest was {near}: beta "
                         f"{book.beta(near, f):+.2f}, purity "
                         f"{book.purity(near, f):.2f} — rejected)")
            print(f"     {f:<8} -> NO USABLE CANDIDATE{extra}")
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
    ap.add_argument("--halflife", type=float, default=None,
                    help="exponential weighting half-life, in PERIODS; 0 => equal "
                         "weight. Default scales with --return-period (252 daily, "
                         "~50 weekly)")
    ap.add_argument("--winsorize", type=float, default=0.005)
    ap.add_argument("--min-obs", type=int, default=None,
                    help="minimum observations per instrument, in PERIODS. Default "
                         "scales with --return-period (250 daily, 50 weekly)")
    ap.add_argument("--out", default=os.path.join("betas", "betas_latest.json"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--skip-stability", action="store_true")
    ap.add_argument("--max-offset", type=int, default=2,
                    help="how many days of date-label shift to test for misalignment")
    ap.add_argument("--min-align-gain", type=float, default=0.10,
                    help="relative overlap gain from a shift before calling a series "
                         "misaligned")
    ap.add_argument("--align-to-reference", action="store_true",
                    help="re-file misaligned series onto the factor calendar. Changes "
                         "only the date label a price is filed under, never a price.")
    ap.add_argument("--ignore-misalignment", action="store_true",
                    help="write betas even from misaligned data. Do not use this "
                         "unless you know exactly why.")
    ap.add_argument("--return-period", default="daily",
                    choices=("daily", "weekly", "biweekly", "monthly"),
                    help="return horizon for beta estimation. Use weekly when the "
                         "synchronicity report says the series are priced at "
                         "different moments within the day (default: daily)")
    args = ap.parse_args()

    halflife = args.halflife if args.halflife and args.halflife > 0 else None

    # Trading days per return period, used for both annualization and the
    # daily-equivalent sigma conversion.
    period_days = {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21}[
        args.return_period
    ]

    # --min-obs and --halflife are expressed in PERIODS, so their daily defaults
    # are wrong for a weekly run. 250 weekly observations is five years, and the
    # stability check (which wants 4x min_obs) then becomes impossible: an 8-year
    # sample is only ~417 weeks. Scale the defaults unless explicitly overridden.
    if args.min_obs is None:
        args.min_obs = max(30, int(round(250 / period_days)))
        if period_days > 1:
            print(f"  --min-obs defaulted to {args.min_obs} "
                  f"({args.return_period} periods, scaled from 250 daily)")
    if args.halflife is None:
        halflife = max(20.0, 252.0 / period_days)
    else:
        halflife = args.halflife if args.halflife > 0 else None

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

    # ---- data integrity, BEFORE any maths -------------------------------
    # Betas from misaligned dates look plausible and are worthless, so this runs
    # first and can stop the whole study.
    ref_col = f"__factor_{ORTHOGONALIZATION_ORDER[0]}"
    if ref_col in frame:
        _hr("0. DATA ALIGNMENT CHECK")
        align_report = diagnose_alignment(
            dates, frame, reference=ref_col,
            max_offset=args.max_offset,
            min_gain=args.min_align_gain,
        )
        print(align_report.render())

        # --align-to-reference is the FIX, so it must not be blocked by the gate
        # that exists to demand a fix.
        if (align_report.misaligned() and not args.ignore_misalignment
                and not args.align_to_reference):
            print("\n" + "=" * 78)
            print("STOPPING: refusing to write betas from misaligned data.")
            print("=" * 78)
            print("\nThe series above are filed under date labels that disagree with")
            print("the factor calendar by a day. Regressing them pairs one day's")
            print("return against another day's factor, which drives R^2 toward 0 and")
            print("betas toward 0 — and the output looks perfectly reasonable, which")
            print("is what makes it dangerous.")
            print("\nOptions:")
            print("  --align-to-reference   snap every series onto the factor calendar")
            print("                         by applying its best offset (recommended;")
            print("                         it corrects a labelling artefact, it does")
            print("                         not alter any price)")
            print("  --ignore-misalignment  proceed anyway and DO NOT trust the result")
            print("\nNothing was written. Any existing beta file is untouched.")
            return 1

        if args.align_to_reference:
            fixed = _apply_offsets(dates, frame, align_report)
            if fixed:
                print(f"\n  Applied label offsets: "
                      + ", ".join(f"{n}{o:+d}" for n, o in fixed.items()))
                print("  (prices unchanged — only the date each price is filed under)")

    # ---- circularity: is a traded instrument its own factor proxy? -------
    circular = find_circular_instruments(
        {f: sym for f, (sym, _s, _d) in FACTOR_PROXIES.items()},
        {name: ysym for name, (ysym, _b, _c) in TRADINGBOT_PORTFOLIO.items()},
    )
    if circular:
        print("\n  *** CIRCULARITY WARNING ***")
        for inst, fac in sorted(circular.items()):
            print(f"    {inst} uses the SAME price series as the {fac} factor proxy")
        print("    These instruments are regressed partly on THEMSELVES, so they")
        print("    will report R2 ~ 1.00 and idiosyncratic vol ~ 0.00. That is an")
        print("    artefact, not a measurement: the risk model will believe they")
        print("    have no unexplained risk and are perfectly hedgeable. Treat")
        print("    their betas as definitional rather than estimated.")

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

    # ---- optional return-period resampling -------------------------------
    if args.return_period != "daily":
        merged = dict(factor_prices)
        merged.update(instrument_prices)
        dates, merged = to_period(dates, merged, args.return_period)
        factor_prices = {k: merged[k] for k in factor_prices}
        instrument_prices = {k: merged[k] for k in instrument_prices}
        print(f"\n  Resampled to {args.return_period}: {len(dates)} periods "
              f"({dates[0]} -> {dates[-1]})")
        print("  Longer periods make an intraday snapshot-time offset negligible,")
        print("  at the cost of sample size. Use this when the daily betas are")
        print("  BIASED (see the synchronicity report), not merely noisy.")

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
    # Annualize with the ACTUAL number of periods per year. Hardcoding 252 here
    # printed ENERGY at 94.8% annual vol on weekly data — sqrt(252) applied to a
    # weekly sigma, i.e. 2.2x too big.
    per_year = 252.0 / float(period_days)
    label = f"{args.return_period} vol"
    print(f"   {'factor':<8}{label:>13}{'ann vol':>10}{'var retained':>14}")
    print("   " + "-" * 45)
    for f in factors.names:
        s = factors.sigma[f]
        print(f"   {f:<8}{s * 100:>12.3f}%{s * math.sqrt(per_year) * 100:>9.1f}%"
              f"{factors.variance_retained[f] * 100:>13.1f}%")
    if period_days > 1:
        print(f"\n   Volatilities above are per {args.return_period} period. They are")
        print(f"   converted to DAILY-equivalent (divided by sqrt({period_days})) before")
        print("   being saved, because the exposure model consumes daily sigmas.")

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

    # ---- synchronicity ---------------------------------------------------
    # Runs against the FIRST factor, which is both the reference calendar and
    # where the live anomaly appeared. Date alignment can pass while this fails:
    # same day, different moment.
    _hr("2b. SYNCHRONICITY CHECK")
    ref_factor = factors.names[0]
    profiles = []
    fser = factors.series[ref_factor]
    for name in sorted(inst_rets):
        series = inst_rets[name]
        y: list = []
        x: list = []
        for pos, row in enumerate(kept):
            if row >= len(series):
                continue
            v = series[row]
            if v is None or not math.isfinite(v):
                continue
            y.append(float(v))
            x.append(fser[pos])
        if len(y) < 100:
            continue
        profiles.append(synchronicity_profile(y, x, 2, name, ref_factor))
    print(render_synchronicity(profiles))
    if args.return_period == "daily":
        print("\n   Caveat: lags here are counted in SURVIVING observations, not")
        print("   calendar days, so for a series with gaps a 'lag 1' may span more")
        print("   than one day. It is a strong signal, not a precise measurement.")

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
        "return_period": args.return_period,
        "period_days": period_days,
        "sigma_basis": "daily",
        "n_factor_obs": factors.n_obs,
        "date_range": [dates[0], dates[-1]],
        "orthogonalization_order": list(ORTHOGONALIZATION_ORDER),
        "factor_proxies": {f: s for f, (s, _sg, _d) in FACTOR_PROXIES.items()},
        "variance_retained": dict(factors.variance_retained),
        "orthogonality_error": factors.orthogonality_error(),
        "skipped": skipped,
    }
    # Convert volatilities to DAILY-equivalent before persisting. Betas are ~
    # horizon-invariant and are left alone; sigmas are not. The exposure model
    # consumes daily sigmas (factor_daily_risk, portfolio_daily_risk), so saving
    # weekly sigmas would silently inflate every reported risk figure by sqrt(5).
    to_save = book
    if period_days > 1:
        to_save = book.scale_sigmas(1.0 / math.sqrt(period_days))
        print(f"\n  Volatilities converted to daily-equivalent "
              f"(divided by sqrt({period_days})) so the exposure model reads them "
              f"correctly.")

    save_betas(to_save, args.out, meta=meta)

    _hr("SAVED")
    print(f"  {args.out}")
    print(f"  {len(book.instruments())} instruments x {len(book.factors)} factors, "
          f"{factors.n_obs} factor observations")
    print("\n  Next: python scripts/measure_exposure.py --loop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
