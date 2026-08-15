"""
Tests for scripts/estimate_betas.py — specifically its FAILURE paths.

A monthly maintenance script that dumps a stack trace when the network hiccups is
a script people stop running. These tests pin the three ways it can legitimately
fail (no data, too little overlapping factor history, too little instrument
history) and assert each produces exit code 1 with an actionable message rather
than a traceback.

The happy path is covered by test_estimate_and_store.py::test_end_to_end_from_prices,
which exercises the same estimation code with synthetic prices built from known
betas.
"""

from __future__ import annotations

import math

import pytest

import scripts.estimate_betas as eb
from src.factors.definitions import FACTOR_PROXIES, ORTHOGONALIZATION_ORDER
from tests.test_math_core import LCG


def prices_from_returns(rets: list[float], start: float = 100.0) -> list[float]:
    px = [start]
    for r in rets:
        px.append(px[-1] * math.exp(r))
    return px


def make_frame(n: int, seed: int = 1, instruments=("GOLD", "EURUSD")):
    """A full fetch result: factor proxy columns plus instrument columns."""
    rng = LCG(seed)
    dates = [f"2020-{1 + (i // 28) % 12:02d}-{1 + i % 28:02d}" for i in range(n + 1)]
    frame: dict[str, list] = {}
    for f in FACTOR_PROXIES:
        frame[f"__factor_{f}"] = prices_from_returns(rng.normals(n, 0.0, 0.01))
    for inst in instruments:
        frame[inst] = prices_from_returns(rng.normals(n, 0.0, 0.012))
    return dates, frame


def run_with_frame(monkeypatch, dates, frame, argv: list[str]) -> int:
    """Run main() with align_price_frame stubbed out — no network."""
    monkeypatch.setattr(eb, "align_price_frame",
                        lambda *_a, **_k: (dates, frame))
    monkeypatch.setattr("sys.argv", ["estimate_betas.py", *argv])
    return eb.main()


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_total_fetch_failure_exits_cleanly(monkeypatch, capsys):
    """No traceback, exit 1, and a message that says what to do."""
    def boom(*_a, **_k):
        raise RuntimeError("no symbols fetched successfully")

    monkeypatch.setattr(eb, "align_price_frame", boom)
    monkeypatch.setattr("sys.argv", ["estimate_betas.py"])

    assert eb.main() == 1

    out = capsys.readouterr().out
    assert "FATAL" in out
    assert "query1.finance.yahoo.com" in out
    assert "any existing beta file is untouched" in out
    assert "pytest" in out, "should point the user at the offline verification"


def test_single_date_exits_cleanly(monkeypatch, capsys):
    assert run_with_frame(monkeypatch, ["2020-01-01"], {"GOLD": [100.0]}, []) == 1
    assert "Nothing to estimate" in capsys.readouterr().out


def test_missing_factor_proxy_exits_cleanly(monkeypatch, capsys):
    dates, frame = make_frame(400)
    del frame["__factor_METALS"]          # simulate one proxy failing to fetch

    assert run_with_frame(monkeypatch, dates, frame, ["--min-obs", "100"]) == 1

    out = capsys.readouterr().out
    assert "FATAL" in out
    assert "METALS" in out
    assert "FACTOR_PROXIES" in out


def test_too_little_factor_history_exits_cleanly(monkeypatch, capsys):
    """Fewer than 30 aligned factor observations is a hard stop."""
    dates, frame = make_frame(20)

    assert run_with_frame(monkeypatch, dates, frame, ["--min-obs", "5"]) == 1

    out = capsys.readouterr().out
    assert "could not build the factor model" in out
    assert "--years" in out


def test_instruments_all_too_short_exits_cleanly(monkeypatch, capsys):
    """Factors build fine, but no instrument clears --min-obs."""
    dates, frame = make_frame(200)

    assert run_with_frame(monkeypatch, dates, frame, ["--min-obs", "5000"]) == 1

    out = capsys.readouterr().out
    assert "could not estimate betas" in out
    assert "noise" in out, "must warn that lowering min-obs buys noise"


# ---------------------------------------------------------------------------
# Happy path through the script itself
# ---------------------------------------------------------------------------


def test_full_run_writes_a_loadable_beta_file(monkeypatch, capsys, tmp_path):
    """End-to-end through main(): stubbed fetch in, usable beta file out."""
    import os

    from src.factors.store import load_betas

    dates, frame = make_frame(900, seed=31,
                              instruments=("GOLD", "SILVER", "EURUSD", "SP500"))
    out_path = os.path.join(str(tmp_path), "betas.json")

    rc = run_with_frame(monkeypatch, dates, frame,
                        ["--min-obs", "200", "--out", out_path])
    assert rc == 0

    text = capsys.readouterr().out
    # The five report sections must all be present.
    assert "RAW CORRELATION OF THE TRADED BOOK" in text
    assert "FACTOR CONSTRUCTION" in text
    assert "MEASURED BETAS" in text
    assert "SPLIT-SAMPLE BETA STABILITY" in text
    assert "HEDGE CANDIDATE QUALITY" in text
    assert "Effective independent bets" in text

    book, meta = load_betas(out_path, max_age_days=1)
    assert set(book.instruments()) >= {"GOLD", "SILVER", "EURUSD"}
    assert book.factors == tuple(ORTHOGONALIZATION_ORDER)
    assert book.factor_sigmas(), "factor sigmas must be persisted with the betas"
    assert meta["orthogonality_error"] < 1e-9
    assert meta["date_range"][0] == dates[0]


def test_skip_stability_flag_omits_that_section(monkeypatch, capsys, tmp_path):
    import os

    dates, frame = make_frame(900, seed=12)
    rc = run_with_frame(
        monkeypatch, dates, frame,
        ["--min-obs", "200", "--skip-stability",
         "--out", os.path.join(str(tmp_path), "b.json")],
    )
    assert rc == 0
    assert "SPLIT-SAMPLE BETA STABILITY" not in capsys.readouterr().out


def test_equal_weight_mode_is_reachable(monkeypatch, capsys, tmp_path):
    """--halflife 0 must mean equal weighting, not a divide-by-zero."""
    import os

    from src.factors.store import load_betas

    dates, frame = make_frame(900, seed=77)
    out_path = os.path.join(str(tmp_path), "b.json")

    assert run_with_frame(monkeypatch, dates, frame,
                          ["--min-obs", "200", "--halflife", "0",
                           "--out", out_path]) == 0

    _book, meta = load_betas(out_path, max_age_days=1)
    assert meta["halflife_days"] is None
    assert "halflife=None" in capsys.readouterr().out


def test_inverted_proxy_sign_is_applied(monkeypatch):
    """Any FACTOR_PROXIES entry with sign=-1 must be inverted before use.

    Nothing ships with sign=-1 today (DXY already rises as the dollar rises), so
    this guards the mechanism against a future proxy swap — e.g. using EURUSD as
    the dollar proxy, where forgetting the inversion would flip every USD beta and
    make the overlay hedge in exactly the wrong direction.
    """
    import os
    import tempfile

    from src.factors.store import load_betas

    dates, frame = make_frame(900, seed=5, instruments=("GOLD",))

    with tempfile.TemporaryDirectory() as td:
        results = {}
        for sign in (+1, -1):
            patched = dict(FACTOR_PROXIES)
            sym, _old, desc = patched["USD"]
            patched["USD"] = (sym, sign, desc)
            monkeypatch.setattr(eb, "FACTOR_PROXIES", patched)

            out = os.path.join(td, f"b{sign}.json")
            monkeypatch.setattr(eb, "align_price_frame",
                                lambda *_a, **_k: (dates, {k: list(v)
                                                           for k, v in frame.items()}))
            monkeypatch.setattr("sys.argv",
                                ["estimate_betas.py", "--min-obs", "200",
                                 "--skip-stability", "--out", out])
            assert eb.main() == 0
            book, _m = load_betas(out, max_age_days=1)
            results[sign] = book.beta("GOLD", "USD")

        assert results[+1] == pytest.approx(-results[-1], rel=1e-6), (
            "inverting the USD proxy must invert the sign of every USD beta"
        )



# ---------------------------------------------------------------------------
# Fetch deduplication
# ---------------------------------------------------------------------------


def test_align_price_frame_fetches_each_symbol_once(monkeypatch):
    """GC=F is both the METALS proxy and the GOLD instrument; CL=F is both ENERGY
    and WTI. Yahoo rate-limits by IP, so each distinct symbol must be fetched
    exactly once and shared across every column that wants it."""
    from src.data import yahoo

    calls: list[str] = []

    def fake_fetch(symbol, **_kwargs):
        calls.append(symbol)
        return yahoo.Bars(symbol, ["2020-01-01", "2020-01-02"], [10.0, 11.0])

    monkeypatch.setattr(yahoo, "fetch_daily", fake_fetch)

    dates, frame = yahoo.align_price_frame(
        {
            "__factor_METALS": "GC=F",
            "GOLD": "GC=F",
            "__factor_ENERGY": "CL=F",
            "WTI": "CL=F",
            "EURUSD": "EURUSD=X",
        },
        verbose=False,
    )

    assert sorted(calls) == ["CL=F", "EURUSD=X", "GC=F"], (
        f"expected 3 distinct fetches, got {calls}"
    )
    # Every requested column is still present and populated.
    assert set(frame) == {"__factor_METALS", "GOLD", "__factor_ENERGY", "WTI", "EURUSD"}
    assert frame["GOLD"] == frame["__factor_METALS"] == [10.0, 11.0]
    assert dates == ["2020-01-01", "2020-01-02"]


def test_align_price_frame_omits_failed_symbols_without_aborting(monkeypatch):
    """One dead hedge candidate must not kill the whole study."""
    from src.data import yahoo

    def fake_fetch(symbol, **_kwargs):
        if symbol == "BAD=X":
            raise RuntimeError("nope")
        return yahoo.Bars(symbol, ["2020-01-01"], [10.0])

    monkeypatch.setattr(yahoo, "fetch_daily", fake_fetch)

    _dates, frame = yahoo.align_price_frame(
        {"GOOD": "GOOD=X", "BROKEN": "BAD=X"}, verbose=False
    )

    assert "GOOD" in frame
    assert "BROKEN" not in frame


def test_align_price_frame_raises_when_everything_fails(monkeypatch):
    from src.data import yahoo

    def fake_fetch(symbol, **_kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(yahoo, "fetch_daily", fake_fetch)

    with pytest.raises(RuntimeError, match="no symbols fetched successfully"):
        yahoo.align_price_frame({"A": "A=X"}, verbose=False)


def test_align_price_frame_unions_mismatched_date_axes(monkeypatch):
    """Instruments trade on different calendars (futures vs FX vs indices). The
    frame must be the UNION of dates with None for gaps, so the caller decides
    how to align rather than losing rows silently."""
    from src.data import yahoo

    series = {
        "A=X": (["2020-01-01", "2020-01-02"], [1.0, 2.0]),
        "B=X": (["2020-01-02", "2020-01-03"], [3.0, 4.0]),
    }

    monkeypatch.setattr(
        yahoo, "fetch_daily",
        lambda symbol, **_k: yahoo.Bars(symbol, *series[symbol]),
    )

    dates, frame = yahoo.align_price_frame({"A": "A=X", "B": "B=X"}, verbose=False)

    assert dates == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert frame["A"] == [1.0, 2.0, None]
    assert frame["B"] == [None, 3.0, 4.0]



# ---------------------------------------------------------------------------
# Alignment gate
# ---------------------------------------------------------------------------


def _misaligned_frame(n: int = 900):
    """A frame where the instrument's labels are one day later than the factors."""
    from datetime import date, timedelta

    rng = LCG(808)

    def bdays(start: str, count: int) -> list[str]:
        d = date.fromisoformat(start)
        out: list[str] = []
        while len(out) < count:
            if d.weekday() < 5:
                out.append(d.isoformat())
            d += timedelta(days=1)
        return out

    factor_days = bdays("2019-01-01", n)
    shifted_days = [
        (date.fromisoformat(d) + timedelta(days=1)).isoformat() for d in factor_days
    ]
    combined = sorted(set(factor_days) | set(shifted_days))
    idx = {d: i for i, d in enumerate(combined)}

    frame: dict[str, list] = {}
    for f in FACTOR_PROXIES:
        col = [None] * len(combined)
        px = prices_from_returns(rng.normals(n - 1, 0.0, 0.01))
        for i, d in enumerate(factor_days):
            col[idx[d]] = px[i]
        frame[f"__factor_{f}"] = col

    inst = [None] * len(combined)
    px = prices_from_returns(rng.normals(n - 1, 0.0, 0.012))
    for i, d in enumerate(shifted_days):
        inst[idx[d]] = px[i]
    frame["GOLD"] = inst

    return combined, frame


def test_misaligned_data_stops_the_study(monkeypatch, capsys, tmp_path):
    """Refuse to write betas from data whose dates do not line up."""
    import os

    dates, frame = _misaligned_frame()
    out = os.path.join(str(tmp_path), "betas.json")

    rc = run_with_frame(monkeypatch, dates, frame,
                        ["--min-obs", "100", "--out", out])

    assert rc == 1
    text = capsys.readouterr().out
    assert "MISALIGNED" in text
    assert "STOPPING" in text
    assert "--align-to-reference" in text
    assert not os.path.exists(out), "no beta file may be written from bad data"


def test_ignore_misalignment_proceeds_but_is_labelled_dangerous(monkeypatch, capsys,
                                                                tmp_path):
    import os

    dates, frame = _misaligned_frame()
    out = os.path.join(str(tmp_path), "betas.json")

    rc = run_with_frame(monkeypatch, dates, frame,
                        ["--min-obs", "100", "--ignore-misalignment", "--out", out])

    assert rc == 0
    assert "MISALIGNED" in capsys.readouterr().out
    assert os.path.exists(out)


def test_align_to_reference_is_not_blocked_by_the_gate(monkeypatch, capsys, tmp_path):
    """--align-to-reference IS the fix, so the gate demanding a fix must not
    block it. This was a real ordering bug: the early return fired first and the
    flag could never take effect."""
    import os

    dates, frame = _misaligned_frame()
    out = os.path.join(str(tmp_path), "betas.json")

    rc = run_with_frame(monkeypatch, dates, frame,
                        ["--min-obs", "100", "--align-to-reference", "--out", out])

    assert rc == 0, "aligning must be allowed to proceed"
    text = capsys.readouterr().out
    assert "Applied label offsets" in text
    assert "prices unchanged" in text
    assert os.path.exists(out)


def test_align_to_reference_restores_overlap(monkeypatch, capsys, tmp_path):
    """After aligning, the instrument must be estimated on far more observations."""
    import os

    from src.factors.store import load_betas

    dates, frame = _misaligned_frame()

    out_bad = os.path.join(str(tmp_path), "bad.json")
    run_with_frame(monkeypatch, dates, {k: list(v) for k, v in frame.items()},
                   ["--min-obs", "50", "--ignore-misalignment", "--out", out_bad])
    bad, _ = load_betas(out_bad, max_age_days=1)

    out_fixed = os.path.join(str(tmp_path), "fixed.json")
    run_with_frame(monkeypatch, dates, {k: list(v) for k, v in frame.items()},
                   ["--min-obs", "50", "--align-to-reference", "--out", out_fixed])
    fixed, _ = load_betas(out_fixed, max_age_days=1)

    # Aligning must reach the FULL factor sample, not merely more of it.
    assert fixed.n_obs("GOLD") > bad.n_obs("GOLD"), (
        f"aligning should recover observations: "
        f"{bad.n_obs('GOLD')} -> {fixed.n_obs('GOLD')}"
    )

    # Note how MILD the observation loss is: a one-weekday shift still lands
    # Mon->Tue, Tue->Wed, Wed->Thu, Thu->Fri on valid weekdays, so only the
    # Friday->Saturday roll is dropped. Roughly 75% of observations survive.
    #
    # That is precisely what makes this bug dangerous. It does not announce
    # itself as missing data -- the row counts look fine. The damage is that the
    # surviving observations are MISPAIRED: one day's instrument return regressed
    # against the next day's factor. R^2 collapses while everything else looks
    # healthy. See test_alignment_diagnostics.test_reproduces_the_live_fx_symptom
    # for the R^2 > 0.99 -> < 0.15 demonstration.
    assert bad.n_obs("GOLD") > 0.5 * fixed.n_obs("GOLD"), (
        "documents that a shift loses only a minority of observations, which is "
        "why row counts alone cannot detect it"
    )


def test_circularity_warning_is_printed(monkeypatch, capsys, tmp_path):
    """GOLD shares GC=F with the METALS proxy in the shipped definitions, which
    makes its R2 definitional rather than measured. The user must be told."""
    import os

    dates, frame = make_frame(900, seed=44, instruments=("GOLD", "EURUSD"))
    rc = run_with_frame(monkeypatch, dates, frame,
                        ["--min-obs", "200", "--skip-stability",
                         "--out", os.path.join(str(tmp_path), "b.json")])

    assert rc == 0
    text = capsys.readouterr().out
    assert "CIRCULARITY WARNING" in text
    assert "GOLD" in text and "METALS" in text
