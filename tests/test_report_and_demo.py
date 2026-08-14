"""
Smoke tests for the renderer and the demo pipeline.

These are not maths tests — they exist so a refactor cannot silently break the
one output a human actually reads, or the --demo path that is the only way to
inspect the tool without a live MT5 account.
"""

from __future__ import annotations

import json
import os

import pytest

from src.exposure.model import compute_exposure
from src.exposure.report import render, render_positions
from src.overlay.decision import hedge_decide

# Reuse the demo fixtures the script ships with, so the test breaks if they rot.
import scripts.measure_exposure as me  # noqa: E402


def build_demo():
    book = me.demo_book()
    positions, specs = me.demo_positions()
    candidates = me.demo_candidates()
    for h in candidates:
        specs.setdefault(h.symbol, h.spec)
    caps = me.caps_from_config({})
    return book, positions, specs, candidates, caps


def test_demo_fixtures_produce_a_concentrated_book():
    """The demo must actually demonstrate the problem, or it is useless as a
    preview of what observe mode will show."""
    book, positions, specs, _cands, _caps = build_demo()
    rep = compute_exposure(positions, specs, book, 10_000.0)

    assert len(rep.positions) == 8
    assert rep.unmapped_symbols == []
    assert 1.0 < rep.effective_bet_count < 6.0, (
        f"demo book reported {rep.effective_bet_count:.2f} effective bets; it should "
        "show meaningful but incomplete diversification"
    )
    assert rep.diversification_ratio > 1.0 / 8 ** 0.5


def test_demo_produces_an_actionable_plan():
    book, positions, specs, candidates, caps = build_demo()
    rep = compute_exposure(positions, specs, book, 10_000.0)
    plan = hedge_decide(rep, book, caps, candidates,
                        avoid_symbols=frozenset(specs) - {h.symbol for h in candidates})

    assert not plan.blocked
    assert plan.has_trades, f"demo should trigger a hedge, got: {plan.summary()}"
    a = plan.trades()[0]
    assert a.action == "open"
    assert a.factor in rep.factors
    assert a.lots > 0


def test_render_contains_the_headline_numbers():
    book, positions, specs, _c, caps = build_demo()
    rep = compute_exposure(positions, specs, book, 10_000.0)
    text = render(rep, caps.factor_caps, "USD")

    assert "FACTOR LEVERAGE" in text
    assert "EFFECTIVE INDEPENDENT BETS" in text
    assert "VARIANCE CONCENTRATION" in text
    assert "DIVERSIFICATION" in text
    for f in rep.factors:
        assert f in text
    assert "BREACH" in text, "the demo book should breach at least one cap"


def test_render_works_without_caps():
    book, positions, specs, _c, _caps = build_demo()
    rep = compute_exposure(positions, specs, book, 10_000.0)
    text = render(rep)
    assert "FACTOR LEVERAGE" in text
    assert "BREACH" not in text


def test_render_empty_book_does_not_crash():
    book, _p, _s, _c, caps = build_demo()
    rep = compute_exposure([], {}, book, 10_000.0)
    text = render(rep, caps.factor_caps, "USD")
    assert "Open positions              : 0" in text


def test_render_flags_unmapped_symbols_loudly():
    from src.exposure.model import InstrumentSpec, Position

    book, _p, _s, _c, caps = build_demo()
    spec = InstrumentSpec("WEIRD", 0.001, 1.0, logical="WEIRD")
    rep = compute_exposure([Position("WEIRD", 1, 1.0, 100.0, logical="WEIRD")],
                           {"WEIRD": spec}, book, 10_000.0)
    text = render(rep, caps.factor_caps, "USD")

    assert "WARNING" in text
    assert "WEIRD" in text


def test_render_positions_lists_every_position():
    book, positions, specs, _c, _caps = build_demo()
    rep = compute_exposure(positions, specs, book, 10_000.0)
    text = render_positions(rep)

    for p in positions:
        assert p.symbol in text
    assert "PER-POSITION DETAIL" in text


def test_append_history_writes_valid_jsonl(tmp_path):
    book, positions, specs, candidates, caps = build_demo()
    rep = compute_exposure(positions, specs, book, 10_000.0)
    plan = hedge_decide(rep, book, caps, candidates)
    path = os.path.join(str(tmp_path), "hist.jsonl")

    me.append_history(path, rep, plan, 10_000.0)
    me.append_history(path, rep, plan, 10_100.0)

    with open(path, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    assert len(rows) == 2
    assert rows[0]["n_positions"] == 8
    assert "USD" in rows[0]["leverage"]
    assert rows[1]["equity"] == 10_100.0
    assert isinstance(rows[0]["would_trade"], bool)


def test_caps_from_empty_config_are_valid_defaults():
    caps = me.caps_from_config({})
    assert caps.factor_caps
    assert caps.unwind_band <= caps.target_fraction


def test_caps_from_config_reads_values():
    caps = me.caps_from_config({
        "caps": {"USD": 2.0, "RISK": 3.0},
        "bands": {"hysteresis": 0.4, "target_fraction": 0.7, "unwind_band": 0.5},
        "limits": {"max_hedges_per_day": 2, "min_purity": 0.8},
    })
    assert caps.factor_caps == {"USD": 2.0, "RISK": 3.0}
    assert caps.hysteresis == pytest.approx(0.4)
    assert caps.max_hedges_per_day == 2
    assert caps.min_purity == pytest.approx(0.8)


EXAMPLE_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "hedge_config.example.yaml",
)


def test_example_config_defaults_are_safe_textually():
    """Runs even without PyYAML. Guards the settings that must never regress:
    the shipped example must default to observe mode, must not collide with
    Tradingbot's magic number, and must not collide with its lock file."""
    with open(EXAMPLE_CONFIG, "r", encoding="utf-8") as fh:
        text = fh.read()

    # Strip comments so a doc mention of 990011 does not trip the check.
    settings = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )

    assert "mode: observe" in settings
    assert "dry_run: true" in settings
    assert "magic_number: 990012" in settings
    assert "990011" not in settings, (
        "no setting may reuse Tradingbot's magic 990011"
    )
    assert "lock_path: logs/hedge.lock" in settings
    assert "bot.lock" not in settings, (
        "the lock path must not collide with Tradingbot's logs/bot.lock"
    )


def test_example_config_is_internally_consistent():
    """The shipped example must not encode a churning configuration."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config", "hedge_config.example.yaml")
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    caps = me.caps_from_config(cfg)          # raises if inconsistent
    assert caps.unwind_band <= caps.target_fraction
    assert cfg["mode"] == "observe", "the shipped example must default to observe"
    assert cfg["dry_run"] is True
    assert cfg["run"]["magic_number"] != 990011, (
        "hedge magic must differ from Tradingbot's 990011"
    )
    assert "hedge.lock" in cfg["run"]["lock_path"], (
        "lock path must differ from Tradingbot's logs/bot.lock"
    )
