"""
Tests for the Tradingbot heartbeat / same-account check in preflight.

This check exists because of a real deployment mistake. Hedgingbot was installed
on a box whose MT5 terminal happened to be logged into the same demo account the
user had manually traded on. Preflight came back 26/26 green. But Tradingbot's
book was not there — the only visible positions were four manual gold trades with
magic 0.

Everything looked correct while the overlay measured the wrong book. That is worse
than a crash: the output is authoritative-looking and describes something else.
These tests pin each branch so the check cannot silently regress.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

import scripts.preflight as pf
from src.exposure.model import Position


class FakeAccount:
    def __init__(self, login: int) -> None:
        self.login = login


class FakeReader:
    """Minimal stand-in for MT5Reader — only what the check touches."""

    def __init__(self, login: int = 1100219238, positions=None) -> None:
        self._login = login
        self._positions = list(positions or [])

    def account(self) -> FakeAccount:
        return FakeAccount(self._login)

    def all_positions(self) -> list[Position]:
        return list(self._positions)


def write_status(path: str, **fields) -> str:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "login": 1100219238,
        "equity": 10_502.26,
        "halted": False,
        "open_positions": [],
    }
    payload.update(fields)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def run_check(cfg: dict, reader) -> pf.Checks:
    c = pf.Checks()
    pf._check_trader_heartbeat(c, cfg, reader)
    return c


def statuses(c: pf.Checks) -> list[tuple[str, str]]:
    return [(s, n) for s, n, _d in c.rows]


def find(c: pf.Checks, needle: str):
    return [(s, n, d) for s, n, d in c.rows if needle.lower() in (n + d).lower()]


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------


def test_relative_path_is_a_hard_failure():
    """The shipped default was `logs/status.json`, which resolves against
    Hedgingbot's own directory and finds nothing forever."""
    c = run_check({"run": {"trader_status_path": "logs/status.json"}}, FakeReader())

    assert pf.FAIL in [s for s, _n in statuses(c)]
    hit = find(c, "RELATIVE")
    assert hit, "must name the problem explicitly"
    assert "absolute" in hit[0][2].lower()


def test_missing_path_warns():
    c = run_check({"run": {}}, FakeReader())
    assert pf.WARN in [s for s, _n in statuses(c)]
    assert find(c, "trader_status_path")


def test_absent_file_is_a_failure_with_the_two_causes(tmp_path):
    path = os.path.join(str(tmp_path), "nope", "status.json")
    c = run_check({"run": {"trader_status_path": path}}, FakeReader())

    hit = find(c, "No Tradingbot heartbeat")
    assert hit
    assert "not running" in hit[0][2]


def test_windows_drive_letter_counts_as_absolute(tmp_path):
    """C:/Tradingbot/... is absolute on Windows but os.path.isabs is False when
    the test runs on Linux, so the check also accepts a drive-letter prefix.
    Without that, a correct Windows config would be reported as broken."""
    c = run_check({"run": {"trader_status_path": "C:/Tradingbot/logs/status.json"}},
                  FakeReader())
    assert not find(c, "RELATIVE"), "a drive-letter path must not be called relative"


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


def test_fresh_heartbeat_passes(tmp_path):
    path = write_status(os.path.join(str(tmp_path), "status.json"))
    c = run_check({"run": {"trader_status_path": path}}, FakeReader())

    hit = find(c, "heartbeat found")
    assert hit and hit[0][0] == pf.PASS
    assert "min ago" in hit[0][2]


def test_stale_heartbeat_warns_and_says_the_overlay_keeps_working(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(
        timespec="seconds")
    path = write_status(os.path.join(str(tmp_path), "status.json"), updated_at=old)

    c = run_check({"run": {"trader_status_path": path}}, FakeReader())

    hit = find(c, "STALE")
    assert hit and hit[0][0] == pf.WARN
    # The overlay must keep measuring even when the trader is dead: broker SL/TP
    # stay live, so open risk does not disappear with the process.
    assert "broker SL/TP stay live" in hit[0][2]


def test_unparseable_timestamp_does_not_crash(tmp_path):
    path = write_status(os.path.join(str(tmp_path), "status.json"),
                        updated_at="not-a-timestamp")
    c = run_check({"run": {"trader_status_path": path}}, FakeReader())
    assert find(c, "heartbeat found")


# ---------------------------------------------------------------------------
# THE check: same account?
# ---------------------------------------------------------------------------


def test_different_accounts_is_a_hard_failure(tmp_path):
    """The failure mode this whole module exists for."""
    path = write_status(os.path.join(str(tmp_path), "status.json"), login=999999)

    c = run_check({"run": {"trader_status_path": path}},
                  FakeReader(login=1100219238))

    hit = find(c, "DIFFERENT ACCOUNTS")
    assert hit and hit[0][0] == pf.FAIL
    assert "999999" in hit[0][1] and "1100219238" in hit[0][1]
    assert "wrong book" in hit[0][2]


def test_same_account_passes(tmp_path):
    path = write_status(os.path.join(str(tmp_path), "status.json"),
                        login=1100219238)
    c = run_check({"run": {"trader_status_path": path}},
                  FakeReader(login=1100219238))

    hit = find(c, "Both bots are on account")
    assert hit and hit[0][0] == pf.PASS


def test_missing_login_field_warns_rather_than_assuming(tmp_path):
    path = os.path.join(str(tmp_path), "status.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"updated_at": datetime.now(timezone.utc).isoformat(),
                   "equity": 1.0}, fh)

    c = run_check({"run": {"trader_status_path": path}}, FakeReader())
    assert find(c, "no login field")


# ---------------------------------------------------------------------------
# Whose positions are we actually looking at?
# ---------------------------------------------------------------------------


MANUAL = [Position("XAUUSD.ecn", -1, 0.25, 4375.96, ticket=t, magic=0)
          for t in (1, 2, 3, 4)]
TRADER = [Position("XAUUSD.ecn", 1, 0.05, 4375.96, ticket=10, magic=990011),
          Position("AUDUSD.ecn", 1, 0.10, 0.708, ticket=11, magic=990011)]
OWN_HEDGE = [Position("EURUSD.ecn", 1, 0.10, 1.157, ticket=20, magic=990012)]


def test_only_manual_positions_warns_that_caps_would_be_miscalibrated(tmp_path):
    """Exactly the live situation: magics=[0], no trader book."""
    path = write_status(os.path.join(str(tmp_path), "status.json"))
    c = run_check({"run": {"trader_status_path": path, "magic_number": 990012}},
                  FakeReader(positions=MANUAL))

    hit = find(c, "Only manual positions")
    assert hit and hit[0][0] == pf.WARN
    assert "MANUAL" in hit[0][2]
    assert "wrong thing" in hit[0][2], (
        "must warn that calibrating caps from this is calibrating to the wrong book"
    )


def test_trader_positions_are_recognised(tmp_path):
    path = write_status(os.path.join(str(tmp_path), "status.json"))
    c = run_check({"run": {"trader_status_path": path, "magic_number": 990012}},
                  FakeReader(positions=TRADER + MANUAL))

    hit = find(c, "bot magic")
    assert hit and hit[0][0] == pf.PASS
    assert "990011" in hit[0][1]


def test_the_overlays_own_hedges_are_not_mistaken_for_the_traders(tmp_path):
    """magic 990012 is ours. Counting it as the trader's book would make the
    overlay believe there is something to protect when there is not."""
    path = write_status(os.path.join(str(tmp_path), "status.json"))
    c = run_check({"run": {"trader_status_path": path, "magic_number": 990012}},
                  FakeReader(positions=OWN_HEDGE))

    assert not find(c, "bot magic"), "own hedges must not count as the trader's book"
    assert find(c, "No open positions at all") or find(c, "Only manual")


def test_empty_book_warns_but_does_not_fail(tmp_path):
    path = write_status(os.path.join(str(tmp_path), "status.json"))
    c = run_check({"run": {"trader_status_path": path}},
                  FakeReader(positions=[]))

    hit = find(c, "No open positions at all")
    assert hit and hit[0][0] == pf.WARN
    assert c.failed == 0


def test_no_reader_skips_the_account_comparison_gracefully(tmp_path):
    """On a machine without MT5 the heartbeat can still be read; the account
    comparison simply cannot be done."""
    path = write_status(os.path.join(str(tmp_path), "status.json"))
    c = run_check({"run": {"trader_status_path": path}}, None)

    assert find(c, "heartbeat found")
    assert not find(c, "Both bots are on account")
    assert not find(c, "DIFFERENT ACCOUNTS")
