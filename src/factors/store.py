"""
Persist and reload a BetaBook, with a staleness guard.

Why staleness is ENFORCED rather than warned about
--------------------------------------------------
Betas are the only thing standing between "hedge 0.4 lots" and "hedge 4 lots".
They are estimated from history and they decay: gold's dollar beta in 2022 is not
gold's dollar beta in 2019. A live overlay sizing hedges from a six-month-old
beta file is not a risk system, it is a random number generator with good
manners. So `load_betas` RAISES on a stale file by default; the caller has to
pass allow_stale=True to consciously override it.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

from .book import BetaBook


class StaleBetasError(RuntimeError):
    """A beta file is older than the caller's tolerance."""


def _write_atomic(path: str, payload: str) -> None:
    """Temp file in the SAME directory, then os.replace.

    Same directory matters — os.replace is only atomic within one filesystem.
    A reader therefore never observes a partially written beta file.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_betas_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_betas(book: BetaBook, path: str, *, meta: dict | None = None) -> None:
    """Write a BetaBook to JSON.

    The factor sigmas travel INSIDE the book. That is not incidental: betas alone
    cannot produce a risk decomposition, and pairing betas from one estimation run
    with sigmas from another is a silent, plausible-looking error that would
    mis-size every hedge. Keeping them in one file makes that mistake impossible.
    """
    payload = book.to_dict()
    payload["schema"] = 1
    payload["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["meta"] = dict(meta or {})
    _write_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def load_betas(
    path: str,
    *,
    max_age_days: float | None = 30.0,
    allow_stale: bool = False,
) -> tuple[BetaBook, dict]:
    """Load a BetaBook and its metadata.

    Raises FileNotFoundError if absent, ValueError if malformed, StaleBetasError
    if older than max_age_days unless allow_stale is set. Metadata always carries
    `created_at` and computed `age_days`.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict) or "betas" not in raw:
        raise ValueError(f"{path} is not a beta file (no 'betas' key)")
    if not raw.get("factors"):
        raise ValueError(f"{path} has no factor list")

    created = str(raw.get("created_at", ""))
    age_days = float("inf")
    if created:
        try:
            ts = datetime.fromisoformat(created)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
        except ValueError:
            age_days = float("inf")

    if max_age_days is not None and age_days > max_age_days and not allow_stale:
        shown = f"{age_days:.1f}" if age_days != float("inf") else "unknown"
        raise StaleBetasError(
            f"betas at {path} are {shown} days old (limit {max_age_days}). "
            "Re-run scripts/estimate_betas.py, or pass allow_stale=True if you "
            "have consciously decided old betas are acceptable."
        )

    meta = dict(raw.get("meta", {}))
    meta["created_at"] = created
    meta["age_days"] = age_days

    return BetaBook.from_dict(raw), meta
