"""
Minimal Yahoo Finance daily-bar loader — pure standard library.

No pandas, no requests. Returns plain dicts of lists, which is exactly what
factors/math_core.py consumes.

Two details carried over from Tradingbot's src/data/yahoo_loader.py, both learned
the hard way there:

  * explicit period1/period2 unix bounds, because `range=max` silently
    downsamples long ranges to monthly bars, and
  * a browser User-Agent, because Yahoo 403s the default urllib agent.

One deliberate improvement: Tradingbot's cache returns a cached file
unconditionally with no staleness check, so a stale CSV silently poisons every
downstream number. Here the cache records when it was written and is ignored once
older than max_age_hours. A stale cache is still used as a LAST resort if the
network fails, but loudly.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data_cache")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


class Bars:
    """Daily bars for one symbol: parallel lists, oldest first."""

    __slots__ = ("symbol", "dates", "close")

    def __init__(self, symbol: str, dates: list[str], close: list[float]) -> None:
        self.symbol = symbol
        self.dates = dates
        self.close = close

    def __len__(self) -> int:
        return len(self.dates)

    def as_map(self) -> dict[str, float]:
        return dict(zip(self.dates, self.close))


def _cache_paths(symbol: str) -> tuple[str, str]:
    safe = (symbol.replace("=", "").replace("^", "").replace("/", "")
            .replace(".", "_").replace(":", "_"))
    base = os.path.join(CACHE_DIR, f"yahoo_{safe}_1d")
    return base + ".csv", base + ".meta.json"


def _read_cache(csv_path: str, symbol: str) -> Bars | None:
    try:
        dates: list[str] = []
        close: list[float] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    c = float(row["close"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(c) and c > 0:
                    dates.append(row["date"])
                    close.append(c)
        return Bars(symbol, dates, close) if dates else None
    except (OSError, csv.Error):
        return None


def _cache_age_hours(meta_path: str) -> float:
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            written = float(json.load(fh).get("written_at", 0.0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return float("inf")
    if written <= 0:
        return float("inf")
    return (time.time() - written) / 3600.0


def fetch_daily(
    symbol: str,
    years: float = 8.0,
    use_cache: bool = True,
    max_age_hours: float = 20.0,
    retries: int = 3,
    timeout: int = 30,
) -> Bars:
    """Daily closes for a Yahoo symbol, oldest first.

    Raises RuntimeError if the symbol cannot be fetched and no cache exists.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    csv_path, meta_path = _cache_paths(symbol)

    if use_cache and os.path.exists(csv_path) and _cache_age_hours(meta_path) <= max_age_hours:
        cached = _read_cache(csv_path, symbol)
        if cached is not None and len(cached) > 0:
            return cached

    period2 = int(time.time())
    period1 = period2 - int(years * 365.25 * 24 * 3600)
    url = (YAHOO_URL.format(symbol=urllib.parse.quote(symbol))
           + f"?interval=1d&period1={period1}&period2={period2}")

    payload = None
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt))

    if payload is None:
        # Fall back to a stale cache rather than failing the whole study, but be
        # loud — silently using old data is how bad betas get shipped.
        stale = _read_cache(csv_path, symbol) if os.path.exists(csv_path) else None
        if stale is not None:
            print(f"  WARNING {symbol}: fetch failed ({last_err!r}); using STALE "
                  f"cache aged {_cache_age_hours(meta_path):.1f}h")
            return stale
        raise RuntimeError(f"could not fetch {symbol}: {last_err!r}")

    try:
        result = payload["chart"]["result"][0]
        stamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError) as exc:
        err = ((payload.get("chart") or {}).get("error")) if isinstance(payload, dict) else None
        raise RuntimeError(f"unexpected Yahoo payload for {symbol}: {err or exc!r}") from exc

    seen: dict[str, float] = {}
    for ts, c in zip(stamps, closes):
        if c is None or ts is None:
            continue
        try:
            cf = float(c)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(cf) or cf <= 0:
            continue
        day = time.strftime("%Y-%m-%d", time.gmtime(int(ts)))
        seen[day] = cf          # later entry wins, dedupes intraday duplicates

    dates = sorted(seen)
    bars = Bars(symbol, dates, [seen[d] for d in dates])
    if len(bars) == 0:
        raise RuntimeError(f"{symbol}: Yahoo returned no usable rows")

    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["date", "close"])
            wr.writerows(zip(bars.dates, bars.close))
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"written_at": time.time(), "symbol": symbol, "rows": len(bars)}, fh)
    except OSError:
        pass    # a read-only cache dir must not break the study

    return bars


def align_price_frame(
    symbols: dict[str, str],
    years: float = 8.0,
    use_cache: bool = True,
    max_age_hours: float = 20.0,
    verbose: bool = True,
) -> tuple[list[str], dict[str, list[float | None]]]:
    """Fetch several symbols and align them on a common date axis.

    symbols : {column_name: yahoo_symbol}

    Returns (dates, {column_name: [price or None per date]}). The date axis is the
    UNION of all observed dates, with None where a series has no observation, so
    downstream alignment decisions stay with the caller — the factor model wants a
    strict common sample while individual instruments do not.

    Columns that fail to fetch are omitted and reported rather than aborting: a
    single missing hedge candidate should not kill the whole beta study.
    """
    # Several column names can map to ONE Yahoo symbol — GC=F is both the METALS
    # factor proxy and the GOLD instrument, CL=F is both ENERGY and WTI. Fetch each
    # distinct symbol once. Yahoo rate-limits by IP, so every avoided request is
    # worth having.
    by_symbol: dict[str, list[str]] = {}
    for name, ysym in symbols.items():
        by_symbol.setdefault(ysym, []).append(name)

    fetched: dict[str, dict[str, float]] = {}
    for ysym, names in by_symbol.items():
        label = names[0] if len(names) == 1 else f"{names[0]} (+{len(names) - 1})"
        try:
            bars = fetch_daily(ysym, years=years, use_cache=use_cache,
                               max_age_hours=max_age_hours)
            price_map = bars.as_map()
            for name in names:
                fetched[name] = price_map
            if verbose:
                print(f"  {label:<16} {ysym:<12} {len(bars):>5} rows  "
                      f"{bars.dates[0]} -> {bars.dates[-1]}")
        except Exception as exc:
            print(f"  {label:<16} {ysym:<12} FAILED: {exc!r}")

    if not fetched:
        raise RuntimeError("no symbols fetched successfully")

    all_dates = sorted({d for m in fetched.values() for d in m})
    frame = {name: [m.get(d) for d in all_dates] for name, m in fetched.items()}
    return all_dates, frame
