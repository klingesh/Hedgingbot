# Hedgingbot — a factor-exposure overlay for [Tradingbot](https://github.com/klingesh/Tradingbot)

**This is not a second strategy bot. It generates no trading signals and it is not
trying to make money.** Its only job is to measure the aggregate *factor risk* of
whatever positions are open and, when one factor exposure gets too large, place the
smallest offsetting trade that brings it back inside a limit.

The goal is **drawdown reduction**, and the honest trade-off is stated up front:

> A working hedge overlay usually **lowers raw return** and lowers drawdown by more.
> If you are expecting it to increase profit, that expectation needs adjusting
> before you turn it on.

---

## The problem it solves

Tradingbot runs 8 slots: GOLD, SILVER, PLATINUM, NATGAS, BRENT, GBPJPY, AUDUSD,
USDJPY. Five are commodities and six have a USD leg. That is not eight independent
bets — but it is *sized* as if it were. Tradingbot's own documentation flags this
and leaves it unmeasured:

> *"Diversification blend is optimistic (selection bias, correlated commodities,
> flat-day volatility dilution). Expect a real but smaller benefit."*
> — `docs/PROJECT_REPORT.md:143`

This repo quantifies that sentence and then acts on it.

On the synthetic demo book (`--demo`, illustrative numbers only):

```
DIVERSIFICATION
  Open positions              : 8
  Portfolio 1-day 1-sigma risk:       532.87  (5.33% of equity)
  Diversification ratio       :        0.579
  EFFECTIVE INDEPENDENT BETS  :         2.99   (out of 8 positions)
```

Eight positions, about three real bets.

---

## The one number that matters: factor leverage

```
leverage[f] = factor_exposure[f] / equity
```

Read as: **"a 1% adverse move in factor f costs leverage[f] percent of my equity."**

`leverage[USD] = 3.2` means a 1% dollar rally costs 3.2% of the account. It is
unit-free, it stays meaningful as the account grows, and it is a limit you can
actually reason about. Every cap in the config is expressed in these units.

Four factors, chosen because this book needs to know whether it is making the same
bet six times, not because more factors would be fancier:

| Factor | Meaning |
|---|---|
| `USD` | Broad dollar direction |
| `RISK` | Risk-on / risk-off (equity beta) |
| `ENERGY` | Oil & gas complex, after removing USD and RISK |
| `METALS` | Gold complex / real rates, after removing USD, RISK and ENERGY |

Factors are **orthogonalized** (sequential Gram-Schmidt). Raw gold and raw DXY
returns are strongly negatively correlated, so un-orthogonalized betas
double-count the same risk and are numerically unstable. After orthogonalization
each factor is independent of the earlier ones, which means betas are stable,
univariate and multivariate betas coincide, and portfolio variance is a plain sum
of squares with no covariance cross-terms.

Because orthogonalization is sequential, **the order is a modelling choice, not a
fact**. We go most-macro-primitive to most-idiosyncratic: `USD → RISK → ENERGY →
METALS`. The consequence: a plain dollar-driven gold rally lands in `USD`, not in
`METALS`. For a metals-heavy book that is the behaviour we want.

---

## What stops it being a spread-donation machine

A naive overlay ("if exposure > cap, hedge to cap") churns: hedge, market wiggles,
unhedge, pay the spread both ways, forever. Six guards prevent that, and they are
the actual product. Each has a test that fails if the guard is removed.

1. **Hysteresis** — open only above `cap × (1 + hysteresis)`. The cap is the edge of
   a band, not a trigger.
2. **Target undershoot** — hedge down to `cap × target_fraction`, never to the cap
   itself. Hedging to the boundary guarantees re-breaching next tick.
3. **Unwind band** — close only below `cap × unwind_band`. With (1) this creates a
   wide dead zone where nothing happens.
4. **Min-lot refusal** — if the required size rounds below `volume_min`, do nothing
   and say why. Never round *up* into a hedge bigger than the risk it removes. Same
   philosophy as Tradingbot's `MinLotPolicy.SKIP`.
5. **Collateral-damage check** — every candidate is simulated against *all* factors
   first. A hedge that fixes USD by pushing RISK into breach is shrunk until it
   stops, or rejected. This is the difference between *reducing* risk and *moving*
   it.
6. **Daily action budget** — a hard ceiling on hedges per day. If the overlay wants
   to trade more than a handful of times a day, the caps are wrong, and the budget
   makes that visible instead of expensive.

Plus two safety gates:

- **Unmapped symbols block everything.** If any open position has no betas, the
  overlay refuses to trade at all rather than hedge a book it cannot fully see. An
  unknown symbol is also charged a pessimistic placeholder volatility, because a
  risk system that reports *zero* risk for something it does not recognise is worse
  than one that errors — it looks fine.
- **The trader's own symbols are never traded.** On a **netting** account an
  opposing order on the same symbol *reduces* the trader's position instead of
  hedging it, silently sabotaging the strategy it is meant to protect.

One deliberate asymmetry: **a drawdown kill switch does not block hedging.**
Tradingbot's kill switch exists to stop the account taking on new *risk*; a hedge
*removes* risk, so blocking it during a drawdown would be backwards — that is when
it is most needed. `flatten` is the separate, opposite instruction.

---

## Status: Phase 1 complete

**Phase 1 is measurement only, and it is deliberately incapable of trading.** The
connector (`src/connectors/mt5_reader.py`) contains no `order_send`, no
`TRADE_ACTION` of any kind. That is a much stronger safety property than a
`dry_run` flag that a YAML typo could flip.

| | Status |
|---|---|
| Factor maths (orthogonalization, betas, purity) | Done, 29 tests |
| Exposure model (notional → factor leverage → risk decomposition) | Done, 19 tests |
| `hedge_decide()` decision layer | Done, 31 tests |
| Beta estimation + persistence with staleness guard | Done, 20 tests |
| Shared portfolio state / combined kill switch | Done, 22 tests |
| Read-only MT5 connector | Done, untested (needs Windows) |
| Observe-mode measurement script | Done |
| **Order placement** | **Not built — Phase 2** |
| Partial close, SL/TP modify, `order_calc_margin` | Not built — Phase 2 |
| Multi-leg backtest validation | Not built — Phase 3 |

`132 passed, 1 skipped`, no third-party runtime dependencies.

---

## Getting started

### 0. Check what is missing

```bash
python scripts/preflight.py
```

Read-only, opens no orders, writes no files. Run it first and again whenever
something does not work. It exists because the three setup steps have non-obvious
ordering dependencies:

| Script | Needs MT5 | Needs internet | Needs config | Needs betas |
|---|---|---|---|---|
| `check_account_mode.py` | **yes** | no | no | no |
| `estimate_betas.py` | no | **yes** | no | no |
| `measure_exposure.py` | **yes** | no | **yes** | **yes** |

So running them in the wrong order, or on the wrong machine, fails in ways whose
error messages do not obviously point at the real cause. Preflight checks each
prerequisite and prints the single command that fixes the first broken thing.

### 1. Answer the account question (Windows, MT5 terminal open)

```bash
python scripts/check_account_mode.py --suffix .ecn --order-check
```

Reports **netting vs hedging** margin mode, whether algo trading is actually
enabled at both account and terminal level, per-symbol contract specs,
`trade_stops_level`, and **swap rates on both directions** — the recurring carry
cost of holding a hedge, which most hedging discussions leave out entirely.

### 2. Measure the betas (needs internet)

```bash
python scripts/estimate_betas.py
```

Prints the raw correlation matrix of the traded book, the orthogonalization
diagnostics, the measured betas with **prior-sign checks** (a gold/USD beta coming
out positive means a wrong ticker, not a market insight), a **split-sample
stability check** (a beta that flips sign between halves is not a number to size a
hedge from), and the hedge-candidate purity table. Writes
`betas/betas_latest.json`.

Re-run monthly. `factors/store.py` **refuses** to load betas older than
`max_age_days` rather than warning, because a six-month-old beta sizing a live
hedge is a random number generator with good manners.

### 3. Preview the output with no account at all

```bash
python scripts/measure_exposure.py --demo
```

### 4. Observe your real book — and stay here for weeks

```bash
python scripts/measure_exposure.py --loop
```

Appends one JSON line per cycle to `logs/exposure_history.jsonl`.

**This step is the whole point of Phase 1.** Do not skip to choosing caps. You
cannot pick them sensibly without knowing what leverage your book actually runs: a
cap above your observed maximum does nothing at all, and a cap far below it hedges
constantly and bleeds spread. Both failures are invisible until measured.

The same applies to `max_hedge_gross_pct`. It defaults to 1000% of equity and that
is not a typo — leveraged CFD notional is inherently a large multiple of equity
(one lot of EURUSD is ~108,000 of notional), so a "sensible-looking" 200% ceiling
would cap the overlay at 0.18 lots on a 10k account and it would spend its life
refusing to hedge. Set it from the `gross_notional` observe mode reports.

---

## How it coexists with Tradingbot

**MT5 is the shared state for positions.** `positions_get()` returns every position
regardless of magic number — Tradingbot's `bot_positions()` filters by magic, which
is right for a strategy bot and wrong for a risk overlay. So the overlay sees the
trader's book natively and no shared position file is needed.

What *is* shared is `logs/portfolio_state.json`: the portfolio-level halt and a
high-water-mark peak equity that only ever rises. This reuses the mechanics
Tradingbot learned the hard way — its `state.py` exists because a 30-second restart
loop silently re-armed a spent kill switch. Clearing a halt requires a human
deleting the file.

Three collisions to avoid, all handled in the shipped config:

| | Tradingbot | Hedgingbot |
|---|---|---|
| magic number | `990011` | `990012` |
| lock file | `logs/bot.lock` | `logs/hedge.lock` |
| status file | `logs/status.json` | `logs/hedge_status.json` |

---

## Layout

```
src/
  factors/
    math_core.py    THE maths — weighted moments, Gram-Schmidt, betas. Pure stdlib.
    estimate.py     Orchestration: prices -> returns -> factors -> BetaBook
    book.py         BetaBook: the container everything reads betas from
    store.py        JSON persistence + staleness guard
    definitions.py  Factors, proxies, prior betas (sanity checks only)
  exposure/
    model.py        positions -> notional -> factor leverage -> risk decomposition
    report.py       human-readable rendering
  overlay/
    decision.py     hedge_decide() — pure, zero I/O, the six guards
  connectors/
    mt5_reader.py   READ-ONLY. No order code exists in Phase 1.
  state/
    shared.py       shared halt + high-water mark, atomic writes
  data/
    yahoo.py        stdlib daily-bar loader with a staleness-aware cache
scripts/
  check_account_mode.py   netting vs hedging, specs, swaps, order pre-check
  estimate_betas.py       the beta + concentration study
  measure_exposure.py     observe mode (--demo works anywhere)
```

### Requirements

**Python 3.9 or newer.** Verified on 3.9, 3.10, 3.11, 3.12, 3.13 and 3.14 — the
demo pipeline produces identical numbers on all of them. `tests/test_python_compatibility.py`
enforces the floor with an AST check, because the usual way to break it (a
module-level `X | None` type alias, which Python evaluates at import time even
with `from __future__ import annotations`) only fails on the *older* interpreter
and with an error message that does not point at the cause. That bug shipped in
the first Phase 1 commit and was caught by running `preflight.py` under 3.9.

`MetaTrader5` is needed only for reading a live account, and only exists on
Windows. `PyYAML` is optional. Nothing else.

### Why no pandas or numpy

These numbers decide whether the bot hedges 0.4 lots or 4 lots. The maths needs to
be verifiable in any environment, readable line by line by a human checking the
algebra, and runnable on a bare VPS with no scientific stack. `math_core.py` is the
single implementation; there is no second code path.

---

## Known gaps and honest caveats

- **Idiosyncratic risks are assumed independent** and added in quadrature. Gold and
  silver residuals are very likely correlated, so true portfolio risk is *higher*
  than reported. Factor exposures and leverages are unaffected; the idio term,
  `portfolio_daily_risk` and the diversification ratio are optimistic.
- **Betas are historical.** They decay and regime-shift. The staleness guard and the
  split-sample stability check mitigate this; they do not solve it.
- **The MT5 connector is untested** — no Windows in CI, and no fake broker yet.
- **No cost model yet.** Spread is used to rank candidates, but the overlay does not
  yet compare the *expected cost* of a hedge against the risk it removes. That is
  Phase 3 work and it is the thing that will decide whether this is worth running.
- **Not validated on real P&L.** Nothing here has been shown to reduce drawdown on
  live or backtested data. Phase 3 is a multi-leg backtest, and the success
  criterion is improved *risk-adjusted* return — not raw return.

---

## Roadmap

- **Phase 1 — measurement.** Complete. Read-only, cannot trade.
- **Phase 2 — execution.** Order placement with idempotency and retries; partial
  close; `TRADE_ACTION_SLTP` modify; `order_calc_margin` so hedge legs are
  margin-aware; hedging-mode detection; flatten-on-kill-switch.
- **Phase 3 — validation.** Multi-leg backtest (Tradingbot's engine is hard-wired to
  one position at a time), a cost model, and a verdict on whether the overlay
  actually improves risk-adjusted return.
