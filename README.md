# Hedgingbot

A factor-exposure overlay for [Tradingbot](https://github.com/klingesh/Tradingbot).

Not a second strategy bot: it generates no signals. It measures the aggregate
factor risk of whatever positions are open and offsets a factor when its exposure
breaches a limit. The goal is drawdown reduction.

See the Phase 1 pull request for the implementation.
