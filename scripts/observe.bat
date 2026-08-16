@echo off
REM ---------------------------------------------------------------------------
REM Run the overlay in OBSERVE mode, restarting it if it ever exits.
REM
REM This places NO orders. It cannot: src/connectors/mt5_reader.py contains no
REM order_send and no TRADE_ACTION of any kind. All this does is measure the
REM factor exposure of whatever is open and append one JSON line per cycle to
REM logs\exposure_history.jsonl.
REM
REM Usage (from the repo root):
REM     scripts\observe.bat
REM
REM Stop it by closing the window, or Ctrl+C twice.
REM
REM Deliberately simple compared with Tradingbot's run_bot.bat: there is no
REM single-instance lock here because nothing is being traded, so two observers
REM running at once is wasteful rather than dangerous. That changes in Phase 2.
REM ---------------------------------------------------------------------------

cd /d "%~dp0.."

if not exist logs mkdir logs

echo ============================================================
echo  Hedgingbot - OBSERVE MODE (read-only, places no orders)
echo ============================================================
echo.
echo  Repo    : %CD%
echo  Log     : logs\observe.log
echo  History : logs\exposure_history.jsonl
echo  Status  : logs\hedge_status.json
echo.
echo  Make sure the MetaTrader 5 terminal is RUNNING and LOGGED IN.
echo.

:loop
echo [%DATE% %TIME%] starting observer >> logs\observe.log
python scripts\measure_exposure.py --loop >> logs\observe.log 2>&1
echo [%DATE% %TIME%] observer exited with code %ERRORLEVEL% >> logs\observe.log

REM A crash is usually the MT5 terminal being closed or losing its connection.
REM Wait, then retry. 30 seconds matches Tradingbot's restart cadence.
echo Observer stopped (code %ERRORLEVEL%). Retrying in 30 seconds...
timeout /t 30 /nobreak >nul
goto loop
