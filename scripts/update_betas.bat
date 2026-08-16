@echo off
REM ---------------------------------------------------------------------------
REM Re-measure the factor betas. Run this MONTHLY.
REM
REM Needs internet. Does NOT need MetaTrader 5, so it can run anywhere.
REM
REM Why monthly: betas decay and correlations regime-shift. gold's dollar beta in
REM 2022 is not gold's dollar beta in 2019. src/factors/store.py REFUSES to load
REM a beta file older than betas.max_age_days (default 30) rather than warning,
REM because a six-month-old beta sizing a live hedge is a random number generator
REM with good manners.
REM
REM --return-period weekly is deliberate, not arbitrary. Yahoo's FX spot series
REM and its futures series are snapshotted hours apart, which attenuates DAILY FX
REM betas by roughly half (EURUSD measured -0.56 daily vs -0.96 weekly, where
REM theory says ~-0.95). Over a week that offset is negligible.
REM ---------------------------------------------------------------------------

cd /d "%~dp0.."

echo ============================================================
echo  Hedgingbot - re-measuring factor betas (weekly returns)
echo ============================================================
echo.

python scripts\estimate_betas.py --no-cache --return-period weekly

if errorlevel 1 (
    echo.
    echo  FAILED - the existing beta file was NOT overwritten.
    echo  Read the FATAL message above; it says which of the three
    echo  likely causes it was and what to try.
    exit /b 1
)

echo.
echo  Done. Verify with:
echo      python scripts\preflight.py
