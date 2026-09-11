@echo off
REM Stage LingBot World generations into Physics-IQ Verified, then evaluate.
REM
REM Thin wrapper over physiq\drive_cam2v.py with --app pinned, so the model to
REM run is chosen by which script you start rather than by a flag you have to
REM remember. Every extra argument is forwarded.
REM
REM   scripts\drive_lingbot.bat --run-name lingbot-op-run_01 --limit 1
REM   scripts\drive_lingbot.bat --run-name lingbot-op-run_01
REM   scripts\drive_lingbot.bat --run-name lingbot-op-run_02 --seed 1
setlocal enabledelayedexpansion

set "HERE=%~dp0.."
pushd "%HERE%"

if not defined FLASHDREAMS_ROOT set "FLASHDREAMS_ROOT=%HERE%\..lashdream_public"
set "DRIVER=%HERE%\physiq\drive_cam2v.py"

REM The driver only reads a CSV and writes JSON/npy; the model itself runs in
REM flashdream's environment via uv, launched by the driver.
set "PY=%HERE%\.venv\Scripts\python.exe"
if not exist "!PY!" set "PY=python"

if not exist "!DRIVER!" (
  echo ERROR: driver not found: !DRIVER! 1>&2
  popd & exit /b 2
)
if not exist "!FLASHDREAMS_ROOT!\integrations_v2\lingbot" (
  echo ERROR: LingBot World integration not found: !FLASHDREAMS_ROOT!\integrations_v2\lingbot 1>&2
  echo        set FLASHDREAMS_ROOT=C:\path	olashdream_public 1>&2
  popd & exit /b 2
)
if not exist "!HERE!\physics-IQ-benchmark-verified\switch-frames" (
  echo ERROR: benchmark data missing: !HERE!\physics-IQ-benchmark-verified\switch-frames 1>&2
  echo        run download_verified_data.sh 1>&2
  popd & exit /b 2
)

echo ============================================================
echo cam2v-lingbot staging into Physics-IQ Verified
echo ============================================================
echo   benchmark : !HERE!
echo   lingbot   : !FLASHDREAMS_ROOT!\integrations_v2\lingbot
echo   driver    : !DRIVER!
echo ============================================================

"!PY!" "!DRIVER!" --benchmark-root "!HERE!" --flashdreams-root "!FLASHDREAMS_ROOT!" --app cam2v-lingbot %*
set "RC=!ERRORLEVEL!"

echo.
echo Videos -^> !HERE!\generated_videos_5secho Now evaluate:  uv run physiq/run_physics_iq.py --input_folders generated_videos_5s/^<run^> --output_folder ^<out^> --descriptions_file descriptions/descriptions_original.csv
popd
exit /b !RC!
