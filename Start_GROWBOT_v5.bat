@echo off
setlocal EnableExtensions
TITLE GROWBOT v5 Experimental - Gantry, RealSense, Optris

REM Run from the folder containing this launcher, regardless of where it was opened.
cd /d "%~dp0"

set "APP=growbot_gantry_realsense_optris_v5.py"
set "ENV_NAME=growbot2024"

if not exist "%APP%" (
    echo ERROR: Cannot find %APP%
    echo Put Start_GROWBOT.bat in the same folder as the GROWBOT v4 Python file.
    goto :ERROR
)

REM Detect an existing Conda environment without requiring Conda on the Windows PATH.
REM This covers the previously confirmed setup and common Miniconda/Anaconda locations.
set "ENV_DIR="
for %%D in (
    "%USERPROFILE%\.conda\envs\%ENV_NAME%"
    "%USERPROFILE%\miniconda3\envs\%ENV_NAME%"
    "%USERPROFILE%\anaconda3\envs\%ENV_NAME%"
    "%USERPROFILE%\Miniconda3\envs\%ENV_NAME%"
    "%USERPROFILE%\Anaconda3\envs\%ENV_NAME%"
    "%LOCALAPPDATA%\miniconda3\envs\%ENV_NAME%"
    "%LOCALAPPDATA%\anaconda3\envs\%ENV_NAME%"
    "%ProgramData%\miniconda3\envs\%ENV_NAME%"
    "%ProgramData%\anaconda3\envs\%ENV_NAME%"
) do (
    if not defined ENV_DIR if exist "%%~D\python.exe" set "ENV_DIR=%%~D"
)

if not defined ENV_DIR (
    echo ERROR: Conda environment "%ENV_NAME%" was not found in standard locations.
    echo Please create it first or edit Start_GROWBOT.bat to set ENV_DIR to its folder.
    echo Example: set "ENV_DIR=C:\Users\YourUser\.conda\envs\growbot2024"
    goto :ERROR
)

REM Provide the DLL and executable directories normally added by Conda activation.
REM Launching its Python directly avoids requiring Anaconda Prompt or conda on PATH.
set "CONDA_PREFIX=%ENV_DIR%"
set "CONDA_DEFAULT_ENV=%ENV_NAME%"
set "PATH=%ENV_DIR%;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%PATH%"

echo ================================================================
echo   GROWBOT v4 - Gantry + RealSense + Optris
 echo ================================================================
echo   Project: %CD%
echo   Python:  %ENV_DIR%\python.exe
echo.
echo Make sure OpenBuilds CONTROL and PIX Connect are closed.
echo To quit: use Shutdown / Disconnect All in GROWBOT, then Ctrl+C here.
echo.

"%ENV_DIR%\python.exe" -m streamlit run "%CD%\%APP%" --server.address 127.0.0.1
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo GROWBOT stopped with error code %EXIT_CODE%.
if "%EXIT_CODE%"=="0" echo GROWBOT has stopped.
echo You can now close this window.
pause
exit /b %EXIT_CODE%

:ERROR
echo.
pause
exit /b 1
