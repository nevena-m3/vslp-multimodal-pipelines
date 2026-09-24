@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "CONDA_CMD=%CONDA_EXE%"
if defined CONDA_CMD if not exist "%CONDA_CMD%" set "CONDA_CMD="
if not defined CONDA_CMD for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%I"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD (
  echo Conda was not found. Run setup_vslp_windows.cmd after installing Miniconda.
  pause
  exit /b 1
)
"%CONDA_CMD%" run --no-capture-output -n vslp-app python --version >nul 2>&1
if errorlevel 1 (
  echo vslp-app is missing. Run setup_vslp_windows.cmd first.
  pause
  exit /b 1
)
if not exist "%LOCALAPPDATA%\VSLP\logs" mkdir "%LOCALAPPDATA%\VSLP\logs"
echo Launching VSLP. Diagnostic log: %LOCALAPPDATA%\VSLP\logs\last-launch.log
"%CONDA_CMD%" run --no-capture-output -n vslp-app python -m vslp.gui.acoustic_app 2>"%LOCALAPPDATA%\VSLP\logs\last-launch.log"
if errorlevel 1 (
  echo VSLP exited with an error. See %LOCALAPPDATA%\VSLP\logs\last-launch.log
  pause
  exit /b 1
)
