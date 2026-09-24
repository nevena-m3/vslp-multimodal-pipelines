@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "CONDA_CMD=%CONDA_EXE%"
if defined CONDA_CMD if not exist "%CONDA_CMD%" set "CONDA_CMD="
if not defined CONDA_CMD for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%I"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD (
  echo Conda was not found. Install Miniconda or run from Anaconda Prompt.
  pause
  exit /b 1
)
"%CONDA_CMD%" run --no-capture-output -n vslp-app python --version >nul 2>&1
if errorlevel 1 (
  echo Creating vslp-app with Python 3.11...
  "%CONDA_CMD%" create -y -n vslp-app python=3.11 pip || goto :failed
)
echo Installing the local VSLP application...
"%CONDA_CMD%" run --no-capture-output -n vslp-app python -m pip install -e ".[gui,silero]" || goto :failed
"%CONDA_CMD%" run --no-capture-output -n vslp-app python -c "import pandas,numpy,scipy,soundfile,librosa,parselmouth,PySide6,pyqtgraph,vslp.gui.acoustic_app.app" || goto :failed
echo VSLP application environment        PASS
echo VSLP package                        PASS
"%CONDA_CMD%" run --no-capture-output -n vslp-app python -c "from vslp.acoustic.alignment.mfa_provider import MfaProfile,MfaProvider; from vslp.acoustic.alignment.profiles import default_profile_path; p=MfaProfile.load(default_profile_path()); r=MfaProvider().inspect_environment(p); print('MFA environment vslp-mfa-334        '+('PASS' if r.get('status')=='AVAILABLE' else r.get('status','FAIL'))); print('MFA version '+r.get('version','?')+'                    '+('PASS' if r.get('version')=='3.3.4' else 'FAIL')); print('english_us_arpa acoustic model       '+('PASS' if 'acoustic' in r.get('resources',{}) else 'FAIL')); print('english_us_arpa dictionary           '+('PASS' if 'dictionary' in r.get('resources',{}) else 'FAIL')); raise SystemExit(0 if r.get('status')=='AVAILABLE' else 1)" || goto :failed
echo.
echo READY TO LAUNCH VSLP
pause
exit /b 0
:failed
echo.
echo Setup failed. See the error above.
pause
exit /b 1
