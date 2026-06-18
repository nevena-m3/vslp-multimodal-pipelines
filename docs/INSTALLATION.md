# Installation and Environment Verification

## Supported Environment

VSLP currently targets Python 3.11 (`>=3.11,<3.12`). Use a dedicated virtual environment. Do not install the research environment into the system Python used by other clinical or laboratory software.

## System Dependencies

### FFmpeg and FFprobe

Required for acoustic ingest and preprocessing.

Windows options include `winget install Gyan.FFmpeg`, Chocolatey, or a vetted institutional binary distribution. Restart the terminal after installation.

macOS:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify:

```text
ffmpeg -version
ffprobe -version
```

### Video and MediaPipe

The kinematics extra installs OpenCV and MediaPipe. Face landmark extraction also requires a compatible MediaPipe Face Landmarker `.task` model. The Kinematics GUI can install or verify the default runtime/model from its Face Landmarks page. Institutional environments may instead provide a vetted local model file.

## Windows Installation

Open PowerShell in the repository root:

```powershell
py -3.11 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[gui,silero,kinematic,dev]"
vslp doctor
```

If PowerShell blocks environment activation, use the interpreter directly:

```powershell
.\.venv-win\Scripts\python.exe -m pip install -e ".[gui,silero,kinematic,dev]"
.\.venv-win\Scripts\python.exe -m vslp.cli.main doctor
```

## macOS or Linux Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e '.[gui,silero,kinematic,dev]'
vslp doctor
```

## Installation Profiles

| Use case | Command |
|---|---|
| Acoustic GUI without Silero | `pip install -e '.[gui]'` |
| Acoustic GUI with Silero | `pip install -e '.[gui,silero]'` |
| Kinematics GUI | `pip install -e '.[gui,kinematic]'` |
| Feature Analysis GUI | `pip install -e '.[gui]'` |
| Full research/development environment | `pip install -e '.[gui,silero,kinematic,dev]'` |

PyTorch installation can vary by operating system, accelerator, and institutional policy. If the default `silero` extra is unsuitable, install the appropriate PyTorch/Torchaudio build first, then install `.[gui]`.

## Launch Verification

Confirm the installed command and GUI list:

```text
vslp --help
vslp gui --help
```

Launch applications individually:

```text
vslp gui acoustic
vslp gui kinematics
vslp gui features
```

## Developer Verification

```text
python -m pytest -q
python -m ruff check src tests
```

For a quick installed-package check:

```text
python -c "import vslp, pandas, numpy, scipy, PySide6; print('VSLP environment OK')"
```

## Updating an Existing Checkout

Preserve or commit local work before updating.

```text
git status
git pull --ff-only
python -m pip install -e '.[gui,silero,kinematic,dev]'
vslp doctor
```

Never overwrite a working repository with a ZIP export. Use Git branches and commits for source updates. Keep input media and generated outputs outside the repository or in ignored local folders.

## Common Problems

### `vslp` is not recognized

Activate the virtual environment or reinstall the project in editable mode. You may always use `python -m vslp.cli.main` from an activated environment.

### PySide6 is missing

Install `pip install -e '.[gui]'`.

### FFmpeg or FFprobe is missing

Install FFmpeg, restart the terminal, and verify both executables are on `PATH`.

### MediaPipe installation fails

Confirm Python 3.11 and a supported operating-system/CPU architecture. Create a clean environment before retrying.

### GUI opens but scientific stages fail

Review the application Run Log, stage error table, and manifest. Confirm prerequisite stage outputs exist and are newer than downstream outputs.
