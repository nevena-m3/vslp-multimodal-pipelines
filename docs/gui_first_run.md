# VSLP Acoustic GUI - First Run

This is the first working desktop GUI for the acoustic backend.

## Install GUI dependencies

From the root of your local repo:

```bash
source .venv/bin/activate
pip install -e '.[gui,silero]'
```

If PyTorch/Torchaudio need to be installed separately on your machine, run:

```bash
pip install torch torchaudio
pip install -e '.[gui]'
```

## Launch the GUI

```bash
vslp gui acoustic
```

## In the GUI

1. Select your **input audio folder**.
2. Select your **output project folder**.
3. Optionally edit the **project name**.
4. Click **Initialize Project**.
5. Click **Run Ingest**.
6. Click **Run Preprocess**.
7. Click **Run Silero Segmentation**.
8. Use the **Reports & Outputs** tab to open generated CSV and HTML outputs.

## Current GUI scope

The GUI currently wraps these backend stages:
- acoustic ingest
- acoustic preprocessing
- Silero segmentation

It does **not** yet include:
- demographics table integration
- feature extraction
- kinematic workflows
- feature analysis GUI
- ML GUI

Those will be added incrementally after the acoustic GUI workflow is stable.
