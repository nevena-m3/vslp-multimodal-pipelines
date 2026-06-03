# VSLP first local development run

This file is intentionally step-by-step. It assumes the user is on macOS and starts from a downloaded repo folder.

## 1. Place test data

Create this folder if it does not already exist:

```text
examples/test_data/audio_inputs/
```

Put your test audio files there. Acceptable examples:

```text
examples/test_data/audio_inputs/example01.wav
examples/test_data/audio_inputs/example02.webm
examples/test_data/audio_inputs/example03.wav
```

Important: if a file has a `.wav` extension but actually contains WebM/Opus data, keep it there anyway. VSLP uses `ffprobe` and `ffmpeg`, so it inspects the actual media container instead of trusting the file extension.

Optional demographics CSV goes here:

```text
examples/test_data/metadata/demographics.csv
```

V1 does not yet join demographics into preprocessing outputs; the folder convention is established now for the next coding stage.

## 2. Install FFmpeg/FFprobe

On macOS with Homebrew:

```bash
brew install ffmpeg
```

Confirm both tools are visible:

```bash
ffmpeg -version
ffprobe -version
```

## 3. Create and activate Python 3.11 environment

From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install VSLP in editable mode:

```bash
pip install -e .
```

For Silero segmentation later, install the optional dependency:

```bash
pip install -e '.[silero]'
```

## 4. Generate synthetic smoke-test data if you do not have files yet

```bash
python scripts/create_test_audio.py
```

This creates one true WAV file and one WebM/Opus file intentionally saved with a `.wav` extension.

## 5. Run project init

```bash
vslp project init examples/test_runs/run_001 --project-name "VSLP smoke test"
```

## 6. Run ingest only

```bash
vslp acoustic ingest examples/test_data/audio_inputs examples/test_runs/run_001
```

Expected main output:

```text
examples/test_runs/run_001/acoustic/001_ingest/tables/audio_ingest_summary.csv
```

## 7. Run preprocessing only

```bash
vslp acoustic preprocess examples/test_data/audio_inputs examples/test_runs/run_001
```

Expected main outputs:

```text
examples/test_runs/run_001/acoustic/002_preprocess/tables/acoustic_preprocess_summary.csv
examples/test_runs/run_001/acoustic/002_preprocess/artifacts/feature_wav/
examples/test_runs/run_001/acoustic/002_preprocess/artifacts/segmentation_wav/
examples/test_runs/run_001/acoustic/002_preprocess/reports/acoustic_preprocess_report.html
examples/test_runs/run_001/acoustic/002_preprocess/logs/stage_manifest.json
```

## 8. Run Silero segmentation

This requires `torch` and may download/cache the Silero model on first use.

```bash
vslp acoustic segment-silero \
  examples/test_runs/run_001/acoustic/002_preprocess/tables/acoustic_preprocess_summary.csv \
  examples/test_runs/run_001
```

Expected main outputs:

```text
examples/test_runs/run_001/acoustic/003_segmentation/tables/acoustic_segmentation_summary.csv
examples/test_runs/run_001/acoustic/003_segmentation/tables/frames/
examples/test_runs/run_001/acoustic/003_segmentation/tables/segments/
examples/test_runs/run_001/acoustic/003_segmentation/tables/boundaries/
examples/test_runs/run_001/acoustic/003_segmentation/plots/
examples/test_runs/run_001/acoustic/003_segmentation/reports/acoustic_segmentation_report.html
examples/test_runs/run_001/acoustic/003_segmentation/logs/stage_manifest.json
```

## 9. Run the first combined backend path

```bash
vslp acoustic run-v1 examples/test_data/audio_inputs examples/test_runs/run_002 --project-name "VSLP combined smoke test"
```

To skip Silero segmentation:

```bash
vslp acoustic run-v1 examples/test_data/audio_inputs examples/test_runs/run_002 --skip-segmentation
```
