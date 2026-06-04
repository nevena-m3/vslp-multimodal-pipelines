# VSLP v0.20.2 - Segmentation lazy Torch import hotfix

This hotfix prevents unit tests, GUI startup, and non-Silero stages from requiring PyTorch at import time.

## Problem

`vslp.acoustic.segment.silero_wrapper` imported `torch` at module import time. This caused tests that only inspected segmentation summary helpers to fail when PyTorch was not installed.

## Fix

PyTorch is now imported lazily only when `build_silero_stage_from_audio()` actually runs Silero.

Silero execution still requires PyTorch/Torchaudio:

```bash
pip install -e '.[silero]'
```

## Test result

```text
12 passed
```
