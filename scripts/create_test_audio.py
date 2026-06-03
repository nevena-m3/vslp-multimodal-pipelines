"""Create small synthetic audio files for smoke-testing VSLP acoustic stages.

This creates:
- a true PCM WAV file;
- a WebM/Opus file intentionally saved with a .wav extension to test ffprobe/ffmpeg
  content-based decoding instead of trusting file extensions.

Usage:
    python scripts/create_test_audio.py
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf


def main() -> None:
    out = Path("examples/test_data/audio_inputs")
    out.mkdir(parents=True, exist_ok=True)

    sr = 16000
    t = np.arange(0, 4.0, 1 / sr)
    x = np.zeros_like(t, dtype=np.float32)
    # Two speech-like sinusoidal bursts with a pause in the middle.
    burst1 = (t >= 0.5) & (t < 1.6)
    burst2 = (t >= 2.2) & (t < 3.4)
    x[burst1] = 0.15 * np.sin(2 * np.pi * 180 * t[burst1]) + 0.04 * np.sin(2 * np.pi * 360 * t[burst1])
    x[burst2] = 0.12 * np.sin(2 * np.pi * 220 * t[burst2]) + 0.03 * np.sin(2 * np.pi * 440 * t[burst2])
    x += 0.003 * np.random.default_rng(42).normal(size=len(x)).astype(np.float32)

    true_wav = out / "SUBJ001_SESSION01_ITERATION01_BAMBOO_true_wav.wav"
    sf.write(true_wav, x, sr, subtype="PCM_16")

    tmp_wav = out / "tmp_for_webm.wav"
    sf.write(tmp_wav, x, sr, subtype="PCM_16")
    mislabeled = out / "SUBJ002_SESSION01_ITERATION01_DDK_webm_content.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp_wav),
        "-c:a",
        "libopus",
        "-f",
        "webm",
        str(mislabeled),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    tmp_wav.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while creating WebM test file:\n{proc.stderr}")

    metadata = Path("examples/test_data/metadata")
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "demographics.csv").write_text(
        "subject_id,session_id,iteration,task,recording_date,diagnosis,severity_score,severity_bin,file_name\n"
        f"SUBJ001,SESSION01,1,bamboo,2026-01-01,ALS,30,mild,{true_wav.name}\n"
        f"SUBJ002,SESSION01,1,ddk,2026-01-02,ALS,35,moderate,{mislabeled.name}\n",
        encoding="utf-8",
    )

    print(f"Created test audio in: {out.resolve()}")
    print(f"Created metadata CSV in: {metadata.resolve()}")


if __name__ == "__main__":
    main()
