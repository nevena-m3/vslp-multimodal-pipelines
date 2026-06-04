"""MediaPipe facial-landmark backend scaffold.

This module is intentionally optional because MediaPipe is a heavier dependency.
"""

from __future__ import annotations


def check_mediapipe_available() -> bool:
    try:
        import mediapipe as mp  # noqa: F401
        return True
    except Exception:
        return False


class MediaPipeLandmarkExtractor:
    """Placeholder for video-to-landmark extraction.

    Planned outputs:
    - per-frame landmark table
    - landmark QC summary
    - normalization manifest
    - diagnostic plots/video overlays
    """

    def __init__(self, model_complexity: int = 1):
        self.model_complexity = model_complexity
        if not check_mediapipe_available():
            raise ImportError("mediapipe is not installed. Install with `pip install -e .[kinematic]`.")

    def extract(self, video_path):
        raise NotImplementedError("MediaPipe landmark extraction will be implemented after acoustic stage contracts are stable.")
