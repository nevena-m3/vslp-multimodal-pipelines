"""Kinematics analysis scaffold for the VSLP GUI."""
from .schemas import LANDMARK_PRESETS, NORMALIZATION_METHODS, AGGREGATION_PROFILES, VideoIngestConfig
from .ingest import run_ingest, discover_videos
from .metadata import link_metadata, load_metadata
from .landmarks import (
    LandmarkRunConfig,
    write_landmark_plan,
    run_mediapipe_landmarks,
    bootstrap_mediapipe_runtime,
    download_default_model,
    mediapipe_capability_note,
    verify_mediapipe_runtime,
)
from .mediapipe_runtime import mediapipe_environment_status, FACE_LANDMARKER_MODEL_URL
from .normalization import NormalizationConfig, write_normalization_config, run_normalization, run_normalization_from_selection
from .video_qc import VideoQCConfig, run_video_qc
from .features import FeatureComputationConfig, run_feature_computation
from .reports import write_scaffold_report

__all__ = [
    "LANDMARK_PRESETS", "NORMALIZATION_METHODS", "AGGREGATION_PROFILES", "VideoIngestConfig",
    "run_ingest", "discover_videos", "link_metadata", "load_metadata", "LandmarkRunConfig",
    "write_landmark_plan", "run_mediapipe_landmarks", "download_default_model",
    "mediapipe_capability_note", "mediapipe_environment_status", "FACE_LANDMARKER_MODEL_URL",
    "NormalizationConfig", "write_normalization_config", "run_normalization", "run_normalization_from_selection",
    "VideoQCConfig", "run_video_qc", "FeatureComputationConfig", "run_feature_computation", "write_scaffold_report",
]
