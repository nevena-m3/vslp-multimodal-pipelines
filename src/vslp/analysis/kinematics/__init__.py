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
from .normalization import write_normalization_config
from .features import KINEMATIC_FEATURE_SPECS, KINEMATIC_FEATURE_GROUPS, DEFAULT_KINEMATIC_FEATURE_IDS, QC_FEATURE_REQUIREMENTS, feature_registry_dataframe
from .reports import write_scaffold_report

__all__ = [
    "LANDMARK_PRESETS", "NORMALIZATION_METHODS", "AGGREGATION_PROFILES", "VideoIngestConfig",
    "run_ingest", "discover_videos", "link_metadata", "load_metadata", "LandmarkRunConfig",
    "write_landmark_plan", "run_mediapipe_landmarks", "download_default_model",
    "KINEMATIC_FEATURE_SPECS", "KINEMATIC_FEATURE_GROUPS", "DEFAULT_KINEMATIC_FEATURE_IDS", "QC_FEATURE_REQUIREMENTS", "feature_registry_dataframe",
    "mediapipe_capability_note", "mediapipe_environment_status", "FACE_LANDMARKER_MODEL_URL",
    "write_normalization_config", "write_scaffold_report",
]
