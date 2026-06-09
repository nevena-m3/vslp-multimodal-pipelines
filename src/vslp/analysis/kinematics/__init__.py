"""Kinematics analysis scaffold for the VSLP GUI."""
from .schemas import LANDMARK_PRESETS, NORMALIZATION_METHODS, AGGREGATION_PROFILES, VideoIngestConfig
from .ingest import run_ingest, discover_videos
from .metadata import link_metadata, load_metadata
from .landmarks import LandmarkRunConfig, write_landmark_plan, mediapipe_capability_note
from .normalization import write_normalization_config
from .reports import write_scaffold_report

__all__ = [
    "LANDMARK_PRESETS", "NORMALIZATION_METHODS", "AGGREGATION_PROFILES", "VideoIngestConfig",
    "run_ingest", "discover_videos", "link_metadata", "load_metadata", "LandmarkRunConfig",
    "write_landmark_plan", "mediapipe_capability_note", "write_normalization_config", "write_scaffold_report",
]
