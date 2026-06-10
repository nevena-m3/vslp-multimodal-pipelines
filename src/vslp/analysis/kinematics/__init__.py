"""Kinematics analysis scaffold for the VSLP GUI."""
from .schemas import LANDMARK_PRESETS, NORMALIZATION_METHODS, NORMALIZATION_METHOD_DETAILS, AGGREGATION_PROFILES, VideoIngestConfig
from .selection import analyze_landmark_selection, write_selected_landmarks
from .ingest import run_ingest, discover_videos, summarize_ingest_manifest, build_format_summary, build_warning_summary
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
from .video_qc import VideoQCConfig, VIDEO_QC_FRAMEWORK_PLACEHOLDER, run_video_qc, write_video_qc_framework_placeholder
from .features import (
    DEFAULT_KINEMATIC_FEATURE_IDS,
    KINEMATIC_FEATURE_GROUPS,
    KINEMATIC_FEATURE_SPECS,
    QC_FEATURE_REQUIREMENTS,
    FeatureComputationConfig,
    feature_registry_dataframe,
    feature_framework_dataframe,
    feature_implementation_audit_dataframe,
    write_feature_framework_catalog,
    run_feature_computation,
)
from .aggregation import TemporalAggregationConfig, aggregation_guide_dataframe, write_aggregation_guide, run_temporal_aggregation
from .reports import write_scaffold_report, write_pipeline_summary_report
from .inspector import KINEMATICS_STAGE_SPECS, artifact_inventory_dataframe, readiness_checklist_dataframe, stage_status_dataframe, write_inspector_inventory

__all__ = [
    "LANDMARK_PRESETS", "NORMALIZATION_METHODS", "NORMALIZATION_METHOD_DETAILS", "AGGREGATION_PROFILES", "VideoIngestConfig",
    "analyze_landmark_selection", "write_selected_landmarks",
    "run_ingest", "discover_videos", "summarize_ingest_manifest", "build_format_summary", "build_warning_summary", "link_metadata", "load_metadata", "LandmarkRunConfig",
    "write_landmark_plan", "run_mediapipe_landmarks", "download_default_model",
    "mediapipe_capability_note", "mediapipe_environment_status", "FACE_LANDMARKER_MODEL_URL",
    "NormalizationConfig", "write_normalization_config", "run_normalization", "run_normalization_from_selection",
    "VideoQCConfig", "VIDEO_QC_FRAMEWORK_PLACEHOLDER", "run_video_qc", "write_video_qc_framework_placeholder", "FeatureComputationConfig", "feature_registry_dataframe", "feature_framework_dataframe", "feature_implementation_audit_dataframe", "write_feature_framework_catalog", "run_feature_computation", "TemporalAggregationConfig", "aggregation_guide_dataframe", "write_aggregation_guide", "run_temporal_aggregation", "write_scaffold_report", "write_pipeline_summary_report", "KINEMATICS_STAGE_SPECS", "artifact_inventory_dataframe", "readiness_checklist_dataframe", "stage_status_dataframe", "write_inspector_inventory",
]
