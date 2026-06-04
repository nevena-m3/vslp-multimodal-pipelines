# VSLP Architecture

VSLP is a stage-oriented multimodal pipeline platform. The backend owns all scientific and engineering logic. GUIs are thin controllers over typed backend services.

## Stage contract

```text
StageInput + StageConfig -> StageOutput + StageReport + StageManifest
```

Each stage writes:

- `tables/`: CSV outputs for human inspection.
- `plots/`: PNG/SVG diagnostic visualizations.
- `logs/`: runtime logs and manifests.
- `reports/`: HTML reports.
- `artifacts/`: model files, processed audio, landmarks, etc.
- `errors/`: file-level and stage-level failures.

## Privacy

The system is local-first. Sensitive audio/video never leaves the machine unless the user explicitly exports it outside the software. Pretrained models are local files and are loaded locally.

## Leakage control

All ML splitting must group by `subject_id`. Longitudinal visits/iterations belong to the subject and cannot cross train/test boundaries.
