# Acoustic GUI Completion Roadmap

The acoustic GUI should mature in layers.

## Layer 1: Validated backend wrapper

Already implemented:
- input/output folder selection
- ingest
- preprocess
- Silero segmentation
- first feature extraction
- reports and output-folder access

## Layer 2: Metadata-aware workflow

Implemented in v0.6:
- optional demographics CSV selection
- filename parsing fallback
- metadata completeness report
- file index generation
- subject/session/iteration/task propagation into feature table

## Layer 3: Feature implementation maturity

Next:
- validate all 73 uploaded features
- implement one subsystem at a time
- add per-feature parameter controls
- add tests and known-answer examples
- add plots specific to each feature family

## Layer 4: Production-grade UX polish

Planned:
- stage dependency warnings
- persistent user settings
- project history
- interactive table previews
- embedded report viewer
- waveform/segmentation preview
- status badges and QC flags
- clinical/simple mode and expert mode separation

## Layer 5: Packaging

Planned:
- macOS app bundle
- Windows installer
- bundled ffmpeg/ffprobe guidance
- local Silero model cache
