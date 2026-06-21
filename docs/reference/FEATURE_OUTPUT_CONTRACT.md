# Feature output contract

Acoustic and kinematic extraction retain their established stage outputs and also write the same canonical delivery structure. The Feature GUI should load the `main` folder rather than selecting stage files individually.

```text
<study_workspace>/
|-- acoustic/feature_handoff/
|   |-- main/
|   `-- supplementary/
`-- kinematics/feature_handoff/
    |-- main/
    `-- supplementary/
```

| Artifact | Required content |
|---|---|
| `feature_values.csv` | One row per recording and task; context/QC plus numeric features |
| `feature_registry.csv` | One row per feature with formula, unit, scope, aggregation, status, and source |
| `feature_status.csv` | One row per recording-feature pair with status and note |
| `feature_export_manifest.json` | Contract version, modality, granularity, canonical paths, and legacy sources |

Optional main files are `qc_features.csv` and `metadata_context.csv`. Their absence is explicit and does not invalidate the four canonical artifacts. `README.md` explains the folder at the point of use.

`supplementary/artifact_catalog.csv` indexes the original stage-specific tables, plots, reports, logs, diagnostics, and intermediate artifacts without copying or moving them. Those artifacts support provenance and review; they are not the default Feature GUI inputs. `feature_handoff/delivery_manifest.json` records copied main outputs, unavailable optional outputs, and the supplementary catalog path.

The shared registry schema is defined in `src/vslp/core/feature_contract.py`. Preferred context fields are `recording_id`, `source_file`, `subject_id`, `session_id`, `protocol_id`, `iteration`, `task`, `recording_date`, and `modality`. Extraction does not invent unavailable metadata.

Kinematic aggregation preserves context and adds `aggregation_level` and `aggregation_method`. Movement-level outputs remain separate from recording-task outputs to prevent pseudoreplication.

Identifiers, paths, processing status, and QC fields are not default biomarkers. QC may support filtering, stratified evaluation, sensitivity analysis, or an explicitly designed quality-aware model. Existing modality-specific filenames remain supported; this contract is additive.
