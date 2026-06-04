# VSLP Acoustic GUI v0.17 - Setup/Ingest refinement

This update refines the Setup block from a product and workflow perspective.

## Design intent

The Setup screen is the project gate. Users must:

1. Select an input audio folder.
2. Select an output project folder.
3. Define a project name.
4. Initialize the project.
5. Run ingest.

Metadata is intentionally not run from the Setup screen. It belongs on the Metadata tab.

## Recursive file discovery

VSLP searches the selected input folder recursively. Files inside subfolders are included.

Recommendation: use one task per input folder when possible. Recursive discovery is supported for convenience, but users should avoid mixing tasks unless metadata is clean and explicit.

## Project initialization rule

Ingest and all downstream stages are blocked until `project_manifest.json` exists in the output project folder.

## Ingest feedback

After ingest, the Setup screen immediately shows:

- number of successfully loaded/probed files;
- number of failed files;
- number of subfolders represented;
- extension distribution;
- detected container/format distribution;
- detected audio codec distribution.

VSLP does not trust file extensions. A `.wav` file that is actually WebM/Opus internally is detected using ffprobe.
