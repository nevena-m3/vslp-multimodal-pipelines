# Feature GUI v0.74 - QC task scope and framework selector

Built from the stable v0.73 distribution-scope cleanup line.

## Scope
This patch only updates QC Integration.

## Adds
- Local Task focus dropdown for QC plots.
- Local QC framework dropdown:
  - Auto / all QC
  - Acoustic QC
  - Kinematic QC
- QC framework summary plot and table.
- QC plots regenerate within the selected local task/framework scope.
- Selected feature x selected QC scatter uses the same local QC scope.

## Safety
- No global filter.
- No mutation of the base analysis dataframe.
- No required metadata.
- No warning-handling changes.
- If no QC table is available, QC plots show empty/explanatory messages.
