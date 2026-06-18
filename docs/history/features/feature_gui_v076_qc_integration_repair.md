# Feature GUI v0.76 - QC integration repair

Built from v0.75 metadata/filename context.

## Fixes
- Restores the original six-family QC framework plot.
- Keeps framework dropdown behavior, but the framework plot remains the conceptual six-family model.
- Adds framework annotation for Auto / all QC, Acoustic QC, and Kinematic QC.
- Repairs Feature and QC metric selectors so they populate from the current local QC scope.
- Feature and QC metric selector changes now update the selected feature x QC scatter when that plot is active.
- Selected scatter errors are shown instead of silently ignored.

## Safety
- No global filter.
- No analysis dataframe mutation.
- No warning-handling changes.
- No dependency on metadata.
