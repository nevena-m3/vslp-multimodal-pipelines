# Development and Release Guide

## Environment

Use Python 3.11 in a dedicated environment and install the full development profile:

```text
pip install -e '.[gui,silero,kinematic,dev]'
```

See [Installation](../getting-started/INSTALLATION.md) for platform details.

## Source Layout

```text
src/vslp/acoustic          Acoustic stages
src/vslp/analysis          Feature and kinematic analysis backends
src/vslp/gui               Desktop applications
src/vslp/cli               Typer CLI
src/vslp/core              Shared project/schema utilities
tests/unit                 Fast unit and source-contract tests
tests/history              Archived implementation-era snapshots; not collected by default
```

## Engineering Rules

- Keep scientific computation outside GUI widgets.
- Prefer typed configuration objects and structured tables over ad hoc strings.
- Preserve source media and row lineage.
- Emit explicit errors and manifests.
- Avoid silent fallback when it changes scientific meaning.
- Keep target, identifier, QC, and provenance columns out of predictor sets by default.
- Add tests proportional to scientific and workflow risk.
- Do not commit participant data, generated study outputs, downloaded models, or credentials.

## Tests

Full suite:

```text
python -m pytest -q
```

The default suite collects `tests/unit`. Pre-v132 Feature GUI source snapshots are preserved under `tests/history/feature_gui` and are not current regression tests.

Focused suites:

```text
python -m pytest tests/unit -k acoustic -q
python -m pytest tests/unit -k kinematics -q
python -m pytest tests/unit -k feature_gui -q
```

Static checks:

```text
python -m ruff check src tests
python -m py_compile src/vslp/cli/main.py
```

The enforced Ruff baseline focuses on syntax/import correctness, undefined names, ambiguous names, and invalid comparisons. The repository still contains legacy GUI formatting debt, especially long lines and compact semicolon statements. Address that debt incrementally in focused formatting changes rather than mixing thousands of mechanical edits into scientific changes.

GUI construction smoke test on a display-less environment:

```powershell
$env:QT_QPA_PLATFORM="offscreen"
python -c "from PySide6.QtWidgets import QApplication; from vslp.gui.features.app import FeatureAnalysisGUI; app=QApplication([]); win=FeatureAnalysisGUI(); win.close()"
```

Real media workflows still require manual platform testing. Unit tests do not validate camera/audio acquisition protocols or scientific calibration.

## Documentation Rules

- `README.md` is the suite entry point.
- Durable instructions live in the sections linked from `docs/README.md`.
- Version-labelled documents belong under `docs/history/` and are not operating instructions.
- Update the relevant SOP and data contract when workflow or output semantics change.
- Commands must be tested on the platform they claim to support.

## Git Workflow

1. Start from an updated branch.
2. Keep source, tests, and durable documentation in the same change when behavior changes.
3. Stage files explicitly in a mixed worktree.
4. Run relevant tests before commit.
5. Use a focused commit message.
6. Push the branch and review the remote diff.

Do not update source by copying a ZIP over a working checkout. See [Updating with Git](../history/shared/update_existing_repo_from_zip.md).

## Release Checklist

- Package metadata and supported Python version are accurate.
- All installed CLI entry points respond to `--help`.
- Relevant tests and smoke checks pass.
- Root README and durable SOPs match implemented behavior.
- Output schemas and manifests identify breaking changes.
- Known scientific limitations are documented.
- No sensitive data or local paths are staged.
- Git status and staged diff contain only intended files.
- Version and release notes are updated according to project policy.
