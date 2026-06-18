# Contributing to VSLP

VSLP is research software for clinical speech and facial-movement analysis. Contributions must preserve scientific traceability, privacy, and leakage-safe downstream use.

## Before Making Changes

- Open or reference a focused issue when possible.
- State the scientific or workflow problem, affected modality/task, and expected behavior.
- Do not include participant data or identifiable media in examples or tests.
- Use synthetic or explicitly approved public test data.

## Development

1. Create a branch from the current development branch.
2. Install the development environment described in `docs/development/DEVELOPMENT.md`.
3. Follow existing backend/GUI boundaries.
4. Add or update tests.
5. Update durable documentation when behavior, outputs, or scientific interpretation changes.
6. Run the relevant test and static-check commands.

## Pull Requests

A pull request should explain:

- what changed and why;
- scientific or user impact;
- affected input/output contracts;
- failure mode or root cause for fixes;
- validation performed;
- remaining limitations.

Keep unrelated refactors and generated files out of the change.

## Scientific Changes

Changes to feature formulas, thresholds, normalization, segmentation, QC rules, reliability, screening, or ML preparation require:

- a documented rationale and source/reference where applicable;
- unit and edge-case tests;
- explicit units and support requirements;
- versioned output/schema consideration;
- an update to the relevant SOP or registry;
- no claim of clinical validity without independent evidence.

## Code Style

- Python 3.11
- Ruff-configured formatting/lint expectations
- Type hints for public APIs and structured configuration
- Small, testable backend functions
- Clear errors instead of silent scientific fallback

## License and Attribution

By contributing, you confirm that you have permission to submit the work under the repository's current research-use terms and institutional policies. Final licensing remains pending institutional confirmation.
