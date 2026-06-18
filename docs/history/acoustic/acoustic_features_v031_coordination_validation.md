# VSLP v0.31 — Coordination feature validation

This release implements the registered coordination features:

- `CPP_F1_comp`
- `CPP_F2_comp`
- `F1_F2_comp`

The implementation is a validated-local, auditable coupling-complexity descriptor. It is not a diagnostic cutoff and should not be interpreted as a direct disease-severity score.

## Method summary

For each recording, VSLP extracts aligned frame trajectories for:

- CPP-like cepstral peak prominence
- F1 formant trajectory
- F2 formant trajectory

The default region is `effective_task`, preserving internal pauses while removing leading and trailing nonspeech.

For each feature pair, the plugin:

1. interpolates only short gaps;
2. robustly z-scales the trajectories;
3. builds a lagged matrix over ±250 ms by default;
4. computes the lagged correlation matrix;
5. summarizes the eigenspectrum using normalized participation ratio.

Formula:

```text
PR_norm = ((Σλ)^2 / Σλ²) / K
```

where `λ` are eigenvalues of the lagged correlation matrix and `K` is the matrix dimension.

## Interpretation

The value is a dimensionless index between approximately 0 and 1.

Lower values indicate that the coupling structure is dominated by fewer effective eigenmodes. Higher values indicate a broader/effective set of modes. Direction is not disease-specific without task, cohort, and validation context.

## Guardrails

The plugin emits low-validity status when:

- the selected audio region is too short;
- CPP/F1/F2 trajectories have too many missing values;
- lagged complete columns are insufficient;
- the correlation matrix is rank-deficient.

## Validation state

This implementation is:

- formula-explicit;
- internally unit-tested;
- auditable through plugin code and feature status notes;
- suitable for exploratory research pipeline testing.

It is not yet externally validated against a published coordination reference implementation. External validation is recommended before publication-level clinical claims.
