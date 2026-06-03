# Acoustic Feature Registry

The initial 73-feature registry is derived from the uploaded Bamboo Passage acoustic-feature extraction notebook. The production implementation uses a plugin architecture so individual features can be corrected, validated, replaced, or extended without changing the pipeline contract.

Current scientific subsystem mapping:

- `respiratory_timing`
- `phonatory`
- `articulatory`
- `resonatory`
- `rhythm`
- `coordination`

Open implementation issue: the current feature notebook uses some proxy measures, especially for CPP, formants, and nasality-style features. These should be validated against accepted definitions before publication-grade claims.
