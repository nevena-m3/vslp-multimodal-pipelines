# Feature GUI v0.72 - warnings import hotfix

This hotfix fixes a runtime crash introduced during constant-correlation warning hardening.

## Fix
- Adds defensive warnings imports.
- Makes the safe Spearman helper import warnings locally.
- Keeps constant-input Spearman warnings suppressed without crashing Run Feature Analysis.

## No UI redesign
This is a focused runtime hotfix.
