# VSLP Feature Analysis GUI v0.40 — Interface Refinement

This update refines the first Feature Analysis GUI shell. It does not change the statistical backend.

## Purpose

The Feature Analysis GUI is a modality-neutral workspace for inspecting acoustic, kinematic, mixed, or generic feature tables before machine learning. It supports optional QC tables, metadata tables, and feature registry/computation-policy tables.

## Interface changes

- Adds a persistent left sidebar aligned with the Acoustic GUI design language.
- Adds stage navigation and status indicators.
- Fixes the branding bar so the Speech Production Lab and University of Toronto logos are scaled to comparable visual size.
- Uses a white branding strip with dark navy title text for logo legibility.
- Improves Project tab structure and visual hierarchy.
- Clarifies the purpose of the optional feature registry / policy table.
- Improves modality dropdown sizing and popup styling.
- Fixes the output-folder open function.

## Feature registry / policy table

This optional input is useful when the feature table is produced by VSLP or another pipeline that can provide feature metadata. It may contain feature names, subsystem labels, units, computation policies, implementation status, expected ranges, or interpretation notes.

When supplied, it improves interpretability and future analyses such as subsystem grouping, expected-range review, and feature reliability reporting. When omitted, the GUI still runs using automatic column detection.

## Current analysis scope

The current GUI performs:

- column role inference
- dataset inventory
- missingness summary
- distribution summary
- robust outlier screening
- feature correlation screening
- feature-QC correlation screening when QC data are supplied
- initial reliability recommendation table
- HTML report generation

This is not model training. ML belongs in the later ML GUI.
