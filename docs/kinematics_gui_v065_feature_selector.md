# VSLP Kinematics GUI v0.65 - Feature Selector

This patch adds an acoustic-style kinematic feature selector to the Features tab.

## Purpose

The selector organizes uploaded kinematic feature formulas into clinically interpretable groups:

- Vertical lip/jaw displacement
- Horizontal lip spread
- Lip aperture geometry
- Jaw lateralization
- Lip symmetry
- Bilateral coordination

Each feature records implementation status, tier, required MediaPipe landmarks, native signal, unit, normalization policy, scalar aggregation rule, interpretation, and source function name.

## Current scientific policy

The uploaded feature formulas use intercanthal-distance normalization with MediaPipe landmarks 243 and 463 as anchor points. Most features depend on mouth/lip/jaw landmarks such as 17, 8, 61, 291, 0 and 152. The GUI now surfaces these dependencies before feature computation.

## QC gate

The feature tab explicitly lists QC requirements that must be evaluated before trusting scalar kinematic features: face-detected fraction, ICD anchor availability, long no-face gaps, coordinate jumps/jitter, and timestamp/frame-rate stability.

## Output

The feature plan is written to:

`kinematics/006_features/tables/feature_computation_plan.json`

This is a planning and policy stage. Numerical feature computation will be wired to the same registry in the next computation patch.
