# VSLP kinematics GUI v0.72 - sidebar status lights and landmark extraction dashboard

## Scope

This patch makes two focused interface improvements after the v0.71 Setup / Project dashboard:

1. Adds a visual stage-status light beside each left-sidebar stage status.
2. Starts the next menu refinement by adding a compact Landmark Extraction readiness dashboard.

## Sidebar status lights

The left pipeline sidebar now shows a small status light beside each stage label:

| Stage state | Sidebar text | Light |
|---|---|---|
| completed / detected | Complete | green |
| completed_with_warnings | Complete - review | amber |
| running | Running | blue |
| failed | Failed | red |
| configured / planned | Configured / Planned | pale blue |
| stale | Stale | purple |
| not run | Not run | muted gray |

This gives immediate visual feedback when a module has completed successfully.

## Landmark Extraction dashboard

The Landmark Extraction tab now starts with a compact readiness section showing:

- runtime status
- plan status
- videos in the landmark manifest
- successful videos
- extraction errors
- mean detected-frame fraction
- extraction status
- next recommended action

The existing MediaPipe configuration controls and extraction result table are preserved.

## Design intent

This patch does not change landmark extraction algorithms. It only improves the GUI workflow state model and makes the second menu easier to interpret before moving into deeper landmark-extraction hardening.
