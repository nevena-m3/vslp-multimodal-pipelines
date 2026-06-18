# Kinematics GUI v0.76 - Landmark Selection Workstation Redesign

This patch redesigns the Landmark Selection menu around an explicit scientific review workflow.

## Design changes

- Replaces unreliable scrubber/step-button navigation with detected-frame bookmarks plus exact-frame loading.
- Organizes the menu into three visual zones:
  - left: video/frame controls, preset selection, region tools, display controls, save actions
  - right: large real-frame MediaPipe overlay canvas
  - bottom: selected landmark table and scientific requirement checks
- Removes continuous/implicit frame loading from the selection workstation. Frame loading is explicit by design so an analyst knows exactly which frame is being reviewed.

## Landmark visibility changes

The overlay now uses calmer review-grade rendering:

- Default display is **All faint**, which shows all landmarks as low-alpha neutral points.
- **Selected + anchors** reduces clutter while preserving normalization and oral-motor context.
- **Selected only** is available for final review screenshots.
- Contrast remains configurable with Subtle / Balanced / High contrast.
- Guide edges are off by default to avoid cluttering the face.

## Frame navigation

Frame navigation now uses:

- detected-frame bookmark dropdown
- exact frame number box
- explicit Load Frame button
- Load middle detected frame helper

The bookmark dropdown is populated from the landmark CSV using `face_detected=True` rows where available. This avoids landing on no-face frames during landmark selection.

## Scientific normalization note

The preferred intercanthal normalization anchors remain inner canthus landmarks `133` and `362`. If those are unavailable, the normalization backend may fall back to outer-eye anchors `33` and `263`, but that fallback is recorded and QC-flagged. Landmark selection should still prefer including `133/362` whenever the downstream method is intercanthal normalization.

## Validation

Validated in the development sandbox with:

```text
python -m pytest tests/unit -k kinematics -q
46 passed

PYTHONPATH=src python -m pytest tests/integration/test_acoustic_preprocess_stage.py -q
1 passed
```
