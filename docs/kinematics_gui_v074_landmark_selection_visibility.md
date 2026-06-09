# Kinematics GUI v0.74 - Landmark selection visibility and frame navigation hardening

## Purpose

This patch improves the Landmark Selection workstation after real user testing showed that MediaPipe points were difficult to see on the face and the frame spinbox arrows were not reliable or comfortable for frame navigation.

## Changes

### High-contrast landmark rendering

The real-frame overlay now draws each unselected landmark with:

- a dark outer halo,
- a bright rim,
- a stronger region-colored fill,
- a larger hover highlight.

Selected landmarks now use:

- a thick black outer halo,
- a white rim,
- a bright green fill,
- a dark label background with white landmark ID text.

This makes landmarks more visible on skin, teeth, shadows, and variable clinical lighting.

### Frame navigation controls

The frame spinbox now hides its small up/down arrows because they were awkward in practice. The workstation now provides explicit buttons:

- `-10`
- `Prev`
- `Next`
- `+10`

The slider remains available for coarse scrubbing. Auto-reload behavior is unchanged: after an overlay has been loaded, moving the frame control reloads the overlay only when auto-reload is enabled.

### Video playback decision

This patch does not add continuous video playback. For landmark selection, frame-by-frame inspection remains scientifically safer because landmark IDs need precise review and accidental playback can obscure poor tracking, occlusion, or frame-specific failure. The recommended near-term design is controlled frame stepping plus slider scrubbing. A future optional preview mode could play a short QC clip, but it should not replace frame inspection.

## Validation

Validated with:

```powershell
PYTHONPATH=src python -m pytest tests/unit/test_kinematics_landmark_selection_visibility.py tests/unit/test_kinematics_landmark_workstation_controls.py tests/unit/test_kinematics_real_frame_selector.py -q
PYTHONPATH=src python -m pytest tests/unit/test_kinematics_*.py -q
```

Expected result in this patch environment:

```text
5 passed
42 passed
```
