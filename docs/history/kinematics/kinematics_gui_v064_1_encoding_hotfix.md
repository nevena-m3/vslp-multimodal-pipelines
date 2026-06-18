# Kinematics GUI v0.64.1 Encoding Hotfix

This hotfix removes non-CP1252 glyphs from the kinematics GUI source so Windows pytest can read the file using the platform default codec. It preserves the v0.64 frame navigation and dark plotting polish behavior.

Affected area: `src/vslp/gui/kinematics/app.py`.

No user-facing workflow changes.
