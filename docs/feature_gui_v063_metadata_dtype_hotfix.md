# Feature GUI v0.63 metadata dtype and design-context hotfix

This patch fixes a Windows/Qt runtime failure where metadata strings were used to fill empty canonical feature-table columns that pandas had inferred as float columns. The GUI now casts canonical fields to object before metadata fill.

It also makes the dataset design overview treat all-empty columns as `empty_column`, not as usable detected metadata, and keeps the design plot compact rather than using large card tiles.
