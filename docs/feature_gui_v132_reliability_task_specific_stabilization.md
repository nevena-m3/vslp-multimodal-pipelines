# Feature GUI v0.132 Reliability task-specific stabilization

This patch makes the Feature Reliability task selector operational for small
task-specific repeated-measures designs.

Changes:

- Task-specific reliability estimates now allow preliminary ICC/MDC screening
  when at least two repeated same-task subject units and four valid observations
  are available.
- Preliminary estimates are labeled in the interpretation text so they are not
  mistaken for high-powered formal reliability modeling.
- The Reliability feature selector now prioritizes features with evaluable
  same-task reliability and repeated valid records in the current task scope.
- Selected-feature trajectory plots now show the task-scope diagnostic notice
  when the selected metric has no repeated valid same-task records.

The Reliability menu should still show diagnostic notices for genuinely sparse
tasks. A task-specific notice is a design limitation, not a GUI failure.

