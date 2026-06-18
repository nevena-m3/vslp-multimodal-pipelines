# Feature GUI v0.133 Recommendations workstation

This patch turns the Recommendations screen into a decision workstation for
feature readiness.

The menu now includes:

- decision, family, and text filters;
- an action-plan table that explains what each recommendation category means;
- a priority-review queue that routes risky features back to the most relevant
  diagnostic menu;
- filtered recommendation, reason, family, and summary tables;
- a decision legend;
- an ML / export handoff summary.

Recommendation labels remain conservative review guidance, not automatic ML
feature selection. Final imputation, scaling, feature selection, model fitting,
and validation still belong inside the downstream ML pipeline.

