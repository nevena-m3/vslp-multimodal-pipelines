# Family 10 DDK contract history

The earlier `family10_ddk_reviewed_v1` calculator used the generic `ddk_rate_syll_s` and `ddk_cycle_mad_s` IDs, a five-event minimum, and a reviewed analysis-window denominator. Those IDs and formulas are retained only in legacy code/tests and are **not selectable production F10 outputs**.

The current frozen downstream feature definition is [family10_scientific_review.md](family10_scientific_review.md): eight target-specific leaves, exactly the first ten accepted reviewed cycles or complete `pa-ta-ka` sequences per trial, onset-to-offset rate denominators, and per-trial temporal variability. F10 now requires the explicit `final_ddk_events.csv` reviewed-event product. The upstream detector and event writer are not production-frozen.
