# Reviewed Trial Confirmation Contract V1

The existing Alignment trial-confirmation action writes observed original-time bounds through `confirmed_trial_rows()`. Each row carries `trial_id`, `sequence_id`, `task_id`, `prompt_id`, `start_sec`, `end_sec`, `trial_status=READY_FOR_ALIGNMENT`, `trial_review_source=HUMAN`, `segmentation_review_hash`, timestamp, and `reviewed_trial_contract_version=REVIEWED_TRIAL_CONFIRMATION_V1`.

The hash is SHA-256 of the frozen Segmentation Review decisions bytes followed by the frozen intervals bytes. It identifies the review state against which the scientist confirmed the trial. Expected protocol repetition count does not create or remove rows.

`load_confirmed_trials()` rejects a missing or changed hash, nonhuman source, nonready status, invalid bounds or task identity, and any legacy row missing the V1 fields. Legacy rows receive `legacy_unverified_trial_confirmation` for feature paths requiring this contract; they are never silently upgraded. This contract is independent of MFA and does not freeze the unfinished Alignment review GUI.
