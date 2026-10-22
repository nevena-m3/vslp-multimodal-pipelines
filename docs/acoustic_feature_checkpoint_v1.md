# Acoustic extraction checkpoint V1

The Acoustic GUI enables **Resume existing run** by default. The CLI can opt in
with `FeatureExtractionConfig(resume_existing_run=True)`. The legacy batch path
remains available for comparison. This is an extraction result cache; it does
not cache CPP tracks or change any feature algorithm.

## Unit and commit marker

The unit is one recording and one executor group (F01/F02 share `voice`; other
active families have one group each). Each unit writes its original family
tables and native audit artifacts beneath `acoustic/005_features/checkpoints_v1/chunks/`.
Only the corresponding JSON checkpoint under `checkpoints_v1/<group>/` marks
the unit complete. It has a schema name, identity key, `COMPLETE` or `FAILED`
state, selected output rows, artifact paths and SHA-256 hashes, provenance, and
a checksum of the checkpoint document. The JSON is written to a temporary file,
flushed and `fsync`ed, then atomically replaced. A process crash before that
replace cannot create a valid complete checkpoint. Failed units retain a
traceback and retry flag; no failure is zero-filled.

At scan time, absent, failed, stale, corrupt, and valid-complete checkpoints are
counted separately. A valid-complete checkpoint must match the current key and
checksum and all referenced artifact hashes. Only valid-complete units are
skipped. Failed, stale, corrupt, or incomplete units are recomputed. A cancelled
run stops before the next unit and retains already completed checkpoints.

## Identity and invalidation

The key includes recording ID; decoded source and analysis audio identity;
task/prompt/repetition fields; per-record reviewed decision and interval rows;
per-record trial choices or DDK events where relevant; per-record canonical
alignment trials, words, and phones where relevant; selected IDs and registry
parameters; relevant external input hashes; executor implementation source
hashes; and numerical dependency versions. Full-file SHA-256 of source and
analysis audio is retained in provenance but does not replace decoded-content
identity. Changing one recording's reviewed intervals invalidates that
recording's units. Changing one family implementation invalidates that family
while unrelated family units remain reusable.

The embedded or imported Final Alignment provider remains authoritative. The
checkpoint layer does not parse raw TextGrid. A provider-level rejection may
invalidate all alignment-dependent units; per-record canonical identities are
used when the provider loads successfully. This is a current limitation of the
provider contract, not a relaxation of alignment freshness.

The assembled status table reports per-record review hashes for scoped runs.
The original family chunk audits retain the whole-file review hashes recorded
when that chunk was computed. This keeps unaffected recording rows stable when
another recording's review changes while preserving the original input-file
provenance in the native audit.

## Assembly and memory

Family output rows and native audits are written after each unit. The final
status and wide result CSVs are assembled one recording at a time from verified
checkpoints, ordered by recording ID and feature ID. The canonical handoff CSVs
are copied from the assembled files. The delivery artifact catalog and GUI
post-run status counts are streamed. Opening the detailed result table still
loads it for interactive review. Per-record metadata and
the frozen segmentation table still reside in memory; this implementation
bounds **result and audit row** retention, rather than claiming constant total
memory for every upstream reader. Family algorithms may still load one full
recording at a time.

## Validation scope

Integration tests cover skip-all reruns, interruption after 10 of 20 units,
resume versus fresh row parity, corruption, one-record review staleness,
family-version invalidation, per-record failure/retry, and F08/F09 numeric/QC
parity with the legacy dispatcher. A separate engineering benchmark runs
10/25/50/100 synthetic short-audio entries against a trivial executor to
measure orchestration overhead and memory; it does not validate full F01-F13
batch runtime. Windows multiprocessing diagnostics and a full heterogeneous
hundreds-recording scientific workload remain separate validation items.
