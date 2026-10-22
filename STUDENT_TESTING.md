# VSLP Alignment + Acoustic student testing (RC1)

## Set up on Windows

```powershell
git clone https://github.com/nevena-m3/vslp-multimodal-pipelines.git
cd vslp-multimodal-pipelines
git checkout student/alignment-acoustic-v2-rc1
py -3.11 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install -e ".[gui,silero,dev]"
```

Python 3.11 is required. The Alignment application additionally needs a separate
Conda environment with Montreal Forced Aligner 3.3.4 and the approved matching
`english_us_arpa` acoustic model and pronunciation dictionary. The GUI's
preflight checks the environment, models, and transcript vocabulary before MFA
runs. See [the MFA profile guide](docs/acoustic_alignment_contract.md) for model
setup; models and participant recordings are not stored in Git.

## Run the two applications

From the activated repository environment, run each command in its own shell:

```powershell
vslp-alignment-gui
vslp-acoustic-gui
```

Equivalent commands are `python -m vslp.gui.alignment_app` and
`python -m vslp.gui.acoustic_app`.

1. In **VSLP Alignment**, add WAVs, choose Bamboo Passage or a supported
   sentence, check the displayed canonical transcript and repetitions, create
   the job, and run preflight/MFA. Review flagged words and phones. Finalize
   accepted recordings, then export **VSLP Alignment Package V1**. Genuine
   mismatches remain unfinalized; a partial package is valid.
2. In **VSLP Acoustic Features**, create or open a project, process and freeze
   segmentation review, and optionally import that Alignment Package. Select
   leaves in F1–F13 and run extraction. Leaves without a satisfied prerequisite
   appear as unavailable with a reason; they do not prevent independent leaves
   from computing. Keep **Resume existing run** enabled to reuse valid completed
   recording/family checkpoints after interruption.
3. Use the Acoustic **Results** inspector for per-recording computed,
   unavailable, warning, and error statuses.

Alignment jobs and packages are written under the workspace selected in the
Alignment app. Acoustic project output contains `acoustic/005_features/tables/`
for results and `acoustic/005_features/checkpoints_v1/` for resumable work. Keep
the original WAVs and package available when reopening a project.

On some Windows/Python installations, CPP multiprocessing may print a
`faulthandler` access-violation diagnostic while the extraction still completes
correctly. If a run fails, report the app and step, recording name, task,
selected feature IDs, concise on-screen error, Technical Log, package status,
and the relevant `stage_manifest.json` and feature status CSV. Do not include
participant audio in a public issue.
