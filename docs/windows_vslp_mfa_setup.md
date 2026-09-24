# Windows application and MFA environments

VSLP and MFA run in separate Python environments. `vslp-app` contains the GUI,
scientific pipeline, pandas, and the declared runtime dependencies. The already
installed `vslp-mfa-334` contains MFA 3.3.4 and registered `english_us_arpa`
resources. VSLP launches MFA through `conda run`; no MFA Python import or PATH
activation is required in the application process.

## One-time setup and normal launch

1. Double-click `setup_vslp_windows.cmd` in the repository root. It creates or
   updates only `vslp-app`, installs the repository, verifies application imports,
   and checks the existing MFA environment and resources.
2. Double-click `Start_VSLP.cmd` for subsequent launches. GUI stderr is retained
   in `%LOCALAPPDATA%\VSLP\logs\last-launch.log`.

Both scripts resolve Conda from `CONDA_EXE`, PATH, then common user-local
Miniconda paths. Neither installs MFA, models, or dictionaries. The module entry
point is `python -m vslp.gui.acoustic_app`; the installed `vslp-acoustic-gui`
console script calls the same startup function.

The Alignment GUI preselects the packaged engineering profile. **Check MFA
Environment** detects the external Conda environment, exact MFA version,
registered acoustic model and dictionary, and required commands. **Run MFA
Self-Test** generates local nonclinical Windows speech or accepts a selected
nonclinical WAV, constructs the corpus, invokes real `mfa validate` and corpus
`mfa align`, parses and freezes ephemeral Alignment tables, then removes the
temporary corpus on success. This is an engineering health check, not an
assessment of clinical boundary accuracy.

The same test is available from `vslp-app` with
`python -m vslp.acoustic.alignment.self_test`. For automated opt-in validation,
set `VSLP_RUN_REAL_MFA=1` and run
`pytest tests/integration/test_mfa_cross_environment_opt_in.py`. Default tests
never download resources or require MFA.

The engineering profile uses MFA's registered model names. The provider records
their installed filesystem paths and SHA256 hashes when discoverable under
`MFA_ROOT_DIR` or the user's standard `Documents\MFA` directory. The selected
dictionary file must be locatable for the explicit preflight OOV check; an
unresolved lexicon fails preflight rather than skipping the OOV policy.
