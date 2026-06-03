# Updating your existing local VSLP repo from a new ZIP

Use **one permanent local folder** for the repo. Do not keep making `v1`, `v2`, `v3` working folders.

Recommended permanent folder:

```bash
~/Projects/vslp-multimodal-pipelines
```

Temporary downloaded ZIPs can stay in `~/Downloads`.

## One-time cleanup

If your real Git repo is already here:

```bash
~/Projects/vslp-multimodal-pipelines
```

keep using that folder.

If your repo is somewhere else, replace the path below with your actual path.

## Standard update procedure

Assume the new ZIP is in Downloads and unzipped as:

```bash
~/Downloads/vslp-multimodal-pipelines-v0.3
```

and your real repo is:

```bash
~/Projects/vslp-multimodal-pipelines
```

Run:

```bash
cd ~/Projects/vslp-multimodal-pipelines

git status
```

If Git says you have changes, save them first:

```bash
git add .
git commit -m "Save local work before VSLP update"
```

Now copy the new version into the existing repo while preserving `.git`:

```bash
rsync -av --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'examples/test_runs' \
  --exclude 'examples/test_data/audio_inputs/*' \
  ~/Downloads/vslp-multimodal-pipelines-v0.3/ \
  ~/Projects/vslp-multimodal-pipelines/
```

Then commit the update:

```bash
cd ~/Projects/vslp-multimodal-pipelines

git status
git add .
git commit -m "Update VSLP pipeline scaffold"
git push
```

## Reinstall after every code update

Activate the environment and reinstall editable package:

```bash
cd ~/Projects/vslp-multimodal-pipelines
source .venv/bin/activate
pip install -e '.[silero]'
```

## Check setup

```bash
vslp doctor
```

If you only want to check non-Silero ingest/preprocess dependencies:

```bash
vslp doctor --no-include-silero
```
