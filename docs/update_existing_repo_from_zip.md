# Updating an Existing VSLP Checkout

This filename is retained for older links, but ZIP-overwrite updates are deprecated. Use Git so changes remain reviewable and recoverable.

## Standard Update

From the repository root:

```text
git status
git fetch origin
git pull --ff-only
```

Then refresh the environment:

```text
python -m pip install -e '.[gui,silero,kinematic,dev]'
vslp doctor
python -m pytest -q
```

## Local Changes Exist

Do not overwrite or delete them. Commit them to a branch or create a reviewed patch before updating. Avoid `git reset --hard` unless loss of local work is explicitly intended and independently backed up.

## Receiving a ZIP Handoff

Treat a ZIP as an import source, not as the working repository:

1. Extract it to a separate temporary directory.
2. Compare it with the Git checkout.
3. Copy only intended files into a branch.
4. Review `git diff`.
5. Run tests.
6. Commit and push through Git.

Never extract a ZIP over the active checkout.
