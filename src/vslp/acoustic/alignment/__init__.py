"""Alignment APIs are loaded only when a linguistic feature needs them.

Importing nonlinguistic trial helpers must not import the MFA provider.
"""

__all__ = [
    "AlignmentConfig", "AlignmentStore", "MFAProvider", "PromptManifest",
    "freeze_alignment", "list_alignment_runs", "load_final_alignment",
    "run_acoustic_alignment",
]


def __getattr__(name: str):
    if name in __all__:
        from . import stage

        return getattr(stage, name)
    raise AttributeError(name)
